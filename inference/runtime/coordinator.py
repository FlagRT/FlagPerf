# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Orchestrate sealed inputs and isolated workers, never device state in this process."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
import traceback
import yaml
from runtime.common import ROOT, Progress, execute, read_json, write_json, digest, seal_sources, assert_source_snapshot
from runtime.config import parser, load
from vendors.device import environment


def output_directory(args, cfg):
    root = Path(args.output).resolve() if args.output else Path(cfg['runtime']['output_root']) / (time.strftime('%Y%m%d-%H%M%S') + f'-{args.command}-{os.getpid()}')
    source = cfg.get('preview',{}).get('resume_from') if args.command == 'preview' else None
    if source and (root.is_relative_to(Path(source).resolve()) or Path(source).resolve().is_relative_to(root)):
        raise ValueError('resume output and source directories must not overlap')
    if args.internal_existing_root:
        if not (root/'host-launch.json').is_file():
            raise ValueError('internal existing root requires a host launch manifest')
        if (root/'result.json').exists():
            raise ValueError('output already contains results')
    else:
        root.mkdir(parents=True,exist_ok=False)
    return root


def invoke_worker(cfg, root, action, destination, prepared=None, include=None, probe=False, timeout=None, repeat_index=None):
    assert_source_snapshot(cfg.get('_source_snapshot'))
    if '_stack' in cfg and action not in ['prepare','export']:
        from copy import deepcopy
        cfg = deepcopy(cfg)
        cfg['_stack']['flaggems'] = 'on' if include is not None else 'off'
    destination.mkdir(parents=True,exist_ok=True)
    worker_config = destination/'worker.yaml'
    worker_config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    argv = [sys.executable,'-u','-m','runtime.worker',action,'--config',str(worker_config),'--output',str(destination)]
    if prepared is not None: argv += ['--prepared',str(prepared)]
    if include is not None: argv += ['--include',json.dumps(include)]
    if probe: argv += ['--probe']
    if repeat_index is not None: argv += ['--repeat-index',str(repeat_index)]
    env = environment(cfg)
    if '_stack' in cfg:
        from vendors.stack import worker_environment
        selected = worker_environment(cfg,root)
        selected.update({k:v for k,v in env.items() if k not in ['PYTHONPATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR']})
        env = selected
    tp = cfg['runtime'].get('parallelism') == 'tp' and action != 'prepare'
    if tp:
        argv = [sys.executable,'-u','-m','torch.distributed.run','--standalone',
                '--nnodes=1',f'--nproc-per-node={len(cfg["runtime"]["devices"])}',
                '--max-restarts=0','--module','runtime.worker'] + argv[4:]
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH','')
    process = execute(argv,destination,timeout or cfg['runtime']['timeout_seconds'],env)
    assert_source_snapshot(cfg.get('_source_snapshot'))
    if tp:
        from runtime.tp import aggregate
        aggregate(cfg,destination,action,process)
    path = destination/'result.json'
    result = read_json(path) if path.exists() else {'status':'failed','error':'worker terminated without result'}
    if process['exit_code'] != 0:
        result['status'] = 'failed'
    result['timed_out'] = process['timed_out']
    result['exit_code'] = process['exit_code']
    result['worker_wall_seconds'] = process.get('wall_seconds_diagnostic_only',0)
    result['cleanup_seconds'] = process.get('cleanup_seconds',0)
    if process['timed_out']: result['error'] = 'worker timeout'
    return result


def records(directory):
    import torch
    state = read_json(directory/'result.json')
    output = {}
    for name in state['artifacts']:
        batch = torch.load(directory/name,map_location='cpu',weights_only=True)
        for i, sample in enumerate(batch['ids']):
            if sample in output: raise ValueError('duplicate sample in outputs')
            output[sample] = {}
            for key, tensor in batch['outputs'].items():
                value = tensor[i]
                if key not in ['pooled','embedding']: value = value[batch['mask'][i].bool()]
                output[sample][key] = value.float().numpy()
    return output


def verified_policy(cfg, root, prepared, mode):
    policy = None
    if cfg[mode]['flaggems'] in ['on','both']:
        policy = yaml.safe_load(Path(cfg['policy']['path']).read_text())
        key = read_json(prepared/'identity-key.json')['key']
        if policy.get('schema_version') != 3 or policy.get('status') != 'verified' or not policy.get('include'):
            raise ValueError('policy requires schema 3 and a verified nonempty FlagGems selection; run a fresh preview')
        if policy.get('identity_key') != key:
            raise ValueError('policy identity mismatch: run preview again in the current environment/configuration')
        if policy.get('verified_selection_sha256') != digest(policy['include']):
            raise ValueError('policy selection changed after verification: run preview again')
        if not isinstance(policy['include'],list) or not all(isinstance(x,str) for x in policy['include']):
            raise ValueError('policy include must be a list of function names')
        (root/'policy.yaml').write_text(yaml.safe_dump(policy,allow_unicode=True,sort_keys=False))
    return policy


def accuracy_diff(cfg, root, prepared, paths, contexts=None):
    if contexts:
        from runtime.stack import assert_observation
        for side in ['off','on']: assert_observation(contexts[side],paths[side])
    if not contexts and paths['off']['identity_key'] != paths['on']['identity_key']:
        raise ValueError('off/on identities differ')
    from analysis.metrics import compare_records
    if cfg['runtime'].get('parallelism') != 'tp':
        return compare_records(records(root/'off'),records(root/'on'),
                               cfg['accuracy']['levels'],cfg['accuracy']['worst_samples'])
    rank_diffs = {}
    expected_ids = {sample['id'] for sample in read_json(prepared/'samples.json')}
    for rank in range(len(cfg['runtime']['devices'])):
        key = str(rank)
        if not contexts and paths['off']['ranks'][key]['identity_key'] != paths['on']['ranks'][key]['identity_key']:
            raise ValueError(f'rank {rank} off/on identities differ')
        off = records(root/'off'/f'rank-{rank}')
        on = records(root/'on'/f'rank-{rank}')
        if set(off) != expected_ids or set(on) != expected_ids:
            raise ValueError(f'rank {rank} outputs do not cover all prepared sample IDs')
        rank_diffs[key] = compare_records(off,on,cfg['accuracy']['levels'],cfg['accuracy']['worst_samples'])
    return {'parallelism':'tp','devices':cfg['runtime']['devices'],
            'comparison_scope':'same_rank_off_on; each rank processes the same logical samples',
            'ranks':rank_diffs}


def accuracy(cfg, root, prepared, progress, contexts=None):
    policy = None if contexts else verified_policy(cfg,root,prepared,'accuracy')
    result = {'status':'completed','paths':{},'comparison':None}
    tp = cfg['runtime'].get('parallelism') == 'tp'
    if tp:
        result.update(parallelism='tp',devices=cfg['runtime']['devices'])
    sides = list(contexts) if contexts else (['off','on'] if cfg['accuracy']['flaggems']=='both' else [cfg['accuracy']['flaggems']])
    for side in sides:
        progress.phase = f'accuracy/{side}'
        ctx = contexts[side] if contexts else {'cfg':cfg,'prepared':prepared,'include':policy['include'] if side=='on' else None}
        value = invoke_worker(ctx['cfg'],root,'forward',root/side,ctx['prepared'],ctx['include'])
        result['paths'][side] = value
        if value['status'] != 'completed':
            result.update(status='failed',error=f'{side}: {value.get("error","worker failed")}')
            break
        if contexts:
            from runtime.stack import assert_observation
            assert_observation(ctx,value)
        if contexts and value.get('numerical_anomalies'):
            result.update(status='failed',error=f'{side}: nonfinite output; see numerical_anomalies')
            break
        if contexts and ctx['include'] is not None:
            from runtime.stack import fully_hit
            if not fully_hit(value,ctx['include']):
                result.update(status='partial',error=f'{side}: selected functions were not observed on every rank')
        if ctx['include'] is not None:
            if tp:
                missing = [rank for rank, calls in value['route']['rank_function_calls'].items()
                           if not all(calls.get(name,0)>0 for name in ctx['include'])]
                if missing:
                    result.update(status='partial',error=f'on has no call for every selected FlagGems function on ranks {missing}')
            elif not value.get('route',{}).get('observed'):
                result.update(status='partial',error='on completed but no actual FlagGems call observed')
    if sides==['off','on'] and all(result['paths'].get(s,{}).get('status')=='completed' for s in sides):
        try:
            diff = accuracy_diff(cfg,root,prepared,result['paths'],contexts)
        except Exception as error:
            result.update(status='failed',error=f'off/on comparison: {error}',exception_type=type(error).__name__)
            return result
        comparisons = diff['ranks'].values() if tp else [diff]
        anomalies = sum(len(item['anomalies']) for item in comparisons)
        invalid_shape = any(row['status']=='shape_mismatch' for item in comparisons for row in item['samples'])
        if contexts:
            axis = next((k for k in ['flaggems','flagtree','flagcx'] if cfg['accuracy'].get(k)=='both'), 'flaggems')
            for comparison in comparisons:
                comparison['reference'] = axis+' off (not an absolute oracle)'
            diff['comparison_component'] = axis
        write_json(root/'comparison.json',diff)
        result['comparison'] = 'comparison.json'
        result['numerical_anomalies'] = anomalies
        result['pairing_status'] = 'invalid_shape' if invalid_shape else 'valid'
        if result['pairing_status'] != 'valid':
            result.update(status='failed',error='off/on output shapes differ; see comparison anomalies')
    return result


def route_summary(native, on, policy):
    """Counts top-level ATen calls in an untimed pass, not hardware kernels."""
    known = {}
    # The native worker already sealed this mapping. Importing FlagGems here
    # would initialize a device backend in the device-free coordinator.
    for candidate in native['candidates']:
        for key in candidate['aten_keys']:
            known.setdefault(key,set()).add(candidate['function'])
    excluded = {item['function'] for item in policy.get('excluded',[])}
    unknown = {item['function'] for item in policy.get('unknown',[])}
    hits = on['route']['actual_function_calls']
    counts = {key:0 for key in ('excluded','unverified','uncovered','allowed_function_observed','ambiguous')}
    details = []
    for key,row in sorted(on['inventory'].items()):
        functions = known.get(key,set())
        if len(functions)>1:
            category = 'ambiguous'
        elif not functions:
            category = 'uncovered'
        else:
            function = next(iter(functions))
            if function in excluded: category = 'excluded'
            elif function in unknown: category = 'unverified'
            elif function in policy['include'] and hits.get(function,0)>0:
                category = 'allowed_function_observed'
            else: category = 'ambiguous'
        counts[category] += row['calls']
        details.append({'aten_key':key,'calls':row['calls'],'category':category,
                        'candidate_functions':sorted(functions)})
    total = sum(counts.values())
    policy_native = sum(counts[k] for k in ('excluded','unverified','uncovered'))
    return {'scope':'untimed on path; top-level device ATen calls; candidate map from untimed native path',
            'counts':counts,'observed_top_level_aten_calls':total,
            'policy_native_call_ratio':policy_native/total if total else None,
            'policy_native_ratio_definition':'(excluded + unverified + uncovered) / observed_top_level_aten_calls',
            'hardware_kernel_fallback_ratio':None,'hardware_kernel_fallback_status':'not_collected',
            'allowed_observed_caveat':'function hit is known, but every call of that key is not individually attributed',
            'details':details}


def performance(cfg, root, prepared, progress, contexts=None):
    from runtime.performance import summarize
    policy = None if contexts else verified_policy(cfg,root,prepared,'performance')
    selection = ('both' if len(contexts)==2 else next(iter(contexts))) if contexts else cfg['performance']['flaggems']
    sides = ['off','on'] if selection=='both' else [selection]
    result = {'status':'completed','level':'total','paths':{},'summary':{},'comparison':None}
    tp = cfg['runtime'].get('parallelism') == 'tp'
    if tp:
        result.update(parallelism='tp',devices=cfg['runtime']['devices'],profiles={})
    if contexts:
        from runtime.stack import audits
        audit_results,summaries,error = audits(root,contexts,invoke_worker,progress)
        result.update(component_audits=audit_results,route_summaries=summaries)
        if error:
            result.update(status='failed',error=error)
            return result
    elif 'on' in sides:
        audits = {}
        for side in ['off','on']:
            progress.phase = f'performance/route-audit/{side}'
            value = invoke_worker(cfg,root,'forward',root/'audit'/side,prepared,
                                  policy['include'] if side=='on' else None,True)
            audits[side] = value
            if value['status']!='completed' or value.get('numerical_anomalies'):
                result.update(status='failed',error=f'{side} audit failed: {value.get("error") or value.get("numerical_anomalies")}')
                result['audits'] = audits
                return result
        result['audits'] = audits
        if audits['off']['identity_key'] != audits['on']['identity_key']:
            raise ValueError('route audit identities differ')
        if not audits['on']['route']['observed']:
            result.update(status='failed',error='on audit did not observe FlagGems function calls')
            return result
        if not all(audits['on']['route']['actual_function_calls'].get(name,0)>0 for name in policy['include']):
            result.update(status='failed',error='on audit did not hit every selected FlagGems function')
            return result
        if tp and not all(calls.get(name,0)>0 for calls in audits['on']['route']['rank_function_calls'].values() for name in policy['include']):
            result.update(status='failed',error='not every rank hit the selected FlagGems functions')
            return result
        result['route_summary'] = route_summary(audits['off'],audits['on'],policy)
        write_json(root/'route-summary.json',result['route_summary'])
    for repeat in range(cfg['performance']['repeats']):
        order = sides if repeat%2==0 else list(reversed(sides))
        for side in order:
            progress.phase = f'performance/{side}/repeat-{repeat}'
            label = f'repeat-{repeat:02}'
            ctx = contexts[side] if contexts else {'cfg':cfg,'prepared':prepared,'include':policy['include'] if side=='on' else None}
            value = invoke_worker(ctx['cfg'],root,'performance',root/side/label,ctx['prepared'],
                                  ctx['include'],repeat_index=repeat)
            result['paths'].setdefault(side,{})[label] = value
            if value['status']!='completed':
                result.update(status='failed',error=f'{side}/{label}: {value.get("error","worker failed")}')
                return result
            if contexts:
                from runtime.stack import assert_observation
                assert_observation(ctx,value)
            if ctx['include'] is not None:
                registered = {name for _,name in value['route']['registered']}
                if not set(ctx['include']) <= registered:
                    result.update(status='failed',error='timed worker registration differs from selected policy')
                    return result
                if tp and any(not set(ctx['include']) <= {n for _,n in rank['route']['registered']} for rank in value['ranks'].values()):
                    result.update(status='failed',error='a TP rank did not register the selected policy')
                    return result
    for side in sides:
        runs = result['paths'][side]
        if len(runs)!=cfg['performance']['repeats']:
            result.update(status='failed',error=f'{side} has incomplete repeats')
            return result
        if len({item['identity_key'] for item in runs.values()})!=1:
            result.update(status='failed',error=f'{side} worker identities differ')
            return result
        rows = [row for value in runs.values() for row in value['batches']]
        result['summary'][side] = summarize(rows)
    if selection=='both':
        off = result['summary']['off']
        on = result['summary']['on']
        if not contexts and result['paths']['off']['repeat-00']['identity_key'] != result['paths']['on']['repeat-00']['identity_key']:
            raise ValueError('off/on performance identities differ')
        if off['batches'] != on['batches'] or off['samples']!=on['samples'] or off['tokens']!=on['tokens']:
            raise ValueError('off/on measurement workloads differ')
        result['comparison'] = {'off_over_on_measured_time':off['measured_seconds']/on['measured_seconds'],
                                'on_over_off_samples_per_second':on['samples_per_second']/off['samples_per_second']}
    if tp:
        for repeat in range(cfg['performance']['repeats']):
            for side in (sides if repeat%2==0 else list(reversed(sides))):
                label = f'repeat-{repeat:02}'
                progress.phase = f'communication/{side}/{label}'
                ctx = contexts[side] if contexts else {'cfg':cfg,'prepared':prepared,'include':policy['include'] if side=='on' else None}
                value = invoke_worker(ctx['cfg'],root,'profile',root/'profiles'/side/label,ctx['prepared'],
                                      ctx['include'],repeat_index=repeat)
                result['profiles'].setdefault(side,{})[label] = value
                if value['status'] != 'completed':
                    result.update(status='partial',error=f'communication {side}/{label}: {value.get("error","incomplete")}')
    return result


def run_export(cfg,root,progress):
    cfg = dict(cfg, _stack={'flaggems':'off','flagtree':'off','flagcx':'off'})
    prepared = root/'prepared'
    state = invoke_worker(cfg,root,'prepare',prepared)
    if state['status'] != 'completed': raise RuntimeError('prepare: '+state.get('error','failed'))
    value = invoke_worker(cfg,root,'export',root/'export',prepared)
    return {'status':value['status'],'export':value}


def run(args, cfg):
    root = output_directory(args,cfg)
    (root/'effective.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    cfg['_source_snapshot'] = seal_sources(root)
    result = {'status':'failed','command':args.command}
    if cfg['runtime'].get('parallelism') == 'tp':
        result.update(parallelism='tp',devices=cfg['runtime']['devices'])
    try:
        with Progress() as progress:
            from runtime.stack import run_stack
            if args.command != 'export':
                result = run_stack(args.command,cfg,root,progress,invoke_worker)
            else:
                result = run_export(cfg,root,progress)
            # Execution is complete; common reporting remains below.
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
    finally:
        result['command'] = args.command
        result['execution_status'] = result.get('status')
        result['analysis_key'] = cfg['_source_snapshot']['analysis_key']
        try:
            assert_source_snapshot(cfg['_source_snapshot'])
            from analysis.assessment import assess
            result['assessment'] = assess(root,result)
            result['analysis_status'] = 'completed'
            write_json(root/'result.json',result)
            from reporting.model import render
            render(root,result)
            assert_source_snapshot(cfg['_source_snapshot'])
        except Exception as error:
            result.update(status='failed', analysis_status='failed', error=str(error))
            policy_path = root/'preview/policy.yaml'
            if policy_path.is_file():
                policy = yaml.safe_load(policy_path.read_text())
                policy.update(status='unverified', verified_selection_sha256=None)
                policy_path.write_text(yaml.safe_dump(policy,sort_keys=False,allow_unicode=True))
                result['verified'] = False
        write_json(root/'result.json',result)
        print(f'Results: {root}',flush=True)
    return 0 if result['status']=='completed' else 1


def main():
    args = parser().parse_args()
    try:
        cfg = load(args)
        if args.command == 'list':
            print((ROOT/'support.json').read_text())
            return 0
        return run(args,cfg)
    except (ValueError,OSError) as error:
        print(f'Configuration error: {error}',file=sys.stderr)
        return 2
