"""Stepped tensor capture and complete finite checks for model-owned execution sessions."""
from contextlib import nullcontext
from pathlib import Path

from runtime.common import write_json, file_hash, read_json
from runtime.workload import sample_view, tensor_tree


def capture(adapter, model, batches, cfg, device, root, inventory, route, scopes, probe,
            evidence_mode, stages):
    import torch
    root = Path(root)
    names = ['all'] if probe else cfg['accuracy']['layers']
    specs = {s.name:s for s in adapter.boundary_specs(cfg, model, names)}
    modules = adapter.selected_modules(model, names) if (probe and not cfg.get('_layer_audit')) or 'layer' in cfg['accuracy']['levels'] else {}
    if cfg.get('_layer_audit'):
        modules = {}
    captured, handles = {}, []
    errors, checks, artifacts, requests = [], [], [], []
    modes = ['performance'] if cfg.get('_layer_audit') else ['accuracy','performance'] if probe else ['accuracy']
    collect = bool(probe and (not cfg.get('_preview_trial') or cfg['_preview_trial']['phase'] == 'baseline'))
    measure = stages.measure if stages else lambda name:nullcontext()
    for name, module in modules.items():
        def hook(_m, _a, out, name=name):
            if name in captured:
                raise ValueError('repeated boundary invocation in one execution step: '+name)
            captured[name] = adapter.module_value(out, specs[name]).detach().cpu()
        handles.append(module.register_forward_hook(hook))
    def checkpoint(complete=False):
        write_json(root/'execution.json', {'schema_version':1, 'complete':complete, 'modes':modes, 'requests':requests})
        write_json(root/'forward-checks.json', {'schema_version':2, 'complete':complete, 'mode':evidence_mode,
                                               'steps':checks, 'execution_sha256':file_hash(root/'execution.json')})
    try:
        for mode in modes:
            for batch_index, batch in enumerate(batches):
                inputs = {k:v.to(device) for k,v in batch['inputs'].items()}
                request = {'batch_index':batch_index, 'mode':mode, 'sample_ids':batch['ids'],
                           'input_shape':adapter.batch_metadata(batch)['input_shape'], 'steps':[], 'samples':{}}
                requests.append(request)
                trajectories = {s:[] for s in batch['ids']}
                with adapter.open_execution(model, inputs, mode) as session:
                    while not session.done:
                        captured.clear()
                        context = dict(session.context)
                        if scopes:
                            scopes.begin({'batch_index':batch_index, **context})
                        try:
                            with measure('forward_and_capture'), torch.inference_mode(), inventory if collect else nullcontext():
                                step = session.advance()
                        finally:
                            if scopes:
                                scopes.active = False
                        with measure('finite_checks'):
                            outputs = tensor_tree(step.outputs, lambda t:t.detach().cpu())
                            values = {**outputs, **captured}
                            adapter.validate_outputs(outputs)
                            regions = tensor_tree(step.regions, lambda t:t.detach().cpu())
                            active = [sid for sid,alive in zip(batch['ids'],context['active_samples']) if alive]
                            wanted = set(step.outputs)|set(modules)
                            if set(values) != wanted:
                                raise ValueError('missing execution-step boundary')
                            selected_tokens = step.token_ids.cpu().tolist()
                            if cfg['runtime'].get('parallelism')=='tp':
                                from runtime.tp import check_execution_step
                                check_execution_step(context,selected_tokens)
                            step_checks = []
                            for i, sid in enumerate(batch['ids']):
                                if sid not in active:
                                    continue
                                trajectories[sid].append(selected_tokens[i])
                                for name,tensor in values.items():
                                    row = sample_view(tensor,i,specs[name],regions)
                                    finite = torch.isfinite(row)
                                    count = int((~finite).sum())
                                    key = {'mode':mode,'batch_index':batch_index,'phase':context['phase'],
                                           'generation_step':context['generation_step'],'sample_id':sid,'boundary':name}
                                    if count:
                                        errors.append({**key,'kind':'nonfinite'})
                                    step_checks.append({**key,'shape':list(row.shape),'dtype':str(row.dtype),
                                                        'checked_elements':row.numel(),'nonfinite_count':count})
                        entry = {**context,'token_ids':selected_tokens,'active_sample_ids':active,
                                 'boundaries':sorted(values)}
                        if evidence_mode != 'lightweight':
                            name = f'batch-{batch_index:04}-{mode}-step-{context["generation_step"]:04}.pt'
                            torch.save({'ids':batch['ids'],'outputs':values,'valid_regions':regions,
                                        'boundary_specs':{n:specs[n].metadata() for n in values},
                                        'execution':entry},root/name)
                            entry['artifact'] = name
                            artifacts.append(name)
                        request['steps'].append(entry)
                        checks.append({'mode':mode,'batch_index':batch_index,**context,
                                       'active_sample_ids':active,'boundaries':sorted(values),'checks':step_checks})
                        with measure('evidence_write'):
                            checkpoint()
                    request['cache'] = session.cache_snapshot()
                    request['samples'] = {sid:{'tokens':trajectories[sid],'stop_reason':reason}
                                           for sid,reason in zip(batch['ids'],session.stop_reasons)}
                checkpoint()
        checkpoint(True)
        return {'execution_kind':'autoregressive','execution':'execution.json','execution_modes':modes,
                'numerical_anomalies':errors,'artifacts':artifacts,'inventory':inventory.ops,
                'inventory_collected':collect, 'forward_evidence':{'mode':evidence_mode,'manifest':'forward-checks.json',
                                                               'sha256':file_hash(root/'forward-checks.json')},
                **({'layer_routes':scopes.evidence()} if scopes else {}),
                'route':route.evidence() if route else {'requested':False,'observed':False,'actual_function_calls':{}}}
    finally:
        for handle in handles:
            handle.remove()


def validate_execution(execution, expected):
    """The same trajectory contract is used by checks and offline reference packages."""
    if (execution.get('schema_version')!=1 or execution.get('complete') is not True
            or execution.get('modes') not in (['accuracy'],['performance'],['accuracy','performance'])):
        raise ValueError('incomplete or unsupported generation execution')
    maximum = expected['generation']['max_new_tokens']
    eos=set(expected['generation']['eos_token_ids'])
    vocabulary=expected['generation']['vocab_size']
    wanted_requests = {(mode,b['batch_index']):b for mode in execution['modes'] for b in expected['batches']}
    seen_requests, wanted_steps = set(), {}
    for request in execution['requests']:
        key = (request['mode'],request['batch_index'])
        if key not in wanted_requests or key in seen_requests:
            raise ValueError('unexpected/duplicate execution request')
        seen_requests.add(key)
        reference = wanted_requests[key]
        if any(request[n] != reference[n] for n in ('sample_ids','input_shape')):
            raise ValueError('execution fixed inputs mismatch')
        steps = request['steps']
        if not steps or len(steps)>maximum or (request['mode']=='performance' and len(steps)!=maximum):
            raise ValueError('execution step coverage mismatch')
        trajectories = {sid:[] for sid in request['sample_ids']}
        alive=[True]*len(request['sample_ids'])
        for index, step in enumerate(steps):
            if (step['generation_step'] != index or step['phase'] != ('prefill' if index == 0 else 'decode')
                    or step.get('mode')!=request['mode'] or step.get('active_samples')!=alive
                    or any(type(a) is not bool for a in step['active_samples'])
                    or step.get('input_shape')!=[len(alive),reference['input_shape'][1] if index==0 else 1]
                    or step.get('kv_length_before')!=(0 if index==0 else reference['input_shape'][1]+index-1)
                    or len(step['token_ids']) != len(request['sample_ids'])
                    or any(type(t) is not int or not 0<=t<vocabulary for t in step['token_ids'])):
                raise ValueError('invalid execution step order/tokens')
            active = [sid for sid,alive in zip(request['sample_ids'],step['active_samples']) if alive]
            if step['active_sample_ids'] != active or not active:
                raise ValueError('invalid active sample transition')
            if request['mode']=='performance' and active != request['sample_ids']:
                raise ValueError('fixed-length performance stopped early')
            if set(step['boundaries']) != set(expected['boundaries']):
                raise ValueError('missing execution boundary coverage')
            for sid,t in zip(request['sample_ids'],step['token_ids']):
                if sid in active: trajectories[sid].append(t)
            if request['mode']=='accuracy':
                alive=[a and t not in eos for a,t in zip(alive,step['token_ids'])]
            wanted_steps[(*key,index)] = (step,reference)
        if request['mode']=='accuracy' and len(steps)<maximum and any(alive):
            raise ValueError('generation stopped before EOS or length limit')
        if set(request['samples']) != set(trajectories):
            raise ValueError('missing generated sample trajectory')
        for sid,tokens in trajectories.items():
            row=request['samples'][sid]
            if row['tokens']!=tokens or row['stop_reason'] not in ('eos','max_new_tokens'):
                raise ValueError('invalid generated trajectory/stop reason')
            if row['stop_reason']=='max_new_tokens' and len(tokens)!=maximum:
                raise ValueError('short generation marked as length-complete')
            if row['stop_reason'] != ('eos' if request['mode']=='accuracy' and tokens[-1] in eos else 'max_new_tokens'):
                raise ValueError('generation EOS/stop reason mismatch')
    if seen_requests != set(wanted_requests):
        raise ValueError('missing execution requests')
    return wanted_steps


def validate_checks(root, result, expected):
    """Verify every expected sample/step/boundary, including steps after divergence."""
    import math
    root = Path(root)
    descriptor = result['forward_evidence']
    if descriptor['manifest'] != 'forward-checks.json' or file_hash(root/descriptor['manifest']) != descriptor['sha256']:
        raise ValueError('execution checks digest mismatch')
    value = read_json(root/descriptor['manifest'])
    execution = read_json(root/'execution.json')
    if (value.get('schema_version') != 2 or value.get('complete') is not True
            or value['mode'] != descriptor['mode'] or value['mode'] not in ('full','lightweight')
            or value['execution_sha256'] != file_hash(root/'execution.json')):
        raise ValueError('incomplete execution checks')
    wanted_steps=validate_execution(execution,expected)
    if (execution['modes'] != result.get('execution_modes') or
            execution['modes'] != expected.get('execution_modes',expected['generation']['preview_modes'])):
        raise ValueError('execution mode summary mismatch')
    full_artifacts,anomalies=[],set()
    for step,_ in wanted_steps.values():
        if value['mode']=='full':
            name=step.get('artifact','')
            path=(root/name).resolve()
            if not name or not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError('missing full execution artifact')
            full_artifacts.append(name)
        elif 'artifact' in step:
            raise ValueError('lightweight execution unexpectedly contains tensors')
    seen_steps=set()
    for row in value['steps']:
        key=(row['mode'],row['batch_index'],row['generation_step'])
        if key not in wanted_steps or key in seen_steps:
            raise ValueError('unexpected/duplicate execution checks')
        seen_steps.add(key)
        step,reference=wanted_steps[key]
        if any(row.get(k)!=step.get(k) for k in ('phase','active_samples','active_sample_ids','boundaries','input_shape','kv_length_before')):
            raise ValueError('step check context differs from execution')
        wanted={(s,n) for s in step['active_sample_ids'] for n in expected['boundaries']}
        seen=set()
        lengths=dict(zip(reference['sample_ids'],reference['valid_tokens']))
        for check in row['checks']:
            pair=(check['sample_id'],check['boundary'])
            if pair not in wanted or pair in seen:
                raise ValueError('unexpected/duplicate step boundary check')
            seen.add(pair)
            spec=expected['boundary_specs'][pair[1]]
            shape=spec.get('phase_shapes',{}).get(step['phase'],spec['sample_shape'])
            if (not isinstance(check['shape'],list) or len(shape)!=len(check['shape'])
                    or any(type(d) is not int or d<1 for d in check['shape'])
                    or not isinstance(check.get('dtype'),str) or not check['dtype'].startswith('torch.')
                    or ('output_dtype' in expected and check['dtype']!=expected['output_dtype'])
                    or any(check.get(k)!=row.get(k) for k in ('mode','batch_index','phase','generation_step'))):
                raise ValueError('invalid step boundary dtype/dimensions/context')
            valid=lengths[pair[0]] if step['phase']=='prefill' else 1
            shape=[valid if d=='valid_tokens' else a if d is None else d for d,a in zip(shape,check['shape'])]
            if (shape!=check['shape'] or math.prod(shape)!=check['checked_elements']
                    or type(check['nonfinite_count']) is not int or not 0<=check['nonfinite_count']<=check['checked_elements']):
                raise ValueError('invalid step boundary shape/count')
            if check['nonfinite_count']:
                anomalies.add((*key,check['sample_id'],check['boundary']))
        if seen!=wanted:
            raise ValueError('missing step boundary checks')
    if seen_steps!=set(wanted_steps):
        raise ValueError('missing execution step checks')
    actual={(a['mode'],a['batch_index'],a['generation_step'],a['sample_id'],a['boundary']) for a in result['numerical_anomalies']}
    if actual!=anomalies or result['artifacts']!=full_artifacts or len(full_artifacts)!=len(set(full_artifacts)):
        raise ValueError('execution anomaly/artifact summary mismatch')
    return value
