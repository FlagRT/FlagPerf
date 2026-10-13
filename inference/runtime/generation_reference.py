"""Schema-3 immutable generation references, with step-indexed CPU tensors."""
from runtime.tensor_io import load_cpu
from copy import deepcopy
from pathlib import Path
import shutil
import time
from runtime.common import read_json, write_json, file_hash, digest


def publish(cfg, root, result, repeat):
    import torch
    from runtime.oracle import contract, selected_boundaries, load_manifest, safe_file
    from runtime.generation_analysis import records
    identity=read_json(root/'prepared/identity.json')
    if (result.get('status')!='completed' or repeat.get('status')!='completed' or repeat.get('numerical_anomalies')
            or identity.get('vendor')!='nvidia' or 'A100' not in identity.get('device_name','')):
        raise ValueError('generation reference requires complete finite A100 producer evidence')
    first=records(root/'off'); second=records(root/'verification')
    if set(first)!=set(second): raise ValueError('generation repeatability sample mismatch')
    for sid,a in first.items():
        b=second[sid]
        if a['tokens']!=b['tokens'] or a['stop_reason']!=b['stop_reason'] or len(a['steps'])!=len(b['steps']):
            raise ValueError('generation token trajectory is not repeatable')
    original=read_json(root/'off/execution.json')
    verification=read_json(root/'verification/execution.json')
    for a,b in zip(original['requests'],verification['requests']):
        if len(a['steps'])!=len(b['steps']): raise ValueError('generation repeatability step mismatch')
        for x,y in zip(a['steps'],b['steps']):
            left=load_cpu(safe_file(root/'off',x['artifact']))
            right=load_cpu(safe_file(root/'verification',y['artifact']))
            if set(left['outputs'])!=set(right['outputs']) or any(
                t.dtype!=right['outputs'][n].dtype or not torch.equal(t,right['outputs'][n]) or
                not bool(torch.isfinite(t).all()) for n,t in left['outputs'].items()):
                raise ValueError('generation hidden states are not finite and bitwise repeatable')
    write_json(root/'repeatability.json',{'status':'bitwise_equal','forward_processes':2,
                                        'scope':'complete independently generated trajectories and all selected hidden states'})
    package=root/'package.incomplete'; package.mkdir()
    for name in ('prepared','outputs','provenance'): (package/name).mkdir()
    boundaries=selected_boundaries(cfg, prepared=root/'prepared')
    execution=deepcopy(original)
    for request in execution['requests']:
        for entry in request['steps']:
            value=load_cpu(safe_file(root/'off',entry['artifact']))
            value['outputs']={n:value['outputs'][n] for n in boundaries}
            value['boundary_specs']={n:value['boundary_specs'][n] for n in boundaries}
            entry['boundaries']=sorted(boundaries)
            value['execution']={k:v for k,v in entry.items() if k!='artifact'}
            torch.save(value,package/'outputs'/entry['artifact'])
    write_json(package/'outputs/execution.json',execution)
    write_json(package/'outputs/index.json',execution['requests'])
    for name in ('inputs.pt','samples.json','workload.json','forward-inputs.json'):
        shutil.copyfile(root/'prepared'/name,package/'prepared'/name)
    workload=read_json(package/'prepared/workload.json')
    # Explicit nested selections may extend the adapter's default boundary set.
    first_step=original['requests'][0]['steps'][0]
    captured=load_cpu(safe_file(root/'off',first_step['artifact']))['boundary_specs']
    workload['boundaries']=[captured[n] for n in boundaries]
    write_json(package/'prepared/workload.json',workload)
    shutil.copyfile(cfg['inputs']['path'],package/'inputs.jsonl')
    shutil.copyfile(root/'prepared/identity.json',package/'provenance/identity.json')
    for name in ('effective.yaml','source-snapshot.json','repeatability.json'):
        shutil.copyfile(root/name,package/'provenance'/name)
    if (root/'comparison-context.json').is_file():
        shutil.copyfile(root/'comparison-context.json',package/'provenance/comparison-context.json')
    for folder in ('prepared','off','verification'):
        for name in ('result.json','run.log','command.json','exit.json','components.json','forward-checks.json','execution.json'):
            if (root/folder/name).is_file():
                dest=package/'provenance'/folder/name;dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(root/folder/name,dest)
    shutil.copytree(root/'source-code',package/'provenance/source-code')
    manifest={'schema_version':3,'kind':'a100_accuracy','status':'complete',
              'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'contract':contract(cfg,identity['model_files']),'boundaries':boundaries,'workload':workload,
              'producer':{'device_name':identity['device_name'],'identity_key':digest(identity),
                          'numerical_settings':identity['oracle_numerics']},
              'repeatability':{'status':'bitwise_equal','forward_processes':2},
              'files':{p.relative_to(package).as_posix():{'bytes':p.stat().st_size,'sha256':file_hash(p)}
                       for p in sorted(package.rglob('*')) if p.is_file()}}
    from runtime.reference_storage import seal_accuracy
    manifest = seal_accuracy(package, manifest)
    load_manifest(package)
    package.rename(root/'package')
    (root/'package-evidence.incomplete').rename(root/'package-evidence')
    result['oracle']={'role':'producer','oracle_id':manifest['oracle_id'],'package':'package',
                     'repeatability':'bitwise_equal','status':'completed'}
    return result


def validate_index(root, manifest):
    from runtime.reference_protocol import package_specs
    from runtime.step_capture import validate_execution
    root=Path(root)
    execution=read_json(root/'outputs/execution.json')
    index=read_json(root/'outputs/index.json')
    if execution.get('complete') is not True or execution.get('modes')!=['accuracy'] or execution['requests']!=index:
        raise ValueError('invalid generation reference execution index')
    ids=[sid for row in index for sid in row['sample_ids']]
    if ids!=[s['id'] for s in read_json(root/'prepared/samples.json')] or len(ids)!=len(set(ids)):
        raise ValueError('generation reference sample coverage mismatch')
    specs=package_specs(root,manifest)
    if set(manifest['boundaries'])!=set(specs) or len(manifest['boundaries'])!=len(set(manifest['boundaries'])):
        raise ValueError('missing generation reference boundary specifications')
    expected=read_json(root/'prepared/forward-inputs.json')
    if expected.get('generation')!=manifest['workload'].get('generation'):
        raise ValueError('generation reference input/workload contract mismatch')
    expected['boundaries']=manifest['boundaries']
    validate_execution(execution,expected)
    files=manifest['files']
    if 'outputs/execution.json' not in files:
        raise ValueError('generation reference execution is not sealed')
    artifacts=[]
    for row in index:
        if not row['steps'] or set(row['samples'])!=set(row['sample_ids']):
            raise ValueError('empty generation reference trajectory')
        for step in row['steps']:
            if 'outputs/'+step['artifact'] not in files or set(step['boundaries'])!=set(manifest['boundaries']):
                raise ValueError('missing generation reference step/boundary')
            artifacts.append(step['artifact'])
    if len(artifacts)!=len(set(artifacts)):
        raise ValueError('duplicate generation reference artifact')


def compare_run(cfg, root, result, manifest):
    from runtime.generation_analysis import records, compare, trajectories, prefix_plan
    from runtime.reference_protocol import package_specs
    from runtime.oracle import selected_boundaries, load_manifest
    from runtime.common import identity_differences
    package=Path(cfg['reference']['path'])
    specs=package_specs(package,manifest)
    selected=selected_boundaries(cfg,prepared=root/'prepared',available=manifest['boundaries'])
    levels={n:s['level'] for n,s in specs.items()}
    reference_trajectories = trajectories(package/'outputs')
    planned, reference_horizons = {}, {sid: 0 for sid in reference_trajectories}
    tp = cfg['runtime'].get('parallelism') == 'tp'
    for side, state in result.get('paths', {}).items():
        ranks = state.get('ranks', {}) if tp else {'single': state}
        for rank, row in ranks.items():
            if row.get('status') != 'completed': continue
            folder = root/side if rank == 'single' else root/side/f'rank-{rank}'
            try:
                horizons = prefix_plan(reference_trajectories, trajectories(folder))
                planned[(side, rank)] = horizons
                for sid, horizon in horizons.items(): reference_horizons[sid] = max(reference_horizons[sid], horizon)
            except (ValueError, KeyError, OSError, TypeError):
                pass  # The original per-candidate path below records the error.
    reference=records(package/'outputs', horizons=reference_horizons)
    def select(data):
        for sample in data.values():
            for step in sample['steps']:
                step['outputs']={n:step['outputs'][n] for n in selected}
        return data
    reference=select(reference)
    comparison={'schema_version':2,'oracle_id':manifest['oracle_id'],'status':'completed',
                'reference':'A100 native PyTorch engineering reference','logical_samples':len(reference),
                'boundaries':selected,'thresholds':None,'candidates':{},'errors':[],
                'candidate_profiles':result.get('component_profiles',{}),
                'scope':'each candidate rank independently; pair-specific common generation prefix, ranks are not new samples'}
    if file_hash(root/'prepared/inputs.pt')!=manifest['files']['prepared/inputs.pt']['sha256']:
        raise ValueError('generation candidate did not use sealed reference tokens')
    tp=cfg['runtime'].get('parallelism')=='tp'
    expected=set(result.get('component_profiles',{})) or set(result.get('paths',{}))
    if not expected: raise ValueError('no generation candidate paths scheduled')
    for side in sorted(expected):
        state=result.get('paths',{}).get(side,{})
        comparison['candidates'][side]={'ranks':{}}
        rows=state.get('ranks',{}) if tp else {'single':state}
        if set(rows)!=({str(i) for i in range(len(cfg['runtime']['devices']))} if tp else {'single'}):
            comparison['errors'].append(side+': missing rank results')
            continue
        for rank,row in rows.items():
            try:
                if row['status']!='completed': raise ValueError('candidate execution incomplete')
                folder=root/side if rank=='single' else root/side/f'rank-{rank}'
                horizons = planned.get((side, rank)) or prefix_plan(reference_trajectories, trajectories(folder))
                diff=compare(reference,select(records(folder, horizons=horizons)),cfg['accuracy']['levels'],cfg['accuracy']['worst_samples'],levels)
                diff['reference']='A100 native PyTorch'
                diff['roles']={'off':'A100 reference','on':'candidate '+side+' rank '+rank}
                from runtime.oracle_analysis import layer_trend
                diff['layer_trend']=layer_trend(diff,manifest['workload'].get('boundary_relations'))
                comparison['candidates'][side]['ranks'][rank]=diff
                if row.get('numerical_anomalies') or any(s['status']!='finite' for s in diff['samples']):
                    comparison['errors'].append(side+'/'+rank+': nonfinite or invalid step output')
            except (ValueError,KeyError,OSError,TypeError,IndexError) as error:
                comparison['errors'].append(f'{side}/{rank}: {error}')
    if load_manifest(package)['oracle_id']!=manifest['oracle_id']:
        raise ValueError('generation reference changed during comparison')
    comparison['provenance_differences']=identity_differences(read_json(package/'provenance/identity.json'),read_json(root/'prepared/identity.json'))
    if comparison['errors']:
        comparison['status']='failed'; result.update(status='failed',error='; '.join(comparison['errors']))
    write_json(root/'oracle-manifest.json',manifest)
    write_json(root/'oracle-producer.json',read_json(package/'provenance/identity.json'))
    write_json(root/'oracle-comparison.json',comparison)
    result['oracle']={'role':'consumer','oracle_id':manifest['oracle_id'],'status':comparison['status'],
                     'comparison':'oracle-comparison.json','logical_samples':len(reference),'thresholds':None}
