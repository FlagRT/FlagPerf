"""CPU-only free-generation alignment, inclusive of the first token divergence."""
from runtime.tensor_io import load_cpu, concatenate
from pathlib import Path
from runtime.common import read_json


def trajectories(folder, mode='accuracy'):
    manifest = read_json(Path(folder)/'execution.json')
    if manifest.get('schema_version') != 1 or manifest.get('complete') is not True:
        raise ValueError('incomplete generation execution')
    output = {}
    for request in manifest['requests']:
        if request['mode'] != mode: continue
        for sid, row in request['samples'].items():
            if sid in output: raise ValueError('duplicate generation sample request')
            output[sid] = row
    if not output: raise ValueError('no generation samples for selected execution mode')
    return output


def prefix_plan(reference, candidate):
    if set(reference) != set(candidate): raise ValueError('generation sample coverage differs')
    output = {}
    for sid, a in reference.items():
        b = candidate[sid]; n = min(len(a['tokens']), len(b['tokens']))
        mismatch = next((i for i in range(n) if a['tokens'][i] != b['tokens'][i]), None)
        if mismatch is None and (len(a['tokens']) != len(b['tokens']) or a['stop_reason'] != b['stop_reason']):
            raise ValueError('equal generation prefix has inconsistent stop/length')
        horizon = mismatch+1 if mismatch is not None else n
        if horizon < 1: raise ValueError('empty comparable generation prefix')
        output[sid] = horizon
    return output


def records(folder, mode='accuracy', *, horizons=None):
    import torch
    from runtime.workload import sample_view
    from runtime.oracle import safe_file
    folder=Path(folder)
    manifest=read_json(folder/'execution.json')
    if manifest.get('schema_version')!=1 or manifest.get('complete') is not True:
        raise ValueError('incomplete generation execution')
    output={}
    for request in manifest['requests']:
        if request['mode']!=mode:
            continue
        for sid,row in request['samples'].items():
            if sid in output:
                raise ValueError('duplicate generation sample request')
            output[sid]={**row,'steps':[]}
        for entry in request['steps']:
            value=load_cpu(safe_file(folder,entry['artifact']))
            if value['ids']!=request['sample_ids'] or value['execution']!={k:v for k,v in entry.items() if k!='artifact'}:
                raise ValueError('generation artifact differs from execution index')
            if set(value['outputs'])!=set(entry['boundaries']):
                raise ValueError('generation artifact boundary coverage mismatch')
            if set(value['boundary_specs'])!=set(value['outputs']):
                raise ValueError('generation artifact lacks boundary specifications')
            for name,tensor in value['outputs'].items():
                spec=value['boundary_specs'][name]
                if (not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu'
                        or tensor.shape[spec['sample_axis']]!=len(value['ids'])):
                    raise ValueError('invalid generation tensor/sample axis')
            for i,sid in enumerate(value['ids']):
                if sid not in entry['active_sample_ids']:
                    continue
                tensors={}
                for n,t in value['outputs'].items():
                    spec=value['boundary_specs'][n]
                    row=sample_view(t,i,spec,value['valid_regions'])
                    shape=spec.get('phase_shapes',{}).get(entry['phase'],spec['sample_shape'])
                    if len(shape)!=row.ndim or any(type(d) is int and d!=actual for d,actual in zip(shape,row.shape)):
                        raise ValueError('invalid generation boundary shape: '+n)
                    # The full trajectory, artifact identity and every shape are
                    # still checked. Only comparable prefixes need CPU arrays.
                    tensors[n]=row.float().numpy() if horizons is None or entry['generation_step'] < horizons[sid] else None
                output[sid]['steps'].append({'phase':entry['phase'],'generation_step':entry['generation_step'],'outputs':tensors})
        for sid in request['sample_ids']:
            if len(output[sid]['steps'])!=len(output[sid]['tokens']):
                raise ValueError('generation trajectory lacks hidden-state steps')
    if not output:
        raise ValueError('no generation samples for selected execution mode')
    return output


def compare(reference, candidate, levels, worst_count, boundary_levels):
    import numpy as np
    from runtime.metrics import compare_records, metrics
    if set(reference)!=set(candidate):
        raise ValueError('generation sample coverage differs')
    left,right={},{}
    phases={p:({}, {}) for p in ('prefill','decode')}
    trajectories,step_metrics=[],[]
    for sid in reference:
        a,b=reference[sid],candidate[sid]
        if len(a['steps'])!=len(a['tokens']) or len(b['steps'])!=len(b['tokens']):
            raise ValueError('generation trajectory lacks step coverage')
        n=min(len(a['tokens']),len(b['tokens']))
        mismatch=next((i for i in range(n) if a['tokens'][i]!=b['tokens'][i]),None)
        if mismatch is None and (len(a['tokens'])!=len(b['tokens']) or a['stop_reason']!=b['stop_reason']):
            raise ValueError('equal generation prefix has inconsistent stop/length')
        horizon=mismatch+1 if mismatch is not None else n
        if horizon<1:
            raise ValueError('empty comparable generation prefix')
        accum_a,accum_b={},{}
        phase_a,phase_b={p:{} for p in phases},{p:{} for p in phases}
        for index in range(horizon):
            sa,sb=a['steps'][index],b['steps'][index]
            if (sa['phase'],sa['generation_step'])!=(sb['phase'],sb['generation_step']) or set(sa['outputs'])!=set(sb['outputs']):
                raise ValueError('generation boundary/step pairing mismatch')
            for name,x in sa['outputs'].items():
                y=sb['outputs'][name]
                level=boundary_levels[name]
                if level not in levels:
                    continue
                if x.shape!=y.shape:
                    raise ValueError('generation boundary shape mismatch: '+name)
                accum_a.setdefault(name,[]).append(x.reshape(-1)); accum_b.setdefault(name,[]).append(y.reshape(-1))
                phase_a[sa['phase']].setdefault(name,[]).append(x.reshape(-1))
                phase_b[sa['phase']].setdefault(name,[]).append(y.reshape(-1))
                step_metrics.append({'sample_id':sid,'phase':sa['phase'],'generation_step':index,
                                     'boundary':name,**metrics(x,y)})
        left[sid]={n:concatenate(v) for n,v in accum_a.items()}
        right[sid]={n:concatenate(v) for n,v in accum_b.items()}
        for phase in phases:
            if phase_a[phase]:
                phases[phase][0][sid]={n:concatenate(v) for n,v in phase_a[phase].items()}
                phases[phase][1][sid]={n:concatenate(v) for n,v in phase_b[phase].items()}
        trajectories.append({'sample_id':sid,'first_divergence_step':mismatch,'compared_steps':horizon,
                             'includes_divergence_step':mismatch is not None,
                             'reference_token':a['tokens'][mismatch] if mismatch is not None else None,
                             'candidate_token':b['tokens'][mismatch] if mismatch is not None else None,
                             'reference_generated_tokens':len(a['tokens']),'candidate_generated_tokens':len(b['tokens']),
                             'reference_stop':a['stop_reason'],'candidate_stop':b['stop_reason']})
    result=compare_records(left,right,levels,worst_count,boundary_levels)
    result['generation']={'alignment':'per_sample_common_input_prefix_including_first_emitted_token_divergence',
                          'trajectories':trajectories,'step_metrics':step_metrics,
                          'phase_comparisons':{p:compare_records(x,y,levels,worst_count,boundary_levels)
                                               for p,(x,y) in phases.items() if x}}
    return result
