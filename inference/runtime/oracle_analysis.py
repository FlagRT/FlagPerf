"""CPU-only A100/device differences. No device initialization or SSH."""
from pathlib import Path
import re
from runtime.common import read_json, write_json, identity_differences, file_hash
from runtime.oracle import load_manifest, load_batches, selected_boundaries, output_batch, safe_file


def reference_records(package, boundaries, *, manifest=None):
    from safetensors.torch import load_file
    import torch
    root=Path(package)
    manifest=load_manifest(root) if manifest is None else manifest
    from runtime.reference_protocol import package_specs, batch_regions
    specs = package_specs(root,manifest)
    batches=load_batches(root/'prepared')
    index=read_json(root/'outputs/index.json')
    if len(batches)!=len(index):
        raise ValueError('oracle batch count differs from input archive')
    records={}
    for row,batch in zip(index,batches):
        if row['ids']!=batch['ids']:
            raise ValueError('oracle output order differs from input archive')
        tensors=load_file(str(safe_file(root,row['file'])),device='cpu')
        if set(tensors)!=set(row['tensors']) or not set(boundaries)<=set(tensors):
            raise ValueError('oracle tensor index coverage mismatch')
        for name,tensor in tensors.items():
            if row['tensors'][name] != {'shape':list(tensor.shape),'dtype':str(tensor.dtype)}:
                raise ValueError('oracle tensor metadata mismatch: '+name)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError('nonfinite oracle tensor')
        for i,sid in enumerate(batch['ids']):
            if sid in records: raise ValueError('duplicate reference sample')
            from runtime.workload import sample_view
            records[sid]={name:sample_view(tensors[name], i, specs[name], batch_regions(batch,specs)).float().numpy()
                          for name in boundaries}
    return records,batches,index


def candidate_records(folder, state, batches, boundaries, index, specs=None):
    names=state.get('artifacts',[])
    if len(names)!=len(batches) or len(names)!=len(set(names)):
        raise ValueError('candidate batch coverage mismatch')
    records={}
    for name,batch,reference in zip(names,batches,index):
        value=output_batch(folder,name,batch,boundaries)
        if specs is not None and value.get('boundary_specs'):
            from runtime.reference_protocol import validate_outputs
            validate_outputs(value,batch,boundaries,specs)
        for boundary in boundaries:
            t=value['outputs'][boundary]
            if {'shape':list(t.shape),'dtype':str(t.dtype)} != reference['tensors'][boundary]:
                raise ValueError('candidate boundary shape/dtype differs (possibly a TP shard): '+boundary)
        for i,sid in enumerate(batch['ids']):
            if sid in records: raise ValueError('duplicate candidate sample')
            from runtime.workload import sample_view, artifact_specs, artifact_regions
            specs, regions = artifact_specs(value), artifact_regions(value)
            records[sid]={key:sample_view(value['outputs'][key], i, specs[key], regions).float().numpy()
                          for key in boundaries}
    return records


def layer_trend(diff, relations=None):
    """Only adjacent complete blocks; no invented significant-divergence threshold."""
    rows=[]
    if relations and relations.get('kind') != 'sequential': return rows
    if relations:
        edges = [tuple(pair) for pair in relations.get('edges',[])]
        order = list(dict.fromkeys(n for pair in edges for n in pair))
        selected = [n for n in order if n in diff['aggregate']]
        for i,name in enumerate(selected):
            m = diff['aggregate'][name]
            previous = selected[i-1] if i else None
            rows.append({'layer':order.index(name),'boundary':name,'mse':m['mse'],'mae':m['mae'],
                         'adjacent_mse_change':m['mse']-diff['aggregate'][previous]['mse']
                         if previous and (previous,name) in edges and m['mse'] is not None and
                         diff['aggregate'][previous]['mse'] is not None else None})
        return rows
    for name,m in diff['aggregate'].items():
        match=re.fullmatch(r'layers\.(\d+)',name)
        if match: rows.append({'layer':int(match.group(1)),'boundary':name,'mse':m['mse'],'mae':m['mae']})
    rows.sort(key=lambda r:r['layer'])
    for i,row in enumerate(rows):
        previous=rows[i-1] if i else None
        row['adjacent_mse_change']=(row['mse']-previous['mse'] if previous and previous['layer']+1==row['layer']
                                    and row['mse'] is not None and previous['mse'] is not None else None)
    return rows


def compare_run(cfg,root,result):
    from runtime.metrics import compare_records
    package=Path(cfg['reference']['path'])
    manifest=load_manifest(package)
    if manifest['schema_version']==3 or manifest.get('workload',{}).get('execution_kind') == 'autoregressive':
        from runtime.generation_reference import compare_run as compare_generation
        return compare_generation(cfg,root,result,manifest)
    result['oracle']={'role':'consumer','oracle_id':manifest['oracle_id'],'status':'failed',
                      'comparison':'oracle-comparison.json','thresholds':None,
                      'retention':{'status':'kept_failure','reason':'incomplete comparison or execution'}}
    # Lightweight evidence is independent of the external package after comparison.
    write_json(root/'oracle-manifest.json',manifest)
    reference_identity=read_json(package/'provenance/identity.json')
    write_json(root/'oracle-producer.json',reference_identity)
    try:
        boundaries=selected_boundaries(cfg, prepared=root/'prepared', available=manifest['boundaries'])
        reference,batches,index=reference_records(package,boundaries,manifest=manifest)
        if file_hash(root/'prepared/inputs.pt')!=manifest['files']['prepared/inputs.pt']['sha256']:
            raise ValueError('candidate execution did not use the sealed oracle tokens')
        tp=cfg['runtime'].get('parallelism')=='tp'
        comparison={'schema_version':1,'oracle_id':manifest['oracle_id'],
                    'reference':'A100 native PyTorch; engineering reference, not mathematical truth',
                    'candidate_profiles':result.get('component_profiles',{}),
                    'parallelism':cfg['runtime'].get('parallelism','single'),
                    'logical_samples':len(reference),'boundaries':boundaries,'thresholds':None,
                    'scope':'each candidate rank against the same complete A100 outputs; ranks are not concatenated or counted as new samples',
                    'candidates':{},'errors':[]}
        expected=set(result.get('component_profiles',{})) or set(result.get('paths',{}))
        if not expected: raise ValueError('no candidate paths were scheduled')
        for side in sorted(expected):
            state=result.get('paths',{}).get(side,{})
            ranks=state.get('ranks',{}) if tp else {'single':state}
            expected_ranks={str(i) for i in range(len(cfg['runtime']['devices']))} if tp else {'single'}
            if set(ranks)!=expected_ranks:
                comparison['errors'].append(side+': missing rank results')
                continue
            target={'ranks':{}}
            comparison['candidates'][side]=target
            for rank in sorted(ranks):
                row=ranks[rank]
                folder=root/side if rank=='single' else root/side/f'rank-{rank}'
                try:
                    if row.get('status')!='completed': raise ValueError('candidate worker did not complete')
                    from runtime.reference_protocol import package_specs
                    specs = package_specs(package,manifest)
                    candidate=candidate_records(folder,row,batches,boundaries,index,specs if manifest.get('protocol_schema_version',manifest['schema_version']) == 2 else None)
                    diff=compare_records(reference,candidate,cfg['accuracy']['levels'],cfg['accuracy']['worst_samples'],
                                         boundary_levels={n:s['level'] for n,s in specs.items()})
                    diff['reference']='A100 native PyTorch'
                    diff['roles']={'off':'A100 reference','on':'candidate '+side+' rank '+rank}
                    diff['layer_trend']=layer_trend(diff,manifest.get('workload',{}).get('boundary_relations'))
                    target['ranks'][rank]=diff
                    if tp:
                        diff['layout_evidence']=row.get('oracle_layout',{})
                    # Cosine undefined for zero vectors is descriptive, but nonfinite data is a failed run.
                    if any(s['status']!='finite' for s in diff['samples']):
                        comparison['errors'].append(side+'/'+rank+': nonfinite or invalid output')
                except (ValueError,OSError,KeyError,TypeError,IndexError) as error:
                    comparison['errors'].append(side+'/'+rank+': '+str(error))
        identity=read_json(root/'prepared/identity.json')
        if load_manifest(package)['oracle_id'] != manifest['oracle_id']:
            raise ValueError('oracle package changed during comparison')
        comparison['provenance_differences']=identity_differences(reference_identity,identity)
        comparison['status']='failed' if comparison['errors'] else 'completed'
        write_json(root/'oracle-comparison.json',comparison)
        result['oracle'].update(status=comparison['status'],logical_samples=len(reference),
                               provenance_differences=comparison['provenance_differences'])
        if comparison['errors']:
            result.update(status='failed',error='oracle comparison: '+'; '.join(comparison['errors']))
        elif result['status']=='completed':
            anomalies=sum(len(row['anomalies']) for target in comparison['candidates'].values() for row in target['ranks'].values())
            result['oracle']['numerical_anomalies']=anomalies
            result['oracle']['metric_classification_version']=2
            result['oracle']['undefined_metric_records']=sum(len(row.get('undefined_metrics', []))
                for target in comparison['candidates'].values() for row in target['ranks'].values())
            status='kept_requested' if cfg['accuracy'].get('keep_tensors') else ('kept_anomalies' if anomalies else 'pending_cleanup')
            result['oracle']['retention']={'status':status}
    except Exception as error:
        result.update(status='failed',error='oracle comparison: '+str(error))
        write_json(root/'oracle-comparison.json',{'status':'failed','oracle_id':manifest['oracle_id'],'error':str(error)})
