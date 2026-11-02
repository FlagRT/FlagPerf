# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Kineto activity projection and exported-clock provenance."""


def activity(event, context, label):
    if event.get('ph')!='X' or event.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):
        return None
    args=event.get('args',{});name=event.get('name','');cat=event['cat']
    comm=bool(args.get('Collective name')) or 'nccl' in name.lower()
    return dict(category='comm' if comm else 'compute' if cat=='kernel' else 'memcpy',
                dev=args.get('device'),stream=args.get('stream'),task_type=cat,
                representation='kernel',collective=args.get('Collective name'),
                step_marker=False,classification_basis='collective_metadata_or_registered_nccl_name' if comm else 'cat')


def clock(directory, capture):
    collector=capture.get('metadata',{}).get('identities',{}).get('collector',{})
    evidence=collector.get('evidence',{})
    bases={o['trace_metadata']['baseTimeNanoseconds'] for o in evidence.get('observations',[])
           if type(o.get('trace_metadata',{}).get('baseTimeNanoseconds')) is int}
    # Older evidence producers store the same top-level metadata under clock.
    bases.update(s['baseTimeNanoseconds'] for s in evidence.get('clock',{}).get('sources',[])
                 if type(s.get('baseTimeNanoseconds')) is int)
    sources=[dict(path='sealed-trace/data/trace.json',sha256=capture['trace']['sha256'])]
    transforms=[]
    if len(bases)==1:
        transforms.append(dict(method='raw_wall_piecewise',base_time_ns=next(iter(bases)),sources=sources,
            formula='export_ns=interpolate_RAW_to_wall(raw_ns)-baseTimeNanoseconds',
            interpretation='https://github.com/pytorch/kineto/blob/31f85df8fbd89c188f14ef10f1ec65379786b943/libkineto/src/output_json.cpp',
            collector_version=collector.get('version')))
    return dict(status='available',clock_valid=False,source_domain='official_export_ns',
        transform_status='not_required',transform_applied=False,export_already_transformed=True,
        alignment_status='official_export_internal',project_host_alignment='unavailable',
        uncertainty_ns=None,uncertainty_status='unavailable',
        reason='RAW to wall samples and exported base are interpreted separately from device synchronization error',
        project_transforms=transforms,sources=sources)
