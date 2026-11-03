# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""PPU kernel projection with conservative clock evidence."""
from flagprof.vendors.nvidia.analysis import activity as kineto_activity


def activity(event,context,label):
    result=kineto_activity(event,context,label)
    if result is not None:
        # Collective identity must be explicit; CUDA-looking names are not
        # PPU collective proof. No kernel-name family is pinned before an
        # isolated real collective observation on the locked stack.
        explicit=event.get('args',{}).get('Collective name')
        result.update(category='comm' if explicit else 'compute' if event.get('cat')=='kernel' else 'memcpy',
                      collective=explicit or None,
                      classification_basis='explicit_collective_metadata' if explicit else 'cat')
    return result


def clock(directory,capture):
    collector=capture.get('metadata',{}).get('identities',{}).get('collector',{})
    evidence=collector.get('evidence',{})
    bases={o['trace_metadata']['baseTimeNanoseconds'] for o in evidence.get('observations',[])
           if type(o.get('trace_metadata',{}).get('baseTimeNanoseconds')) is int}
    sources=[dict(path='sealed-trace/data/trace.json',sha256=capture['trace']['sha256'])]
    transforms=[]
    if collector.get('facts',{}).get('collector',{}).get('version')=='2.10.0' and len(bases)==1:
        transforms.append(dict(method='raw_wall_piecewise',base_time_ns=next(iter(bases)),sources=sources,
            formula='export_ns=interpolate_RAW_to_wall(raw_ns)-baseTimeNanoseconds',
            interpretation='docs/thead/inference-v2.md#时钟验证',collector_version='2.10.0',
            validation='PPU pinned torch.profiler; host record_function marker bracketed by independent wall-clock samples'))
    return dict(status='available',clock_valid=False,source_domain='official_export_ns',transform_status='not_required',
        transform_applied=False,export_already_transformed=True,alignment_status='official_export_internal',
        project_host_alignment='unavailable',uncertainty_ns=None,uncertainty_status='unavailable',
        reason='PPU exported host timestamps independently checked; device synchronization error bound remains unknown',
        sources=sources,project_transforms=transforms)
