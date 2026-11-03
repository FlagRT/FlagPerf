# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""P800 kernel projection with conservative clock evidence."""
from flagprof.vendors.nvidia.analysis import activity as kineto_activity
import re

# Frozen exporter names isolated with real float AllReduce and OneRec BF16
# TP2. Arithmetic reduce/gather kernels are deliberately outside this family.
ALLREDUCE = re.compile(r'^void kl3_all_reduce<(?:float|bfloat16)>\(')
def activity(event,context,label):
    result=kineto_activity(event,context,label)
    if result is not None:
        # Collective identity must be explicit; CUDA-looking names are not BKCL proof.
        explicit=event.get('args',{}).get('Collective name')
        registered=event.get('cat')=='kernel' and bool(ALLREDUCE.match(event.get('name','')))
        result.update(category='comm' if explicit or registered else 'compute' if event.get('cat')=='kernel' else 'memcpy',
                      collective=explicit or ('allreduce' if registered else None),
                      classification_basis='explicit_collective_metadata' if explicit else 'pinned_isolated_allreduce_kernel' if registered else 'cat')
    return result
def clock(directory,capture):
    collector=capture.get('metadata',{}).get('identities',{}).get('collector',{})
    evidence=collector.get('evidence',{})
    bases={o['trace_metadata']['baseTimeNanoseconds'] for o in evidence.get('observations',[])
           if type(o.get('trace_metadata',{}).get('baseTimeNanoseconds')) is int}
    sources=[dict(path='sealed-trace/data/trace.json',sha256=capture['trace']['sha256'])]
    transforms=[]
    if collector.get('facts',{}).get('collector',{}).get('version')=='2.9.0+cu129' and len(bases)==1:
        transforms.append(dict(method='raw_wall_piecewise',base_time_ns=next(iter(bases)),sources=sources,
            formula='export_ns=interpolate_RAW_to_wall(raw_ns)-baseTimeNanoseconds',
            interpretation='docs/kunlunxin/inference-v2.md#时钟验证',collector_version='2.9.0+cu129',
            validation='P800 pinned exporter; nine independently bracketed host markers in three captures'))
    return dict(status='available',clock_valid=False,source_domain='official_export_ns',transform_status='not_required',
        transform_applied=False,export_already_transformed=True,alignment_status='official_export_internal',
        project_host_alignment='unavailable',uncertainty_ns=None,uncertainty_status='unavailable',
        reason='P800 exported host timestamps independently checked; device synchronization error bound remains unknown',
        sources=sources,project_transforms=transforms)
