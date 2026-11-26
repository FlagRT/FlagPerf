# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Reuse the explicitly shared Kineto execution syntax; P800 identity remains separate."""
def initialize(rank):
    import os
    if os.environ.get('XPU_ENABLE_PROFILER_TRACING')=='1':
        selected=os.environ.get('XPU_CUPTI_ENABLE_DEVICE')
        if selected not in (None,str(rank)):
            raise ValueError('P800 profiler device selection differs from rank')
        os.environ['XPU_CUPTI_ENABLE_DEVICE']=str(rank)
    import torch
    device='cuda:'+str(rank)
    torch.cuda.set_device(device)
    return device


def physical_identity(device):
    """P800 identity via torch properties (NVML-compat UUID when exposed);
    unknown is recorded honestly when the compat stack omits it."""
    import torch
    try:
        props = torch.cuda.get_device_properties(device)
        uuid = str(getattr(props, 'uuid', '') or '')
    except Exception as error:
        return dict(status='unknown', reason=str(error)[:120])
    if not uuid:
        return dict(status='unknown', reason='no uuid exposed by compat stack')
    if not uuid.startswith(('GPU-', 'P800-')):
        uuid = 'P800-'+uuid
    return dict(entity='chip:'+uuid, uuid=uuid, status='available')
