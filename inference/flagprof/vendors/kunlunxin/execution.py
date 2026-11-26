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
    """P800 identity for monitor registration.

    The monitor binds a target to telemetry entities by entity string, so the
    UUID must be the exact string the telemetry provider records (the XPU
    stack's NVML-compatible library). The framework-reported UUID only
    selects which device to match; unknown is recorded honestly when the
    library is unavailable.
    """
    import torch
    try:
        props = torch.cuda.get_device_properties(device)
        torch_uuid = str(getattr(props, 'uuid', '') or '')
    except Exception as error:
        return dict(status='unknown', reason=str(error)[:120])
    from flagprof.telemetry.nvml_compat import logical_identity
    identity = logical_identity(torch.device(device).index or 0, torch_uuid or None)
    if identity is None:
        return dict(status='unknown',
                    reason='NVML-compatible library unavailable or UUID unmatched',
                    torch_uuid=torch_uuid or None)
    identity['logical_device'] = str(device)
    return identity
