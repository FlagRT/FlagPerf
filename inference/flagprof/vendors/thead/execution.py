# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""PPU worker device binding; no tracing gate variable exists on the locked stack."""
def initialize(rank):
    import torch
    device='cuda:'+str(rank)
    torch.cuda.set_device(device)
    return device


def physical_identity(device):
    """Device identity from the CUDA-compat stack for monitor registration;
    returns unknown when the compat stack exposes no UUID (honest, not silent)."""
    import torch
    try:
        props = torch.cuda.get_device_properties(device)
        uuid = str(getattr(props, 'uuid', '') or '')
    except Exception as error:
        return dict(status='unknown', reason=str(error)[:120])
    if not uuid:
        return dict(status='unknown', reason='no uuid exposed by compat stack')
    if not uuid.startswith(('GPU-', 'PPU-')):
        uuid = 'PPU-'+uuid
    return dict(entity='chip:'+uuid, uuid=uuid, status='available')
