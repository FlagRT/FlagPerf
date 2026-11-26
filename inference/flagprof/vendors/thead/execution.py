# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""PPU worker device binding; no tracing gate variable exists on the locked stack."""
def initialize(rank):
    import torch
    device='cuda:'+str(rank)
    torch.cuda.set_device(device)
    return device


def physical_identity(device):
    """Device identity for monitor registration.

    The monitor binds a target to telemetry entities by entity string, so the
    UUID must be the exact string the telemetry provider records. Both sides
    read it from the same NVML-compatible library; the framework-reported
    UUID only selects which device to match. When the library is unavailable
    the identity is reported unknown rather than a fabricated prefix.
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
