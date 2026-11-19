# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Device predicates shared by producers and offline consumers; no torch import."""


from importlib import import_module


def device_types(dev):
    values = dev.values() if isinstance(dev, dict) else [dev] if dev is not None else []
    return {value.split(':', 1)[0] for value in values}


def accelerator(dev):
    return bool(device_types(dev) - {'cpu', 'meta'})


def execution_device(requested=None):
    """Resolve the PyTorch device only when executing a workload."""
    import os
    assigned = os.environ.get('FLAGPROF_DEVICE') if os.environ.get('FLAGPROF_LAUNCH_WORKER') == '1' else None
    if assigned:
        if requested and requested not in (assigned, assigned.split(':')[0]):
            raise ValueError(f'--device {requested} conflicts with rank assignment {assigned}')
        return assigned
    if requested:
        return requested
    import torch
    from .vendors import torch_runtimes
    runtimes = torch_runtimes()
    for extension in filter(None, runtimes.values()):
        try:
            import_module(extension)
        except ModuleNotFoundError as error:
            if error.name != extension:
                raise
    available = [family for family in runtimes if getattr(torch, family, None)
                 and getattr(torch, family).is_available()]
    if len(available) > 1:
        raise ValueError('multiple available accelerator runtimes; select --device')
    if not available:
        return 'cpu'
    runtime = getattr(torch, available[0])
    return f'{available[0]}:{runtime.current_device()}'


def synchronize_device(device):
    import torch
    from .vendors import torch_runtimes
    extension = torch_runtimes().get(device.split(':')[0])
    if extension:
        import_module(extension)
    target = torch.device(device)
    if target.type == 'cpu':
        return lambda: None
    runtime = getattr(torch, target.type)
    runtime.set_device(target)
    return lambda: runtime.synchronize(target)
