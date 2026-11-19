# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Standard-library correlation and recorded trace support."""

import importlib
from functools import lru_cache
import shutil


def vendor_names():
    from flagprof.registry import BUILTINS
    return tuple(d['id'] for d in BUILTINS if d['kind'] == 'vendor')


def monitored_vendors():
    from flagprof.registry import BUILTINS
    providers = {d['id'] for d in BUILTINS if d['kind'] == 'telemetry'}
    return tuple(v for v in vendor_names() if 'telemetry.' + v in providers)


def adapter(vendor, module):
    if vendor not in vendor_names():
        raise ValueError('unknown hardware adapter: ' + str(vendor))
    return importlib.import_module('flagprof.vendors.' + vendor + '.' + module)


@lru_cache(maxsize=1)
def torch_runtimes():
    """Map supported PyTorch device types to optional registration modules."""
    runtimes = {}
    for vendor in vendor_names():
        execution = adapter(vendor, 'execution')
        runtimes[execution.TORCH_DEVICE] = getattr(execution, 'TORCH_EXTENSION', None)
    return runtimes


def launch_backend(vendor, backend):
    profiles = {name: adapter(name, 'execution') for name in vendor_names()}
    if vendor is None:
        candidates = [name for name, profile in profiles.items()
                      if (backend is None or backend in profile.BACKENDS) and shutil.which(profile.SMI)]
        if len(candidates) != 1:
            raise ValueError('select --vendor for this host; detected compatible adapters: ' + str(candidates))
        vendor = candidates[0]
    if vendor not in profiles:
        raise ValueError('unknown --vendor: ' + str(vendor))
    supported = profiles[vendor].BACKENDS
    backend = backend or supported[0]
    if backend not in supported:
        raise ValueError(f'{vendor} supports --comm-backend ' + ', '.join(supported))
    return vendor, backend


@lru_cache(maxsize=None)
def is_kineto(vendor):
    """Trace syntax is shared independently of device identity and qualification."""
    if vendor not in vendor_names():
        raise ValueError('unknown hardware adapter: ' + str(vendor))
    return importlib.import_module('flagprof.vendors.' + vendor).TRACE_FAMILY == 'kineto'
