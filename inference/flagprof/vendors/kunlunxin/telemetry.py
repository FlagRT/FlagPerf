# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Kunlunxin P800 telemetry over the CUDA-compat stack's NVML-compatible
interface.

The XPU stack bundles an NVML-ABI-compatible library inside its Python
packages (libxpunvidia-ml.so, driver ABI 515.58.kunlun): utilization,
memory, temperature and power are queryable with the standard nvml*
symbols (verified on P800 OAM x8, driver 5.0.21.47). This provider reuses
the NVIDIA NVML reader pointed at that library.
"""


def _site_packages():
    """The running interpreter's purelib, resolved without importing site.

    The monitor worker runs with -S, where importlib.find_spec cannot see
    installed packages, but sysconfig still reports the interpreter's own
    layout.
    """
    import sysconfig
    from pathlib import Path
    try:
        return Path(sysconfig.get_paths()['purelib'])
    except (KeyError, TypeError):
        return None


def _candidate_libraries():
    """Libraries to try, in order of preference.

    1. FLAGPROF_TELEMETRY_LIBRARY, an explicit operator override;
    2. the XPU vendor tree's standard install path;
    3. the NVML-ABI library the vendor's stack bundles under
       triton/backends/*/*/so, derived from the running interpreter.
    """
    import glob
    import os
    candidates = []
    override = os.environ.get('FLAGPROF_TELEMETRY_LIBRARY')
    if override:
        candidates.append(override)
    candidates += ['/usr/local/xpu/lib/libxpunvidia-ml.so', 'libxpunvidia-ml.so']
    purelib = _site_packages()
    if purelib is not None:
        candidates += sorted(glob.glob(str(purelib/'triton'/'backends'/'*'/'*'/'so'/'libxpunvidia-ml.so*')))
    return tuple(candidates)


def Provider(devices='all', source='auto', **kwargs):
    from ..nvidia.telemetry import NVML
    from flagprof.telemetry.contract import ReadError
    if source not in ('auto', 'nvml'):
        raise ValueError('kunlunxin telemetry source must be auto or nvml')
    last = None
    for library in _candidate_libraries():
        try:
            provider = NVML(devices=devices, source='nvml', library=library, **kwargs)
            provider.inventory['source'] = 'kunlunxin-nvml-compat'
            provider.inventory['library'] = library
            return provider
        except (OSError, ReadError) as error:
            last = str(error)
    raise ReadError('Kunlunxin NVML-compatible library not found: ' + str(last))
