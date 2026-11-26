# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""PPU telemetry over the CUDA-compat stack's NVML-compatible interface.

The PPU driver ships an NVML-ABI-compatible library (at
/usr/lib64/libnvidia-ml.so on the verified host): utilization, memory,
temperature and power are queryable with the standard nvml* symbols
(verified on PPU-ZW810E, driver 1.5.1-1d747a). This provider reuses the
NVIDIA NVML reader with that library; per-vendor identity naming stays
honest (PPU-ZW810E).
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
    2. the platform loader's standard sonames;
    3. the NVML-ABI library the vendor's Python stack bundles under
       triton/backends/*/*/so (PPU images ship it as libxpunvidia-ml.so,
       the CUDA-compat ABI alias).
    No installation-specific paths are hard-coded: the third group is
    derived from the running interpreter.
    """
    import glob
    import os
    candidates = []
    override = os.environ.get('FLAGPROF_TELEMETRY_LIBRARY')
    if override:
        candidates.append(override)
    candidates += ['/usr/lib64/libnvidia-ml.so', '/usr/lib/libnvidia-ml.so',
                   '/usr/lib64/libnvidia-ml.so.1', '/usr/lib/libnvidia-ml.so.1',
                   '/usr/local/lib64/libnvidia-ml.so', 'libnvidia-ml.so.1']
    purelib = _site_packages()
    if purelib is not None:
        candidates += sorted(glob.glob(str(purelib/'triton'/'backends'/'*'/'*'/'so'/'libxpunvidia-ml.so')))
    return tuple(candidates)


def Provider(devices='all', source='auto', **kwargs):
    from ..nvidia.telemetry import NVML
    from flagprof.telemetry.contract import ReadError
    if source not in ('auto', 'nvml'):
        raise ValueError('thead telemetry source must be auto or nvml')
    last = None
    for library in _candidate_libraries():
        try:
            provider = NVML(devices=devices, source='nvml', library=library, **kwargs)
            provider.inventory['source'] = 'thead-nvml-compat'
            provider.inventory['library'] = library
            return provider
        except (OSError, ReadError) as error:
            last = str(error)
    raise ReadError('PPU NVML-compatible library not found: ' + str(last))
