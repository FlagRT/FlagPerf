# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""PPU telemetry over the CUDA-compat stack's NVML-compatible interface.

The PPU driver ships /usr/lib64/libnvidia-ml.so, an NVML-ABI-compatible
library: utilization, memory, temperature and power are queryable with the
standard nvml* symbols (verified on PPU-ZW810E, driver 1.5.1-1d747a).
This provider reuses the NVIDIA NVML reader with that library; per-vendor
identity naming stays honest (PPU-ZW810E).
"""


def _candidate_libraries():
    from pathlib import Path
    return ('/usr/lib64/libnvidia-ml.so', '/usr/lib/libnvidia-ml.so',
            '/usr/lib64/libnvidia-ml.so.1', '/usr/lib/libnvidia-ml.so.1',
            str(Path('/usr/local/bin').parent/'lib64'/'libnvidia-ml.so.1'),
            'libnvidia-ml.so.1')


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
