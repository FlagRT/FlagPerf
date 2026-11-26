# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""PPU telemetry over the CUDA-compat stack's NVML-compatible interface.

The PPU driver ships an NVML-ABI-compatible library (at
/usr/lib64/libnvidia-ml.so on the verified host): utilization, memory,
temperature and power are queryable with the standard nvml* symbols
(verified on PPU-ZW810E, driver 1.5.1-1d747a). This provider reuses the
NVIDIA NVML reader with that library; per-vendor identity naming stays
honest (PPU-ZW810E).

Library discovery, UUID reads and error aggregation are shared with the
target-side identity (flagprof.telemetry.nvml_compat). Diagnose an
unreachable library with ``flagprof telemetry check --vendor thead``.
"""


def Provider(devices='all', source='auto', **kwargs):
    from ..nvidia.telemetry import NVML
    from flagprof.telemetry.contract import ReadError
    from flagprof.telemetry.nvml_compat import candidate_libraries
    if source not in ('auto', 'nvml'):
        raise ValueError('thead telemetry source must be auto or nvml')
    attempts = []
    for library in candidate_libraries():
        try:
            provider = NVML(devices=devices, source='nvml', library=library, **kwargs)
            provider.inventory['source'] = 'thead-nvml-compat'
            provider.inventory['library'] = library
            return provider
        except (OSError, ReadError) as error:
            attempts.append(library + ': ' + str(error))
    raise ReadError('PPU NVML-compatible library unavailable; tried ' +
                    str(len(attempts)) + ' path(s): ' + ' | '.join(attempts))
