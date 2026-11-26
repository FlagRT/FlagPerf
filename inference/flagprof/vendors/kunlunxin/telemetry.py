# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Kunlunxin P800 telemetry over the CUDA-compat stack's NVML-compatible
interface.

The XPU stack bundles an NVML-ABI-compatible library inside its Python
packages (libxpunvidia-ml.so, driver ABI 515.58.kunlun): utilization,
memory, temperature and power are queryable with the standard nvml*
symbols (verified on P800 OAM x8, driver 5.0.21.47). This provider reuses
the NVIDIA NVML reader pointed at that library.

Library discovery, UUID reads and error aggregation are shared with the
target-side identity (flagprof.telemetry.nvml_compat). Diagnose an
unreachable library with ``flagprof telemetry check --vendor kunlunxin``.
"""

# The XPU vendor tree's documented install paths, tried before the shared
# standard sonames.
XPU_LIBRARY_PATHS = ('/usr/local/xpu/lib/libxpunvidia-ml.so', 'libxpunvidia-ml.so')


def Provider(devices='all', source='auto', **kwargs):
    from ..nvidia.telemetry import NVML
    from flagprof.telemetry.contract import ReadError
    from flagprof.telemetry.nvml_compat import candidate_libraries
    if source not in ('auto', 'nvml'):
        raise ValueError('kunlunxin telemetry source must be auto or nvml')
    attempts = []
    for library in candidate_libraries(extra=XPU_LIBRARY_PATHS):
        try:
            provider = NVML(devices=devices, source='nvml', library=library, **kwargs)
            provider.inventory['source'] = 'kunlunxin-nvml-compat'
            provider.inventory['library'] = library
            return provider
        except (OSError, ReadError) as error:
            attempts.append(library + ': ' + str(error))
    raise ReadError('Kunlunxin NVML-compatible library unavailable; tried ' +
                    str(len(attempts)) + ' path(s): ' + ' | '.join(attempts))
