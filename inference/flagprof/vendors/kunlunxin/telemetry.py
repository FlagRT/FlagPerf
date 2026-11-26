# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Kunlunxin P800 telemetry over the CUDA-compat stack's NVML-compatible
interface.

The XPU stack ships libxpunvidia-ml.so (NVML-ABI compatible, driver ABI
515.58.kunlun) inside the triton XPU backend and torch_xmlir dependency
directories: utilization, memory, temperature and power are queryable with
the standard nvml* symbols (verified on P800 OAM x8, driver 5.0.21.47).
This provider reuses the NVIDIA NVML reader pointed at that library.
"""


def _candidate_libraries():
    """Locate the NVML-compatible shared library inside the XPU stack.

    The image ships it under the triton XPU backend
    (site-packages/triton/backends/xpu/xpu3/so/libxpunvidia-ml.so*) and in
    the torch_xmlir dependency tree; host installs may also provide it.
    """
    import glob
    from pathlib import Path
    import importlib.util
    candidates = []
    spec = importlib.util.find_spec('triton')
    if spec and spec.origin:
        root = Path(spec.origin).parent
        candidates += sorted(glob.glob(str(root/'backends'/'*'/'*'/'so'/'libxpunvidia-ml.so'), recursive=True))
        candidates += sorted(glob.glob(str(root/'backends/**/so/libxpunvidia-ml.so*'), recursive=True))
    candidates += ['/usr/local/xpu/lib/libxpunvidia-ml.so', 'libxpunvidia-ml.so']
    return candidates


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
