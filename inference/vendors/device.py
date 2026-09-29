# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Device policy is selected before importing torch, never inferred from availability."""
import os


def environment(cfg):
    vendor = cfg['runtime']['vendor']
    env = os.environ.copy()
    env.update({k:str(v) for k,v in cfg['vendors'][vendor]['env'].items()})
    env.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    # In the container the single mapped physical device is logical device zero.
    visible = 'ASCEND_RT_VISIBLE_DEVICES' if vendor == 'ascend' else 'CUDA_VISIBLE_DEVICES'
    if os.environ.get('FLAGPERF_CONTAINER_DEVICE') != '1':
        env[visible] = (','.join(map(str,cfg['runtime']['devices']))
                        if cfg['runtime'].get('parallelism') == 'tp' else str(cfg['runtime']['device']))
    return env


def initialize(cfg):
    if cfg['runtime']['vendor'] == 'ascend':
        import torch_npu  # noqa: F401
    import torch
    backend = torch.npu if cfg['runtime']['vendor'] == 'ascend' else torch.cuda
    if not backend.is_available():
        raise RuntimeError('selected device backend is unavailable; no CPU fallback')
    rank = int(os.environ.get('LOCAL_RANK','0')) if cfg['runtime'].get('parallelism') == 'tp' else 0
    backend.set_device(rank)
    torch.manual_seed(cfg['runtime']['seed'])
    backend.manual_seed_all(cfg['runtime']['seed'])
    device = f'npu:{rank}' if cfg['runtime']['vendor'] == 'ascend' else f'cuda:{rank}'
    if cfg['runtime'].get('parallelism') == 'tp' and 'LOCAL_RANK' in os.environ:
        from runtime.tp import initialize_group
        initialize_group(cfg)
    return device, backend
