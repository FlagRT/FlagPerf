# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Reuse the explicitly shared Kineto execution syntax; P800 identity remains separate."""
TORCH_DEVICE = 'cuda'
MODEL_MARKERS = ('P800', 'KUNLUN')
VISIBLE_DEVICES_ENV = 'XPU_VISIBLE_DEVICES'
SMI = 'xpu-smi'
BACKENDS = ('nccl', 'flagcx')


def prepare_capture_environment(env, rank=None):
    if rank is None:
        rank = env.get('XPU_CUPTI_ENABLE_DEVICE', 0)
    env['XPU_ENABLE_PROFILER_TRACING'] = '1'
    env['XPU_CUPTI_ENABLE_DEVICE'] = str(rank)


def initialize(rank):
    import os
    if os.environ.get('XPU_ENABLE_PROFILER_TRACING')=='1':
        selected=os.environ.get('XPU_CUPTI_ENABLE_DEVICE')
        if selected not in (None,str(rank)):
            raise ValueError('P800 profiler device selection differs from rank')
        os.environ['XPU_CUPTI_ENABLE_DEVICE']=str(rank)
    import torch
    device='cuda:'+str(rank)
    torch.cuda.set_device(device)
    return device
