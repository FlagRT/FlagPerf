# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Existing Torch vendor routes exposed through the common operation contract.

Non-NVIDIA container device options must be supplied explicitly; this adapter
never guesses a device namespace or claims hardware validation.
"""
import importlib
import os
from pathlib import Path


class Adapter:
    def __init__(self, name):
        self.name = name

    def default_image(self):
        return None

    def runtime_identity(self, image, info):
        return {'image': image, 'image_id': info['Id'], 'validation': 'identity-recorded; hardware-unverified'}

    def bootstrap(self, device_id):
        if self.name == 'cambricon':
            importlib.import_module('torch_mlu')
        elif self.name == 'kunlunxin':
            importlib.import_module('torch_xmlir')
        elif self.name in ('metax', 'iluvatar'):
            pass  # Their vendor PyTorch builds expose the existing CUDA interface.
        import torch
        # The checked-in xpytorch029 install verifies tensors through .cuda().
        backend = 'mlu' if self.name == 'cambricon' else 'cuda'
        api = getattr(torch, backend)
        api.set_device(device_id)
        if backend == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
        self.api = api
        self.device = torch.device(f'{backend}:{device_id}')
        return self.device

    def identity(self):
        return {'vendor':self.name, 'device':str(self.device), 'hardware_validation':'not-established'}

    def synchronize(self):
        self.api.synchronize(self.device)

    def kernel_time(self, fn, cfg):
        if self.name != 'nvidia':
            return None
        from triton.testing import do_bench
        return do_bench(fn, warmup=cfg['KERNELWARMUP'], rep=cfg['KERNELITERS'], return_mode='median') / 1000

    def environment(self, device, mode="probe", local=False):
        env = {'DO_NOT_TRACK': '1', 'TORCH_DEVICE_BACKEND_AUTOLOAD': '0'}
        if self.name == 'kunlunxin':
            env.update(CUDART_DUMMY_REGISTER='1', TRITON_XPU_ARCH='3', XPURT_DISPATCH_MODE='PROFILING')
        return env

    def measurement_fallback(self, log):
        return False

    def docker_options(self, device, args):
        if self.name == 'nvidia':
            return ['--gpus', f'device={device}']
        if not args.container_device:
            raise ValueError(f'{self.name}: supply --container-device for the vendor runtime')
        return []

    def lease_root(self):
        return Path('/tmp') / f'flagperf-operation-{self.name}-leases'

    def preflight(self, root, devices, label='preflight'):
        if self.name != 'nvidia':
            raise RuntimeError(f'{self.name}: automatic host occupancy checking is not integrated; use local execution in an externally reserved runtime')
        import subprocess
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
        mapping = dict(line.replace(' ', '').split(',') for line in text.strip().splitlines())
        selected = [mapping[str(d)] for d in devices]
        running = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if any(line.split(',')[0].strip() in selected for line in running.splitlines()):
            raise RuntimeError('selected NVIDIA device is occupied')
        return {'device_ids': devices, 'uuid': selected, 'occupancy': 'idle'}

    def diagnose(self, error, root=None):
        return None

    def route(self, probe, log):
        calls = probe.get('calls', [])
        gems = [x for x in calls if 'flag_gems' in x['path']]
        on_device = bool(probe.get('output_devices')) and all(d == probe.get('device') and d != 'cpu' for d in probe['output_devices'])
        return {'status': 'partial' if on_device else 'failed', 'output_device': probe.get('output_devices'),
                'implementation': gems, 'reason': 'device placement and Python calls recorded; kernel dispatch not proven'}
