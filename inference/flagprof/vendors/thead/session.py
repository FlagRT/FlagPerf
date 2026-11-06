# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Official PyTorch profiler with sourced physical PPU identity."""
import json
import os
from pathlib import Path
from flagprof.vendors.nvidia.session import NvidiaProfiler
from flagprof.vendors.session import validate_chrome
from .format import Taxonomy

class TheadProfiler(NvidiaProfiler):
    # The PPU stack and NVIDIA share the profiler ownership domain: never overlap.
    activity_source='cuda'
    def identity(self,config):
        import torch
        selected=torch.cuda.current_device()
        hardware=hardware_identity(selected)
        result=super().identity(config)
        result['hardware']=hardware
        return result
    def validate(self,path):
        taxonomy=Taxonomy()
        result=validate_chrome(path,'trace.json',lambda e,p:e.get('ph')=='X' and e.get('cat') in taxonomy.contract['roles']['device_categories'])
        result['configuration']=dict(requested=dict(status='available',values=self._requested,source='explicit profiler API arguments'),
            actual=dict(status='unknown',reason='derive only from sealed PPU exporter metadata'),comparison=dict(status='unavailable',mismatches=[]))
        return result


def hardware_identity(selected=None):
    """Also identify host-only runs without requiring profiler environment."""
    import torch
    selected=torch.cuda.current_device() if selected is None else selected
    path=os.environ.get('FLAGPROF_THEAD_IDENTITY')
    if not path:raise ValueError('FLAGPROF_THEAD_IDENTITY must name a sourced host identity JSON')
    with Path(path).open('rb') as stream:
        raw=stream.read(65537)
    if len(raw)>65536:raise ValueError('PPU identity byte budget')
    host=json.loads(raw)
    devices=host.get('devices',[])
    if not isinstance(devices,list) or selected>=len(devices):raise ValueError('PPU logical device outside physical selection')
    device=devices[selected]
    if host.get('source')!='ppu-smi' or device.get('model')!='PPU-ZW810E':raise ValueError('sourced PPU-ZW810E model required')
    observed=str(torch.cuda.get_device_properties(selected).uuid).lower().removeprefix('gpu-')
    if observed!=str(device.get('uuid','')).lower().removeprefix('gpu-'):raise ValueError('PPU physical/framework UUID mismatch')
    for field in ('pci_bdf','driver','host_device_node'):
        if not device.get(field):raise ValueError('missing sourced PPU '+field)
    return dict(device,logical_device=selected,source='ppu-smi host query joined to torch.cuda UUID',
                            compatibility_name=torch.cuda.get_device_name(selected))
