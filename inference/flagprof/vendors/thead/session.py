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


# Sourced-identity admission pairs each accepted vendor smi tool with the
# device model strings that tool reports. The framework UUID cross-check below
# remains the real identity anchor; this table only admits identities that
# plausibly come from a real vendor query.
SOURCED_IDENTITY = {
    'ppu-smi': ('PPU-ZW810E',),
    'xpu-smi': ('P800', 'P800 OAM', 'GPU'),
    'npu-smi': ('Ascend910B', 'Ascend910C', 'Ascend910', 'Atlas A2'),
}


def validate_sourced_identity(host, device, selected):
    """Validate a sourced host identity against the admitted smi/model pairs."""
    source=host.get('source')
    accepted=SOURCED_IDENTITY.get(source)
    if accepted is None or device.get('model') not in accepted:
        raise ValueError('sourced identity model mismatch for source %r (accepted smi tools: %s)'
                         % (source, ', '.join(sorted(SOURCED_IDENTITY))))
    for field in ('pci_bdf', 'driver', 'host_device_node'):
        if not device.get(field):raise ValueError('missing sourced '+str(source)+' '+field)


def hardware_identity(selected=None):
    """Also identify host-only runs without requiring profiler environment."""
    import torch
    selected=torch.cuda.current_device() if selected is None else selected
    path=os.environ.get('FLAGPROF_THEAD_IDENTITY')
    if not path:raise ValueError('FLAGPROF_THEAD_IDENTITY must name a sourced host identity JSON')
    with Path(path).open('rb') as stream:
        raw=stream.read(65537)
    if len(raw)>65536:raise ValueError('identity byte budget')
    host=json.loads(raw)
    devices=host.get('devices',[])
    if not isinstance(devices,list) or selected>=len(devices):raise ValueError('logical device outside physical selection')
    device=devices[selected]
    validate_sourced_identity(host, device, selected)
    observed=str(torch.cuda.get_device_properties(selected).uuid).lower().removeprefix('gpu-')
    if observed!=str(device.get('uuid','')).lower().removeprefix('gpu-'):raise ValueError('physical/framework UUID mismatch')
    return dict(device,logical_device=selected,source=str(host.get('source'))+' host query joined to torch.cuda UUID',
                            compatibility_name=torch.cuda.get_device_name(selected))
