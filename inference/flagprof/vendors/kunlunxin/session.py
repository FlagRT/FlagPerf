# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Official XPyTorch profiler with sourced physical P800 identity."""
import json
import os
from pathlib import Path
from flagprof.vendors.nvidia.session import NvidiaProfiler
from flagprof.vendors.session import validate_chrome
from .format import Taxonomy

class KunlunxinProfiler(NvidiaProfiler):
    # XPyTorch and NVIDIA share the profiler ownership domain: never overlap.
    activity_source='cuda'
    def approve_prerequisites(self,prerequisites):
        import torch
        expected={'XPU_ENABLE_PROFILER_TRACING':'1','XPU_CUPTI_ENABLE_DEVICE':str(torch.cuda.current_device())}
        return (isinstance(prerequisites,list) and len(prerequisites)==2 and
                {p.get('name'):p.get('value') for p in prerequisites}==expected and
                all(os.environ.get(k)==v for k,v in expected.items()))
    def identity(self,config):
        import torch
        selected=torch.cuda.current_device()
        required={'XPU_ENABLE_PROFILER_TRACING':'1','XPU_CUPTI_ENABLE_DEVICE':str(selected)}
        if any(os.environ.get(k)!=v for k,v in required.items()):
            raise ValueError('P800 profiler prerequisites: '+repr(required)+'; set before importing torch')
        hardware=hardware_identity(selected)
        result=super().identity(config)
        result['hardware']=hardware
        result['collector_prerequisites']=[dict(name=k,value=v,reason='XPyTorch official device tracing',scope='diagnostic capture process') for k,v in required.items()]
        result['collector']['prerequisites']=required
        return result
    def validate(self,path):
        taxonomy=Taxonomy()
        result=validate_chrome(path,'trace.json',lambda e,p:e.get('ph')=='X' and e.get('cat') in taxonomy.contract['roles']['device_categories'])
        result['configuration']=dict(requested=dict(status='available',values=self._requested,source='explicit profiler API arguments'),
            actual=dict(status='unknown',reason='derive only from sealed P800 exporter metadata'),comparison=dict(status='unavailable',mismatches=[]))
        return result


def hardware_identity(selected=None):
    """Also identify host-only runs without requiring profiler environment."""
    import torch
    selected=torch.cuda.current_device() if selected is None else selected
    path=os.environ.get('FLAGPROF_P800_IDENTITY')
    if not path:raise ValueError('FLAGPROF_P800_IDENTITY must name a sourced host identity JSON')
    with Path(path).open('rb') as stream:
        raw=stream.read(65537)
    if len(raw)>65536:raise ValueError('P800 identity byte budget')
    host=json.loads(raw)
    devices=host.get('devices',[])
    if not isinstance(devices,list) or selected>=len(devices):raise ValueError('P800 logical device outside physical selection')
    device=devices[selected]
    if host.get('source')!='xpu-smi' or device.get('model')!='P800':raise ValueError('sourced P800 model required')
    observed=str(torch.cuda.get_device_properties(selected).uuid).lower().removeprefix('gpu-')
    if observed!=str(device.get('uuid','')).lower().removeprefix('gpu-'):raise ValueError('P800 physical/framework UUID mismatch')
    for field in ('pci_bdf','driver','xpu_smi_runtime','host_device_node'):
        if not device.get(field):raise ValueError('missing sourced P800 '+field)
    return dict(device,logical_device=selected,source='xpu-smi host query joined to torch.cuda UUID',
                            compatibility_name=torch.cuda.get_device_name(selected))
