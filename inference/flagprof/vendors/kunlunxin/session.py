# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Official XPyTorch profiler with sourced physical P800 identity."""
import os
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
    from flagprof.vendors.smi_identity import hardware_identity as identify
    return identify(selected, command='xpu-smi', prefix='XPU', env_key='FLAGPROF_P800_IDENTITY',
                    node_pattern='xpu[0-9]*', model_names={'P800 OAM': 'P800', 'P800': 'P800'}, runtime='XPU-RT Version')
