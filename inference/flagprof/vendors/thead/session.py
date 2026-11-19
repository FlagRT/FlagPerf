# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Official PyTorch profiler with sourced physical PPU identity."""
import os
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
    from flagprof.vendors.smi_identity import hardware_identity as identify
    return identify(selected, command='ppu-smi', prefix='PPU', env_key='FLAGPROF_THEAD_IDENTITY',
                    node_pattern='alixpu_ppu[0-9]*', model_names={'PPU-ZW810E': 'PPU-ZW810E'})
