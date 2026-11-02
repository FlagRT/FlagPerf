"""Official torch.profiler backend. Importing this module does not load torch."""
from __future__ import annotations
import platform
from pathlib import Path
from typing import Any,Callable,Mapping
from copy import deepcopy
from .format import Taxonomy
from flagprof.vendors.session import SessionConfig,validate_chrome,requested_configuration

class NvidiaProfiler:
    activity_source='cuda'

    def __init__(self, *, hardware_identity: Mapping[str,Any] | None = None) -> None:
        if hardware_identity is not None and not hardware_identity.get('source'):
            raise ValueError('hardware_identity requires a source')
        self.hardware_identity=deepcopy(dict(hardware_identity)) if hardware_identity is not None else dict(status='unknown',reason='no sourced driver/firmware/physical assignment supplied')
        self._requested=None

    def conflict(self) -> str:
        import torch
        # Detection only: recorded private symbol, never used to start/stop.
        flag=getattr(torch.autograd.profiler,'_is_profiler_enabled',None)
        if flag is None:return 'not_detectable'
        return 'conflict' if (flag() if callable(flag) else flag) else 'detectable'

    def identity(self, config: SessionConfig) -> dict[str,Any]:
        import torch
        self._requested=requested_configuration(config,['CPU','CUDA'])
        return dict(hardware=dict(model=torch.cuda.get_device_name(),logical_device=torch.cuda.current_device(),
                                 source='torch.cuda public device API',additional=self.hardware_identity),
                    collector=dict(name='torch.profiler',version=torch.__version__,cuda=torch.version.cuda,
                                   python=platform.python_version(),activities=['CPU','CUDA'],requested_configuration=self._requested),
                    collector_prerequisites=[],private_detection_symbols=['torch.autograd.profiler._is_profiler_enabled'])

    def set_destination(self, path: Path) -> None:
        self.destination=path

    def create(self, config: SessionConfig, on_ready: Callable[[Any],None]) -> Any:
        import torch
        self._requested=requested_configuration(config,['CPU','CUDA'])
        return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0,warmup=0,active=config.active,repeat=config.windows),
            record_shapes=config.record_shapes,with_stack=config.with_stack,profile_memory=config.profile_memory,
            with_flops=config.with_flops,with_modules=config.with_modules,on_trace_ready=on_ready)

    def export(self, profiler: Any, path: Path) -> None:
        profiler.export_chrome_trace(str(path/'trace.json'))

    def create_final(self,config,path,on_ready):
        import torch
        if not torch.cuda.is_available():raise ValueError('CUDA unavailable')
        return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            **{k:getattr(config,k) for k in ('record_shapes','with_stack','profile_memory','with_flops','with_modules')})

    def export_final(self,profiler,path):self.export(profiler,path)

    def validate(self, path: Path) -> dict[str,Any]:
        taxonomy=Taxonomy()
        result=validate_chrome(path,'trace.json',lambda e,p:e.get('ph')=='X' and e.get('cat') in taxonomy.contract['roles']['device_categories'])
        result['configuration']=dict(
            requested=dict(status='available',values=self._requested,source='explicit profiler API arguments') if self._requested is not None else dict(status='unknown',reason='request not observed'),
            actual=dict(status='unknown',reason='no independent actual configuration in Chrome export'),
            comparison=dict(status='unavailable',mismatches=[]))
        return result
