# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Official torch_npu.profiler backend; SDK imports are lazy."""
from __future__ import annotations
import platform
import json
from pathlib import Path
from typing import Any,Callable,Mapping
from copy import deepcopy
from flagprof.vendors.session import SessionConfig,validate_chrome,requested_configuration
from flagprof.vendors.contracts import load_contract

class AscendProfiler:
    activity_source='npu'

    def __init__(self, *, hardware_identity: Mapping[str,Any] | None = None,
                 profiler_level: str = 'Level0', data_simplification: bool = True) -> None:
        if hardware_identity is not None and not hardware_identity.get('source'):
            raise ValueError('hardware_identity requires a source')
        self.hardware_identity=deepcopy(dict(hardware_identity)) if hardware_identity is not None else dict(status='unknown',reason='no sourced driver/firmware/physical assignment supplied')
        if profiler_level not in ('Level0','Level1','Level2') or type(data_simplification) is not bool:
            raise ValueError('invalid profiler level or data simplification')
        self.profiler_level=profiler_level;self.data_simplification=data_simplification
        self._requested=None
        self.contract=load_contract(__file__,'4')

    def requested(self, config: SessionConfig) -> dict[str,Any]:
        self._requested=dict(requested_configuration(config,['CPU','NPU']),
            profiler_level=self.profiler_level,data_simplification=self.data_simplification)
        return deepcopy(self._requested)

    def conflict(self) -> str:
        # This installed torch_npu version exposes no reliable active-session
        # query. Do not probe by starting an overlapping profiler.
        return 'not_detectable'

    def identity(self, config: SessionConfig) -> dict[str,Any]:
        import torch,torch_npu
        return dict(hardware=dict(model=torch.npu.get_device_name(),logical_device=torch.npu.current_device(),
                                 source='torch.npu public device API',additional=self.hardware_identity),
                    collector=dict(name='torch_npu.profiler',version=torch_npu.__version__,torch=torch.__version__,
                                   python=platform.python_version(),activities=['CPU','NPU'],
                                   requested_configuration=self.requested(config)),
                    collector_prerequisites=[],private_detection_symbols=[])

    def set_destination(self, path: Path) -> None:
        import torch_npu
        # Public factory sets the next official capture directory. It is
        # called only after the preceding window's manifest was fsynced.
        self.handler=torch_npu.profiler.tensorboard_trace_handler(str(path),analyse_flag=True,async_mode=False)

    def create(self, config: SessionConfig, on_ready: Callable[[Any],None]) -> Any:
        import torch_npu
        p=torch_npu.profiler
        self.requested(config)
        return p.profile(activities=[p.ProfilerActivity.CPU,p.ProfilerActivity.NPU],
            schedule=p.schedule(wait=0,warmup=0,active=config.active,repeat=config.windows),
            record_shapes=config.record_shapes,with_stack=config.with_stack,profile_memory=config.profile_memory,
            with_flops=config.with_flops,with_modules=config.with_modules,
            experimental_config=p._ExperimentalConfig(profiler_level=getattr(p.ProfilerLevel,self.profiler_level),
                                                       data_simplification=self.data_simplification),
            on_trace_ready=on_ready)

    def export(self, profiler: Any, path: Path) -> None:
        self.handler(profiler)

    def create_final(self,config,path,on_ready):
        import torch,torch_npu
        torch.npu.init();p=torch_npu.profiler
        handler=p.tensorboard_trace_handler(str(path),analyse_flag=False)
        def ready(profiler):
            on_ready(profiler);handler(profiler)
        return p.profile(activities=[p.ProfilerActivity.CPU,p.ProfilerActivity.NPU],
            **{k:getattr(config,k) for k in ('record_shapes','with_stack','profile_memory','with_flops','with_modules')},
            experimental_config=p._ExperimentalConfig(profiler_level=getattr(p.ProfilerLevel,self.profiler_level),
                                                       data_simplification=self.data_simplification),on_trace_ready=ready)

    def export_final(self,profiler,path):
        # The SDK delivered raw during stop; analysis runs after owner exit.
        pass

    def validate(self, path: Path) -> dict[str,Any]:
        result = validate_chrome(path,'**/ASCEND_PROFILER_OUTPUT/trace_view.json',
            lambda e,p:e.get('ph')=='X' and p.get(e.get('pid')) in self.contract['roles']['device_process_names'])
        actual=[]
        for file in sorted(path.glob('**/profiler_info*.json')):
            try:
                info=json.loads(file.read_bytes())
                actual.append(dict(source=str(file.relative_to(path)),config=info['config'],
                                   cann_version=info.get('cann_version'),torch_npu_version=info.get('torch_npu_version')))
            except (ValueError,KeyError,OSError) as error:
                result.update(valid=False,reasons=result['reasons']+['invalid_profiler_info: '+str(error)])
        result['actual_configuration']=actual
        if not actual: result.update(valid=False,reasons=result['reasons']+['profiler_info_missing'])
        values=[];mismatches=[]
        for item in actual:
            common=item['config'].get('common_config',{})
            extra=item['config'].get('experimental_config',{})
            value={k:common.get(k) for k in ('record_shapes','with_stack','profile_memory','with_flops','with_modules')}
            value.update(profiler_level=extra.get('_profiler_level'),data_simplification=extra.get('_data_simplification'),
                         activities=[str(x).split('.')[-1] for x in common.get('activities',[])],
                         schedule={k:common.get('schedule',{}).get(k) for k in ('wait','warmup','active','repeat')})
            values.append(dict(source=item['source'],values=value,sdk_version=item.get('cann_version')))
            if self._requested is not None:
                for key,expected in self._requested.items():
                    if value.get(key)!=expected or type(value.get(key)) is not type(expected):
                        mismatches.append(dict(field=key,requested=expected,actual=value.get(key),source=item['source']))
        if mismatches:
            result.update(valid=False,reasons=result['reasons']+[
                ('configuration_missing:' if x['actual'] is None else 'configuration_mismatch:')+x['field']
                for x in mismatches])
        known=bool(values) and all(v['values']==values[0]['values'] for v in values)
        result['configuration']=dict(
            requested=dict(status='available',values=self._requested,source='explicit profiler API arguments') if self._requested is not None else dict(status='unknown',reason='request not observed'),
            actual=dict(status='available',values=values[0]['values'],sources=values) if known else dict(status='unknown',reason='actual configuration missing or conflicting',sources=values),
            comparison=dict(status='mismatch' if mismatches else 'match' if known and self._requested is not None else 'unavailable',mismatches=mismatches))
        return result
