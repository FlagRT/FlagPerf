"""Lazy ctypes NVML provider; no CUDA context or torch import."""
import ctypes as C
import time
from flagprof.telemetry.contract import Call, Metric, Observation, ReadError, raw_clock


class Memory(C.Structure):
    _fields_ = [('version',C.c_uint)]+[(n,C.c_ulonglong) for n in ('total','reserved','free','used')]


class Utilization(C.Structure):
    _fields_ = [('gpu',C.c_uint),('memory',C.c_uint)]


class Violation(C.Structure):
    _fields_ = [('reference',C.c_ulonglong),('duration',C.c_ulonglong)]


class Value(C.Union):
    _fields_ = [('dVal',C.c_double),('uiVal',C.c_uint),('ulVal',C.c_ulong),('ullVal',C.c_ulonglong),
                ('sllVal',C.c_longlong),('siVal',C.c_int),('usVal',C.c_ushort)]


class PCI(C.Structure):
    _fields_ = [('legacy',C.c_char*16)]+[(n,C.c_uint) for n in ('domain','bus','device','device_id','subsystem_id')]+[('bus_id',C.c_char*32)]


class Field(C.Structure):
    _fields_ = [('fieldId',C.c_uint),('scopeId',C.c_uint),('timestamp',C.c_longlong),
                ('latencyUsec',C.c_longlong),('valueType',C.c_int),('nvmlReturn',C.c_int),('value',Value)]


def checked(code, api):
    if code:
        raise ReadError('unsupported' if code in (3,13) else 'read_error',api+': '+str(code))


class NVML:
    container_safe = True

    def __init__(self, devices='all', source='auto', library='libnvidia-ml.so.1', processes=False):
        if source not in ('auto','nvml'):
            raise ValueError('NVIDIA source must be auto, nvml or nvidia-smi')
        self.lib=C.CDLL(library)
        checked(self.lib.nvmlInit_v2(),'nvmlInit_v2')
        self.targets={}
        self.metrics=[];self._calls=[];self.unavailable={};self.symbols={}
        count=C.c_uint()
        checked(self.lib.nvmlDeviceGetCount_v2(C.byref(count)),'nvmlDeviceGetCount_v2')
        requested=None if devices=='all' else set(devices.split(','))
        entities=[]
        for index in range(count.value):
            handle=C.c_void_p()
            checked(self.lib.nvmlDeviceGetHandleByIndex_v2(index,C.byref(handle)),'nvmlDeviceGetHandleByIndex_v2')
            uuid=self._string('nvmlDeviceGetUUID',handle)
            if requested is not None and str(index) not in requested and uuid not in requested:
                continue
            entity='chip:'+uuid
            pci=PCI()
            rc=self.lib.nvmlDeviceGetPciInfo_v3(handle,C.byref(pci)) if hasattr(self.lib,'nvmlDeviceGetPciInfo_v3') else 13
            entities.append(dict(entity=entity,uuid=uuid,index=index,name=self._string('nvmlDeviceGetName',handle),
                                 pci=pci.bus_id.decode() if rc==0 else None,pci_returncode=rc,power_domain=entity,
                                 power_sampling=self._power_sampling(handle)))
            self._device(handle,entity)
            if processes:
                from .telemetry_process import attach,health
                attach(self,handle,entity);health(self,handle,entity)
        if not entities:raise ValueError('no selected NVML device discovered')
        semantics={(e['power_sampling']['kind'],e['power_sampling']['window_ns']) for e in entities}
        if len(semantics)==1:
            from dataclasses import replace
            kind,window=next(iter(semantics))
            self.metrics=[replace(m,kind=kind,window_ns=window) if m.name=='power.draw' else m for m in self.metrics]
        self.inventory=dict(source='nvml',container_safe=True,entities=entities,unavailable=self.unavailable,
                            symbols=self.symbols,driver_version=self._string('nvmlSystemGetDriverVersion'),
                            library_version=self._string('nvmlSystemGetNVMLVersion'),call_timeout_s=3,
                            unit_validation='NVML ABI: bytes, mW, MHz, mJ; sample timestamps retain their source clock')

    def _power_sampling(self,handle):
        # NVML documents GA100 and pre-Ampere as instantaneous. Other Ampere
        # and newer devices return a one-second mean. Query capability once.
        result=dict(kind='window',window_ns=None,compute_capability=None)
        fn=getattr(self.lib,'nvmlDeviceGetCudaComputeCapability',None)
        if fn is None:return result
        fn.restype=C.c_int;fn.argtypes=[C.c_void_p,C.POINTER(C.c_int),C.POINTER(C.c_int)]
        major=C.c_int();minor=C.c_int()
        if fn(handle,C.byref(major),C.byref(minor))!=0 or major.value<2:return result
        capability=(major.value,minor.value);instant=major.value<8 or capability==(8,0)
        return dict(kind='instant' if instant else 'window',window_ns=None if instant else 1_000_000_000,
                    compute_capability=list(capability))

    def _string(self,api,*args):
        value=C.create_string_buffer(128)
        checked(getattr(self.lib,api)(*args,value,len(value)),api)
        return value.value.decode()

    def _add(self,handle,entity,names,args,typ,fields,*,group='fast',kind='instant',factor=1,original_unit=''):
        api=next((name for name in names if hasattr(self.lib,name)),None)
        for name,_,unit in fields:
            if not any(m.name==name for m in self.metrics):
                self.metrics.append(Metric(name,unit,'chip',api or names[0],kind=kind,group=group,
                                           factor=factor,original_unit=original_unit,busy_indicator=name=='nvidia.gpu.activity',
                    roles=('restriction.cumulative.'+name.rsplit('.',1)[-1],) if name.startswith('nvidia.violation.') else
                          ('activity',) if name=='nvidia.gpu.activity' else ('power',) if name=='power.draw' else
                          ('temperature',) if name=='temperature.chip' else (),
                    restriction_masks={'thermal':0x60,'power':0x4} if name=='nvidia.clock.reasons' else {}))
        if api is None:
            for name,_,_ in fields:self.unavailable[name]='symbol_missing'
        else:self.symbols[names[0]]=api
        function=getattr(self.lib,api) if api else None
        if function:
            function.restype=C.c_int
            function.argtypes=[C.c_void_p]+[C.c_uint]*len(args)+[C.POINTER(typ)]
        def read():
            value=typ()
            if typ is Memory:value.version=C.sizeof(Memory)|(2<<24)
            start=raw_clock()
            rc=function(handle,*args,C.byref(value)) if function else 13
            end=raw_clock()
            values={name:convert(value) if rc==0 else None for name,convert,_ in fields}
            reason='unsupported' if rc in (3,13) else 'read_error'
            miss={name:reason for name,v in values.items() if v is None}
            for name,v in values.items():
                if v is not None and (v<0 or (name.endswith('.activity') and v>100)):
                    values[name]=None;miss[name]='invalid'
            if typ is Memory and rc == 0 and (value.used > value.total or
                    abs(value.used+value.free+value.reserved-value.total) > 2*1048576):
                for name in values:values[name]=None;miss[name]='invalid'
            detail=dict(returncode=rc)
            if typ is Violation and rc==0:detail['source_reference_us']=value.reference
            return Observation(entity,values,start,end,miss,api or names[0],detail=detail)
        self._calls.append(Call(entity+':'+names[0]+str(args),entity,entity,tuple(n for n,_,_ in fields),read,group,api or names[0]))

    def _device(self,handle,entity):
        self._add(handle,entity,['nvmlDeviceGetMemoryInfo_v2'],(),Memory,[
            ('memory.used',lambda x:x.used,'B'),('memory.total',lambda x:x.total,'B'),('nvidia.memory.system_reserved',lambda x:x.reserved,'B')])
        self._add(handle,entity,['nvmlDeviceGetUtilizationRates'],(),Utilization,[
            ('nvidia.gpu.activity',lambda x:x.gpu,'%'),('nvidia.memory.activity',lambda x:x.memory,'%')],kind='window')
        for api,metric,unit,factor,group,args,typ in [
            ('nvmlDeviceGetPowerUsage','power.draw','W',.001,'fast',(),C.c_uint),
            ('nvmlDeviceGetEnforcedPowerLimit','power.limit','W',.001,'slow',(),C.c_uint),
            ('nvmlDeviceGetTotalEnergyConsumption','energy.total','J',.001,'fast',(),C.c_ulonglong),
            ('nvmlDeviceGetTemperature','temperature.chip','C',1,'slow',(0,),C.c_uint),
            ('nvmlDeviceGetClockInfo','clock.compute','Hz',1e6,'fast',(1,),C.c_uint),
            ('nvmlDeviceGetClockInfo','clock.memory','Hz',1e6,'fast',(2,),C.c_uint),
            ('nvmlDeviceGetMaxClockInfo','clock.compute.max','Hz',1e6,'slow',(1,),C.c_uint)]:
            self._add(handle,entity,[api],args,typ,[(metric,lambda x,f=factor:x.value*f,unit)],factor=factor,group=group,
                      kind='window' if metric=='power.draw' else 'cumulative' if metric=='energy.total' else 'instant')
        self._add(handle,entity,['nvmlDeviceGetCurrentClocksEventReasons','nvmlDeviceGetCurrentClocksThrottleReasons'],(),C.c_ulonglong,
                  [('nvidia.clock.reasons',lambda x:x.value,'bitmask')])
        for policy,name in [(0,'power'),(1,'thermal')]:
            self._add(handle,entity,['nvmlDeviceGetViolationStatus'],(policy,),Violation,
                      [('nvidia.violation.'+name,lambda x:x.duration,'ns')],group='slow',kind='cumulative')
        self.metrics.append(Metric('temperature.memory','C','chip','nvmlDeviceGetFieldValues',group='slow')) if not any(m.name=='temperature.memory' for m in self.metrics) else None
        def field():
            value=Field(fieldId=82)
            start=raw_clock();wall=time.time_ns()
            rc=self.lib.nvmlDeviceGetFieldValues(handle,1,C.byref(value));end=raw_clock()
            rc=rc or value.nvmlReturn
            names=('dVal','uiVal','ulVal','ullVal','sllVal','siVal','usVal')
            valid=rc==0 and 0<=value.valueType<len(names)
            number=getattr(value.value,names[value.valueType]) if valid else None
            sample={}
            # FieldValues uses epoch microseconds. A clock bracket checks the
            # conversion on each read; stale source time is not replaced by now.
            wall_end=time.time_ns()
            conversion=abs((wall_end-wall)-(end-start))<=1_000_000
            converted=value.timestamp*1000-wall+start
            if valid and conversion and 0<=converted<=end and value.timestamp>0:
                sample['temperature.memory']=converted
            return Observation(entity,{'temperature.memory':number},start,end,
                {} if valid else {'temperature.memory':'unsupported' if rc==3 else 'read_error'},
                'nvmlDeviceGetFieldValues',sample,dict(source_timestamp_us=value.timestamp,source_clock='unix_us',
                    conversion_valid=conversion,wall_raw_offset_ns=wall-start,source_latency_us=value.latencyUsec,returncode=rc))
        self._calls.append(Call(entity+':field:82',entity,entity,('temperature.memory',),field,'slow'))

    def calls(self):return list(self._calls)


def Provider(devices='all', source='auto', **kwargs):
    from .telemetry_smi import Provider as SMI
    if source == 'nvidia-smi':
        return SMI(devices=devices)
    try:
        return NVML(devices=devices, source=source, **kwargs)
    except (OSError, ReadError) as error:
        if source != 'auto':
            raise
        result=SMI(devices=devices)
        result.inventory['initial_source_failure']=str(error)
        return result
