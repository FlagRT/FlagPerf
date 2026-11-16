"""NVML installed nvml.h ABI: v3 uses nvmlProcessInfo_t (v2 struct), 24 bytes."""
import ctypes as C
from flagprof.telemetry.contract import Call, Metric, Observation, ReadError, raw_clock
from flagprof.telemetry.process_memory import observations


class Process(C.Structure):
    _fields_=[('pid',C.c_uint),('used',C.c_ulonglong),('gpu_instance',C.c_uint),('compute_instance',C.c_uint)]


def attach(provider,handle,entity):
    if not any(m.name=='process.device_memory' for m in provider.metrics):
        provider.metrics.extend([Metric('process.device_memory','B','process_device','nvmlDeviceGetComputeRunningProcesses_v3',group='slow'),
                                 Metric('process.count','count','chip','nvmlDeviceGetComputeRunningProcesses_v3',group='slow')])
    api=next((n for n in ('nvmlDeviceGetComputeRunningProcesses_v3','nvmlDeviceGetComputeRunningProcesses_v2') if hasattr(provider.lib,n)),None)
    if api is None:
        provider.unavailable['process.device_memory']='NVML process enumeration symbol missing'
        return
    function=getattr(provider.lib,api)
    function.restype=C.c_int;function.argtypes=[C.c_void_p,C.POINTER(C.c_uint),C.POINTER(Process)]
    def read():
        capacity=1024;rows=(Process*capacity)();count=C.c_uint(capacity);targets=dict(provider.targets)
        target_devices=getattr(provider,'target_devices',None);start=raw_clock()
        code=function(handle,C.byref(count),rows);end=raw_clock()
        if code or count.value>capacity:raise ReadError('unsupported' if code in (3,13) else 'read_error',dict(api=api,rc=code,count=count.value,capacity=capacity))
        return observations(entity,[(r.pid,r.used) for r in rows[:count.value]],targets,start,end,api,target_devices=target_devices)
    provider._calls.append(Call(entity+':processes',entity,entity,('process.count',),read,'slow'))


def health(provider,handle,entity):
    """Volatile uncorrectable ECC delta. Existing nonzero totals are not new errors."""
    api='nvmlDeviceGetTotalEccErrors';function=getattr(provider.lib,api,None)
    name='nvidia.ecc.uncorrected'
    if function is None:
        provider.unavailable[name]='symbol_missing';return
    function.restype=C.c_int;function.argtypes=[C.c_void_p,C.c_int,C.c_int,C.POINTER(C.c_ulonglong)]
    if not any(m.name==name for m in provider.metrics):
        provider.metrics.append(Metric(name,'count','chip',api,kind='cumulative',group='slow',roles=('error.cumulative',)))
    previous=[None]
    def read():
        start=raw_clock();value=C.c_ulonglong();code=function(handle,1,0,C.byref(value));end=raw_clock()
        events=[]
        if code==0:
            if previous[0] is not None and value.value>previous[0]:
                events=[dict(t='health',ts=end,kind='ecc',sev='error',detail=dict(entity=entity,
                    previous=previous[0],current=value.value,source=api,normalization='nvml-ecc/1'))]
            elif previous[0] is not None and value.value>=previous[0]:
                events=[dict(t='health',ts=end,kind='heartbeat',sev='info',detail=dict(entity=entity,
                    scope='device_health',healthy=True,source=api))]
            previous[0]=value.value
        else:
            previous[0]=None
            events=[dict(t='health',ts=end,kind='heartbeat',sev='info',detail=dict(entity=entity,
                scope='device_health',healthy=None,source=api))]
        return Observation(entity,{name:value.value if code==0 else None},start,end,
            {} if code==0 else {name:'unsupported' if code in (3,13) else 'read_error'},api,health=events)
    provider._calls.append(Call(entity+':ecc',entity,entity,(name,),read,'slow'))
