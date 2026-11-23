"""DCMI 26.1.1 proc_mem_info: int PID, unsigned long byte count; max 32 entries."""
import ctypes as C
from flagprof.telemetry.contract import Call, Metric, Observation, ReadError, raw_clock
from flagprof.telemetry.process_memory import observations


class Process(C.Structure):
    _fields_=[('pid',C.c_int),('used',C.c_ulong)]


def attach(provider,card,chip,entity):
    api='dcmi_get_device_resource_info';function=getattr(provider.lib,api,None)
    if not any(m.name=='process.device_memory' for m in provider.metrics):
        provider.metrics.extend([Metric('process.device_memory','B','process_device',api,group='slow'),
                                 Metric('process.count','count','chip',api,group='slow')])
    if function is None:
        provider.process_unavailable['process.device_memory']='DCMI process enumeration symbol missing';return
    function.restype=C.c_int
    function.argtypes=[C.c_int,C.c_int,C.POINTER(Process),C.POINTER(C.c_int)]
    def read():
        rows=(Process*32)();count=C.c_int();targets=dict(provider.targets)
        target_devices=getattr(provider,'target_devices',None);start=raw_clock()
        code=function(card,chip,rows,C.byref(count));end=raw_clock()
        if code or not 0<=count.value<=32:raise ReadError('unsupported' if code in (-8255,-8013) else 'read_error',dict(api=api,rc=code,count=count.value,capacity=32))
        return observations(entity,[(r.pid,r.used) for r in rows[:count.value]],targets,start,end,api,target_devices=target_devices)
    provider._calls.append(Call(entity+':processes','card:'+str(card)+':chip:'+str(chip),entity,('process.count',),read,'slow'))


def health(provider,card,chip,entity):
    api='dcmi_get_device_errorcode_v2';function=getattr(provider.lib,api,None)
    if function is None:
        provider.process_unavailable['device.error.codes']='DCMI error-list symbol missing';return
    function.restype=C.c_int
    function.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_uint),C.c_uint]
    name='ascend.error.count'
    if not any(m.name==name for m in provider.metrics):provider.metrics.append(Metric(name,'count','chip',api,group='slow'))
    def read():
        count=C.c_int();codes=(C.c_uint*128)();start=raw_clock()
        code=function(card,chip,C.byref(count),codes,128);end=raw_clock()
        valid=code==0 and 0<=count.value<=128
        errors=[int(v) for v in codes[:count.value] if v] if valid else []
        events=[dict(t='health',ts=end,kind='device_error',sev='error',detail=dict(entity=entity,
            codes=errors,source=api,normalization='dcmi-error-list/1'))] if errors else []
        if not valid:
            events=[dict(t='health',ts=end,kind='heartbeat',sev='info',detail=dict(entity=entity,
                scope='device_health',healthy=None,source=api))]
        return Observation(entity,{name:len(errors) if valid else None},start,end,
            {} if valid else {name:'unsupported' if code in (-8255,-8013) else 'read_error'},api,health=events,
            detail=dict(returncode=code,error_codes=errors))
    provider._calls.append(Call(entity+':errors','card:'+str(card)+':chip:'+str(chip),entity,(name,),read,'slow'))
