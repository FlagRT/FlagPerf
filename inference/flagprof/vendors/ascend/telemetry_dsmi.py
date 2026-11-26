# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Chip telemetry from the devices visible to the driver in this process."""
import ctypes as C
from functools import partial

from flagprof.telemetry.contract import Call, Metric, Observation, raw_clock
from .telemetry import HBM, checked
from .telemetry_health import health_events


LIBRARY = 'libdrvdsmi_host.so'
HBM_API = 'dsmi_get_hbm_info'
SCALARS = (
    (Metric('temperature.chip', 'C', source_api='dsmi_get_device_temperature',
            roles=('temperature',), group='slow'), C.c_int, ()),
    (Metric('ascend.health', 'code', source_api='dsmi_get_device_health', group='slow'), C.c_uint, ()),
    (Metric('ascend.aicore.activity', '%', source_api='dsmi_get_device_utilization_rate',
            roles=('activity',), kind='window', profiler_conflict=True, busy_indicator=True), C.c_uint, (2,)),
)


def bind(lib, name, *types):
    fn = getattr(lib, name)
    fn.argtypes = list(types)
    fn.restype = C.c_int
    return fn


class Provider:
    container_safe = True

    def __init__(self, devices='all'):
        lib = C.CDLL(LIBRARY)
        count = C.c_int()
        checked(bind(lib, 'dsmi_get_device_count', C.POINTER(C.c_int))(C.byref(count)), 'dsmi_get_device_count')
        if not 0 < count.value <= 4096:
            raise ValueError('invalid visible DSMI device count: '+str(count.value))
        logical_ids = (C.c_int * count.value)()
        checked(bind(lib, 'dsmi_list_device', C.POINTER(C.c_int), C.c_int)(logical_ids, count.value), 'dsmi_list_device')
        physical_id = bind(lib, 'dsmi_get_phyid_from_logicid', C.c_uint, C.POINTER(C.c_uint))
        self.hbm = bind(lib, HBM_API, C.c_int, C.POINTER(HBM))
        selected = None if devices == 'all' else set(map(int, devices.split(',')))
        self.metrics = [Metric(name, 'B', source_api=HBM_API, original_unit='KiB', factor=1024)
                        for name in ('memory.used', 'memory.total')]
        sensors = []
        unavailable = {'power.draw': 'card power domain not established by chip enumeration',
                       'process.device_memory': 'process enumeration not collected by DSMI provider'}
        for metric, typ, arguments in SCALARS:
            if not hasattr(lib, metric.source_api):
                unavailable[metric.name] = 'driver symbol missing: '+metric.source_api
                continue
            fn = bind(lib, metric.source_api, C.c_int, *([C.c_int] * len(arguments)), C.POINTER(typ))
            sensors.append((metric, fn, typ, arguments))
            self.metrics.append(metric)
        self._calls = []
        entities = []
        visible = set()
        for logical in logical_ids:
            physical = C.c_uint(-1)
            checked(physical_id(logical, C.byref(physical)), 'dsmi_get_phyid_from_logicid')
            if logical < 0 or physical.value == C.c_uint(-1).value or physical.value in visible:
                raise ValueError('ambiguous DSMI device identity')
            visible.add(physical.value)
            if selected is not None and physical.value not in selected:
                continue
            entity = 'chip:'+str(physical.value)
            entities.append(dict(entity=entity, phyid=physical.value, logical_id=logical,
                                 identity_source='dsmi_get_phyid_from_logicid'))
            self._calls.append(Call(entity+':hbm', entity, entity, ('memory.used', 'memory.total'),
                                    partial(self.read_hbm, logical, entity), source_api=HBM_API))
            for metric, fn, typ, arguments in sensors:
                self._calls.append(Call(entity+':'+metric.name, entity, entity, (metric.name,),
                                        partial(self.read_scalar, logical, entity, metric, fn, typ, arguments),
                                        metric.group, metric.source_api))
        if not entities or (selected is not None and not selected <= visible):
            raise ValueError('selected physical device missing from DSMI inventory')
        self.inventory = dict(source='dsmi', container_safe=True, entities=entities, unavailable=unavailable,
                              minimum_interval_ms=100, call_timeout_s=3, calibration_rounds=3)

    def calls(self):
        return list(self._calls)

    def read_hbm(self, logical, entity):
        info = HBM()
        start = raw_clock()
        rc = self.hbm(logical, C.byref(info))
        end = raw_clock()
        valid = rc == 0 and info.total > 0 and info.used <= info.total
        values = {'memory.used': info.used * 1024 if valid else None,
                  'memory.total': info.total * 1024 if valid else None}
        miss = {} if valid else dict.fromkeys(values, 'read_error' if rc else 'invalid')
        return Observation(entity, values, start, end, miss, HBM_API, detail=dict(returncode=rc))

    @staticmethod
    def read_scalar(logical, entity, metric, fn, typ, arguments):
        value = typ()
        start = raw_clock()
        rc = fn(logical, *arguments, C.byref(value))
        end = raw_clock()
        valid = (rc == 0 and (metric.unit != '%' or value.value <= 100)
                 and (metric.name != 'ascend.health' or value.value in (0, 1, 2, 3)))
        number = value.value if valid else None
        miss = {} if valid else {metric.name: 'read_error' if rc else 'invalid'}
        return Observation(entity, {metric.name: number}, start, end, miss, metric.source_api, detail=dict(returncode=rc),
                           health=health_events(entity, number, end) if metric.name == 'ascend.health' else [])
