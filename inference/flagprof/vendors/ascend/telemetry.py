"""Read-only DCMI telemetry. ABI follows dcmi_interface_api.h (26.1.1)."""
import ctypes as C
from flagprof.telemetry.contract import Call, Metric, Observation, ReadError, raw_clock


class HBM(C.Structure):
    _fields_ = [('total', C.c_ulonglong), ('frequency', C.c_uint), ('used', C.c_ulonglong),
                ('temperature', C.c_int), ('bandwidth', C.c_uint)]


class Multi(C.Structure):
    _fields_ = [(n, C.c_uint) for n in ('aic', 'aiv', 'aicore', 'npu')]+[('reserved', C.c_uint*8)]


class PCI(C.Structure):
    _fields_ = [(n, C.c_uint) for n in ('vendor', 'subvendor', 'device', 'subdevice')]+[
        ('domain', C.c_int), ('bus', C.c_uint), ('slot', C.c_uint), ('function', C.c_uint),
        ('reserved', C.c_ubyte*32)]


def checked(code, api):
    if code:
        raise ReadError('unsupported' if code in (-8255,-8013) else 'read_error', api+': '+str(code))


class DCMI:
    container_safe = False

    def __init__(self, devices='all', source='auto', library='/usr/local/dcmi/libdcmi.so', processes=False):
        if source not in ('auto','dcmi'):
            raise ValueError('Ascend source must be auto, dcmi or npu-smi')
        self.lib = C.CDLL(library)
        checked(self.lib.dcmi_init(), 'dcmi_init')
        self.targets = {}
        self.process_unavailable = {}
        self.metrics = []
        self._calls = []
        count, cards = C.c_int(), (C.c_int*64)()
        checked(self.lib.dcmi_get_card_list(C.byref(count), cards, len(cards)), 'dcmi_get_card_list')
        entities = []
        selected = None if devices == 'all' else set(map(int, devices.split(',')))
        for card in cards[:count.value]:
            maximum, mcu, cpu = C.c_int(), C.c_int(), C.c_int()
            checked(self.lib.dcmi_get_device_id_in_card(card,C.byref(maximum),C.byref(mcu),C.byref(cpu)), 'dcmi_get_device_id_in_card')
            members = []
            for chip in range(maximum.value):
                unit = C.c_int()
                rc = self.lib.dcmi_get_device_type(card, chip, C.byref(unit))
                if rc or unit.value != 0:  # NPU_TYPE from header
                    continue
                logical, physical = C.c_int(), C.c_uint()
                checked(self.lib.dcmi_get_device_logic_id(C.byref(logical), card, chip), 'dcmi_get_device_logic_id')
                checked(self.lib.dcmi_get_device_phyid_from_logicid(logical, C.byref(physical)), 'dcmi_get_device_phyid_from_logicid')
                phy = physical.value
                members.append('chip:'+str(phy))
                if selected is not None and phy not in selected:
                    continue
                pci = PCI()
                checked(self.lib.dcmi_get_device_pcie_info_v2(card,chip,C.byref(pci)), 'dcmi_get_device_pcie_info_v2')
                entity = dict(entity='chip:'+str(phy), phyid=phy, card_id=card, chip_id=chip,
                              power_domain='card:'+str(card), pci='%04x:%02x:%02x.%x'%(pci.domain,pci.bus,pci.slot,pci.function))
                entities.append(entity)
                self._chip(card, chip, entity['entity'])
                if processes:
                    from .telemetry_process import attach,health
                    attach(self,card,chip,entity['entity']);health(self,card,chip,entity['entity'])
            chosen = [e for e in entities if e.get('card_id') == card and e['entity'].startswith('chip:')]
            if chosen:
                power_entity = 'card:'+str(card)
                entities.append(dict(entity=power_entity, members=members, power_domain=power_entity))
                self._add(card, 0, power_entity, 'dcmi_get_device_power_info', (), C.c_int,
                          [('power.draw', lambda x:x.value*.1, 'W')], scope='card', factor=.1, original_unit='0.1 W',kind='window')
        if not self._calls:
            raise ValueError('no selected DCMI chip discovered')
        self.inventory = dict(entities=entities, source='dcmi', container_safe=False, call_timeout_s=3, minimum_interval_ms=150.378,
                              sampling_calibration='T15.1 host-16x100-corrected: 2 * round p99 75.18898 ms',
                              unit_validation='M0: HBM MiB, power 0.1 W, frequency MHz',
                              unavailable={'power.limit':'A3 returned 950000 outside documented 150000..600000; unit unconfirmed',
                                           'ascend.acg':'ACG not listed among supported A3 LP commands',
                                           'restriction.cumulative':'DCMI registered source has no validated cumulative restriction duration',
                                           'restriction.instant.named':'raw DCMI reason mask retained; named bit meanings not validated',
                                           **self.process_unavailable})
        for name, api in [('driver_version','dcmi_get_driver_version'),('library_version','dcmi_get_dcmi_version')]:
            buf = C.create_string_buffer(128)
            rc = getattr(self.lib, api)(buf, len(buf))
            self.inventory[name] = buf.value.decode(errors='replace') if rc == 0 else dict(rc=rc)
        if self.inventory.get('driver_version') == '26.1.1':
            from dataclasses import replace
            self.metrics = [replace(m,profiler_conflict=False) if m.name=='ascend.npu.activity' else m for m in self.metrics]
            self.inventory['type13_calibration']='T15.1: final median 13 vs off 20; windowed 10 vs outside 17, same workload'

    def _add(self, card, chip, entity, api, args, typ, fields, *, scope='chip', factor=1,
             original_unit='', group='fast', kind='instant', conflict=False):
        names = tuple(x[0] for x in fields)
        for name, _, unit in fields:
            if not any(m.name == name for m in self.metrics):
                self.metrics.append(Metric(name, unit, scope, api, kind=kind, group=group,
                    factor=factor, original_unit=original_unit, profiler_conflict=conflict,
                    busy_indicator=name=='ascend.aicore.activity',
                    roles=('activity',) if name in ('ascend.aicore.activity','ascend.npu.activity') else
                          ('power',) if name=='power.draw' else ('temperature',) if name=='temperature.chip' else ()))
        function = getattr(self.lib, api)
        function.restype = C.c_int
        function.argtypes = [C.c_int, C.c_int]+[C.c_int]*len(args)+[C.POINTER(typ)]
        def read():
            value = typ(); start = raw_clock()
            rc = function(card, chip, *args, C.byref(value)); end = raw_clock()
            reason = 'unsupported' if rc in (-8255,-8013) else 'read_error'
            values = {name: convert(value) if rc == 0 else None for name,convert,_ in fields}
            miss = {name: reason for name,v in values.items() if v is None}
            for name,v in values.items():
                if v is not None and (v < 0 or (name.endswith('.activity') and v > 100)):
                    values[name] = None; miss[name] = 'invalid'
            if typ is HBM and rc == 0 and value.used > value.total:
                values['memory.used']=None; miss['memory.used']='invalid'
            from .telemetry_health import health_events
            return Observation(entity, values, start, end, miss, api, detail=dict(returncode=rc),
                               health=health_events(entity, values.get('ascend.health'), end))
        self._calls.append(Call(entity+':'+api+str(args), 'card:'+str(card)+':chip:'+str(chip), entity, names, read, group, api))

    def _chip(self, card, chip, entity):
        self._clock_reasons(card,chip,entity)
        self._add(card,chip,entity,'dcmi_get_device_hbm_info',(),HBM,[
            ('memory.used',lambda x:x.used*1048576,'B'),('memory.total',lambda x:x.total*1048576,'B'),
            ('temperature.memory',lambda x:x.temperature,'C')])
        self._add(card,chip,entity,'dcmi_get_device_multi_utilization_rate',(),Multi,
            [('ascend.'+name+'.activity',lambda x,n=name:getattr(x,n),'%') for name in ('aic','aiv','aicore','npu')],
            kind='window',conflict=True)
        for kind,name in [(14,'cube'),(10,'hbm'),(3,'aicpu')]:
            self._add(card,chip,entity,'dcmi_get_device_utilization_rate',(kind,),C.c_uint,
                      [('ascend.'+name+'.activity',lambda x:x.value,'%')],kind='window',conflict=True)
        for kind,name in [(7,'compute'),(6,'memory'),(9,'compute.max')]:
            self._add(card,chip,entity,'dcmi_get_device_frequency',(kind,),C.c_uint,
                      [('clock.'+name,lambda x:x.value*1e6,'Hz')],factor=1e6,original_unit='MHz')
        self._add(card,chip,entity,'dcmi_get_device_temperature',(),C.c_int,
                  [('temperature.chip',lambda x:x.value,'C')],group='slow')
        self._add(card,chip,entity,'dcmi_get_device_health',(),C.c_uint,
                  [('ascend.health',lambda x:x.value,'code')],group='slow')

    def _clock_reasons(self,card,chip,entity):
        name='ascend.clock.reasons';api='dcmi_get_device_info'
        if not any(m.name==name for m in self.metrics):
            self.metrics.append(Metric(name,'bitmask','chip',api,window_ns=100_000_000))
        function=self.lib.dcmi_get_device_info
        function.argtypes=[C.c_int,C.c_int,C.c_int,C.c_uint,C.c_void_p,C.POINTER(C.c_uint)]
        function.restype=C.c_int
        def read():
            value=C.c_ulonglong();size=C.c_uint(C.sizeof(value));start=raw_clock()
            rc=function(card,chip,8,9,C.byref(value),C.byref(size));end=raw_clock()
            valid=rc==0 and size.value==C.sizeof(value)
            return Observation(entity,{name:value.value if valid else None},start,end,
                {} if valid else {name:'invalid' if rc==0 else 'read_error'},api,detail=dict(returncode=rc,main=8,sub=9))
        self._calls.append(Call(entity+':lp:9','card:'+str(card)+':chip:'+str(chip),entity,(name,),read,source_api=api))

    def calls(self):
        return list(self._calls)


def Provider(devices='all', source='auto', **kwargs):
    from .telemetry_smi import Provider as SMI
    if source == 'npu-smi':
        return SMI(devices=devices)
    try:
        return DCMI(devices=devices, source=source, **kwargs)
    except (OSError, ReadError) as error:
        if source != 'auto':
            raise
        result=SMI(devices=devices)
        result.inventory['initial_source_failure']=str(error)
        return result
