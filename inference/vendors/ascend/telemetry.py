"""Read-only DCMI telemetry for the Ascend HDK A3 ABI.

Physical /dev/davinci IDs are mapped by DCMI, never by arithmetic. A3 power
is a shared management-card domain, including any unselected sibling chip.
The collector process owns the C library so a blocked driver cannot block inference.
"""
import ctypes as C
import re
import subprocess
import time

from vendors.telemetry import TelemetryProvider, metric, observation


class HBM(C.Structure):
    _fields_ = [('memory_size', C.c_ulonglong), ('freq', C.c_uint),
                ('memory_usage', C.c_ulonglong), ('temp', C.c_int),
                ('bandwith_util_rate', C.c_uint)]


UTIL = {2: 'aicore', 12: 'vector', 13: 'npu', 14: 'cube', 10: 'hbm_bandwidth'}
METRICS = [metric('ascend.'+name+'.activity', '%',
                  'DCMI native '+name+' utilization; not achieved peak FLOPS',
                  profiler_conflict=True) for name in UTIL.values()]
METRICS += [metric('power.draw', 'W', 'A3 management-card power including both chips', 'card'),
            metric('temperature.chip', 'degC', 'chip sensor temperature'),
            metric('clock.compute', 'MHz', 'current AI Core clock'),
            metric('clock.memory', 'MHz', 'current HBM clock'),
            metric('memory.used', 'B', 'device HBM used; not framework allocated/reserved'),
            metric('memory.total', 'B', 'device HBM capacity'),
            metric('memory.free', 'B', 'capacity minus device HBM used'),
            metric('temperature.memory', 'degC', 'HBM sensor temperature'),
            metric('ascend.health', 'code', 'native DCMI health code; zero means healthy')]
DEFS = {m['name']: m for m in METRICS}


class AscendTelemetry(TelemetryProvider):
    def __init__(self, library=None):
        self.lib = library or C.CDLL('/usr/local/dcmi/libdcmi.so')
        self.targets = []
        if self.lib.dcmi_init() != 0:
            raise RuntimeError('DCMI initialization failed; check device access and driver permissions')

    def discover(self, devices):
        self.targets = []
        entities = {}
        for rank, physical in enumerate(devices):
            card, chip = C.c_int(), C.c_int()
            code = self.lib.dcmi_get_card_id_device_id_from_phyid(
                C.byref(card), C.byref(chip), C.c_uint(physical))
            if code:
                raise RuntimeError(f'DCMI physical device mapping failed: {physical}, code={code}')
            count, mcu, cpu = C.c_int(), C.c_int(), C.c_int()
            code = self.lib.dcmi_get_device_id_in_card(card, C.byref(count), C.byref(mcu), C.byref(cpu))
            if code or count.value != 2:
                raise RuntimeError('telemetry currently validates the A3 two-chip management-card ABI only')
            entity = f'ascend:card:{card.value}:chip:{chip.value}'
            power = f'ascend:card:{card.value}'
            self.targets.append((entity, card.value, chip.value))
            entities[entity] = dict(scope='chip', physical_device=physical, card=card.value,
                                    chip=chip.value, ranks=[rank], power_domain=power)
            entry = entities.setdefault(power, dict(scope='card', card=card.value, ranks=[],
                includes_unselected_sibling=True, attribution='shared device domain, not process power'))
            entry['ranks'].append(rank)
        version = C.create_string_buffer(256)
        code = self.lib.dcmi_get_driver_version(version, 256)
        self.inventory = dict(provider='ascend.dcmi.a3', driver_version=version.value.decode(errors='replace') if not code else None,
            entities=entities, metrics=METRICS,
            unavailable={'power.limit': 'not exposed by this provider',
                         'clock.throttle_reason': 'not exposed by this provider',
                         'memory.read_bytes_per_second': 'not exposed by this provider',
                         'memory.write_bytes_per_second': 'not exposed by this provider'},
            hbm_source_unit='HDK A3 MB; 2**20 bytes per unit, validated against device capacity',
            resolution='source averaging window and refresh period are unknown; polling cadence is not resolution')
        return self.inventory

    def _scalar(self, entity, name, function, card, chip, *args, scale=1):
        start, raw = time.perf_counter_ns(), C.c_int()
        code = getattr(self.lib, function)(card, chip, *args, C.byref(raw))
        return observation(entity, DEFS[name], raw.value*scale if code == 0 else None,
            start, raw=raw.value if code == 0 else None,
            error=f'DCMI return code {code}' if code else None, source=function)

    def read(self):
        rows, cards = [], set()
        for entity, card, chip in self.targets:
            for kind, name in UTIL.items():
                rows.append(self._scalar(entity, 'ascend.'+name+'.activity',
                    'dcmi_get_device_utilization_rate', card, chip, kind))
            rows.append(self._scalar(entity, 'temperature.chip', 'dcmi_get_device_temperature', card, chip))
            for kind, name in [(7, 'clock.compute'), (6, 'clock.memory')]:
                rows.append(self._scalar(entity, name, 'dcmi_get_device_frequency', card, chip, kind))
            rows.append(self._scalar(entity, 'ascend.health', 'dcmi_get_device_health', card, chip))
            start, info = time.perf_counter_ns(), HBM()
            code = self.lib.dcmi_get_device_hbm_info(card, chip, C.byref(info))
            invalid = info.memory_usage > info.memory_size
            for name, raw, scale in [('memory.total', info.memory_size, 2**20),
                ('memory.used', info.memory_usage, 2**20),
                ('memory.free', info.memory_size-info.memory_usage, 2**20),
                ('temperature.memory', info.temp, 1)]:
                rows.append(observation(entity, DEFS[name], raw*scale if not code and not invalid else None,
                    start, raw=raw if not code else None, source='dcmi_get_device_hbm_info',
                    error=f'DCMI return code {code}' if code else 'invalid HBM capacity' if invalid else None))
            if card not in cards:
                cards.add(card)
                rows.append(self._scalar(f'ascend:card:{card}', 'power.draw',
                    'dcmi_get_device_power_info', card, 0, scale=0.1))
        return rows


def label_blocks(text, trailing_chip=False):
    """Common prints Chip ID first; usages prints it after the chip's fields."""
    result, fields, chip = {}, {}, None
    for line in text.splitlines():
        if ':' not in line: continue
        key, value = (s.strip() for s in line.split(':', 1))
        if key in ('NPU ID', 'Chip Count'): continue
        if key == 'Chip Name':
            if chip is not None: result[chip] = fields
            fields, chip = {}, None
            continue
        if key == 'Chip ID':
            if trailing_chip:
                result[int(value)] = fields
                fields, chip = {}, None
            else:
                if chip is not None: result[chip] = fields
                fields, chip = {}, int(value)
        else:
            fields[key] = value
    if chip is not None: result[chip] = fields
    return result


class SMITelemetry(TelemetryProvider):
    """Verified A3 label-based fallback. Rounded occupancy never becomes used bytes."""
    def command(self, *args):
        proc = subprocess.run(['npu-smi', 'info', *args], capture_output=True, text=True, timeout=3)
        if proc.returncode:
            raise RuntimeError(f'npu-smi {args}: {proc.returncode}: {proc.stderr.strip()}')
        return proc.stdout

    def discover(self, devices):
        raw = self.command('-m')
        lines = [re.split(r'\s{2,}', line.strip()) for line in raw.splitlines() if line.strip()]
        header = lines[0]
        indices = [header.index(n) for n in ('NPU ID','Chip ID','Chip Phy-ID','Chip Name')]
        mapping, counts = {}, {}
        for line in lines[1:]:
            if len(line) != len(header): continue
            card, chip, physical, name = [line[i] for i in indices]
            if not physical.isdigit(): continue
            card, chip, physical = int(card), int(chip), int(physical)
            mapping[physical] = (card,chip)
            counts.setdefault(card,set()).add(chip)
        self.targets, entities = [], {}
        for rank, physical in enumerate(devices):
            card, chip = mapping[physical]
            if counts[card] != {0,1}: raise ValueError('CLI fallback validates A3 two-chip cards only')
            entity, power = f'ascend:card:{card}:chip:{chip}', f'ascend:card:{card}'
            self.targets.append((entity,card,chip))
            entities[entity] = dict(scope='chip',physical_device=physical,card=card,chip=chip,ranks=[rank],power_domain=power)
            entities.setdefault(power,dict(scope='card',card=card,ranks=[],includes_unselected_sibling=True))['ranks'].append(rank)
        self.definitions = [d for d in METRICS if d['name'] in
            ['power.draw','clock.compute','temperature.chip','memory.total'] or d['name'].startswith('ascend.') and d['name'] != 'ascend.health']
        self.definitions.append(metric('memory.capacity_used', '%', 'rounded HBM capacity occupancy; not bandwidth'))
        return dict(provider='ascend.npu_smi.a3',entities=entities,metrics=self.definitions,
                    mapping_raw=raw,unavailable={name:'not exposed by CLI fallback' for name in
                        ('memory.used','memory.free','clock.memory','temperature.memory','power.limit','clock.throttle_reason')},
                    resolution='source windows unknown; actual query cadence may exceed requested interval')

    def read(self):
        rows=[]
        usage={'Aicore Usage Rate(%)':'ascend.aicore.activity','Aivector Usage Rate(%)':'ascend.vector.activity',
               'Aicube Usage Rate(%)':'ascend.cube.activity','NPU Utilization(%)':'ascend.npu.activity',
               'HBM Bandwidth Usage Rate(%)':'ascend.hbm_bandwidth.activity',
               'HBM Capacity(MB)':'memory.total','HBM Usage Rate(%)':'memory.capacity_used'}
        common={'Aicore curFreq(MHZ)':'clock.compute','Temperature(C)':'temperature.chip'}
        definitions={d['name']:d for d in self.definitions}
        for card in sorted({c for _,c,_ in self.targets}):
            for mode,fields in [('usages',usage),('common',common)]:
                start=time.perf_counter_ns()
                text=self.command('-t',mode,'-i',str(card))
                blocks=label_blocks(text,mode=='usages')
                first=len(rows)
                for entity,c,chip in self.targets:
                    if c!=card: continue
                    block=blocks.get(chip,{})
                    for label,name in fields.items():
                        raw=block.get(label)
                        try: value=float(raw)*(2**20 if name=='memory.total' else 1)
                        except (TypeError,ValueError): value=None
                        rows.append(observation(entity,definitions[name],value,start,raw=raw,
                            error='missing or invalid CLI field' if value is None else None,
                            source=f'npu-smi info -t {mode} -i {card}: {label}'))
                if mode=='common':
                    raw=blocks.get(0,{}).get('NPU Real-time Power(W)')
                    try: value=float(raw)
                    except (TypeError,ValueError): value=None
                    rows.append(observation(f'ascend:card:{card}',DEFS['power.draw'],value,start,raw=raw,
                        error='missing or invalid CLI power' if value is None else None,source='npu-smi common: NPU Real-time Power(W)'))
                if len(rows)>first: rows[first]['raw_query']=text
        return rows


class AutoTelemetry(TelemetryProvider):
    def discover(self, devices):
        try:
            self.provider=AscendTelemetry()
            return self.provider.discover(devices)
        except (OSError, AttributeError, RuntimeError) as error:
            self.provider=SMITelemetry()
            result=self.provider.discover(devices)
            result['fallback_reason']=str(error)
            return result
    def read(self): return self.provider.read()
    def close(self): self.provider.close()
