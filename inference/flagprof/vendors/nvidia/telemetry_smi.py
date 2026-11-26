"""CSV fallback with named fields and explicit unsupported values."""
import csv
import io
import math
import subprocess
from flagprof.telemetry.contract import Call,Metric,Observation,raw_clock

FIELDS=('index','uuid','pci.bus_id','memory.used','memory.total','utilization.gpu','utilization.memory',
        'power.draw','power.limit','temperature.gpu','clocks.sm','clocks.mem')
METRICS=(('memory.used','B',1048576),('memory.total','B',1048576),('nvidia.gpu.activity','%',1),
         ('nvidia.memory.activity','%',1),('power.draw','W',1),('power.limit','W',1),
         ('temperature.chip','C',1),('clock.compute','Hz',1e6),('clock.memory','Hz',1e6))


def parse(text):
    result=[]
    for row in csv.reader(io.StringIO(text)):
        if not row:continue
        if len(row)!=len(FIELDS):raise ValueError('incomplete nvidia-smi fields')
        row=[v.strip() for v in row]
        if not row[1].startswith('GPU-'):raise ValueError('GPU UUID missing')
        values={};miss={}
        for raw,(metric,unit,factor) in zip(row[3:],METRICS):
            if raw in ('N/A','[N/A]','Not Supported','[Not Supported]'):
                values[metric]=None;miss[metric]='unsupported'
            else:
                value=float(raw)*factor
                if not math.isfinite(value) or value<0 or (unit=='%' and value>100):raise ValueError('invalid nvidia-smi metric')
                values[metric]=value
        if values['memory.used'] is not None and values['memory.total'] is not None and values['memory.used']>values['memory.total']:
            raise ValueError('invalid nvidia-smi memory')
        result.append(dict(entity='chip:'+row[1],index=int(row[0]),uuid=row[1],pci=row[2].lower(),
                           power_domain='chip:'+row[1],values=values,miss=miss))
    if not result:raise ValueError('empty nvidia-smi result')
    return result


class Provider:
    container_safe=True
    def __init__(self,devices='all'):
        start=raw_clock();rows=parse(self.query());duration=raw_clock()-start
        selected=None if devices=='all' else set(devices.split(','))
        self.initial=[r for r in rows if selected is None or str(r['index']) in selected or r['uuid'] in selected]
        if not self.initial:raise ValueError('selected GPU missing')
        self.metrics=[Metric(n,u,'chip','nvidia-smi',factor=f,kind='window' if n.endswith('.activity') else 'instant',
            roles=('activity',) if n=='nvidia.gpu.activity' else ('power',) if n=='power.draw' else
                  ('temperature',) if n=='temperature.chip' else ()) for n,u,f in METRICS]
        self.inventory=dict(source='nvidia-smi',container_safe=True,initial_round_ns=duration,
            entities=[{k:v for k,v in r.items() if k not in ('values','miss')} for r in self.initial],
            interval_quantum_ms=1000, minimum_interval_ms=max(100,2*duration/1e6),call_timeout_s=max(3,5*duration/1e9),
            unavailable={'energy.total':'not_in_fallback_csv','nvidia.violation.thermal':'not_in_fallback_csv',
                         'process.device_memory':'not_in_fallback_csv','nvidia.ecc.uncorrected':'not_in_fallback_csv'})

    def query(self):
        return subprocess.run(['nvidia-smi','--query-gpu='+','.join(FIELDS),'--format=csv,noheader,nounits'],
                              capture_output=True,text=True,check=True,timeout=3).stdout

    def read(self):
        start=raw_clock();reason=None
        try:rows=parse(self.query())
        except ValueError:rows=[];reason='fallback_unparsable'
        except subprocess.TimeoutExpired:rows=[];reason='timeout'
        except (subprocess.SubprocessError,OSError):rows=[];reason='read_error'
        end=raw_clock();byid={r['entity']:r for r in rows};points=[]
        for registered in self.initial:
            row=byid.get(registered['entity'])
            values=row['values'] if row else {k:None for k in registered['values']}
            miss=row['miss'] if row else {k:reason or 'fallback_unparsable' for k in values}
            points.append(Observation(registered['entity'],values,start,end,miss,'nvidia-smi'))
        return points

    def calls(self):return [Call('nvidia-smi','node','node',(),self.read,source_api='nvidia-smi')]
