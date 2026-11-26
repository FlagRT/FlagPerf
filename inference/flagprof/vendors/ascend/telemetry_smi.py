"""Explicit npu-smi fallback, one node query per round."""
import re
import subprocess
from flagprof.telemetry.contract import Call,Metric,Observation,ReadError,raw_clock


def parse(text):
    lines=text.splitlines();result=[]
    for index,line in enumerate(lines[:-1]):
        columns=[c.strip() for c in line.split('|')[1:-1]]
        if len(columns)!=3 or not re.fullmatch(r'\d+\s+Ascend\S+',columns[0]):continue
        card=int(columns[0].split()[0]);health=columns[1]
        if index+1>=len(lines):raise ValueError('truncated npu-smi chip')
        detail=[c.strip() for c in lines[index+1].split('|')[1:-1]]
        if len(detail)!=3 or not re.fullmatch(r'\d+\s+\d+',detail[0]):raise ValueError('chip identity missing')
        chip,phy=map(int,detail[0].split())
        pci=detail[1]
        if not re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',pci):raise ValueError('PCI identity missing')
        first=columns[2].split()
        values=re.fullmatch(r'(\d+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)',detail[2])
        if not values or len(first)<2:raise ValueError('resource fields missing')
        activity,_,_,used,total=map(int,values.groups())
        if used>total or activity>100:raise ValueError('invalid npu-smi values')
        result.append(dict(entity='chip:'+str(phy),phyid=phy,card_id=card,chip_id=chip,pci=pci.lower(),
            power_domain='card:'+str(card),values={'memory.used':used*1048576,'memory.total':total*1048576,
                'temperature.chip':float(first[1]),'ascend.aicore.activity':activity,
                'ascend.health':{'OK':0,'Warning':1,'Alarm':2,'Critical':3}.get(health)},
            power=None if first[0]=='-' else float(first[0])))
    if not result or 'Process' not in text:
        raise ValueError('incomplete npu-smi table')
    return result


class Provider:
    container_safe=False
    def __init__(self,devices='all'):
        self.selected=None if devices=='all' else set(map(int,devices.split(',')))
        start=raw_clock();raw=self.query();self.initial=parse(raw);elapsed=raw_clock()-start
        self.initial=[r for r in self.initial if self.selected is None or r['phyid'] in self.selected]
        if not self.initial:raise ValueError('selected npu-smi device missing')
        entities=[{k:v for k,v in r.items() if k not in ('values','power')} for r in self.initial]
        for card in sorted({r['card_id'] for r in self.initial}):
            entities.append(dict(entity='card:'+str(card),members=[r['entity'] for r in parse(raw) if r['card_id']==card]))
        self.metrics=[Metric(n,u,'chip','npu-smi info',kind='window' if n.endswith('.activity') else 'instant',
            profiler_conflict=n.endswith('.activity'),
            roles=('activity',) if n.endswith('.activity') else ('temperature',) if n=='temperature.chip' else ()) for n,u in [('memory.used','B'),('memory.total','B'),
            ('temperature.chip','C'),('ascend.aicore.activity','%'),('ascend.health','code')]]
        self.metrics.append(Metric('power.draw','W','card','npu-smi info',roles=('power',)))
        self.inventory=dict(entities=entities,source='npu-smi',container_safe=False,initial_round_ns=elapsed,
                            interval_quantum_ms=1000, minimum_interval_ms=max(100,2*elapsed/1e6),call_timeout_s=max(3,5*elapsed/1e9),
                            unavailable={'clock.compute':'not_in_fallback_table','clock.memory':'not_in_fallback_table',
                                         'process.device_memory':'not_in_fallback_table','device.error_codes':'not_in_fallback_table'})

    def query(self):
        result = subprocess.run(['npu-smi','info'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                text=True,timeout=3)
        if result.returncode:
            raise ReadError('read_error', f'npu-smi info exited {result.returncode}: {result.stdout.strip()[:256]}')
        return result.stdout

    def calls(self):
        return [Call('npu-smi','node','node',(),self.read,source_api='npu-smi info')]

    def read(self):
        start=raw_clock();reason=None
        try:rows=parse(self.query())
        except ValueError:rows=[];reason='fallback_unparsable'
        except ReadError as error:rows=[];reason=error.reason
        except subprocess.TimeoutExpired:rows=[];reason='timeout'
        except (subprocess.SubprocessError,OSError):rows=[];reason='read_error'
        end=raw_clock();byid={r['entity']:r for r in rows};points=[]
        powers={r['power_domain']:r['power'] for r in rows if r['power'] is not None}
        for registered in self.initial:
            row=byid.get(registered['entity']);values=row['values'] if row else {k:None for k in registered['values']}
            miss={k:reason or 'fallback_unparsable' for k,v in values.items() if v is None}
            from .telemetry_health import health_events
            points.append(Observation(registered['entity'],values,start,end,miss,'npu-smi info',
                health=health_events(registered['entity'],values.get('ascend.health'),end)))
            if row and row['power'] is not None:powers[registered['power_domain']]=row['power']
        for domain in {r['power_domain'] for r in self.initial}:
            value=powers.get(domain)
            points.append(Observation(domain,{'power.draw':value},start,end,
                {} if value is not None else {'power.draw':reason or 'unsupported'},'npu-smi info'))
        return points
