"""Versioned minute trends. Raw evidence remains the only rule/recovery input."""
import json
from pathlib import Path
from flagprof.store.manifest import atomic_json
from .retention import CapacityError, reserve
from .session import edit


class Minutes:
    def __init__(self, root):
        self.root=Path(root); self.minute=None; self.rows={}

    def add(self, event):
        if event.get('t')!='counter' or event.get('ts') is None: return
        minute=event['ts']//60_000_000_000
        if self.minute is not None and minute>self.minute:
            self.flush()
        if self.minute is None: self.minute=minute
        if minute<self.minute: return  # Late reads stay in raw evidence; no invented minute relocation.
        for name,value in event['values'].items():
            key=(event['dev'],name)
            row=self.rows.setdefault(key,dict(entity=key[0],metric=key[1],count=0,missing=0,total=0.,minimum=None,maximum=None))
            if value is None:
                row['missing']+=1;continue
            row['count']+=1;row['total']+=value
            row['minimum']=value if row['minimum'] is None else min(value,row['minimum'])
            row['maximum']=value if row['maximum'] is None else max(value,row['maximum'])

    def flush(self):
        if self.minute is None:return
        output=dict(schema='monitor-minute/1',minute=self.minute,source='raw_observations',
                    semantics='sample mean for trend only; not time weighted or a substitute for raw evidence',rows=[])
        for row in self.rows.values():
            output['rows'].append(dict(row,mean=row['total']/row['count'] if row['count'] else None))
        amount=2*len(json.dumps(output).encode())+4096
        with edit(self.root) as state:
            folder=self.root/'aggregates';folder.mkdir(exist_ok=True)
            existing=sorted(folder.glob('*.json'),key=lambda p:int(p.stem.split('-')[0]))
            total=sum(p.stat().st_size for p in existing)
            for path in existing:
                age=(self.minute-int(path.stem.split('-')[0]))*60
                if total+amount>state['config']['aggregate_bytes'] or age>state['config']['aggregate_seconds']:
                    total-=path.stat().st_size;path.unlink()
            if total+amount>state['config']['aggregate_bytes']:raise CapacityError('minute_aggregate_budget')
            reserve(self.root,state,amount)
            # Restart contributions remain separate; never overwrite an earlier partial minute.
            import uuid
            atomic_json(folder/(str(self.minute)+'-'+uuid.uuid4().hex[:8]+'.json'),output)
            state['reserved_bytes']-=amount
        self.minute=None;self.rows={}
