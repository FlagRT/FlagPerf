"""Target Store tailing into the node's ordered, durable diagnostic input journal."""
import json
import time
from pathlib import Path
from flagprof.store import Reader
from flagprof.delivery_refs import resolve
from flagprof.telemetry.contract import raw_clock
from .retention import CapacityError


class Inputs:
    def __init__(self, root, writer):
        self.root=Path(root);self.writer=writer;self.readers={};self.cursors={}
        # Recover only committed source positions from the persisted input journal.
        from flagprof.store import StoreView
        with StoreView(root) as view:
            for event in view.iter_monitor_events(('health',)):
                detail=event.get('detail',{})
                if detail.get('scope')=='business_input':
                    identity=detail['source_identity']
                    key=tuple(identity[:-1])
                    self.cursors[key]=max(self.cursors.get(key,-1),identity[-1])

    def poll(self):
        sources={}
        paths={}
        for kind in ('active','history'):
            for registration in (self.root/'targets'/kind).glob('*.json'):
                row=json.loads(registration.read_text())
                if not row.get('diagnostic_identity'):continue
                for relative in row.get('references',[]):
                    paths[resolve(registration, relative, base=self.root).parent]=row['diagnostic_identity']['run_id']
        for path in set(self.readers)-set(paths):
            self.readers.pop(path).close()
        for run_id in paths.values():sources[run_id]='unavailable'
        for path in sorted(paths):
            run_id=paths[path]
            try:
                if not (path/'manifest.json').exists():continue
                if json.loads((path/'manifest.json').read_text())['run_id']!=run_id:
                    continue
                reader=self.readers.setdefault(path,Reader(path,retain=False)) if path not in self.readers else self.readers[path]
                count=0;batch=[];positions=[];deadline=time.monotonic()+.05
                exhausted=True
                for stored in reader.stream():
                    count+=1
                    event=stored.record
                    if event.get('t')=='health' and event.get('detail',{}).get('scope')=='business':
                        identity=stored.identity;key=tuple(identity[:-1])
                        if identity[-1]>self.cursors.get(key,-1):
                            batch.append(dict(t='health',kind=event['kind'],sev=event['sev'],ts=raw_clock(),
                                detail=dict(scope='business_input',source_identity=list(identity),record=event)))
                            positions.append((key,identity[-1]))
                    if count>=4096 or len(batch)>=256 or time.monotonic()>=deadline:
                        exhausted=False
                        break
                source_status='ready' if exhausted and not reader.report.get('truncated') and not any(
                    w.get('dropped') or w.get('state')=='lost' for w in reader.report.get('writers',{}).values()) else 'backlog'
                for record in batch:record['detail']['source_status']=source_status
            except CapacityError:
                raise
            except (OSError,ValueError,KeyError):
                sources[run_id]='unavailable'
                reader=self.readers.pop(path,None)
                if reader:reader.close()
                continue
            # A destination failure may follow a committed partial batch. Stop
            # the collector; restart rebuilds watermarks from that journal.
            # Retrying it as a source read gap would duplicate the prefix.
            self.writer.emit_many(batch)
            for key,sequence in positions:self.cursors[key]=sequence
            sources[run_id]=source_status
        self.writer.emit(dict(t='health',kind='heartbeat',sev='info',ts=raw_clock(),
            detail=dict(scope='business_tick',sources=sources)))
        return all(status=='ready' for status in sources.values())

    def close(self):
        for reader in self.readers.values():reader.close()
