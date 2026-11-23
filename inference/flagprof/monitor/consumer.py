"""One bounded public Reader at a time; recoverable cursor and deterministic rules."""
import json
from pathlib import Path
from flagprof.store import Reader
from .reading import normalize
from .retention import CapacityError, read_lease, publish_control
from .rules import VERSION
from .business import engine as Engine
from .session import edit, load


def validate_rules(root, engine):
    path=Path(root)/'rules.json'
    if path.exists():
        declared=json.loads(path.read_text())
        if declared.get('rule_version')!=VERSION or declared.get('config_version')!=engine.config_id:
            raise ValueError('monitor rule version or configuration mismatch')
        if declared.get('diagnostic_rule_version','business-rules/1')!='business-rules/1':
            raise ValueError('diagnostic rule version mismatch')


class Consumer:
    def __init__(self, root):
        self.root=Path(root); session=load(root)
        path=self.root/'consumer.json'
        previous=json.loads(path.read_text()) if path.exists() else {}
        self.engine=Engine(session['inventory'],session['config'],previous.get('engine'))
        validate_rules(root,self.engine)
        if previous and previous.get('config_version')!=self.engine.config_id:
            raise ValueError('rule configuration changed: use a new session')
        self.cursors=previous.get('cursors',{});self.reader=None;self.lease=None;self.segment=None

    def _open(self, segment):
        self._close_reader()
        self.segment=segment['id'];self.lease=read_lease(self.root,self.segment)
        directory=self.lease.__enter__()
        self.reader=Reader(directory,retain=False)

    def _close_reader(self):
        if self.reader:self.reader.close();self.reader=None
        if self.lease:self.lease.__exit__(None,None,None);self.lease=None
        self.segment=None

    def poll(self):
        session=load(self.root)
        identities={p.stem for kind in ('active','history') for p in (self.root/'targets'/kind).glob('*.json')}
        self.engine.state['exit_ids']={k:v for k,v in self.engine.state['exit_ids'].items() if k in identities}
        self.engine.state['episodes']={k:v for k,v in self.engine.state['episodes'].items()
            if not v.get('target') or v['target'] in identities}
        live={s['id'] for s in session['segments']}
        self.cursors={key:value for key,value in self.cursors.items() if key in live}
        for segment in session['segments']:
            cursor=self.cursors.setdefault(segment['id'],dict(writers={},closed=False))
            if cursor['closed']:continue
            if self.segment!=segment['id']:self._open(segment)
            checkpoint=self.root/'checkpoints'/(segment['id']+'.json')
            if 'diagnosis' in session['config'] and not checkpoint.exists():
                publish_control(self.root,checkpoint,dict(schema='business-checkpoint/1',
                    config_version=self.engine.config_id,engine=self.engine.state,
                    cursor=cursor['writers']))
            changed=False
            for stored in self.reader.stream():
                record=stored.record
                if record.get('t') not in ('counter','health') or 'seq' not in record:continue
                key=str(stored.writer_id)
                if record['seq']<=cursor['writers'].get(key,-1):continue
                event=normalize(stored)
                for finding in self.engine.feed(event):
                    folder=self.root/'findings'/segment['id']
                    publish_control(self.root,folder/(finding['id']+'-'+str(finding['revision'])+'.json'),finding)
                    with edit(self.root) as state:
                        begin=finding['window'][0]-int(state['config']['pre_seconds']*1e9)
                        end=finding['observed_at']+int(state['config']['post_seconds']*1e9)
                        state['pins']['finding:'+finding['id']]=dict(window=[begin,end],finding_id=finding['id'],
                            missing_pre=begin<state['started_raw_ns'],requested_pre_seconds=state['config']['pre_seconds'],
                            requested_post_seconds=state['config']['post_seconds'])
                        reference=dict(version=2,kind='finding',pin_key='finding:'+finding['id'],session_id=state['id'],
                            root='..',session_path='..',window=[begin,end],worker_proc_key=None,
                            devices=[dict(entity=e['entity'],status='available') for e in state['inventory'].get('entities',[])],
                            finding_id=finding['id'],revision=finding['revision'],status='fixed')
                    publish_control(self.root,self.root/'references'/(finding['id']+'.json'),reference)
                cursor['writers'][key]=record['seq'];changed=True
            if segment['status']=='closed':
                cursor['closed']=True;changed=True;self._close_reader()
            if changed:
                publish_control(self.root,self.root/'consumer.json',dict(schema='monitor-consumer/1',
                    rule_version=VERSION,config_version=self.engine.config_id,cursors=self.cursors,engine=self.engine.state))

    def close(self): self._close_reader()


def replay(session, out):
    """Replay raw records through the identical engine in Store sequence order."""
    root=Path(session);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    state=load(root);engine=Engine(state['inventory'],state['config'])
    validate_rules(root,engine)
    if state['segments'] and 'diagnosis' in state['config']:
        checkpoint=root/'checkpoints'/(state['segments'][0]['id']+'.json')
        if checkpoint.exists():
            initial=json.loads(checkpoint.read_text())
            if initial['schema']!='business-checkpoint/1' or initial['config_version']!=engine.config_id:
                raise ValueError('diagnostic checkpoint version mismatch')
            if initial['cursor']:
                raise ValueError('checkpoint must precede first input in the archived segment')
            engine=Engine(state['inventory'],state['config'],initial['engine'])
        elif state.get('status')=='archived':
            raise ValueError('diagnostic archive lacks initial checkpoint')
    count=0
    with (out/'findings.jsonl').open('w') as stream:
        for segment in state['segments']:
            with read_lease(root,segment['id']) as directory,Reader(directory,retain=False) as reader:
                for stored in reader.stream():
                    if stored.record.get('t') not in ('counter','health') or 'seq' not in stored.record:continue
                    for finding in engine.feed(normalize(stored)):
                        stream.write(json.dumps(finding,sort_keys=True)+'\n');count+=1
    from flagprof.store.manifest import atomic_json
    result=dict(schema='monitor-replay/1',rule_version=VERSION,config_version=engine.config_id,findings=count,
                raw_segments=len(state['segments']),session_id=state['id'])
    atomic_json(out/'replay.json',result)
    return result
