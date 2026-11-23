import copy
import json
import tempfile
import unittest
from pathlib import Path
from flagprof.monitor.business import Engine
from flagprof.monitor.session import configuration
from flagprof.probes.config import resolve


class BusinessTests(unittest.TestCase):
    def engine(self,state=None):
        return Engine(dict(descriptors=[]),dict(configuration(),diagnosis={}),state)

    def event(self, action, seconds, run='run', rank=0, **extra):
        identity=dict(run_id=run,rank=rank,group_id='group',proc_key={'pid':rank+1})
        return dict(t='health',kind='heartbeat',ts=int(seconds*1e9),evidence_id=['event','node',1,seconds],
            detail=dict(scope='business_input',source_identity=['event',run,1,seconds],
                record=dict(t='health',kind='heartbeat',ts=int(seconds*1e9),
                    detail=dict(action=action,identity=identity,started_ns=0,**extra))))

    def tick(self, seconds, sources=None):
        return dict(t='health',kind='heartbeat',ts=int(seconds*1e9),evidence_id=['event','node',2,seconds],
            detail=dict(scope='business_tick',sources=sources or {'run':'ready'}))

    def test_config(self):
        _,c=resolve('perf-lite',dict(diagnose='on',monitor_root='/tmp/example'))
        self.assertEqual((c['monitor'],c['monitor_topology']),('on','agent'))
        for kw in ({'monitor':'off'},{'monitor_topology':'local'},{}):
            with self.assertRaises(ValueError):resolve('perf-lite',dict(diagnose='on',**kw))
        with self.assertRaises(ValueError):resolve('accuracy',dict(diagnose='on',monitor_root='/tmp/example'))
        self.assertEqual(resolve('perf-lite')[1]['diagnose'],'off')

    def test_heartbeat_channel_gap_and_recovery(self):
        e=self.engine();e.feed(self.event('start',0,period_ns=10**9))
        self.assertEqual(e.feed(self.tick(11,{'run':'unavailable'})),[])
        rows=e.feed(self.tick(12));self.assertEqual(rows[0]['rule'],'hang.heartbeat')
        self.assertEqual(rows[0]['rank'],0)
        self.assertEqual(e.feed(self.tick(13,{'run':'backlog'}))[0]['state'],'unknown')
        self.assertEqual(e.feed(self.event('heartbeat',14,period_ns=10**9,active={},calls={})),[])
        self.assertEqual(e.feed(self.event('heartbeat',15,period_ns=10**9,active={},calls={}))[0]['state'],'resolved')

    def test_progress_load_and_scope_exit(self):
        e=self.engine();row=dict(eligible=False,name='load',last_progress=0)
        self.assertEqual(e.feed(self.event('heartbeat',70,period_ns=10**9,active={'1':row},calls={})),[])
        row['eligible']=True;row['name']='forward'
        self.assertEqual(e.feed(self.event('heartbeat',71,period_ns=10**9,active={'1':row},calls={}))[0]['rule'],'hang.no_progress')
        e.feed(self.event('leave',72,token='1'))
        self.assertEqual(e.feed(self.event('heartbeat',73,period_ns=10**9,active={},calls={}))[0]['state'],'resolved')

    def test_communication_peer_and_async(self):
        for asynchronous in (True,False):
            e=self.engine()
            e.feed(self.event('start',0,period_ns=10**9))
            e.feed(self.event('start',0,run='peer',rank=1,period_ns=10**9))
            e.feed(self.event('comm_enter',1,key='group/model:0',call=dict(group='group/model',sequence=0,
                role='model',asynchronous=asynchronous,since=10**9,members=[0,1])))
            rows=e.feed(self.tick(62,{'run':'ready','peer':'ready'}))
            comm=[x for x in rows if x['rule']=='comm.timeout']
            self.assertEqual(len(comm),0 if asynchronous else 1)
            if comm:self.assertEqual(comm[0]['detail']['device_completion'],'unavailable')

    def test_backlogged_and_stale_heartbeat_cannot_confirm_no_progress(self):
        for state in ('backlog','unavailable','stale'):
            e=self.engine();event=self.event('heartbeat',70,period_ns=10**9,
                active={'1':dict(eligible=True,name='forward',last_progress=0)},calls={})
            event['detail']['source_status']='ready' if state=='stale' else state
            if state=='stale':event['ts']=90*10**9
            self.assertEqual(e.feed(event),[])

    def test_wait_receipt_ends_missing_heartbeat_checks_after_source_drain(self):
        e=self.engine();start=self.event('start',0,period_ns=10**9);e.feed(start)
        exit_event=dict(t='health',kind='process_exit',ts=10**9,evidence_id=['event','node',0,3],
            detail=dict(source='subprocess.wait',target_id='target',returncode=0,
                diagnostic_identity=start['detail']['record']['detail']['identity']))
        e.feed(exit_event)
        self.assertEqual(e.feed(self.tick(30,{'run':'backlog'})),[])
        self.assertEqual(e.feed(self.tick(31)),[])
        self.assertTrue(e.business['targets']['run']['ended'])

    def test_unknown_diagnostic_event_version_is_rejected(self):
        event=self.event('oom',1);event['detail']['record']['detail']['version']=2
        with self.assertRaisesRegex(ValueError,'diagnostic version'):self.engine().feed(event)

    def test_communication_domains_peer_entry_and_missing_source(self):
        for peer_key,peer_status,role in [('group/control:0','ready','model'),('group/model:0','ready','model'),('group/model:0','unavailable','model'),('group/model:0','ready','unknown')]:
            e=self.engine()
            e.feed(self.event('start',0,period_ns=10**9));e.feed(self.event('start',0,run='peer',rank=1,period_ns=10**9))
            call=dict(group='group/model',sequence=0,role=role,asynchronous=False,since=10**9,members=[0,1],api='barrier')
            e.feed(self.event('comm_enter',1,key='group/model:0',call=call))
            peer_call=dict(call,group=peer_key.rsplit(':',1)[0],role='control' if '/control:' in peer_key else 'model')
            e.feed(self.event('comm_enter',2,run='peer',rank=1,key=peer_key,call=peer_call))
            rows=[r for r in e.feed(self.tick(62,{'run':'ready','peer':peer_status})) if r['rule']=='comm.timeout' and r['rank']==0]
            if peer_status!='ready' or role=='unknown':self.assertEqual(rows,[])
            else:self.assertEqual(rows[0]['detail']['peer_ranks_observed_in_call'],[1] if peer_key=='group/model:0' else [])

    def test_normal_progress_and_call_below_deadline_do_not_alert(self):
        e=self.engine();e.feed(self.event('start',0,period_ns=10**9))
        self.assertEqual(e.feed(self.event('heartbeat',59,period_ns=10**9,active={'scope':dict(eligible=True,name='forward',last_progress=0)},calls={})),[])
        self.assertEqual(e.feed(self.event('heartbeat',65,period_ns=10**9,active={'scope':dict(eligible=True,name='forward',last_progress=64*10**9)},calls={})),[])
        self.assertEqual(e.feed(self.event('heartbeat',130,period_ns=10**9,active={},calls={})),[])

    def test_persisted_restart_exact(self):
        events=[self.event('start',0,period_ns=10**9),self.tick(11),self.event('oom',12),
                self.tick(13,{'run':'unavailable'})]
        a=self.engine();expected=[]
        for event in events:expected+=a.feed(event)
        b=self.engine();actual=b.feed(events[0]);b=self.engine(json.loads(json.dumps(b.state)))
        for event in events[1:]:actual+=b.feed(event)
        self.assertEqual(expected,actual)


if __name__=='__main__':unittest.main()

class ProbeTests(unittest.TestCase):
    def test_oom_typed_deduplicated_does_not_retain_exception(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from flagprof.probes.diagnostic import DiagnosticProbe
        class OOM(RuntimeError):pass
        rows=[]
        writer=SimpleNamespace(run_id='run',config=SimpleNamespace(heartbeat_period_ns=10**9),emit=rows.append)
        probe=DiagnosticProbe(SimpleNamespace(writer=writer,rank=1,state={'process':{'pid':7}}))
        with patch.dict(sys.modules,{'torch':SimpleNamespace(OutOfMemoryError=OOM)}):
            probe.exception(RuntimeError('out of memory'))
            error=OOM('device allocation failed')
            probe.exception(error);probe.exception(error)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['kind'],'oom')
        self.assertEqual(rows[0]['detail']['identity']['rank'],1)
        self.assertEqual(len(probe.seen),0)

    def test_thread_progress_does_not_advance_other_thread(self):
        from types import SimpleNamespace
        from flagprof.probes.diagnostic import DiagnosticProbe
        writer=SimpleNamespace(run_id='run',config=SimpleNamespace(heartbeat_period_ns=10**9),emit=lambda x:True)
        probe=DiagnosticProbe(SimpleNamespace(writer=writer,rank=0,state={'process':{'pid':7}}))
        probe.enter('own','forward');probe.enter('other','forward')
        probe.active['other']['thread']=-1
        probe.progress()
        self.assertEqual(probe.active['own']['progress'],1)
        self.assertEqual(probe.active['other']['progress'],0)
