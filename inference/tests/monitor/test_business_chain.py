import json
import os
from pathlib import Path
import tempfile
import time
import sys
import unittest
from flagprof.record import record
from flagprof.monitor.consumer import replay
from flagprof.monitor.commands import publish
from flagprof.delivery import restore
from flagprof.monitor.report import generate
from tests.monitor import test_agent
from tests.store.html_cells import Cells


class ChainTests(unittest.TestCase):
    stop=staticmethod(test_agent.AgentTests.stop)
    start_agent=test_agent.AgentTests.start_agent

    def test_probe_startup_failure_keeps_attachment_and_target_running(self):
        import threading
        from unittest.mock import patch
        from flagprof.runtime import attach, current_attachment
        from flagprof.store import create_run
        def failed_start(probe):
            probe.thread=threading.Thread(target=lambda:None)
            raise RuntimeError('injected thread start failure')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=create_run(root/'target')
            with patch('flagprof.probes.diagnostic.DiagnosticProbe.start',failed_start):
                handle=attach(run,'nvidia',preset='perf-lite',config=dict(diagnose='on',monitor_root=str(root/'absent')))
            try:
                self.assertFalse(handle.closed)
                self.assertIs(current_attachment(),handle)
                self.assertIsNone(handle.context.diagnostic)
                self.assertEqual(handle.state['diagnostic_status'],'startup_unavailable:RuntimeError')
            finally:handle.close()

    def test_agent_unavailable_preserves_target_exit_and_reports_gap(self):
        from flagprof.monitor.diagnostic_report import panel
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            code=record([sys.executable,'-c','raise SystemExit(7)'],store=root/'target',vendor='nvidia',preset='perf-lite',
                config=dict(diagnose='on',monitor_root=str(root/'missing-node')))
            self.assertEqual(code,7)
            run=next((root/'target').glob('*/manifest.json')).parent
            self.assertEqual(panel(run)['status'],'unavailable')

    def test_worker_preserves_explicit_system_exit_codes(self):
        from unittest.mock import patch
        from flagprof.launch import worker
        from types import SimpleNamespace
        for code in (0,7):
            with tempfile.TemporaryDirectory() as directory:
                settings=dict(root=directory,group_id='group',vendor='nvidia',config={},preset='none',
                    command=[sys.executable,'-c','raise SystemExit('+str(code)+')'])
                name='flagprof.vendors.nvidia.execution'
                previous=sys.modules.get(name)
                sys.modules[name]=SimpleNamespace(initialize=lambda rank:'cpu')
                try:
                    with patch.object(sys,'argv',[]), patch.dict(os.environ):
                        self.assertEqual(worker(settings,0),code)
                finally:
                    if previous is None:sys.modules.pop(name,None)
                    else:sys.modules[name]=previous
                state=json.loads((Path(directory)/'rank-0/worker.json').read_text())
                self.assertEqual(state['status'],'complete' if code==0 else 'failed')

    def test_wait_receipt_replacing_inflight_registration_end_is_delivered(self):
        from types import SimpleNamespace
        from flagprof.monitor.node import drain_notifications
        from flagprof.monitor.session import configuration, edit, initialize
        from flagprof.monitor.subscriptions import Server
        from flagprof.store.manifest import proc_key
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize(root, 'nvidia', 'all', configuration())
            with edit(root) as state:
                state['status'] = 'running'
                state['config']['queue_capacity'] = 1
            server = SimpleNamespace(root=root)
            reference = dict(worker_proc_key=proc_key(), window=[1, None], root='.', session_path='.')
            message = dict(reference=reference, path=str(root/'ref.json'))
            Server.handle(server, dict(action='register', **message))
            Server.handle(server, dict(action='finish', **message))
            def publish_exit():
                Server.handle(server, dict(action='exit', exit=dict(source='subprocess.wait', returncode=7), **message))
            events = []
            writer = SimpleNamespace(emit=events.append, writer=SimpleNamespace(flush=publish_exit))
            drain_notifications(root, writer, 1)
            drain_notifications(root, writer, 1)
            self.assertEqual([row['detail']['source'] for row in events], ['registration_end', 'subprocess.wait'])
            self.assertEqual(events[1]['detail']['returncode'], 7)
            self.assertFalse(list((root/'notifications').glob('*.json')))

    def test_failed_rank_and_launcher_terminated_peer_keep_separate_receipts(self):
        from unittest.mock import patch
        import subprocess
        from flagprof.launch import launch
        from flagprof.report.group import generate_group
        body="""import json,os,sys,time
from pathlib import Path
from flagprof.store import create_run
from flagprof.runtime import attach
settings=json.loads(Path(sys.argv[1]).read_text());rank=int(sys.argv[2])
root=Path(sys.argv[1]).resolve().parent;part=root/('rank-'+str(rank));part.mkdir()
run=create_run(part/'delivery',identity={'rank':rank,'group_id':settings['group_id']})
(part/'worker.json').write_text(json.dumps(dict(rank=rank,group_id=settings['group_id'],status='running',run=str(run.relative_to(root)))))
h=attach(run,'nvidia',preset='perf-lite',config=settings['config'],rank=rank)
h._monitor_bind([dict(status='available',entity='chip:0')])
if rank==0:
    time.sleep(1)
    h.close(normal=False,returncode=7)
    raise SystemExit(7)
time.sleep(30)
"""
        native=subprocess.Popen
        def spawn(command,*args,**kwargs):
            if 'flagprof.launch' in command:
                i=command.index('flagprof.launch')
                command=[sys.executable,'-B','-c',body]+command[i+1:]
            return native(command,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);agent=self.start_agent(root/'node')
            try:
                with patch('subprocess.Popen',spawn):
                    code=launch([sys.executable,'-c','pass'],store=root/'group',vendor='nvidia',comm_backend='nccl',preset='perf-lite',
                        config=dict(diagnose='on',monitor_root=str(root/'node')),timeout=10)
                self.assertEqual(code,2)
                time.sleep(.5)
            finally:self.stop(agent)
            online=[json.loads(p.read_text()) for p in (root/'node/findings').glob('*/*.json')]
            exits={r['rank']:r for r in online if r['rule']=='proc.exit'}
            self.assertEqual(set(exits),{0,1})
            self.assertEqual(exits[0]['termination_reason'],'target_exit')
            self.assertEqual(exits[1]['termination_reason'],'peer_failure')
            self.assertNotEqual(exits[0]['run_id'],exits[1]['run_id'])
            replay(root/'node',root/'replay')
            actual=[json.loads(s) for s in (root/'replay/findings.jsonl').read_text().splitlines()]
            self.assertEqual(sorted(online,key=lambda r:(r['id'],r['revision'])),sorted(actual,key=lambda r:(r['id'],r['revision'])))
            group=json.loads((root/'group/group.json').read_text())
            self.assertTrue(all(row['run'] for row in group['ranks']))
            # A missing rank's Store must leave the other rank's diagnosis visible.
            missing=root/'group'/group['ranks'][1]['run'];missing.rename(missing.with_name(missing.name+'-held'))
            generate_group(root/'group/group.json',root/'report')
            result=json.loads((root/'report/result.json').read_text())
            self.assertEqual({r['rank'] for r in result['diagnosis']['findings'] if r['rule']=='proc.exit'},{0,1})

    def test_scope_burst_is_drained_without_watchdog_restart(self):
        from flagprof.monitor.session import load
        from flagprof.store import StoreView
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);agent=self.start_agent(root/'node')
            try:
                body="""import flagprof
h=flagprof.current_attachment()
h._monitor_bind([dict(status='available',entity='chip:0')])
for i in range(1600):
    with flagprof.scope('short-'+str(i)): pass
"""
                code=record([sys.executable,'-c',body],store=root/'targets',vendor='nvidia',preset='perf-lite',
                    config=dict(diagnose='on',monitor_root=str(root/'node')))
                self.assertEqual(code,0)
                deadline=time.monotonic()+12
                while time.monotonic()<deadline:
                    with StoreView(root/'node') as view:
                        events=list(view.iter_monitor_events(('health',)))
                    if any(e.get('detail',{}).get('record',{}).get('detail',{}).get('action')=='finish' for e in events):break
                    time.sleep(.2)
                else:self.fail('target committed prefix was not drained')
            finally:self.stop(agent)
            self.assertEqual(load(root/'node')['gaps'],[])
            online=[json.loads(p.read_text()) for p in (root/'node/findings').glob('*/*.json')]
            replay(root/'node',root/'replay')
            actual=[json.loads(s) for s in (root/'replay/findings.jsonl').read_text().splitlines()]
            self.assertEqual(sorted(online,key=lambda r:(r['id'],r['revision'])),
                sorted(actual,key=lambda r:(r['id'],r['revision'])))
            with StoreView(root/'node') as view:
                inputs=[e for e in view.iter_monitor_events(('health',)) if e.get('detail',{}).get('scope')=='business_input']
            identities=[tuple(e['detail']['source_identity']) for e in inputs]
            self.assertEqual(len(identities),len(set(identities)))
            self.assertEqual(sum(e['detail']['record']['detail']['action']=='leave' for e in inputs),1600)

    def test_real_target_scope_to_restored_replay(self):
        from flagprof.monitor.session import load
        from flagprof.store import StoreView
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            agent=self.start_agent(root/'node',diagnosis={'progress_timeout_s':.1})
            body="""import time,threading
import flagprof
h=flagprof.current_attachment()
h._monitor_bind([dict(status='available',entity='chip:0')])
lock=threading.Lock();lock.acquire()
def release():
    time.sleep(2.2)
    lock.release()
helper=threading.Thread(target=release);helper.start()
with flagprof.scope('blocked'):
    lock.acquire()
lock.release();helper.join()
time.sleep(1.2)
"""
            code=record([sys.executable,'-c',body],store=root/'targets',vendor='nvidia',preset='perf-lite',
                config=dict(diagnose='on',monitor_root=str(root/'node')))
            try:
                self.assertEqual(code,0)
                reference,=(root/'targets').glob('*/monitor-ref.json')
                target=reference.parent
                ref=json.loads(reference.read_text())
                identity=ref['diagnostic_identity']
                self.assertEqual(identity['run_id'],target.name)
                self.assertTrue(identity.get('proc_key'))
                self.assertEqual(identity['proc_key'],ref['worker_proc_key'])
                deadline=time.monotonic()+15
                with StoreView(target) as view:
                    while True:
                        states={r['state'] for r in view.iter_diagnostic_findings()
                                if r['rule']=='hang.no_progress' and r['run_id']==target.name
                                and r['proc_key']==identity['proc_key']}
                        coverage=list(view.iter_diagnostic_coverage())
                        if 'resolved' in states and any(row['run_id']==target.name and row.get('returncode')==code
                                                        for row in coverage):
                            break
                        if agent.poll() is not None or time.monotonic()>=deadline:
                            self.fail(f"run={target.name} proc={identity['proc_key']} stall_states={states} "
                                      f"coverage={coverage} agent_exit={agent.poll()} "
                                      f"subscription_errors={ref.get('errors')}")
                        time.sleep(.1)
            finally:
                self.stop(agent)
            session=load(root/'node')
            self.assertEqual((agent.returncode,session['status']),(0,'stopped'),
                             (session.get('supervisor_error'),session.get('worker_error'),session['segments']))
            self.assertTrue(session['segments'] and all(s['status']=='closed' for s in session['segments']),
                            session['segments'])
            online=[json.loads(p.read_text()) for p in (root/'node/findings').glob('*/*.json')]
            self.assertTrue(any(x['rule']=='hang.no_progress' for x in online),online)
            replay(root/'node',root/'replay')
            actual=[json.loads(s) for s in (root/'replay/findings.jsonl').read_text().splitlines()]
            order=lambda x:(x['id'],x['revision'])
            self.assertEqual(sorted(online,key=order),sorted(actual,key=order))
            publish(target,root/'archive')
            result=restore(root/'archive',root/'restore')
            from flagprof.monitor.reference import resolve_reference
            _, restored, _ = resolve_reference(result['mapping']['monitor-reference'])
            replay(restored,root/'restored-replay')
            actual=[json.loads(s) for s in (root/'restored-replay/findings.jsonl').read_text().splitlines()]
            self.assertEqual(sorted(online,key=order),sorted(actual,key=order))
            generate(restored,root/'report')
            diagnosis=json.loads((root/'report/result.json').read_text())['diagnosis']
            online_stalls=[row for row in online if row['rule']=='hang.no_progress']
            reported_stalls=[row for row in diagnosis['findings'] if row['rule']=='hang.no_progress']
            self.assertEqual(sorted(reported_stalls,key=order),sorted(online_stalls,key=order))
            references={tuple(ref) for row in reported_stalls for ref in row['evidence']}
            self.assertTrue(references)
            self.assertLessEqual(references,{tuple(row['identity']) for row in diagnosis['evidence']})

            page=Cells()
            page.feed((root/'report/index.html').read_text())
            labels=[''.join(text) for (panel,_,field),(_,text) in page.values.items()
                    if panel=='diagnosis' and field=='event_label']
            self.assertIn('执行进度停滞',labels)
            self.assertIn('执行进度停滞 · 已解除',labels)
            self.assertGreater(page.rows.get('diagnosis.evidence',0),0)


if __name__=='__main__':unittest.main()
