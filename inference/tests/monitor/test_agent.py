import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from flagprof.monitor.leases import Leases, device_keys
from flagprof.monitor.lifecycle import Monitor
from flagprof.monitor.reference import bind
from flagprof.monitor.session import configuration, load
from flagprof.monitor.subscriptions import request
from flagprof.store.manifest import proc_key


class AgentTests(unittest.TestCase):
    def start_agent(self,root,**options):
        root.mkdir()
        config=root/'config.json';config.write_text(json.dumps(dict(interval_ms=100,segment_seconds=1,min_free_bytes=0,**options)))
        child=subprocess.Popen([sys.executable,'-B','-S','-m','tests.monitor.fake_agent','--root',str(root),'--config',str(config)])
        self.addCleanup(self.stop,child)
        until=time.monotonic()+10
        while time.monotonic()<until:
            try:
                if load(root)['status']=='running':return child
            except (OSError,ValueError):pass
            if child.poll() is not None:self.fail('agent failed: '+str(child.returncode))
            time.sleep(.02)
        self.fail('agent start timeout')

    @staticmethod
    def stop(child):
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)

    def test_agent_registration_rollover_and_exit_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);agent=self.start_agent(root/'node')
            target=root/'target';target.mkdir()
            monitor=Monitor(target,target/'monitor-ref.json','nvidia',dict(monitor='on',monitor_topology='agent',monitor_root=str(root/'node')))
            ref=bind(monitor.path,worker=proc_key(),devices=[dict(status='available',entity='chip:0')])
            self.assertEqual(ref['version'],2)
            time.sleep(1.2)
            monitor.target_exit(dict(returncode=7,source='subprocess.wait'))
            monitor.target_exit(dict(returncode=7,source='subprocess.wait'))
            monitor.close()
            self.assertFalse(list((root/'node/targets/active').glob('*.json')))
            history=json.loads(next((root/'node/targets/history').glob('*.json')).read_text())
            self.assertEqual(history['exit']['returncode'],7)
            self.stop(agent)
            state=load(root/'node');self.assertEqual(state['status'],'stopped')
            self.assertGreaterEqual(len(state['segments']),2)
            from flagprof.store import StoreView
            with StoreView(target) as view:
                points=list(view.iter_monitor_events(('counter',)))
            self.assertGreater(len(points),4)
            self.assertEqual(len(points),len({tuple(p['evidence_id']) for p in points}))

    def test_cross_root_shared_power_domain_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            # Both chip selectors include the shared card key.
            state=dict(boot_id='test',vendor='ascend')
            a=device_keys(state,dict(entities=[dict(entity='chip:0',power_domain='card:0')]))
            b=device_keys(state,dict(entities=[dict(entity='chip:1',power_domain='card:0')]))
            leases=Path(directory)/'leases'
            with Leases().acquire(a,leases):
                with self.assertRaises(BlockingIOError):Leases().acquire(b,leases)
            with Leases().acquire(b,leases):pass

    def test_explicit_unavailable_agent_does_not_spawn_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            monitor=Monitor(root,root/'monitor-ref.json','nvidia',dict(monitor='on',monitor_topology='agent',monitor_root=str(root)))
            self.assertIsNone(monitor.child)
            self.assertEqual(json.loads(monitor.path.read_text())['status'],'failed')
            monitor.close()

    def test_collector_restart_produces_new_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'node';agent=self.start_agent(root)
            probe=json.loads((root/'probe.json').read_text())
            os.kill(probe['process']['pid'],signal.SIGKILL)
            until=time.monotonic()+10
            while time.monotonic()<until:
                state=load(root)
                if state['restarts'] and state['status']=='running':break
                time.sleep(.02)
            self.assertEqual(state['status'],'running')
            self.assertEqual(len(state['restarts']),1)
            self.assertTrue(state['gaps'])
            self.stop(agent)
            self.assertTrue(all(s['status']=='closed' for s in load(root)['segments']))
            self.assertEqual(load(root)['reserved_bytes'],0)

    def test_capacity_and_duplicate_start_do_not_disrupt_existing_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);agent=self.start_agent(root/'node',max_targets=1)
            workers=[]
            for i in range(2):
                child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
                workers.append(child);self.addCleanup(self.stop,child)
            identity=request(root/'node',dict(action='hello'))
            def registration(child):
                return dict(action='register',session_id=identity['session_id'],path=str(root/str(child.pid)),
                    reference=dict(worker_proc_key=proc_key(child.pid),devices=[dict(status='available',entity='chip:0')],
                                   window=[__import__('flagprof.telemetry.contract',fromlist=['raw_clock']).raw_clock(),None]))
            first=registration(workers[0]);request(root/'node',first)
            with self.assertRaisesRegex(ValueError,'active_target_limit'):request(root/'node',registration(workers[1]))
            duplicate=subprocess.run([sys.executable,'-B','-S','-m','tests.monitor.fake_agent','--root',str(root/'node'),
                '--config',str(root/'node/config.json')],capture_output=True,timeout=5)
            self.assertNotEqual(duplicate.returncode,0)
            self.assertEqual(load(root/'node')['status'],'running')
            request(root/'node',dict(first,action='finish'))
            request(root/'node',registration(workers[1]))
            self.assertEqual(len(list((root/'node/targets/active').glob('*.json'))),1)
            self.stop(agent)

    def test_real_record_entry_uses_existing_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);agent=self.start_agent(root/'node')
            from flagprof.record import record
            code=record([sys.executable,'-c',"from flagprof import current_attachment; h=current_attachment(); h._monitor_bind([dict(status='available',entity='chip:0')]); raise SystemExit(7)"],
                store=root/'target',vendor='nvidia',preset='monitor',config=dict(monitor_topology='agent',monitor_root=str(root/'node')))
            self.assertEqual(code,7)
            self.assertFalse((root/'target/monitor').exists())
            histories=list((root/'node/targets/history').glob('*.json'))
            self.assertEqual(len(histories),1)
            data=json.loads(histories[0].read_text());self.assertEqual(data['exit']['returncode'],7)
            self.assertIn(data['process_coverage'],('observed','worker_not_observed'))
            self.stop(agent)

    def test_two_rank_launch_shares_session_and_records_overlap(self):
        from unittest.mock import patch
        from flagprof.launch import launch
        from flagprof.store import StoreView
        from tests.monitor.test_entries import BODY
        actual=subprocess.Popen
        def spawn(command,*args,**kwargs):
            command=list(command)
            if 'flagprof.launch' in command:
                command.insert(command.index('flagprof.launch'),'tests.monitor.fake_target')
            return actual(command,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);agent=self.start_agent(root/'node')
            with patch('subprocess.Popen',spawn):
                code=launch([sys.executable,'-c',BODY],store=root/'group',vendor='nvidia',comm_backend='nccl',preset='perf-lite',
                    config=dict(monitor='on',monitor_topology='agent',monitor_root=str(root/'node'),probes='none'),timeout=30)
            self.assertEqual(code,0)
            paths=list((root/'group').glob('rank-*/delivery/*/monitor-ref.json'))
            self.assertEqual(len(paths),2)
            sessions=set()
            for path in paths:
                with StoreView(path.parent) as view:
                    state=view.monitor_status();sessions.add(state['reference']['session_id'])
                    self.assertEqual(state['reference']['shared_devices'],['chip:0'])
            self.assertEqual(len(sessions),1)
            histories=[json.loads(p.read_text()) for p in (root/'node/targets/history').glob('*.json')]
            self.assertEqual(len(histories),2)
            self.assertTrue(all(row['exit']['returncode']==0 for row in histories))
            self.stop(agent)


if __name__=='__main__':unittest.main()
