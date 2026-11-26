"""Outcome, retry, and SDK-call contracts for the default final lifecycle."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from flagprof.runtime import attach, current_attachment
from flagprof.store import create_run, StoreView, finalize
from flagprof.offline.service import import_bundle
from flagprof.offline.bundle import make_bundle
from flagprof.offline.process import run_owned
from flagprof.registry import Registry

FIXTURE=Path(__file__).resolve().parents[1]/'correlate/fixtures/t8.1/nvidia.json'


class FinalBackend:
    activity_source='final-test'
    def __init__(self,failure=None):self.failure=failure;self.starts=0;self.stops=0;self.exports=0
    def conflict(self):return 'not_detectable'
    def identity(self,config):
        if self.failure=='identity':raise RuntimeError('identity')
        return dict(collector=dict(requested_configuration=dict(record_shapes=True)))
    def create_final(self,config,path,on_ready):
        if self.failure=='create':raise RuntimeError('create')
        return self
    def start(self):
        self.starts+=1
        if self.failure=='start':raise RuntimeError('start')
    def stop(self):
        self.stops+=1
        if self.failure=='stop':raise RuntimeError('stop')
    def export_final(self,profiler,path):
        self.exports+=1
        if self.failure=='export':raise RuntimeError('export')
        (path/'trace.json').write_bytes(FIXTURE.read_bytes())


class FinalLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def attached(self,backend):
        with patch.object(Registry,'create',return_value=(backend,{})):
            return attach(create_run(self.root),'nvidia')
    def test_default_final_step_compatibility_and_duplicate_close(self):
        backend=FinalBackend();handle=self.attached(backend)
        self.assertIs(current_attachment(),handle)
        for _ in range(20):self.assertTrue(handle.step())
        self.assertEqual(backend.stops,0)
        state=handle.close();before=(handle.capture/'producer.json').read_bytes()
        self.assertEqual(handle.close(),state);self.assertEqual(before,(handle.capture/'producer.json').read_bytes())
        meta=json.loads(before);self.assertTrue(meta['normal_close']);self.assertEqual(meta['step_calls'],0)
        self.assertEqual(meta['compatibility_steps'],20)
        self.assertEqual((backend.starts,backend.stops,backend.exports),(1,1,1))
    def test_exception_is_not_a_normal_capture(self):
        backend=FinalBackend();handle=self.attached(backend)
        with self.assertRaisesRegex(ValueError,'business'):
            with handle:raise ValueError('business')
        self.assertFalse(json.loads((handle.capture/'producer.json').read_bytes())['normal_close'])
        self.assertEqual(backend.stops,1);self.assertEqual(handle.state['status'],'partial')
    def test_atexit_without_exit_evidence_stays_unknown(self):
        handle=self.attached(FinalBackend());handle._exit_close()
        meta=json.loads((handle.capture/'producer.json').read_bytes())
        self.assertFalse(meta['normal_close']);self.assertIsNone(meta['target_returncode'])
    def test_sdk_failures_never_publish_normal_capture(self):
        for failure in ('identity','create','start','stop','export'):
            with self.subTest(failure=failure):
                backend=FinalBackend(failure);handle=self.attached(backend);handle.close()
                meta=json.loads((handle.capture/'producer.json').read_bytes())
                self.assertFalse(meta['normal_close']);self.assertEqual(handle.state['status'],'failed')
                self.assertLessEqual(backend.starts,1);self.assertLessEqual(backend.stops,1)
    def test_sdk_step_rejected(self):
        handle=self.attached(FinalBackend())
        with self.assertRaises(RuntimeError):handle.session.profiler.step()
        handle.close();self.assertFalse(handle.session.meta['normal_close'])
    def test_window_config_requires_explicit_mode(self):
        with self.assertRaises(ValueError):attach(create_run(self.root),'nvidia',config=dict(windows=1))
    def test_owned_child_timeout_and_signal_exit(self):
        cp=run_owned([sys.executable,'-c','import time;time.sleep(30)'],timeout=.1)
        self.assertEqual(cp.returncode,124);self.assertTrue(cp.timed_out)
        cp=run_owned([sys.executable,'-c','import os,signal;os.kill(os.getpid(),signal.SIGKILL)'],timeout=5)
        self.assertEqual(cp.returncode,-signal.SIGKILL)
    def test_migrate_resume_default_consumer_and_cleanup(self):
        root=self.root/'delivery';root.mkdir();raw=root/'raw';raw.mkdir()
        source=raw/'trace.json';source.write_bytes(FIXTURE.read_bytes())
        make_bundle(raw,source,root/'capture',vendor='nvidia',metadata=dict(normal_close=True,target_returncode=0))
        first=import_bundle(root/'capture',store=root/'runs',run_id='migration-one')
        old={str(f.relative_to(root/'runs/migration-one')):f.read_bytes() for f in (root/'runs/migration-one').rglob('*') if f.is_file()}
        moved=self.root/'moved';root.rename(moved)
        second=import_bundle(moved/'capture',store=moved/'runs',run_id='migration-two')
        self.assertEqual(second['reused'],['derived'])
        self.assertEqual(old,{str(f.relative_to(moved/'runs/migration-one')):f.read_bytes() for f in (moved/'runs/migration-one').rglob('*') if f.is_file()})
        with StoreView(moved/'runs/migration-two') as view:
            self.assertEqual(sum(1 for _ in view.iter_device_events()),second['count'])
            self.assertEqual(view.inspect()[0]['integrity'],'verified')
            self.assertEqual(view.ledger(view.traces()[0]['trace_uid'],limit=1)['total'],second['source_records'])
        self.assertEqual(finalize(moved/'runs/migration-two')['status'],'complete')
        from flagprof.offline.workspace import private_name
        self.assertEqual(list((moved/'runs'/private_name()/'tmp').iterdir()), [])
        with (moved/'capture/sealed-trace/data/trace.json').open('ab') as source:source.write(b' ')
        with StoreView(moved/'runs/migration-two') as view:
            self.assertEqual(view.traces()[0]['status'],'failed')
            self.assertEqual(view.inspect()[0]['integrity'],'failed')

if __name__=='__main__':unittest.main()
