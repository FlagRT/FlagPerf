import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
from flagprof.monitor.lifecycle import Monitor
from flagprof.monitor.__main__ import supervise
from flagprof.monitor.reference import bind
from flagprof.store import create_run


class FailureTests(unittest.TestCase):
    def test_reclaim_timeout_records_result_without_target_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'monitor-ref.json';bind(path,root=root,topology='local')
            owner=Monitor(root,path,'nvidia',dict(monitor='off'))
            owner.enabled=True
            child=types.SimpleNamespace(terminate=lambda:None,wait=lambda **_:(_ for _ in ()).throw(subprocess.TimeoutExpired('monitor',5)))
            owner.child=child;owner.close()
            self.assertTrue(json.loads(path.read_text())['reclaim']['timed_out'])
            owner.close()  # Teardown remains idempotent.

    def test_watchdog_terminates_then_kills_stuck_sampler(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=create_run(root/'monitor');actions=[]
            class Child:
                returncode=None
                def poll(self):return self.returncode
                def terminate(self):actions.append('terminate')
                def kill(self):actions.append('kill');self.returncode=-9
                def wait(self,timeout=None):
                    if self.returncode is None:raise subprocess.TimeoutExpired('sampler',timeout)
                    return self.returncode
            args=argparse.Namespace(root=str(root),run=str(run),vendor='nvidia',devices='all',interval_ms=1000,source='auto')
            with patch('subprocess.Popen',return_value=Child()),patch('signal.signal'),patch('time.monotonic',side_effect=[0,11]):
                self.assertEqual(supervise(args),-9)
            self.assertEqual(actions,['terminate','kill'])
            self.assertTrue(json.loads((run/'inventory.json').read_text())['watchdog']['timed_out'])


if __name__=='__main__':unittest.main()
