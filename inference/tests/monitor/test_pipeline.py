import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from flagprof.record import record
from flagprof.report import generate
from flagprof.store import StoreView


class PipelineTests(unittest.TestCase):
    def test_local_record_report_and_off(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            actual=subprocess.Popen
            def spawn(command,*args,**kwargs):
                command=list(command)
                if 'flagprof.monitor' in command:
                    command[command.index('flagprof.monitor')]='tests.monitor.fake_entry'
                return actual(command,*args,**kwargs)
            body='''import time
from flagprof import current_attachment
from flagprof.runner.generate import interval
handle=current_attachment()
handle.state['device_identity']={'status':'available','entity':'chip:0','logical_device':'fake'}
handle._monitor_bind([handle.state['device_identity']])
with interval(handle,'measure','phase',{}):time.sleep(.7)
'''
            with patch('subprocess.Popen',spawn):
                result=record([sys.executable,'-c',body],store=root/'on',vendor='nvidia',preset='monitor',
                              config=dict(monitor_interval_ms=100,monitor_topology='local'))
            self.assertEqual(result,0)
            target=next(p.parent for p in (root/'on').glob('*/record-status.json'))
            ref=json.loads((target/'monitor-ref.json').read_text())
            self.assertEqual(ref['devices'][0]['entity'],'chip:0')
            monitor=(target/ref['monitor_path']).resolve()
            with StoreView(monitor) as view:
                points=[s.record for s in view.iter_events(('counter',))]
            self.assertGreaterEqual(len(points),3)
            generate(target,root/'report')
            data=json.loads((root/'report/result.json').read_text())['results']['resource']
            self.assertNotEqual(data['status'],'unavailable',data)
            self.assertTrue(any(r['name']=='measure' and r['mean']==200 for r in data['data']))
            result=record([sys.executable,'-c','pass'],store=root/'off',vendor='nvidia',preset='perf-lite',
                          config=dict(probes='none',monitor='off'))
            self.assertEqual(result,0)
            self.assertFalse(list((root/'off').rglob('monitor-ref.json')))
            self.assertFalse((root/'off/monitor').exists())

    def test_monitor_spawn_failure_preserves_target_returncode(self):
        with tempfile.TemporaryDirectory() as folder:
            actual=subprocess.Popen
            def spawn(command,*args,**kwargs):
                if 'flagprof.monitor' in command:raise OSError('injected monitor start failure')
                return actual(command,*args,**kwargs)
            root=Path(folder)
            with patch('subprocess.Popen',spawn):
                code=record([sys.executable,'-c','raise SystemExit(7)'],store=root,vendor='nvidia',preset='perf-lite',
                            config=dict(probes='none',monitor='on',monitor_topology='local'))
            self.assertEqual(code,7)
            status=next(root.glob('*/record-status.json'))
            self.assertEqual(json.loads(status.read_text())['target_exit']['returncode'],7)
            ref=json.loads((status.parent/'monitor-ref.json').read_text())
            self.assertTrue(any('monitor start failure' in e for e in ref['errors']))


if __name__=='__main__':unittest.main()
