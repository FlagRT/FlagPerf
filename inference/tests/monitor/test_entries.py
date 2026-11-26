import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from flagprof.offline.capture import record_final, finish_capture
from flagprof.offline.service import import_bundle
from flagprof.monitor.reference import resolve_reference
from flagprof.launch import launch
from flagprof.report import generate

BODY="""import time
from flagprof import current_attachment
from flagprof.runner.generate import interval
h=current_attachment();h.register_device('cuda:0')
with interval(h,'measure','phase',{}):time.sleep(.35)
"""

class EntryTests(unittest.TestCase):
    def test_final_and_launch_references(self):
        actual=subprocess.Popen
        def spawn(command,*args,**kwargs):
            command=list(command)
            if 'flagprof.monitor' in command:
                command[command.index('flagprof.monitor')]='tests.monitor.fake_entry'
            for module in ('flagprof.offline.capture','flagprof.launch'):
                if module in command:
                    index=command.index(module);command[index:index]=['tests.monitor.fake_target'];break
            return actual(command,*args,**kwargs)
        with tempfile.TemporaryDirectory() as folder,patch('subprocess.Popen',spawn):
            root=Path(folder)
            config=dict(monitor='on',monitor_topology='local',monitor_interval_ms=100,probes='none')
            code=record_final([sys.executable,'-c',BODY],store=root/'final',vendor='nvidia',config=config,capture_only=True)
            self.assertEqual(code,0)
            capture=next((root/'final').glob('capture-*'))
            target=next((capture/'collection').iterdir())
            self.assertTrue((target/'monitor-ref.json').exists())
            generate(target,root/'final-report')
            result=json.loads((root/'final-report/result.json').read_text())['results']['resource']
            self.assertIn(result['status'],('ok','partial'))
            self.assertGreater(result['coverage']['counter_events'],0)
            finish_capture(capture)
            imported=import_bundle(capture,store=root/'analysis',run_id='imported')
            imported_run=Path(imported['run'])
            self.assertEqual(resolve_reference(imported_run/'monitor-ref.json')[1],resolve_reference(target/'monitor-ref.json')[1])
            moved=root.with_name(root.name+'-moved')
            root.rename(moved);self.addCleanup(__import__('shutil').rmtree,moved)
            root=moved
            generate(root/'analysis/imported',root/'moved-report')
            self.assertNotEqual(json.loads((root/'moved-report/result.json').read_text())['results']['resource']['status'],'unavailable')
            code=launch([sys.executable,'-c',BODY],store=root/'group',vendor='nvidia',comm_backend='nccl',preset='perf-lite',config=config,timeout=30)
            self.assertEqual(code,0)
            refs=[json.loads(p.read_text()) for p in (root/'group').glob('rank-*/delivery/*/monitor-ref.json')]
            self.assertEqual(len(refs),2)
            self.assertEqual(len({r['monitor_run_id'] for r in refs}),1)
            self.assertTrue(all(r['devices'][0]['entity']=='chip:0' for r in refs))

if __name__=='__main__':unittest.main()
