import json
from pathlib import Path
import tempfile
import unittest
from flagprof.analyzers.timeline import export
from flagprof.analyzers.workspace import Workspace
from flagprof.store import create_run,Writer,StoreView,finalize

class TimelineTests(unittest.TestCase):
    def test_structure_clock_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=create_run(root/'store')
            with Writer(run) as writer:
                for i in range(3):writer.emit(dict(t='span',ph='x',cat='phase',corr=i+1,name='measure',ts=100000+i*1000,dur=500))
            finalize(run)
            with StoreView(run) as view:
                work=Workspace(view,root/'work');work.ingest()
                meta=export(work,root/'trace.json',max_events=2)
                trace=json.loads((root/'trace.json').read_text())
                self.assertTrue(meta['truncated']);self.assertEqual(meta['events'],2)
                self.assertEqual([e['ts'] for e in trace['traceEvents']],[0,1])
                self.assertTrue(all(e['ph']=='X' and e['dur']==.5 and 'pid' in e and 'tid' in e for e in trace['traceEvents']))
                meta=export(work,root/'all.json',max_events=3);self.assertFalse(meta['truncated'])
                work.close()

    def test_root_module_inside_phase_is_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=create_run(root/'store')
            with Writer(run) as writer:
                writer.emit(dict(t='span',ph='x',cat='phase',corr=1,name='measure',ts=100,dur=100))
                writer.emit(dict(t='span',ph='x',cat='module',corr=2,parent=1,name='root',ts=110,dur=80))
                writer.emit(dict(t='span',ph='x',cat='module',corr=3,parent=2,name='child',ts=120,dur=30))
            finalize(run)
            with StoreView(run) as view:
                work=Workspace(view,root/'work');work.ingest();export(work,root/'trace.json');work.close()
            events=json.loads((root/'trace.json').read_text())['traceEvents']
            self.assertEqual([e['name'] for e in events],['measure','root'])

if __name__=='__main__':unittest.main()
