"""Read a currently open monitor writer through the public report path."""
import json
from pathlib import Path
import tempfile
import unittest
from flagprof.monitor.reference import bind
from flagprof.store import Writer,WriterConfig,create_run,finalize
from flagprof.store.manifest import atomic_json,proc_key
from flagprof.telemetry.contract import raw_clock
from flagprof.report import generate


class LiveTests(unittest.TestCase):
    def test_committed_prefix_remains_usable_while_monitor_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);monitor=create_run(root/'monitor');target=create_run(root/'targets-run')
            begin=raw_clock();end=begin+300_000_000
            ref=bind(target/'monitor-ref.json',root=root,monitor=monitor,worker=proc_key(),
                     devices=[dict(status='available',entity='chip:0')]);ref['window']=[begin,end]
            atomic_json(target/'monitor-ref.json',ref)
            atomic_json(monitor/'inventory.json',dict(status='running',boot_id=proc_key()['boot_id'],
                started_raw_ns=begin-1,ended_raw_ns=None,entities=[dict(entity='chip:0')],
                descriptors=[dict(name=name,unit=unit,kind='instant') for name,unit in
                    [('memory.used','B'),('clock.compute','Hz'),('clock.compute.max','Hz'),
                     ('nvidia.gpu.activity','%'),('temperature.chip','C'),('nvidia.violation.thermal','ns')]]))
            with Writer(target) as writer:
                writer.emit(dict(t='span',ph='x',cat='phase',name='measure',corr=1,ts=begin,dur=end-begin))
            finalize(target)
            with Writer(monitor,kind='counter',config=WriterConfig(compress=False)) as writer:
                for i in range(3):
                    writer.emit(dict(t='counter',src='device',dev='chip:0',ts=begin+i*100_000_000,
                        values={'memory.used':10+i,'clock.compute':[50,60,100][i],'clock.compute.max':100,
                                'nvidia.gpu.activity':20,'temperature.chip':40,'nvidia.violation.thermal':[0,4,4][i]},
                        miss={},ext=dict(telemetry=dict(interval_ns=100_000_000))))
                writer.flush()
                generate(target,root/'report')
                result=json.loads((root/'report/result.json').read_text())['results']['resource']
                self.assertEqual(result['coverage']['reference']['status'],'running')
                self.assertEqual(result['coverage']['counter_events'],3)
                self.assertEqual(next(r['mean'] for r in result['data'] if r['name']=='measure' and r['metric']=='memory.used'),11)
                self.assertEqual(result['coverage']['process_metrics'],'worker_not_observed')
                frequency=result['coverage']['frequency'][0]
                self.assertEqual(frequency['context']['nvidia.gpu.activity']['mean'],20)
                self.assertEqual(frequency['context']['temperature.chip']['mean'],40)
                self.assertEqual(frequency['restrictions']['thermal']['evidence'],'mechanism')
                self.assertEqual(frequency['evidence'],'time_candidate')


if __name__=='__main__':unittest.main()
