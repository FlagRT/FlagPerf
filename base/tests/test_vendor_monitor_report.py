"""Non-Ascend telemetry, explicit rank mapping and offline report migration."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from benchmark_monitor import finalize_benchmark_monitor, write_monitor_terminal_summary
from benchmark_report_schema import selected_samples
from executors.common import write_json
from generate_benchmark_report import generate_and_record
from base.vendors.protocol import DeviceBinding


class Provider:
    name='fixture'
    def monitor_policy(self,enabled):
        return {'enabled':enabled,'vendor':'fixture','collector':'fixture meter','target_resource':'fixture-memory',
                'required_samples_per_target':10,'automatic_workload_extension':False,
                'metric_fields':[{'key':'watts','label':'Board power','unit':'W'}]}
    def rank_target(self,targets,selected,rank,binding):
        return next((item for item in targets if item['device_id']==binding.resource_key),None)
    def measurement_identity(self,target,binding):
        return {'device_id':binding.resource_key,'framework_logical_id':binding.framework_logical_id}


class Monitor:
    origin_monotonic_s=100.0
    targets=[{'device_id':'fixture/u-a'},{'device_id':'fixture/u-b'}]
    def finish(self,result_dir,monitor_dir,windows,extensions,**kwargs):
        self.windows=windows
        return {'status':'passed','reasons':[],'targets':self.targets,'workload_windows':windows}


class VendorMonitorReportTests(unittest.TestCase):
    def event(self,kind,start,end,**extra):
        return {'schema_version':1,'kind':kind,'started_monotonic_ns':int((100+start)*1e9),
                'finished_monotonic_ns':int((100+end)*1e9),**extra}

    def test_monitor_uses_explicit_binding_for_reordered_framework_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);monitor=Monitor()
            write_json(root/'benchmark-events/torchrun-window.json',self.event('torchrun-window',2,18))
            bindings=[]
            for rank,(uuid,logical) in enumerate([('u-b',1),('u-a',0)]):
                bindings.append(DeviceBinding('fixture',6 if rank==0 else 2,None,uuid,'/fixture/node','/fixture/node',
                                              rank,logical,'fixture:'+str(logical),rank,'fixture/'+uuid))
                write_json(root/f'benchmark-events/measurement-rank-{rank}.json',
                           self.event('measurement-window',4,15,rank=rank,local_rank=rank,world_size=2))
            result=finalize_benchmark_monitor(monitor,root,self.event('container-window',1,20),[6,2],2,
                                              provider=Provider(),bindings=bindings)
            self.assertEqual(result['status'],'passed')
            self.assertEqual([item['device_id'] for item in monitor.windows],['fixture/u-b','fixture/u-a'])
            self.assertEqual([item['framework_logical_id'] for item in monitor.windows],[1,0])
            self.assertNotIn('npu_id',json.dumps(result))

    def test_duplicate_rank_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            binding=DeviceBinding('fixture',2,None,'u-a','/fixture','/fixture',0,1,'fixture:1',0,'fixture/u-a')
            with self.assertRaisesRegex(RuntimeError,'unique and complete'):
                finalize_benchmark_monitor(Monitor(),Path(tmp),self.event('container-window',1,20),[2,6],2,
                                           provider=Provider(),bindings=[binding,binding])

    def test_reports_cover_all_statuses_without_hardware_or_provider_lookup(self):
        for status in ('passed','partial','failed','skipped'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                write_json(root/'summary.json',{'schema_version':3,'kind':'benchmark','vendor':'fixture',
                    'vendor_display_name':'Fixture Accelerator','status':status,'case':'example','run_id':'fixture-run',
                    'execution_status':'passed' if status in ('passed','partial') else status,
                    'skip_reason':'fixture capability' if status=='skipped' else None})
                write_json(root/'benchmark-result.json',{'schema_version':1,'status':status,'metrics':[], 'missing_ranks':[1] if status=='partial' else []})
                monitor=write_monitor_terminal_summary(root,status='not-run',enabled=False,targets=[],reason='fixture',provider=Provider())
                with patch('base.vendors.registry.get_provider',side_effect=AssertionError('report must use stored evidence')):
                    generate_and_record(root)
                    first=(root/'report.md').read_bytes(),(root/'report_monitor.md').read_bytes()
                    generate_and_record(root)
                self.assertEqual(first,((root/'report.md').read_bytes(),(root/'report_monitor.md').read_bytes()))
                self.assertEqual(json.loads((root/'summary.json').read_text())['status'],status)
                self.assertNotIn(b'npu-smi',first[1])
                self.assertIn(b'fixture meter',first[1])

    def test_new_report_uses_metric_units_and_measurement_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'summary.json',{'schema_version':3,'kind':'benchmark','vendor':'fixture','status':'passed','case':'example'})
            write_json(root/'benchmark-result.json',{'schema_version':1,'metrics':[]})
            sample={'device_id':'fixture/u-a','started_offset_s':4.1,'finished_offset_s':4.7,'valid':True,'values':{'watts':180}}
            wrong={**sample,'device_id':'fixture/u-b','values':{'watts':999}}
            outside={**sample,'started_offset_s':30,'finished_offset_s':31,'values':{'watts':888}}
            samples=[sample,wrong,outside]
            windows=[{'role':'measurement','device_id':'fixture/u-a','started_offset_s':4,'finished_offset_s':15}]
            self.assertEqual(selected_samples(samples,windows),[sample])
            path=root/'benchmark-monitor/samples.jsonl';path.parent.mkdir()
            path.write_text(''.join(json.dumps(v)+'\n' for v in samples))
            write_json(root/'benchmark-monitor/summary.json',{'schema_version':2,'vendor':'fixture','status':'passed',
                'policy':Provider().monitor_policy(True),'workload_windows':windows,'parsed_samples':{'path':'benchmark-monitor/samples.jsonl'}})
            metadata=generate_and_record(root)
            report=(root/'report_monitor.md').read_text()
            self.assertIn('Board power | W | 1 | 180',report)
            self.assertNotIn('999',report)
            self.assertNotIn('888',report)
            self.assertTrue(any(asset['path'].endswith('benchmark-monitor-usage.svg') for asset in metadata['assets']))


if __name__=='__main__': unittest.main()
