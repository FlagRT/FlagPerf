"""P800 Toolkit contracts and failure paths, independent of accelerator availability."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))
from executors.toolkit import ToolkitExecutor, ToolkitRunRequest, parse_args
from toolkits._common.kunlunxin.P800 import contract as c
from toolkits._common.kunlunxin.P800 import evidence as e
from toolkits._common.kunlunxin.P800.evidence_runner import resolve_native, parse_record
from toolkits._common.kunlunxin.P800.report import generate_and_record, metric_summary
from executors.p800_toolkit import finalize_monitor, compose_status
from toolkits._common.kunlunxin.P800 import evidence_runner


class ContractTests(unittest.TestCase):
    def test_twelve_cases(self):
        self.assertEqual(len(c.CASES), 12)

    def test_sweep_cardinality_and_modes(self):
        points = c.case_points('interconnect-h2d-latency', [0, 4])
        self.assertEqual(len(points), 32)
        self.assertEqual({p['bytes'] for p in points}, {512,4096,65536,1048576})
        self.assertEqual({(p['pinned'],p['async']) for p in points}, set(c.MODES))

    def test_pairs_are_unordered_and_never_infer_reverse(self):
        points = c.case_points('interconnect-P2P_intraserver', [6,0,4,5])
        self.assertEqual(len(points), 12)
        self.assertTrue(all(p['source'] < p['destination'] for p in points))
        self.assertEqual(c.case_points('interconnect-P2P_intraserver', [4]), [])

    def test_payload_modes_and_smoke_scope(self):
        from executors.p800_toolkit import settings_for
        options=settings_for(SimpleNamespace(p800_options={"smoke":True,"p2p_payload_bytes":268435456}))
        self.assertEqual(options["p2p_payload_bytes"],1048576)
        points=c.case_points('interconnect-P2P_intraserver',[0,5],p2p_payload_bytes=268435456)
        self.assertEqual({p['bytes'] for p in points},{268435456})
        self.assertEqual({p['bytes'] for p in c.case_points('interconnect-P2P_intraserver',[0,5])},{33554432})
        points=c.case_points('main_memory-bandwidth',[5],1000)
        self.assertEqual([p['mode'] for p in points],['d2d','d2d-kernel'])
        record={**self.sample(),'mode':'d2d-kernel'}
        metric=c.metric_from_native(record,points[1],'main_memory-bandwidth','stdout',128,3)
        self.assertEqual(metric['value'],10.0)
        self.assertEqual(metric['source_kind'],'native-xblas')
        from toolkits._common.kunlunxin.P800.evidence_runner import target_name
        self.assertNotEqual(target_name(points[0],1),target_name(points[1],1))

    def test_invalid_selection(self):
        for ids in ([4,4], [], [-1], [8]):
            with self.assertRaises(ValueError): c.validate_selection(ids)
        c.validate_selection([1]); c.validate_selection([0,1,2])

    def test_identity_join_uses_pci_not_ordinal(self):
        host=[{'host_physical_id':6,'pci_bdf':'0000:b6:00.0','uuid':'six'},
              {'host_physical_id':4,'pci_bdf':'0000:84:00.0','uuid':'four'}]
        observed=[{'native_id':0,'pci_bdf':'00000000:84:00.0'}, {'native_id':1,'pci_bdf':'00000000:b6:00.0'}]
        self.assertEqual(resolve_native(host,observed)[6]['native_id'],1)
        with self.assertRaises(ValueError): resolve_native(host,observed[:1])
        observed[1]['pci_bdf']=observed[0]['pci_bdf']
        with self.assertRaises(ValueError): resolve_native(host,observed)

    def sample(self):
        return {'schema_version':1,'mode':'h2d','correctness':True,'checked_values':1000,'payload_bytes':1000,
                'samples_ns':[100,200,300], 'started_monotonic_ns':1000,'finished_monotonic_ns':2000,'timer':'native'}

    def test_unit_formula_and_no_factor_of_two(self):
        record=self.sample()
        point={'mode':'h2d','bytes':1000,'source':4}
        metric=c.metric_from_native(record,point,'interconnect-h2d','stdout',128,3)
        self.assertEqual(metric['value'],5.0)
        self.assertEqual(metric['unit'],'GB/s')
        self.assertEqual(c.metric_from_native(record,point,'interconnect-h2d-latency','stdout',128,3)['value'],200)

    def test_reject_exit_zero_without_correct_data(self):
        for changes in ({'correctness':False},{'samples_ns':[]},{'samples_ns':[float('nan')]*3},
                        {'samples_ns':[0,100,200]},{'payload_bytes':512},{'mode':'d2h'},
                        {'finished_monotonic_ns':1001}):
            with self.assertRaises(ValueError):
                c.validate_native({**self.sample(),**changes},{'mode':'h2d','bytes':1000},128,3)

    def test_chunk_protocol_requires_matching_mode_payload_and_count(self):
        point={'mode':'h2d','bytes':1000,'source':4,'pinned':True,'async':True}
        record={**self.sample(),'async_chunk_bytes':256,'api_calls_per_sample':4}
        metric=c.metric_from_native(record,point,'interconnect-h2d','stdout',128,3)
        self.assertEqual(metric['value'],5.0)
        self.assertEqual(metric['submission_scope'],'chunked pinned async logical payload')
        for changes in ({'api_calls_per_sample':3},{'async_chunk_bytes':1000},{'async_chunk_bytes':-1},{'checked_values':1}):
            with self.assertRaises(ValueError): c.validate_native({**record,**changes},point,128,3)
        for changes in ({'pinned':False},{'async':False},{'mode':'d2d'}):
            modified={**point,**changes}
            with self.assertRaises(ValueError): c.validate_native({**record,'mode':modified['mode']},modified,128,3)

    def test_bidirectional_uses_joint_time_and_checks_both_payloads(self):
        point={'mode':'p2p-bidir','bytes':1000,'source':4,'destination':5}
        record={**self.sample(),'mode':'p2p-bidir','checked_values':2000,'api_calls_per_sample':2}
        self.assertEqual(c.metric_from_native(record,point,'interconnect-P2P_intraserver','stdout',128,3)['value'],10.0)
        with self.assertRaises(ValueError): c.validate_native({**record,'checked_values':1000},point,128,3)
        with self.assertRaises(ValueError): c.validate_native({**record,'api_calls_per_sample':1},point,128,3)

    def test_computation_dtype_shape_and_sampled_correctness(self):
        point={'mode':'gemm','dtype':'FP32','source':4}
        record={**self.sample(),'mode':'gemm','dtype':'FP32','output_dtype':'FP32','dimension':2048,'checked_values':257}
        self.assertEqual(c.metric_from_native(record,point,'computation-FP32','stdout',2048,3)['unit'],'TFLOPS')
        for changes in ({'dtype':'FP16'},{'output_dtype':'INT32'},{'dimension':128},{'checked_values':256}):
            with self.assertRaises(ValueError): c.validate_native({**record,**changes},point,2048,3)

    def test_native_json_missing_or_ambiguous(self):
        for text in ('success','P800_METRIC {}\nP800_METRIC {}'):
            with self.assertRaises(ValueError): parse_record(text,'P800_METRIC ')

    def test_failure_propagates(self):
        self.assertEqual(c.layer_status(['passed','failed']),'failed')
        self.assertEqual(c.layer_status(['passed','not-run']),'partial')
        self.assertEqual(c.layer_status([]),'not-run')


class FacadeTests(unittest.TestCase):
    def request(self, *flags):
        return ToolkitRunRequest.from_namespace(parse_args(['--config',str(BASE/'configs/kunlunxin_p800_xpytorch29.yaml'),*flags]))

    def test_release_snapshot_does_not_qualify_a_failed_run(self):
        from executors import p800_toolkit as executor
        from executors.common import ConfigurationError
        with tempfile.TemporaryDirectory() as temp:
            request = self.request('--physical-device-ids', '4', '--result-root', temp)
            plan = ToolkitExecutor().plan(request)
            commands = Mock()
            commands.checked.side_effect = RuntimeError('fixture image inspection failed')
            container = Mock()
            container.cleanup.return_value = {'status': 'passed', 'container_absent': True}
            with patch.object(executor, 'HostCommands', return_value=commands), \
                 patch.object(executor, 'ManagedContainer', return_value=container), \
                 patch('subprocess.run', side_effect=AssertionError('hardware access forbidden')):
                self.assertEqual(executor.execute(request, plan), 1)
            summary = json.loads(next(Path(temp).glob('*/summary.json')).read_text())
            self.assertTrue(summary['validated'])
            self.assertEqual(summary['release_stage'], 'validated')
            self.assertEqual(summary['validation_status'], 'partial')
            self.assertEqual(summary['qualification_status'], 'not-qualified')
            self.assertEqual(summary['measurement_status'], 'not-run')
            self.assertEqual(summary['status'], 'failed')
            commands.checked.assert_called_once()
            container.create.assert_not_called()
            plan['runtime_lock']['image_manifest']['validated'] = False
            with patch.object(Path, 'mkdir', side_effect=AssertionError('write before gate')):
                with self.assertRaisesRegex(ConfigurationError, 'candidate'):
                    executor.execute(request, plan)

    def test_static_plan_never_calls_subprocess(self):
        with patch('subprocess.run',side_effect=AssertionError('dry-run touched system')):
            plan=ToolkitExecutor().plan(self.request('--physical-device-ids','0,4-6','--dry-run'))
        self.assertEqual(plan['suite'],'kunlunxin-p800-toolkit')
        self.assertEqual(plan['permissions']['network'],'none')
        self.assertEqual(plan['logical_point_counts']['interconnect-P2P_intraserver'],12)
        self.assertEqual(plan['repeat_count'],5)
        self.assertEqual(plan['settings']['p2p_payload_bytes'],33554432)
        expanded=ToolkitExecutor().plan(self.request('--physical-device-ids','0,4-6','--p2p-payload-bytes','268435456','--dry-run'))
        self.assertEqual(expanded['settings']['p2p_payload_bytes'],268435456)
        self.assertEqual(expanded['point_counts']['interconnect-P2P_intraserver'],60)

    def test_card_one_selection_allowed(self):
        expanded=ToolkitExecutor().plan(self.request('--physical-device-ids','0,1,4','--dry-run'))
        self.assertEqual(expanded['excluded_physical_ids'],[])
        self.assertEqual(expanded['point_counts']['computation-FP32'],15)
        full=ToolkitExecutor().plan(self.request('--physical-device-ids','0-7','--dry-run'))
        self.assertEqual(full['point_counts']['computation-FP32'],40)

    def test_aliases_and_vendor_flags_rejected(self):
        for flags in (('--npu-ids','4'),('--physical-device-ids','4','--allow-disruptive-dmi'),
                      ('--physical-device-ids','4','--case','not-real'),('--physical-device-ids','4','--samples','-1')):
            with self.assertRaises(RuntimeError): ToolkitExecutor().plan(self.request(*flags))

    def test_smoke_scope_and_busy_authorization_are_explicit(self):
        plan=ToolkitExecutor().plan(self.request('--physical-device-ids','4','--smoke','--allow-busy-devices'))
        self.assertEqual(plan['settings']['matrix_size'],128)
        self.assertTrue(plan['permissions']['allow_busy_devices'])


class EvidenceTests(unittest.TestCase):
    def test_setup_failure_is_failed_and_keeps_each_requested_case(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            e.write_json(root/'context.json',{'host':{'devices':[{'host_physical_id':4}]},
                'settings':{'payload_bytes':1024,'repeat':1},'cases':list(c.CASES),'run_id':'fixture',
                'conda_prefix':'/fixture','image_identity':{}})
            with patch.object(evidence_runner,'prepare',side_effect=RuntimeError('compiler failed')):
                self.assertEqual(evidence_runner.main(['--context',str(root/'context.json'),'--output',str(root/'evidence')]),1)
            manifest=json.loads((root/'evidence/manifest.json').read_text())
            self.assertEqual(manifest['measurement_status'],'failed')
            self.assertEqual(set(manifest['cases']),set(c.CASES))
            self.assertTrue(all(v['measurement_status']=='failed' for v in manifest['cases'].values()))

    def test_monitor_only_counts_complete_samples_inside_measurement_window(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            base={'device_id':'kunlunxin/test','valid':True}
            samples=[{**base,'started_offset_s':1+i*.5,'finished_offset_s':1.1+i*.5} for i in range(10)]
            monitor=SimpleNamespace(origin_monotonic_s=100,samples=samples,finish=lambda *a,**k:None)
            target={'target':'point/repeat-1','measurement_status':'passed','bindings':[{'uuid':'test','host_physical_id':4}],
                    'measurement_window':{'started_monotonic_s':101,'finished_monotonic_s':107,'role':'measurement'}}
            manifest={'cases':{'interconnect-h2d':{'targets':[target]}}}
            request=SimpleNamespace(compute_monitor='on',data_movement_monitor='on')
            finalize_monitor(root,manifest,monitor,request)
            self.assertEqual(target['monitoring_status'],'passed')
            samples[0]['started_offset_s']=.99
            finalize_monitor(root,manifest,monitor,request)
            self.assertEqual(target['monitor_sample_counts'],{'4':9})
            self.assertEqual(target['monitoring_status'],'partial')
            samples[0]['started_offset_s']=1
            samples[-1]['valid']=False
            finalize_monitor(root,manifest,monitor,request)
            self.assertEqual(target['monitoring_status'],'partial')
            self.assertEqual(target['invalid_monitor_samples'],1)

    def test_timeout_keeps_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            result=e.command([sys.executable,'-u','-c','import time; print("started",flush=True); time.sleep(5)'],root/'case',root,timeout=.1)
            self.assertEqual(result['returncode'],124)
            self.assertTrue(result['timed_out'])
            self.assertIn('started',(root/result['stdout']['path']).read_text())
            with self.assertRaises(FileExistsError): e.command(['anything'],root/'case',root)

    def test_final_status_composes_measurement_monitoring_diagnosis(self):
        self.assertEqual(compose_status('passed','passed','passed'),'passed')
        self.assertEqual(compose_status('failed','passed','passed'),'failed')
        self.assertEqual(compose_status('passed','partial','passed'),'partial')
        self.assertEqual(compose_status('not-run','not-run','not-run'),'partial')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            target={'target':'point/repeat-1','measurement_status':'passed','diagnosis_status':'passed','monitoring_status':'pending-host-finalization',
                    'bindings':[{'uuid':'test','host_physical_id':4}],
                    'measurement_window':{'started_monotonic_s':101,'finished_monotonic_s':107,'role':'measurement'}}
            case={'targets':[target],'measurement_status':'passed','diagnosis_status':'passed','status':'partial'}
            manifest={'cases':{'interconnect-h2d':case},'measurement_status':'passed','monitoring_status':'not-run'}
            monitor=SimpleNamespace(origin_monotonic_s=100,
                                    samples=[{'device_id':'kunlunxin/test','valid':True,'started_offset_s':1+i*.5,'finished_offset_s':1.1+i*.5} for i in range(10)],
                                    finish=lambda *a,**k:None)
            finalize_monitor(root,manifest,monitor,SimpleNamespace(compute_monitor='on',data_movement_monitor='on'))
            self.assertEqual(target['monitoring_status'],'passed')
            self.assertEqual(case['status'],'passed')
            self.assertEqual(manifest['status'],'passed')

    def test_hash_index_detects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'test').write_text('original');e.index(root)
            self.assertEqual(e.validate_index(root),[])
            (root/'test').write_text('changed')
            self.assertEqual(e.validate_index(root),['test'])

    def test_case_summary_uses_original_compact_format_and_splits_d2d(self):
        from toolkits._common.kunlunxin.P800.report import case_metric_rows
        base = {'point': {'source': 4, 'mode': 'd2d', 'bytes': 1000},
                'field': 'bandwidth', 'unit': 'GB/s', 'value': 10}
        kernel = {**base, 'point': {**base['point'], 'mode': 'd2d-kernel'}, 'value': 2090}
        rows = case_metric_rows('main_memory-bandwidth', {'metrics': [base, kernel]})
        self.assertEqual([title for title, _ in rows], ['D2D设备内存读写带宽', 'XRE runtime memcpy 拷贝吞吐'])
        self.assertEqual(metric_summary(rows[0][1]), 'n=1，2090.000–2090.000 GB/s，中位数 2090.000')
        self.assertEqual(metric_summary(rows[1][1]), 'n=1，10.000–10.000 GB/s，中位数 10.000')
        self.assertEqual(metric_summary([base, {**base, 'value': 12}]), 'n=2，10.000–12.000 GB/s，中位数 11.000')

    def test_report_uses_recorded_policy_and_does_not_infer_missing_policy(self):
        for source, excluded in (('resolved-plan.json', []), ('resolved-plan.json', [1]),
                                 ('summary', []), ('toolkit-evidence/toolkit-contract.json', [1]),
                                 (None, None)):
            with self.subTest(source=source, excluded=excluded), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                summary = {'suite': 'kunlunxin-p800-toolkit', 'status': 'partial', 'run_id': 'fixture'}
                policy = {'excluded_physical_ids': excluded}
                if source == 'summary':
                    summary['resolved_plan'] = policy
                elif source:
                    e.write_json(root / source, policy)
                if source == 'resolved-plan.json':
                    e.write_json(root / 'toolkit-evidence/toolkit-contract.json', {'excluded_physical_ids': [7]})
                e.write_json(root / 'summary.json', summary)
                e.write_json(root / 'toolkit-evidence/manifest.json', {'schema_version': 1, 'cases': {}, 'settings': {}})
                generate_and_record(root)
                text = (root / 'report.md').read_text()
                expected = json.dumps(excluded) if source else '未记录'
                self.assertIn('| 排除物理卡（本轮保存的策略） | ' + expected + ' |', text)
                self.assertNotIn('卡 1 已排除', text)
                self.assertEqual(json.loads((root / 'summary.json').read_text())['status'], 'partial')

    def test_kernel_chart_uses_read_write_and_group_cv_uses_sample_sd(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            point={'source':4,'mode':'d2d-kernel','bytes':1000}
            metrics=[{'point':point,'field':'bandwidth','unit':'GB/s','value':v,'repetition':i,'source':'raw.log'} for i,v in enumerate([10,10,10,10,11],1)]
            case={'metrics':metrics,'targets':[{'target':'kernel','point':point}]}
            e.write_json(root/'summary.json',{'suite':'kunlunxin-p800-toolkit','status':'partial','run_id':'fixture'})
            e.write_json(root/'toolkit-evidence/manifest.json',{'schema_version':1,'cases':{'main_memory-bandwidth':case},'settings':{}})
            e.write_json(root/'toolkit-evidence/cases/main_memory-bandwidth/kernel/samples.json',{'samples_ns':[4]})
            generate_and_record(root)
            chart=(root/'report-assets/main_memory-bandwidth-kernel-samples.svg').read_text()
            self.assertIn('>500</text>',chart)
            self.assertIn('4.3844%',(root/'report.md').read_text())
            self.assertEqual(json.loads((root/'summary.json').read_text())['status'],'partial')

    def test_public_report_splits_only_case_names_and_restores_original_summary(self):
        from generate_toolkit_report import generate_and_record as generate_public
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metrics = []
            for device, mode, value in ((4, 'd2d', 11), (7, 'd2d', 12),
                                        (4, 'd2d-kernel', 2100), (7, 'd2d-kernel', 2200)):
                for repetition in range(1, 6):
                    metrics.append({'point': {'source': device, 'mode': mode, 'bytes': 536870912},
                                    'field': 'bandwidth', 'unit': 'GB/s', 'value': value,
                                    'repetition': repetition, 'source': 'raw.log'})
            case = {'metrics': metrics, 'targets': [{'point': m['point'], 'target': str(i)}
                                                   for i, m in enumerate(metrics)],
                    'measurement_status': 'passed', 'monitoring_status': 'partial',
                    'diagnosis_status': 'passed', 'status': 'partial', 'duration_s': 327}
            e.write_json(root/'summary.json', {'suite': 'kunlunxin-p800-toolkit',
                                              'status': 'partial', 'run_id': 'fixture'})
            manifest_path = root/'toolkit-evidence/manifest.json'
            e.write_json(manifest_path, {'schema_version': 1, 'settings': {}, 'cases': {
                'main_memory-bandwidth': case}})
            original_manifest = manifest_path.read_bytes()
            generate_public(root)
            text = (root/'report.md').read_text()
            self.assertIn('| Case | 测量 | 监控 | 诊断 | 最终 | 指标摘要 | 范围 | 耗时 | 限制/错误 |', text)
            rows = [line for line in text.splitlines() if line.startswith(('| D2D设备内存读写带宽 |', '| XRE runtime memcpy 拷贝吞吐 |'))]
            self.assertEqual(len(rows), 2)
            for row in rows:
                columns = [column.strip() for column in row.strip('|').split('|')]
                self.assertEqual(columns[1:5], ['通过', '部分完成', '通过', '部分完成'])
                self.assertEqual(columns[6:], ['20 个厂商执行目标', '5 min 27.0 s', '—'])
            self.assertIn('| n=10，2100.000–2200.000 GB/s，中位数 2150.000 |', rows[0])
            self.assertIn('| n=10，11.000–12.000 GB/s，中位数 11.500 |', rows[1])
            self.assertNotIn('<br>', ''.join(rows))
            self.assertNotIn('card 4', ''.join(rows))
            self.assertNotIn('Case / 测量模式', text)
            self.assertNotIn('#### Scopy', text)
            self.assertNotIn('D2D 按模式分行展示', text)
            self.assertNotIn('进入的是 runtime 的通用复制路径', text)
            self.assertTrue((root/'report-assets/main_memory-bandwidth-GB-s.svg').is_file())
            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            self.assertEqual(json.loads((root/'summary.json').read_text())['status'], 'partial')
            self.assertEqual(e.validate_index(root), [])

    def test_report_regeneration_preserves_status_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            e.write_json(root/'summary.json',{'suite':'kunlunxin-p800-toolkit','status':'failed','run_id':'fixture'})
            e.write_json(root/'toolkit-evidence/manifest.json',{'schema_version':1,'cases':{},'settings':{}})
            generate_and_record(root)
            first={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
            generate_and_record(root)
            self.assertEqual(first,{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()})
            self.assertEqual(json.loads((root/'summary.json').read_text())['status'],'failed')
            self.assertEqual(e.validate_index(root),[])


if __name__=='__main__': unittest.main()
