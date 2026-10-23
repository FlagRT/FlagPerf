import json
from pathlib import Path
import tempfile
import unittest
import copy
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import cli
from vendors import get_vendor
from vendors.thead.adapter import Adapter


class TheadOperationTests(unittest.TestCase):
    def test_profile_and_vendor_selection(self):
        adapter = get_vendor('thead', 'torch210-ppu')
        self.assertIsInstance(adapter, Adapter)
        self.assertEqual(adapter.lock['expected_device_ids'], list(range(16)))
        self.assertFalse(adapter.lock['validated'])

    def test_dry_run_has_locked_profile_and_both_paths(self):
        args = cli.parser().parse_args([
            'run', '--vendor', 'thead', '--runtime-profile', 'torch210-ppu',
            '--device-ids', '6', '--case', 'abs', '--dtype', 'FP32',
            '--oplib', 'both', '--workload', 'smoke', '--dry-run'])
        adapter, plan = cli.resolve(args)
        self.assertIsInstance(adapter, Adapter)
        self.assertEqual(len(plan['tasks']), 2)
        self.assertTrue(all(t['runtime_profile'] == 'torch210-ppu' for t in plan['tasks']))
        self.assertTrue(all(t['runtime_config']['sharing_policy'] == 'exclusive-required' for t in plan['tasks']))

    def test_privileged_and_extra_mapping_are_rejected(self):
        for extra in (['--allow-privileged-root'], ['--container-device', '/dev/test'],
                      ['--mount', '/tmp/a:/tmp/b:ro']):
            argv = ['run', '--vendor', 'thead', '--runtime-profile', 'torch210-ppu',
                    '--device-ids', '6', '--case', 'abs', '--dtype', 'FP32', *extra]
            args = cli.parser().parse_args(argv)
            with self.assertRaises(ValueError): cli.resolve(args)

    def test_route_is_conservative(self):
        adapter = Adapter()
        result = adapter.route({'device': 'cuda:0', 'output_devices': ['cuda:0'], 'calls': []}, '')
        self.assertEqual(result['status'], 'partial')
        self.assertIn('not proof', result['boundary'])

    def test_runtime_identity_rejects_other_image(self):
        adapter = Adapter()
        with self.assertRaises(ValueError): adapter.runtime_identity('wrong', {'Id': 'sha256:' + '0' * 64})

    def test_selection_rejects_out_of_range_and_duplicate_ids(self):
        for ids in ('16', '-1', '6,6'):
            args = cli.parser().parse_args(['run', '--vendor', 'thead', '--device-ids', ids,
                                           '--case', 'abs', '--dry-run'])
            with self.assertRaises(ValueError): cli.resolve(args)

    def test_shared_opt_in_is_explicit_and_cannot_qualify_performance(self):
        args = cli.parser().parse_args(['run', '--vendor', 'thead', '--device-ids', '6',
                                       '--case', 'abs', '--allow-shared-device', '--dry-run'])
        adapter, plan = cli.resolve(args)
        self.assertEqual(plan['tasks'][0]['runtime_config']['sharing_policy'], 'shared-authorized')
        identity = dict(vendor_runtime={}, packages={}, case_sha256='a', input_sha256='b', registration={}, runtime_config={}, device='cuda:0', input_devices=['cuda:0'])
        item = {**identity, 'status': 'passed', 'measurement': identity,
                'runtime_config': {'execution': 'docker', 'sharing_policy': 'shared-authorized'}}
        probe = {**identity, 'runtime_config': item['runtime_config']}
        item['measurement'] = probe
        adapter.qualify_result(item, probe)
        self.assertEqual(item['status'], 'partial')
        self.assertEqual(item['performance_qualification']['status'], 'partial')

    def test_foreign_handle_environment_cannot_bypass_policy(self):
        from vendors.thead.host import HostSession
        session = object.__new__(HostSession)
        with patch.dict('os.environ', {'THEAD_ALLOW_FOREIGN_HANDLES': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'does not accept'):
                session.observe('test')

    def test_all_applicable_ppu_inputs_and_module_state_construct_on_cpu(self):
        import torch
        from runtime.worker import build, tensors
        args = cli.parser().parse_args(['run', '--vendor', 'thead', '--device-ids', '6',
                                       '--dtype', 'FP32', 'FP16', 'BF16', 'INT32', 'INT16', 'BOOL', 'INT64',
                                       '--oplib', 'both', '--workload', 'smoke', '--dry-run'])
        _, plan = cli.resolve(args)
        count = 0
        for task in plan['tasks']:
            if not task['applicable'] or task['oplib'] != 'nativetorch':
                continue
            with self.subTest(case=task['case'], dtype=task['dtype']):
                fn, inputs, backward, work = build(task)
                values = tensors(inputs)
                if isinstance(fn, torch.nn.Module):
                    values += list(fn.parameters()) + list(fn.buffers())
                self.assertTrue(values)
                self.assertTrue(all(t.device.type == 'cpu' for t in values))
                self.assertGreater(work(1), 0)
                count += 1
        self.assertEqual(count, 154)

    def test_replay_refreshes_profile_and_current_case_identity(self):
        args = cli.parser().parse_args(['diagnose', '--source-task', '/tmp/source', '--replay'])
        task = {'case': 'abs', 'device_id': 6, 'runtime_config': {'old': True}}
        Adapter().prepare_replay(task, args)
        self.assertNotIn('old', task['runtime_config'])
        self.assertIn('case_source_sha256', task['runtime_config'])

    def test_memory_identity_drift_withholds_displayed_values(self):
        item = {'memory': {'status': 'measured', 'peak_allocated_bytes': 42}, 'measurement': {}}
        Adapter().qualify_supplement(item, 'memory')
        self.assertEqual(item['memory']['status'], 'partial')
        self.assertNotIn('peak_allocated_bytes', item['memory'])


class CorrelationTests(unittest.TestCase):
    def fixture(self):
        events = [
            dict(ph='X', cat='user_annotation', name='ppu_target_first_0', pid=1, tid=2, ts=0, dur=100),
            dict(ph='X', cat='user_annotation', name='ppu_dispatch_0', pid=1, tid=2, ts=10, dur=50),
            dict(ph='X', cat='cuda_runtime', name='cudaLaunchKernel', pid=1, tid=2, ts=20, dur=1, args={'correlation': 7}),
            dict(ph='s', cat='ac2g', id=7, pid=1, tid=2, ts=20),
            dict(ph='X', cat='kernel', name='abs_kernel', pid=4, tid=5, ts=500, dur=3, args={'correlation': 7, 'device': 0, 'stream': 1}),
            dict(ph='f', cat='ac2g', id=7, pid=4, tid=5, ts=500)]
        rows = [dict(index=0, op='aten::abs', device_input=True, device_output=True, cpu_transfer=False)]
        return events, rows

    def test_complete_semantic_dispatch_requires_correlated_device_kernel(self):
        from vendors.thead.dispatch import analyze_dispatch
        events, rows = self.fixture()
        self.assertEqual(analyze_dispatch(events, rows)['status'], 'passed')

    def test_driver_api_launch_is_correlated_for_flaggems(self):
        from vendors.thead.dispatch import analyze_dispatch
        events, rows = self.fixture()
        launch = events[2]
        launch['cat'] = 'cuda_driver'
        launch['name'] = 'cuLaunchKernelEx'
        self.assertEqual(analyze_dispatch(events, rows)['status'], 'passed')

    def test_missing_wrong_or_unattributed_device_evidence_stays_partial(self):
        from vendors.thead.dispatch import analyze_dispatch
        for index in (2, 3, 4, 5):
            events, rows = self.fixture()
            del events[index]
            self.assertEqual(analyze_dispatch(events, rows)['status'], 'partial')
        events, rows = self.fixture()
        events[4]['args']['device'] = 1
        self.assertEqual(analyze_dispatch(events, rows)['status'], 'partial')

    def test_copy_only_cannot_prove_computation(self):
        from vendors.thead.dispatch import analyze_dispatch
        events, rows = self.fixture()
        rows[0]['op'] = 'aten::copy_'
        self.assertEqual(analyze_dispatch(events, rows)['status'], 'partial')

    def test_cpu_transfer_fails(self):
        from vendors.thead.dispatch import analyze_dispatch
        events, rows = self.fixture()
        rows[0]['cpu_transfer'] = True
        self.assertEqual(analyze_dispatch(events, rows)['status'], 'failed')

    def test_missing_cached_route_cannot_pass(self):
        adapter = Adapter()
        probe = {'vendor_route': {'protocol': 'ppu-routing-v1', 'dispatch': {}, 'windows': {}},
                 'device': 'cuda:0', 'output_devices': ['cuda:0']}
        self.assertEqual(adapter.route(probe, 'OPERATION_TARGET_BEGIN\nOPERATION_TARGET_END')['status'], 'partial')

    def test_native_counter_parser_preserves_units_and_rejects_missing_values(self):
        from vendors.thead.profiling import parse_counter_csv
        text = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n"0","abs","dram__bytes.sum","byte","1024"\n"1","abs","unknown","%","N/A"\n'
        records = parse_counter_csv(text)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['Metric Unit'], 'byte')
        self.assertEqual(parse_counter_csv('no metrics'), [])


if __name__ == '__main__':
    unittest.main()
