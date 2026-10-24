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

    def test_large_cold_trace_joins_flows_without_rescanning_per_kernel(self):
        from vendors.thead.trace import analyze_events
        class CountedEvents(list):
            scans = 0
            def __iter__(self):
                self.scans += 1
                return super().__iter__()
        events = CountedEvents([dict(ph='X', cat='user_annotation', name='operation_target_0',
                                    pid=1, tid=2, ts=0, dur=10000)])
        for corr in range(1, 257):
            events.extend([
                dict(ph='X', cat='cuda_driver', name='cuLaunchKernelEx', pid=1, tid=2,
                     ts=corr * 2, dur=1, args={'correlation': corr}),
                dict(ph='s', cat='ac2g', id=corr, pid=1, tid=2, ts=corr * 2),
                dict(ph='X', cat='kernel', name='autotune', pid=4, tid=5, ts=10000 + corr * 2,
                     dur=1, args={'correlation': corr, 'device': 0, 'stream': 1}),
                dict(ph='f', cat='ac2g', id=corr, pid=4, tid=5, ts=10000 + corr * 2)])
        result = analyze_events(events, count=1)
        self.assertEqual(result['status'], 'measured')
        self.assertEqual(result['kernel_count'], 256)
        self.assertLess(events.scans, 20)

    def test_standard_driver_launch_requires_the_same_semantic_correlation(self):
        from vendors.thead.dispatch import analyze_dispatch
        for op in ('aten::mv', 'aten::prod'):
            with self.subTest(op=op):
                events, rows = self.fixture()
                rows[0]['op'] = op
                events[2].update(cat='cuda_driver', name='cuLaunchKernel')
                self.assertEqual(analyze_dispatch(events, rows)['status'], 'passed')
                for removed in (3, 5):
                    incomplete = copy.deepcopy(events)
                    del incomplete[removed]
                    self.assertEqual(analyze_dispatch(incomplete, rows)['status'], 'partial')
                events[4]['args']['device'] = 1
                self.assertEqual(analyze_dispatch(events, rows)['status'], 'partial')

    def test_driver_api_names_do_not_qualify_copies_or_misplaced_launches(self):
        from vendors.thead.dispatch import analyze_dispatch
        for category, name in (('cuda_driver', 'cuMemcpyDtoD'),
                               ('cuda_runtime', 'cuLaunchKernel'),
                               ('cuda_driver', 'cuLaunchKernelUnknown')):
            with self.subTest(category=category, name=name):
                events, rows = self.fixture()
                events[2].update(cat=category, name=name)
                self.assertEqual(analyze_dispatch(events, rows)['status'], 'partial')
        events, rows = self.fixture()
        events[2].update(cat='cuda_driver', name='cuLaunchKernel', tid=99)
        events[3]['tid'] = 99
        self.assertEqual(analyze_dispatch(events, rows)['status'], 'partial')

    def test_indexed_flow_join_still_rejects_duplicate_and_wrong_timestamps(self):
        from vendors.thead.dispatch import analyze_dispatch
        for index in (3, 5):
            with self.subTest(index=index, problem='duplicate'):
                events, rows = self.fixture()
                events.append(copy.deepcopy(events[index]))
                self.assertEqual(analyze_dispatch(events, rows)['status'], 'partial')
            with self.subTest(index=index, problem='timestamp'):
                events, rows = self.fixture()
                events[index]['ts'] += 1
                self.assertEqual(analyze_dispatch(events, rows)['status'], 'partial')

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

    def test_native_counter_parser_handles_live_acu_progress_and_report_footer(self):
        from vendors.thead.profiling import parse_counter_csv
        header = '"ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"\n'
        metric = '"0","abs","Occupancy","Active Warps","%","3.59"\n'
        footer = 'Processing [> ] 0 %\rProcessing [===] 100 %\n[Asight]: Trace written into the output file: target.acurep\n'
        malformed = '"1","abs","Occupancy","missing value"\n'
        nonfinite = '"2","abs","Occupancy","invalid","%","inf"\n'
        missing_unit = '"3","abs","Occupancy","no unit","","2"\n'
        records = parse_counter_csv(header + metric + malformed + nonfinite + missing_unit + footer)
        self.assertEqual(records, parse_counter_csv(header + metric))
        self.assertEqual(len(records), 1)
        self.assertEqual(parse_counter_csv(header + footer), [])

    def test_asight_sections_use_sdk_identifiers_not_filenames(self):
        from vendors.thead.profiling import ASIGHT_SECTIONS
        self.assertEqual(ASIGHT_SECTIONS, (
            'SpeedOfLight', 'MemoryWorkloadAnalysis', 'Occupancy',
            'WarpStateStats', 'LaunchStats'))

    def test_counter_capture_requests_unit_bearing_details_and_checks_identity(self):
        from vendors.thead.profiling import collect_counters, ASIGHT_SECTION_NAMES
        from types import SimpleNamespace
        identity = dict(vendor_runtime={}, packages={}, case_sha256='case', input_sha256='inputs',
                        registration={}, runtime_config={}, device='cuda:2', input_devices=['cuda:2'])
        csv = '"ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"\n'
        csv += ''.join('"0","abs","' + section + '","native metric","%","3.59"\n'
                       for section in ASIGHT_SECTION_NAMES.values())
        window = {'calls': 5, 'device': 2, 'status': 'completed'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'measurement.json').write_text(json.dumps(identity))
            original = Path.is_file
            def is_file(path):
                return str(path) == '/usr/local/PPU_SDK/asight/bin/acu' or original(path)
            def launch(command, **kwargs):
                self.assertEqual(command[command.index('--page') + 1], 'details')
                self.assertEqual(command[command.index('--devices') + 1], '2')
                folder = root / 'profiling/native-counters'
                kwargs['stdout'].write(csv)
                kwargs['stdout'].flush()
                (folder / 'profile-target-identity.json').write_text(json.dumps(identity))
                (folder / 'profile-window.json').write_text(json.dumps(window))
                (folder / 'target.acurep').write_bytes(b'native evidence')
                return SimpleNamespace(returncode=0, wait=lambda **kw: 0, poll=lambda: 0)
            with patch.object(Path, 'is_file', is_file), \
                    patch('vendors.thead.profiling.subprocess.Popen', side_effect=launch):
                result = collect_counters(root, {'worker_device': 2})
            self.assertEqual(result['status'], 'measured')
            self.assertEqual(result['metric_records'], 5)
            self.assertEqual(result['missing_sections_by_kernel'], {})
            rows = json.loads((root / 'profiling/native-counters/metrics.json').read_text())
            self.assertEqual(rows[0]['Metric Unit'], '%')
            self.assertEqual(rows[0]['Section Name'], 'PPU Speed Of Light Throughput')
            for index, invalid in enumerate(({}, {'calls': 4}, {'device': 0}, {'device': True},
                                             {'status': 'started'}, {'calls': 5.0})):
                with self.subTest(window=invalid):
                    root = Path(tmp) / ('bad-window-' + str(index))
                    root.mkdir()
                    (root / 'measurement.json').write_text(json.dumps(identity))
                    window = {**{'calls': 5, 'device': 2, 'status': 'completed'}, **invalid} if invalid else {}
                    with patch.object(Path, 'is_file', is_file), \
                            patch('vendors.thead.profiling.subprocess.Popen', side_effect=launch):
                        result = collect_counters(root, {'worker_device': 2})
                    self.assertEqual(result['status'], 'partial')
                    self.assertIn('target window', result['reason'])
            window = {'calls': 5, 'device': 2, 'status': 'completed'}
            root = Path(tmp) / 'missing-sections'
            root.mkdir()
            (root / 'measurement.json').write_text(json.dumps(identity))
            csv = '\n'.join(csv.splitlines()[:2]) + '\n'
            with patch.object(Path, 'is_file', is_file), \
                    patch('vendors.thead.profiling.subprocess.Popen', side_effect=launch):
                result = collect_counters(root, {'worker_device': 2})
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(len(result['missing_sections_by_kernel']['0']), 4)
            root = Path(tmp) / 'identity-drift'
            root.mkdir()
            (root / 'measurement.json').write_text(json.dumps({**identity, 'packages': {'different': '1'}}))
            with patch.object(Path, 'is_file', is_file), \
                    patch('vendors.thead.profiling.subprocess.Popen', side_effect=launch):
                result = collect_counters(root, {'worker_device': 2})
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['mismatches'], ['packages'])

    def test_cpu_workers_drop_device_node_creation_capability_without_device_mapping(self):
        import subprocess
        args = cli.parser().parse_args(['run', '--vendor', 'thead', '--device-ids', '6'])
        with tempfile.TemporaryDirectory() as tmp:
            pool = cli.Workers(Path(tmp), Adapter(), args, 'locked-image', 'run')
            for phase in ('reference', 'reference-check', 'diagnose'):
                with patch.object(cli.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as launch:
                    pool.start({'device_id': 6, 'oplib': 'nativetorch'}, phase)
                argv = launch.call_args.args[0]
                self.assertIn('--cap-drop=ALL', argv)
                self.assertIn('--security-opt=no-new-privileges', argv)
                self.assertNotIn('--device', argv)

    def test_full_profile_cleans_owned_workers_before_capture_and_stops_on_failure(self):
        adapter = Adapter()
        self.assertTrue(adapter.isolate_workers({'profiling_mode': 'full'}, 'profiling'))
        self.assertFalse(adapter.isolate_workers({'profiling_mode': 'timeline'}, 'profiling'))
        self.assertFalse(adapter.isolate_workers({'profiling_mode': 'full'}, 'measure'))
        args = cli.parser().parse_args(['run', '--vendor', 'thead', '--device-ids', '6'])
        with tempfile.TemporaryDirectory() as tmp:
            pool = cli.Workers(Path(tmp), adapter, args, 'image', 'run')
            with patch.object(pool, 'close', side_effect=cli.CleanupError('not confirmed')) as cleanup, \
                    patch.object(pool, 'start') as start:
                with self.assertRaises(cli.CleanupError):
                    pool.phase(Path(tmp), {'profiling_mode': 'full'}, 'profiling')
                cleanup.assert_called_once_with()
                start.assert_not_called()

    def test_asight_preload_requires_counter_phase_and_locked_mapped_library(self):
        import hashlib
        from vendors.thead.bootstrap import checked_preload
        body = b'locked pthread'
        locked = {'path': '/usr/lib/x86_64-linux-gnu/libpthread.so.0',
                  'sha256': hashlib.sha256(body).hexdigest()}
        profile = {'asight_preload': locked}
        env = {'LD_PRELOAD': 'libpthread.so.0', 'FLAGPERF_PPU_COUNTER_TARGET': '1'}
        maps = '0-1 r-xp 0 00:00 0 ' + locked['path'] + '\n'
        with patch.dict('os.environ', env, clear=True), \
                patch.object(Path, 'read_text', return_value=maps), \
                patch.object(Path, 'read_bytes', return_value=body):
            self.assertEqual(checked_preload('profile-target', profile)['sha256'], locked['sha256'])
            for phase in ('reference', 'probe', 'measure', 'profiling'):
                with self.assertRaises(RuntimeError):
                    checked_preload(phase, profile)
            with patch.dict('os.environ', {'LD_PRELOAD': 'libunknown.so'}):
                with self.assertRaises(RuntimeError):
                    checked_preload('profile-target', profile)
            with patch.object(Path, 'read_text', return_value=maps.replace('/usr/lib/', '/tmp/')):
                with self.assertRaises(RuntimeError):
                    checked_preload('profile-target', profile)
            with patch.object(Path, 'read_bytes', return_value=b'changed'):
                with self.assertRaises(RuntimeError):
                    checked_preload('profile-target', profile)
            with patch.dict('os.environ', {'FLAGPERF_PPU_COUNTER_TARGET': '0'}):
                with self.assertRaises(RuntimeError):
                    checked_preload('profile-target', profile)

    def test_reviewed_counter_preload_does_not_change_invariant_runtime_identity(self):
        from types import SimpleNamespace
        adapter = Adapter()
        adapter.device = 'cuda:0'
        adapter.framework_observation = {'uuid': 'locked'}
        torch = SimpleNamespace(__version__='2.10.0',
            cuda=SimpleNamespace(get_device_name=lambda device: 'PPU'),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
        record = dict(executable='python', profile_sha256='profile', packages={}, hook_hashes={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bootstrap.json'
            with patch.dict('sys.modules', {'torch': torch}), \
                    patch.dict('os.environ', {'PPU_OPERATION_BOOTSTRAP_RECORD': str(path)}):
                path.write_text(json.dumps({**record, 'reviewed_preload': {}}))
                measured = adapter.identity()
                path.write_text(json.dumps({**record, 'reviewed_preload': {'name': 'libpthread.so.0'}}))
                self.assertEqual(adapter.identity(), measured)
if __name__ == '__main__':
    unittest.main()
