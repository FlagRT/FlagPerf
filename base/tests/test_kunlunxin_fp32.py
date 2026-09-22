import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'benchmarks'))
from benchmarks.case_assets import read_config, resolve_case_assets
from benchmarks.fp32_contract import validate_config
from drivers import correctness, fp32, kunlunxin, utils
from executors.benchmark import BenchmarkRunRequest, BenchmarkExecutor, add_cli_arguments
from executors.bounded_benchmark import BenchmarkWorkload, validate_metric
from executors import preflight as lifecycle
from executors.common import DeviceLease
from base.vendors.kunlunxin.provider import KunlunxinProvider, binding_records
from base.vendors.protocol import ConfigurationError

PROFILE = BASE / 'configs/kunlunxin_p800_xpytorch29.yaml'
CASE = BASE / 'benchmarks/computation-FP32/kunlunxin/P800'
UID = '00000000-0000-0000-0000-000000000001'
DEVICE = dict(host_physical_id=6, uuid=UID, pci_bdf='0000:11:00.0', device_minor=5,
              device_major=240, host_device_node='/dev/xpu5', container_node='/dev/xpu5')


def request(*extra):
    parser = argparse.ArgumentParser()
    add_cli_arguments(parser)
    args = parser.parse_args(['--config', str(PROFILE), '--case', 'computation-FP32:P800',
                              '--physical-device-ids', '6', '--timeout', '300', *extra])
    return BenchmarkRunRequest.from_namespace(args)


class DriverTests(unittest.TestCase):
    def setUp(self):
        kunlunxin._binding = None
        kunlunxin._identity = None

    def fixture(self, root, **changes):
        context = {'kind': 'benchmark', 'run_id': 'fixture', 'host': {'devices': [DEVICE]}}
        raw = json.dumps(context).encode()
        (root / 'context.json').write_bytes(raw)
        record = {'run_id': 'fixture', 'context_sha256': hashlib.sha256(raw).hexdigest(),
                  'bindings': binding_records([DEVICE], [UID]), 'observed_uuids': [UID]}
        record.update(changes)
        (root / 'binding.json').write_text(json.dumps(record))
        return {'FLAGPERF_HOST_CONTEXT': str(root / 'context.json'),
                'FLAGPERF_RUNTIME_BINDINGS': str(root / 'binding.json'), 'CUDA_VISIBLE_DEVICES': '0', 'USE_FLAGGEMS': '0'}

    def test_binding_rechecked_and_initialization_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            env = self.fixture(Path(temp))
            with patch.dict(os.environ, env, clear=True), patch.object(kunlunxin.importlib, 'import_module'), \
                 patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'device_count', return_value=1), \
                 patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(uuid=UID)), \
                 patch.object(torch.cuda, 'set_device') as select:
                kunlunxin.initialize()
                kunlunxin.initialize()
                select.assert_called_once_with(0)
                with patch.object(torch.cuda, 'synchronize', side_effect=RuntimeError('sync error')):
                    with self.assertRaisesRegex(RuntimeError, 'sync error'):
                        kunlunxin.synchronize()

    def test_stale_binding_and_wrong_uuid_rejected(self):
        for change in ({'run_id': 'stale'}, {'context_sha256': 'stale'}, {'bindings': []}, {'observed_uuids': ['wrong']}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                env = self.fixture(Path(temp), **change)
                with patch.dict(os.environ, env, clear=True), patch.object(kunlunxin.importlib, 'import_module'), \
                     patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'device_count', return_value=1), \
                     patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(uuid=UID)), \
                     patch.object(torch.cuda, 'set_device') as select:
                    with self.assertRaises(RuntimeError):
                        kunlunxin.initialize()
                    select.assert_not_called()

    def test_extension_missing_propagates(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, self.fixture(Path(temp)), clear=True), \
             patch.object(kunlunxin.importlib, 'import_module', side_effect=ImportError('missing extension')):
            with self.assertRaises(ImportError):
                kunlunxin.initialize()

    def test_logical_device_not_derived_from_rank(self):
        kunlunxin._binding = {'framework_local_rank': 0, 'framework_logical_id': 7, 'framework_device_name': 'cuda:7'}
        self.assertEqual(str(kunlunxin.device(0)), 'cuda:7')
        with self.assertRaises(RuntimeError):
            kunlunxin.device(7)

    def test_dispatch_unknown_vendor_and_error_classification(self):
        with patch.object(kunlunxin, 'initialize') as initialize:
            utils.bootstrap_vendor('kunlunxin/P800')
            initialize.assert_called_once()
        with self.assertRaises(ValueError):
            utils.host_device_sync('unknown')
        self.assertTrue(kunlunxin.is_out_of_memory(torch.OutOfMemoryError('oom')))
        self.assertFalse(kunlunxin.is_out_of_memory(RuntimeError('sync timeout')))


class ContractTests(unittest.TestCase):
    def test_dry_run_has_no_commands_runtime_import_or_result_directory(self):
        with patch('subprocess.run', side_effect=AssertionError('side effect')), \
             patch.object(kunlunxin, 'initialize', side_effect=AssertionError('runtime import')), redirect_stdout(io.StringIO()):
            self.assertEqual(BenchmarkExecutor().execute(request('--dry-run')), 0)

    def test_invalid_config_and_selection_fail_closed(self):
        config = read_config(CASE / 'case_config.yaml')
        for change in ({'M': 99999}, {'ITERS': 0}, {'ATOL': float('nan')}, {'RTOL': 0.01},
                       {'DIST_BACKEND': 'nccl'}, {'FAULT_MODE': 'cpu-wait'}, {'IMPLEMENTATION': 'gems'}):
            with self.subTest(change=change), self.assertRaises(ConfigurationError):
                validate_config({**config, **change})
        for extra in (('--physical-device-ids', '6,7'), ('--nproc-per-node', '2'), ('--allow-privileged-root',), ('--timeout', '601')):
            with self.subTest(extra=extra), self.assertRaises(ConfigurationError):
                BenchmarkExecutor().plan(request(*extra))

    def test_snapshot_roundtrip_keeps_host_worker_contract(self):
        from benchmarks.case_assets import verify_worker_assets
        workload = BenchmarkWorkload(request('--case-config', str(CASE / 'case_config.smoke.yaml')), KunlunxinProvider())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload.prepare(root)
            actual = verify_worker_assets(BASE, root / 'control/case-assets.json', 'computation-FP32:P800', 'kunlunxin')
            expected_m = read_config(CASE / 'case_config.smoke.yaml')['M']
            self.assertEqual(actual['merged_config']['M'], expected_m)
            (root / 'control/case-config/override.yaml').write_text('M: 99\n')
            with self.assertRaises(ConfigurationError):
                verify_worker_assets(BASE, root / 'control/case-assets.json', 'computation-FP32:P800', 'kunlunxin')

    def test_candidate_and_window_gate_before_commands(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.run', side_effect=AssertionError('device access')):
            with self.assertRaises(ConfigurationError):
                BenchmarkExecutor().execute(request('--result-dir', str(Path(temp) / 'run')))
            with self.assertRaises(ConfigurationError):
                BenchmarkExecutor().execute(request('--allow-candidate-runtime', '--result-dir', str(Path(temp) / 'run')))
            self.assertFalse((Path(temp) / 'run').exists())


class NumericalTests(unittest.TestCase):
    def test_seed_reference_tolerance_and_nonfinite(self):
        left, right = correctness.input_pair(17, 29, 11, 519)
        repeated = correctness.input_pair(17, 29, 11, 519)
        self.assertTrue(torch.equal(left, repeated[0]))
        reference = left.double() @ right.double()
        actual = left @ right
        self.assertTrue(correctness.compare(actual, reference, atol=1e-4, rtol=1e-4)['passed'])
        for value in (0.1, float('nan'), float('inf')):
            modified = actual.clone()
            modified[0, 0] = actual[0, 0] + value
            self.assertFalse(correctness.compare(modified, reference, atol=1e-4, rtol=1e-4)['passed'])

    def test_large_shape_sampling_and_full_output_finiteness(self):
        left, right = correctness.input_pair(40, 33, 50, 519)
        actual = left @ right
        record = correctness.sampled_reference(left, right, actual, atol=1e-4, rtol=1e-4)
        self.assertTrue(record['passed'])
        self.assertEqual(record['reduction_size'], 33)
        actual[1, 1] = float('nan')
        self.assertFalse(correctness.sampled_reference(left, right, actual, atol=1e-4, rtol=1e-4)['passed'])

    def test_device_and_dtype_gate(self):
        for tensor, target in ((torch.ones(1), torch.device('cuda:0')), (torch.ones(1).double(), torch.device('cpu'))):
            with self.assertRaises(RuntimeError):
                fp32.require_placement([tensor], target)

    def run_cpu_fixture(self, temp, clock=(1000000000, 2000000000), failure=False):
        root = Path(temp)
        # Timer and correctness semantics are independent of the benchmark scale, which is
        # now the Ascend-parity 8192-cubed; keep this fixture explicitly small.
        config = {**read_config(CASE / 'case_config.yaml'), **read_config(CASE / 'case_config.smoke.yaml')}
        config.update(M=32, N=32, K=32, ITERS=4, WARMUP=1)
        calls = []
        driver = Mock()
        driver.device.return_value = torch.device('cpu')
        driver.evidence.return_value = {'run_id': 'fixture', 'context_sha256': 'hash'}
        driver.synchronize.side_effect = lambda: calls.append('sync')
        (root / 'context.json').write_text(json.dumps({'case_assets_sha256': 'case-hash'}))
        env = {'FLAGPERF_BENCHMARK_OUTPUT': str(root), 'FLAGPERF_HOST_CONTEXT': str(root / 'context.json')}
        iterator = iter(clock)
        def tick():
            calls.append('timer')
            return next(iterator)
        original = correctness.compare
        with patch.dict(os.environ, env), patch.object(fp32, 'benchmark_measurement_start', side_effect=lambda: calls.append('start')), \
             patch.object(fp32, 'benchmark_measurement_finish', side_effect=lambda token: calls.append('finish')), \
             patch.object(fp32.time, 'perf_counter_ns', side_effect=tick), \
             patch.object(fp32, 'compare', side_effect=(lambda *args, **kw: {**original(*args, **kw), 'passed': False}) if failure else original):
            value = fp32.run_verified_fp32(driver, config, 0, 1, 0)
        return calls, value

    def test_timing_excludes_reference_and_includes_final_sync(self):
        with tempfile.TemporaryDirectory() as temp:
            calls, value = self.run_cpu_fixture(temp)
            self.assertEqual(calls[-6:], ['sync', 'start', 'timer', 'sync', 'timer', 'finish'])
            self.assertEqual(value, 2 * 32**3 * 4 / 1e12)
            self.assertEqual(json.loads((Path(temp) / 'correctness-rank-0.json').read_text())['status'], 'passed')

    def test_bad_clock_and_correctness_never_write_success_metric(self):
        for clock, failure in (((1, 1), False), ((2, 1), False), ((1, 2), True)):
            with self.subTest(clock=clock, failure=failure), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(RuntimeError):
                    self.run_cpu_fixture(temp, clock, failure)
                self.assertFalse((Path(temp) / 'metric-rank-0.json').exists())
                self.assertEqual(json.loads((Path(temp) / 'correctness-rank-0.json').read_text())['status'], 'failed')


def rank_records(context, context_hash, binding, config):
    identity = {'schema_version': 1, 'status': 'passed', 'rank': 0, 'run_id': context['run_id'],
                'context_sha256': context_hash, 'binding': binding, 'worker_pid': 2}
    numerical = {'passed': True, 'finite': True, 'atol': config['ATOL'], 'rtol': config['RTOL'],
                 'full_output_finite': True, 'reduction_size': config['N']}
    correctness_record = {**identity, 'seed': config['SEED'], 'dtype': 'float32', 'reference_dtype': 'float64',
                          'small_cases': [numerical] * 3, 'shape_before': numerical, 'shape_after': numerical}
    operations = 2 * config['M'] * config['N'] * config['K'] * config['ITERS']
    metric = {**identity, 'world_size': 1, 'metric': 'computation-FP32', 'unit': 'TFLOPS',
              'case_assets_sha256': context['case_assets_sha256'], 'shape': [config['M'], config['N'], config['K']],
              'iterations': config['ITERS'], 'warmup': config['WARMUP'], 'mode': config['MODE'],
              'implementation': config['IMPLEMENTATION'], 'elapsed_seconds': 20., 'operations': operations,
              'value': operations / 20 / 1e12, 'started_monotonic_ns': 1000000000, 'finished_monotonic_ns': 21000000000}
    return metric, correctness_record


class MetricTests(unittest.TestCase):
    def test_rejects_tampered_rank_context_and_formula(self):
        config = read_config(CASE / 'case_config.yaml')
        context = {'run_id': 'fixture', 'case_assets_sha256': 'case'}
        binding = binding_records([DEVICE], [UID])[0]
        metric, record = rank_records(context, 'hash', binding, config)
        validate_metric(metric, record, context, 'hash', binding, config)
        for change in ({'rank': 1}, {'run_id': 'stale'}, {'value': float('nan')}, {'value': 0},
                       {'value': metric['value'] * 2}, {'operations': 1}, {'elapsed_seconds': -1}, {'iterations': 2}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_metric({**metric, **change}, record, context, 'hash', binding, config)
        for change in ({'status': 'failed'}, {'small_cases': []}, {'seed': 0}, {'dtype': 'float16'}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_metric(metric, {**record, **change}, context, 'hash', binding, config)


class BoundedIntegrationTests(unittest.TestCase):
    def run_fixture(self, mode):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            leases, calls = [], []
            config = resolve_case_assets(BASE, 'computation-FP32:P800', 'kunlunxin', CASE / 'case_config.smoke.yaml')['merged_config']
            request_value = replace(request('--case-config', str(CASE / 'case_config.smoke.yaml')),
                result_dir=root, allow_candidate_runtime=True, monitor='on' if mode == 'partial' else 'off',
                reservation_end=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
                reservation_reference='synthetic CPU test')
            def lease_factory(ids, **kwargs):
                kwargs.update(root=Path(temp) / 'locks', compatibility_paths=[Path(temp) / 'compat.lock'])
                lease = DeviceLease(ids, **kwargs)
                leases.append(lease)
                return lease
            def inspect(config, selected, commands, directory):
                calls.append(directory.name)
                if directory.name != 'host-preflight':
                    self.assertTrue(leases[0]._streams)
                return {'devices': [deepcopy(DEVICE)]}
            def wait(deadline):
                if mode == 'timeout':
                    raise TimeoutError('bounded fixture timeout')
                if mode == 'interrupt':
                    raise KeyboardInterrupt('bounded fixture interrupt')
                raw = (root / 'control/host-context.json').read_bytes()
                context = json.loads(raw)
                context_hash = hashlib.sha256(raw).hexdigest()
                binding = binding_records([DEVICE], [UID])[0]
                output = root / 'artifacts'
                (output / 'runtime-bindings.json').write_text(json.dumps({'schema_version': 1, 'run_id': context['run_id'],
                    'context_sha256': context_hash, 'bindings': [binding], 'observed_uuids': [UID]}))
                (output / 'probe.json').write_text(json.dumps({'status': 'passed', 'run_id': context['run_id'],
                    'context_sha256': context_hash, 'binding': binding}))
                metric, numerical = rank_records(context, context_hash, binding, config)
                if mode == 'stale':
                    metric['context_sha256'] = 'stale'
                for name, record in [('metric', metric), ('correctness', numerical)]:
                    if not (mode == 'missing' and name == 'correctness'):
                        (output / (name + '-rank-0.json')).write_text(json.dumps(record))
                if mode != 'missing-event':
                    (output / 'benchmark-events').mkdir()
                    (output / 'benchmark-events/measurement-rank-0.json').write_text(json.dumps({
                        'kind': 'measurement-window', 'case': 'computation-FP32:P800', 'rank': 0, 'local_rank': 0,
                        'world_size': 1, 'pid': 2, 'started_monotonic_ns': 999999999, 'finished_monotonic_ns': 21000000001}))
                line = "[FlagPerf Result]Rank 0's computation-FP32=" + str(metric['value']) + 'TFLOPS\n'
                (root / 'container-logs.json').write_text(json.dumps({'returncode': 0, 'stdout': line * (2 if mode == 'duplicate' else 1), 'stderr': ''}))
                return 0
            container = Mock()
            container.cid = 'a' * 64
            container.wait.side_effect = wait
            container.inspect.return_value = {'Id': container.cid, 'Image': 'fixture', 'State': {'Pid': 123, 'Running': True}}
            container.cleanup.return_value = {'container_absent': mode != 'cleanup-unknown', 'status': 'passed'}
            manifest = json.loads((BASE / 'vendors/kunlunxin/xpytorch_2.9_p800_candidate/image-manifest.json').read_text())
            image = {'Id': manifest['image_id'], 'RepoDigests': manifest['repo_digests'], 'Architecture': 'amd64'}
            commands = Mock()
            commands.checked.side_effect = lambda argv, **kw: {'stdout': json.dumps([image]) if argv[0] == 'docker' else 'fixture'}
            monitor = Mock()
            monitor.origin_monotonic_s = 0
            monitor.finish.return_value = {'status': 'partial'}
            with patch.object(lifecycle, 'HostCommands', return_value=commands), \
                 patch.object(lifecycle, 'ManagedContainer', return_value=container), \
                 patch.object(lifecycle, 'DeviceLease', side_effect=lease_factory), \
                 patch.object(KunlunxinProvider, 'inspect_host', side_effect=inspect), \
                 patch.object(KunlunxinProvider, 'create_monitor', return_value=monitor), redirect_stdout(io.StringIO()):
                code = BenchmarkExecutor().execute(request_value)
            summary = json.loads((root / 'summary.json').read_text())
            self.assertFalse(leases[0]._streams)
            self.assertTrue(container.cleanup.called)
            if mode != 'cleanup-unknown':
                self.assertEqual(calls[-1], 'host-postflight')
            before = (root / 'summary.json').read_bytes()
            report = (root / 'report.md').read_bytes()
            from generate_benchmark_report import generate_benchmark_report
            generate_benchmark_report(root)
            self.assertEqual(before, (root / 'summary.json').read_bytes())
            self.assertEqual(report, (root / 'report.md').read_bytes())
            return code, summary

    def test_complete_single_rank_evidence(self):
        code, summary = self.run_fixture('normal')
        self.assertEqual(code, 0)
        self.assertEqual(summary['correctness_status'], 'passed')

    def test_missing_duplicate_and_stale_evidence_never_pass(self):
        for mode in ('missing', 'duplicate', 'stale'):
            with self.subTest(mode=mode):
                code, summary = self.run_fixture(mode)
                self.assertEqual(code, 1)
                self.assertEqual(summary['status'], 'failed')


    def test_missing_event_or_partial_monitor_is_partial(self):
        for mode in ('missing-event', 'partial'):
            with self.subTest(mode=mode):
                code, summary = self.run_fixture(mode)
                self.assertEqual(code, 2)
                self.assertEqual(summary['status'], 'partial')

    def test_timeout_interrupt_and_cleanup_uncertainty(self):
        for mode in ('timeout', 'interrupt', 'cleanup-unknown'):
            with self.subTest(mode=mode):
                code, summary = self.run_fixture(mode)
                self.assertEqual(code, 1)
                self.assertEqual(summary['status'], 'failed')


class QualificationStatisticsTests(unittest.TestCase):
    def fixtures(self, parent, values):
        config = read_config(CASE / 'case_config.yaml')
        contract = json.dumps({'merged_config': config}).encode()
        binding = binding_records([DEVICE], [UID])[0]
        directories = []
        for index, value in enumerate(values):
            root = parent / str(index)
            (root / 'artifacts').mkdir(parents=True)
            (root / 'control').mkdir()
            (root / 'benchmark-monitor').mkdir()
            run_id = 'run-' + str(index)
            context = {'run_id': run_id, 'case_assets_sha256': hashlib.sha256(contract).hexdigest()}
            raw_context = json.dumps(context).encode()
            metric, correct = rank_records(context, hashlib.sha256(raw_context).hexdigest(), binding, config)
            elapsed = metric['operations'] / value / 1e12
            metric.update(value=value, elapsed_seconds=elapsed,
                          finished_monotonic_ns=metric['started_monotonic_ns'] + round(elapsed * 1e9))
            metric['elapsed_seconds'] = (metric['finished_monotonic_ns'] - metric['started_monotonic_ns']) / 1e9
            metric['value'] = metric['operations'] / metric['elapsed_seconds'] / 1e12
            summary = {key: 'passed' for key in ('status', 'execution_status', 'measurement_status', 'correctness_status',
                'measurement_evidence_status', 'monitoring_status', 'postflight_status', 'cleanup_status')}
            summary.update(run_id=run_id, lease_released=True, qualification={'mode': 'qualification'},
                           runtime={'image_id': 'fixed-image'}, device_bindings=[binding])
            documents = {'summary.json': summary, 'benchmark-result.json': {'status': 'passed', 'metrics': [metric]},
                'benchmark-monitor/summary.json': {'status': 'passed', 'primary_sample_counts_by_target': {binding['resource_key']: 20}},
                'code-identity.json': {'source_sha256': {'code': 'fixed'}}, 'artifacts/metric-rank-0.json': metric,
                'artifacts/correctness-rank-0.json': correct}
            for filename, content in documents.items():
                (root / filename).write_text(json.dumps(content))
            (root / 'control/host-context.json').write_bytes(raw_context)
            (root / 'control/case-assets.json').write_bytes(contract)
            directories.append(root)
        return directories

    def test_all_five_values_and_sample_deviation_used(self):
        from qualification import summarize
        import statistics
        values = [4.0, 4.01, 3.99, 4.02, 3.98]
        with tempfile.TemporaryDirectory() as temp:
            result = summarize(self.fixtures(Path(temp), values))
            self.assertEqual(result['status'], 'passed')
            self.assertAlmostEqual(result['sample_standard_deviation'], statistics.stdev(values), places=8)
            self.assertAlmostEqual(result['median'], 4)
            self.assertEqual(result['ddof'], 1)

    def test_unstable_runs_not_trimmed(self):
        from qualification import summarize
        with tempfile.TemporaryDirectory() as temp:
            result = summarize(self.fixtures(Path(temp), [4, 4, 4, 4, 2]))
            self.assertEqual(result['status'], 'unstable')
            self.assertEqual(len(result['runs']), 5)

    def test_incomplete_mixed_code_and_tampered_evidence_rejected(self):
        from qualification import summarize
        for mode in ('duplicate', 'failed', 'mixed-code', 'missing-correctness', 'tampered-metric'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                directories = self.fixtures(Path(temp), [4] * 5)
                if mode == 'duplicate':
                    directories[-1] = directories[0]
                elif mode == 'failed':
                    path = directories[0] / 'summary.json'
                    record = json.loads(path.read_text())
                    record['status'] = 'failed'
                    path.write_text(json.dumps(record))
                elif mode == 'mixed-code':
                    (directories[0] / 'code-identity.json').write_text(json.dumps({'source_sha256': {'code': 'different'}}))
                elif mode == 'missing-correctness':
                    (directories[0] / 'artifacts/correctness-rank-0.json').unlink()
                else:
                    path = directories[0] / 'artifacts/metric-rank-0.json'
                    record = json.loads(path.read_text())
                    record['value'] = 999
                    path.write_text(json.dumps(record))
                with self.assertRaises((ValueError, RuntimeError, OSError)):
                    summarize(directories)

if __name__ == '__main__':
    unittest.main()
