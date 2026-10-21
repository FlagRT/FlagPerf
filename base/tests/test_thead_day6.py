"""CPU-only contracts for the thead PPU Day 6 memory/communication cases."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'benchmarks'))
from benchmarks import day6_contract as contract
from benchmarks.case_assets import read_config, resolve_case_assets
from base.vendors.protocol import ConfigurationError
from base.vendors.thead import container_probe
from base.vendors.thead import provider as thead_provider
from base.vendors.thead.provider import TheadProvider, binding_records

PPU_UUID = '011ef111-4522-040e-0000-0000c08f6579'
DEVICE = {'host_physical_id': 0, 'pci_bdf': '00000001:c9:00.0', 'uuid': PPU_UUID,
          'host_device_node': '/dev/alixpu_ppu0', 'container_node': '/dev/alixpu_ppu0', 'device_minor': 1}

DAY6_CASES = ('main_memory-bandwidth:PPU', 'main_memory-capacity:PPU',
              'interconnect-MPI_intraserver:PPU', 'interconnect-P2P_intraserver:PPU')


def thead_case_config(case, mode=None):
    directory = BASE / 'benchmarks' / case.rpartition(':')[0] / 'thead/PPU'
    config = read_config(directory / 'case_config.yaml')
    if mode:
        config.update(read_config(directory / ('case_config.' + mode + '.yaml')))
    return config


def make_request(case, nproc=None, timeout=300, privileged=False):
    return type('R', (), {'case': case, 'allow_privileged_root': privileged,
                          'context': type('C', (), {'timeout': timeout, 'result_root': None})(),
                          'result_dir': None, 'nproc_per_node': nproc})()


class TheadDay6ContractTests(unittest.TestCase):
    def test_ppu_selectors_extend_without_disturbing_the_p800_order(self):
        legacy = ('main_memory-bandwidth:P800', 'main_memory-capacity:P800',
                  'interconnect-P2P_intraserver:P800', 'interconnect-MPI_intraserver:P800')
        self.assertEqual(contract.CASES[:4], legacy)
        for case in DAY6_CASES:
            self.assertIn(case, contract.CASES)
        self.assertEqual(contract.METRIC_NAMES['main_memory-bandwidth:P800'],
                         contract.METRIC_NAMES['main_memory-bandwidth:PPU'])
        for case in contract.MEMORY_CASES:
            self.assertEqual(contract.expected_ranks(case), 1)
        for case in contract.COMMUNICATION_CASES:
            self.assertEqual(contract.expected_ranks(case), 2)

    def test_communication_backend_follows_the_chip(self):
        self.assertEqual(contract.communication_backend('P800'), 'cpu:gloo,cuda:flagcx')
        self.assertEqual(contract.communication_backend('PPU'), 'cpu:gloo,cuda:nccl')
        mpi = dict(thead_case_config('interconnect-MPI_intraserver:PPU'))
        contract.validate_config(mpi, 'interconnect-MPI_intraserver:PPU')
        with self.assertRaises(ConfigurationError):
            contract.validate_config({**mpi, 'DIST_BACKEND': 'cpu:gloo,cuda:flagcx'},
                                     'interconnect-MPI_intraserver:PPU')
        p800 = dict(mpi, DIST_BACKEND='cpu:gloo,cuda:flagcx')
        contract.validate_config(p800, 'interconnect-MPI_intraserver:P800')
        with self.assertRaises(ConfigurationError):
            contract.validate_config(p800, 'interconnect-MPI_intraserver:PPU')

    def test_allreduce_coefficient_bound_follows_the_chip(self):
        self.assertEqual(contract.max_allreduce_world_size('interconnect-MPI_intraserver:P800'), 8)
        self.assertEqual(contract.max_allreduce_world_size('interconnect-MPI_intraserver:PPU'), 16)
        # The historical two-argument call keeps the P800 bound.
        for size in range(2, 9):
            self.assertEqual(contract.allreduce_timed_coefficient(0, size), size - 1)
        with self.assertRaises(ValueError):
            contract.allreduce_timed_coefficient(8, 9)
        self.assertEqual(contract.allreduce_timed_coefficient(0, 16, 16), 15)
        self.assertEqual(contract.allreduce_timed_coefficient(15, 16, 16), -1)
        with self.assertRaises(ValueError):
            contract.allreduce_timed_coefficient(16, 17, 16)

    def test_interserver_cases_stay_p800_unsupported_records(self):
        for case in contract.INTERSERVER_CASES:
            self.assertNotIn(case, contract.CASES)
            path = BASE / 'benchmarks' / case.rpartition(':')[0] / 'kunlunxin/P800/runtime_requirements.json'
            record = json.loads(path.read_text())
            self.assertIs(record['supported'], False)


class TheadDay6AssetTests(unittest.TestCase):
    def test_day6_chip_assets_validate(self):
        scopes = {'main_memory-bandwidth': {'nnodes': 1, 'nproc_per_node': 1},
                  'main_memory-capacity': {'nnodes': 1, 'nproc_per_node': 1},
                  'interconnect-MPI_intraserver': {'nnodes': 1, 'min_nproc_per_node': 2, 'max_nproc_per_node': 16},
                  'interconnect-P2P_intraserver': {'nnodes': 1, 'nproc_per_node': 2, 'min_nproc_per_node': 2}}
        for case in DAY6_CASES:
            with self.subTest(case=case):
                assets = resolve_case_assets(BASE, case, 'thead')
                merged = assets['merged_config']
                self.assertEqual(merged['IMPLEMENTATION'], 'native-xpytorch')
                self.assertEqual(merged['MODE'],
                                 'capacity' if case.rpartition(':')[0] == 'main_memory-capacity' else 'qualification')
                self.assertEqual(merged['DTYPE'], 'float32')
                contract.validate_config(merged, case)
                self.assertEqual(assets['requirements']['process_scope'], scopes[case.rpartition(':')[0]])
                self.assertTrue(assets['entrypoint']['relative_path'].endswith('thead/PPU/main.py'))

    def test_override_layers_stay_bounded(self):
        for case in DAY6_CASES:
            family = case.rpartition(':')[0]
            with self.subTest(case=case):
                smoke = thead_case_config(case, 'smoke')
                self.assertEqual(smoke['MODE'], 'smoke')
                if family != 'main_memory-capacity':
                    self.assertEqual(smoke['ITERS'], 2)
                contract.validate_config({**thead_case_config(case), **smoke}, case)
                if family != 'main_memory-capacity':
                    calibration = thead_case_config(case, 'calibration')
                    self.assertEqual(calibration['MODE'], 'calibration')
                    self.assertLessEqual(calibration['ITERS'], 64)
                    contract.validate_config({**thead_case_config(case), **calibration}, case)

    def test_capacity_stays_the_only_high_risk_asset(self):
        assets = resolve_case_assets(BASE, 'main_memory-capacity:PPU', 'thead')
        self.assertIs(assets['requirements']['high_risk'], True)
        assets = resolve_case_assets(BASE, 'main_memory-bandwidth:PPU', 'thead')
        self.assertIs(assets['requirements'].get('high_risk'), None)


class TheadDay6ProviderTests(unittest.TestCase):
    def test_probe_and_provider_registries_match(self):
        registered = ('computation-FP32:PPU', 'computation-FP16:PPU', 'computation-BF16:PPU',
                      'computation-INT8:PPU', 'computation-FP64:PPU', 'computation-TF32:PPU',
                      'interconnect-h2d:PPU', 'interconnect-d2h:PPU',
                      'main_memory-bandwidth:PPU', 'main_memory-capacity:PPU',
                      'interconnect-MPI_intraserver:PPU', 'interconnect-P2P_intraserver:PPU')
        self.assertEqual(container_probe.PERFORMANCE_CASES, thead_provider.PERFORMANCE_CASES)
        self.assertEqual(container_probe.PERFORMANCE_CASES, registered)

    def test_day6_request_bounds_are_per_family(self):
        provider = TheadProvider()
        assets = {case: resolve_case_assets(BASE, case, 'thead') for case in DAY6_CASES}
        provider.validate_benchmark(make_request('main_memory-bandwidth:PPU'), assets['main_memory-bandwidth:PPU'])
        provider.validate_benchmark(make_request('main_memory-capacity:PPU'), assets['main_memory-capacity:PPU'])
        provider.validate_benchmark(make_request('interconnect-P2P_intraserver:PPU', nproc=2),
                                    assets['interconnect-P2P_intraserver:PPU'])
        for nproc in (2, 8, 16):
            provider.validate_benchmark(make_request('interconnect-MPI_intraserver:PPU', nproc=nproc),
                                        assets['interconnect-MPI_intraserver:PPU'])
        for kwargs, case in (({'nproc': 2}, 'main_memory-bandwidth:PPU'),
                             ({'nproc': 16}, 'interconnect-P2P_intraserver:PPU'),
                             ({'nproc': 1}, 'interconnect-MPI_intraserver:PPU'),
                             ({'nproc': 17}, 'interconnect-MPI_intraserver:PPU'),
                             ({'timeout': 20}, 'main_memory-bandwidth:PPU'),
                             ({'privileged': True}, 'interconnect-MPI_intraserver:PPU')):
            with self.subTest(case=case, **kwargs), self.assertRaises(ConfigurationError):
                provider.validate_benchmark(make_request(case, **kwargs), assets[case])

    def test_merged_config_drift_is_rejected(self):
        case = 'interconnect-MPI_intraserver:PPU'
        assets = resolve_case_assets(BASE, case, 'thead')
        broken = {'requirements': assets['requirements'],
                  'merged_config': {**assets['merged_config'], 'Melements': 257}}
        with self.assertRaises(ConfigurationError):
            TheadProvider().validate_benchmark(make_request(case, nproc=2), broken)


class TheadDay6ExecutionTests(unittest.TestCase):
    def run_bandwidth(self, case, selector_env=True):
        binding = binding_records([DEVICE], [PPU_UUID])[0]
        config = thead_case_config(case)
        config.update(Melements=1, ITERS=4, WARMUP=1)
        context = {'case': case, 'run_id': 'fixture', 'case_assets_sha256': 'case-hash', 'nproc_per_node': 1}
        from drivers import day6
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'context.json').write_text(json.dumps({'case_assets_sha256': 'case-hash'}))
            driver = Mock()
            driver.memory_info.return_value = (1 << 40, 1 << 41)
            driver.device.return_value = torch.device('cpu')
            driver.evidence.return_value = {'binding': binding, 'bindings': [binding],
                                            'run_id': 'fixture', 'context_sha256': 'context'}
            driver.binding.return_value = binding
            calls = []
            driver.synchronize.side_effect = lambda: calls.append('sync')

            def tick():
                calls.append('timer')
                return calls.count('timer') * 10 ** 9

            env = {'FLAGPERF_BENCHMARK_OUTPUT': str(root), 'FLAGPERF_HOST_CONTEXT': str(root / 'context.json')}
            if selector_env:
                env['FLAGPERF_BENCHMARK_CASE'] = case
            with patch.dict(os.environ, env), patch.object(day6.time, 'perf_counter_ns', side_effect=tick), \
                 patch.object(day6, 'benchmark_measurement_start', side_effect=lambda: calls.append('start')), \
                 patch.object(day6, 'benchmark_measurement_finish', side_effect=lambda token: calls.append('finish')):
                gb, gib = day6.run_memory_bandwidth(driver, config, 0, 1, 0)
            payload = contract.payload_bytes(1)
            self.assertAlmostEqual(gb, 2 * payload * 4 / 1.0 / 1e9)
            metric = json.loads((root / 'metric-rank-0.json').read_text())
            correctness = json.loads((root / 'correctness-rank-0.json').read_text())
            self.assertEqual(metric['case'], case)
            self.assertEqual(correctness['case'], case)
            contract.validate_metric(metric, correctness, context, 'context', binding, config, case)
            self.assertEqual((metric['unit'], metric['metric'], metric['world_size']), ('GB/s', 'device-memory-bandwidth', 1))
            self.assertEqual(correctness['status'], 'passed')

    def test_bandwidth_chain_accepts_the_ppu_selector(self):
        self.run_bandwidth('main_memory-bandwidth:PPU')

    def test_bandwidth_chain_keeps_the_p800_default(self):
        # No FLAGPERF_BENCHMARK_CASE in the environment: the historical P800
        # selector stays the default and the chain validates unchanged.
        with patch.dict(os.environ, {}):
            os.environ.pop('FLAGPERF_BENCHMARK_CASE', None)
            self.run_bandwidth('main_memory-bandwidth:P800', selector_env=False)

    def test_case_family_guard_rejects_mismatched_selector(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'context.json').write_text(json.dumps({'case_assets_sha256': 'case-hash'}))
            driver = Mock()
            env = {'FLAGPERF_BENCHMARK_OUTPUT': str(root), 'FLAGPERF_HOST_CONTEXT': str(root / 'context.json'),
                   'FLAGPERF_BENCHMARK_CASE': 'interconnect-MPI_intraserver:PPU'}
            with patch.dict(os.environ, env):
                from drivers import day6
                with self.assertRaises(RuntimeError):
                    day6.run_memory_bandwidth(driver, thead_case_config('main_memory-bandwidth:PPU'), 0, 1, 0)


class TheadDay6MetricTests(unittest.TestCase):
    CONTEXT = {'run_id': 'run-day6-test', 'case_assets_sha256': 'a' * 64, 'nproc_per_node': 16}
    CONTEXT_HASH = 'context'
    BINDING = 'binding-day6-test'

    def capacity_records(self):
        config = thead_case_config('main_memory-capacity:PPU')
        metric = {'schema_version': 1, 'status': 'passed', 'run_id': self.CONTEXT['run_id'],
                  'context_sha256': self.CONTEXT_HASH, 'rank': 0, 'binding': self.BINDING,
                  'world_size': 1, 'case_assets_sha256': self.CONTEXT['case_assets_sha256'],
                  'mode': config['MODE'], 'implementation': config['IMPLEMENTATION'],
                  'metric': 'main_memory-capacity', 'unit': 'GB', 'value': 32 * (1 << 20) / 1e9,
                  'value_gib_s': 32 / 1024, 'held_mib': 32,
                  'started_monotonic_ns': 10 ** 12, 'finished_monotonic_ns': 10 ** 12 + 5 * 10 ** 9,
                  'elapsed_seconds': 5.0}
        correctness = {'schema_version': 1, 'status': 'passed', 'run_id': self.CONTEXT['run_id'],
                       'context_sha256': self.CONTEXT_HASH, 'rank': 0, 'binding': self.BINDING,
                       'released': True, 'release_verified': True, 'free_before_mib': 65536,
                       'free_after_bytes': 65536 * (1 << 20), 'held_mib': 32,
                       'trail': [{'stage': 'held', 'request_mib': 32, 'held_total_mib': 32}]}
        return config, metric, correctness

    def test_capacity_metric_recomputes_for_the_ppu(self):
        case = 'main_memory-capacity:PPU'
        config, metric, correctness = self.capacity_records()
        validated = contract.validate_metric(metric, correctness, self.CONTEXT, self.CONTEXT_HASH,
                                             self.BINDING, config, case)
        self.assertEqual(validated['metric'], 'main_memory-capacity')
        correctness['free_after_bytes'] = 0
        with self.assertRaises(RuntimeError):
            contract.validate_metric(metric, correctness, self.CONTEXT, self.CONTEXT_HASH,
                                     self.BINDING, config, case)

    def test_sixteen_rank_coefficient_evidence_is_enforced(self):
        case = 'interconnect-MPI_intraserver:PPU'
        config = thead_case_config(case)
        payload = contract.payload_bytes(config['Melements'])
        elapsed = 16.0

        def records(rank):
            algbw, algbw_gib, busbw, busbw_gib = contract.allreduce_bandwidth(payload, config['ITERS'], elapsed, 16)
            common = {'schema_version': 1, 'status': 'passed', 'run_id': self.CONTEXT['run_id'],
                      'context_sha256': self.CONTEXT_HASH, 'rank': rank, 'binding': self.BINDING,
                      'world_size': 16, 'case_assets_sha256': self.CONTEXT['case_assets_sha256'],
                      'mode': config['MODE'], 'implementation': config['IMPLEMENTATION']}
            started = 10 ** 12
            metric = {**common, 'metric': 'allreduce-algbw', 'unit': 'GB/s', 'value': algbw,
                      'value_gib_s': algbw_gib, 'busbw_gb_s': busbw, 'busbw_gib_s': busbw_gib,
                      'iterations': config['ITERS'], 'warmup': config['WARMUP'],
                      'started_monotonic_ns': started, 'finished_monotonic_ns': started + int(elapsed * 10 ** 9),
                      'elapsed_seconds': elapsed, 'message_bytes': payload, 'total_bytes': payload * config['ITERS']}
            checks = [{'phase': phase, 'passed': True, 'elements': payload // 4,
                       'mismatch_count': 0, 'first_mismatch': None}
                      for phase in ('cold', 'second', 'post-loop')]
            correctness = {**common, 'checks': checks, 'timed_input_coefficient':
                           contract.allreduce_timed_coefficient(rank, 16, 16)}
            return metric, correctness

        for rank in (0, 15):
            metric, correctness = records(rank)
            contract.validate_metric(metric, correctness, self.CONTEXT, self.CONTEXT_HASH,
                                     self.BINDING, config, case)
        metric, correctness = records(3)
        correctness['timed_input_coefficient'] = 1
        with self.assertRaises(RuntimeError):
            contract.validate_metric(metric, correctness, self.CONTEXT, self.CONTEXT_HASH,
                                     self.BINDING, config, case)

    def test_stdout_pairs_cover_the_ppu_selectors(self):
        for case in DAY6_CASES:
            with self.subTest(case=case):
                pairs = contract.expected_stdout_pairs(case, {'value': 1.0, 'value_gib_s': 1.0, 'busbw_gb_s': 1.0})
                self.assertTrue(pairs)
                for name, unit, value in pairs:
                    self.assertIsInstance(value, float)
                    self.assertTrue(unit in ('GB/s', 'GiB/s', 'GB', 'GiB'))


class TheadDay6WorkerTests(unittest.TestCase):
    def test_rank_limit_follows_the_vendor(self):
        import benchmark_worker
        run_bound_case, vendor_rank_limit = benchmark_worker.run_bound_case, benchmark_worker.vendor_rank_limit
        self.assertEqual(vendor_rank_limit('thead'), 16)
        self.assertEqual(vendor_rank_limit('kunlunxin'), 8)
        for vendor, nproc in (('thead', 17), ('kunlunxin', 9)):
            with self.subTest(vendor=vendor, nproc=nproc), self.assertRaises(RuntimeError):
                run_bound_case({'nproc_per_node': nproc, 'kind': 'benchmark', 'vendor': vendor},
                               Path('context.json'), Path('out'))


if __name__ == '__main__':
    unittest.main()
