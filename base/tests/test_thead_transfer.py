"""CPU-only contracts for the thead PPU transfer cases."""
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
from benchmarks import transfer_contract
from benchmarks.case_assets import read_config, resolve_case_assets
from benchmarks.transfer_contract import CASES, contract, validate_config
from drivers import transfer as transfer_driver
from executors.bounded_benchmark import validate_metric
from base.vendors.thead import container_probe
from base.vendors.thead import provider as thead_provider

H2D = 'interconnect-h2d:PPU'
D2H = 'interconnect-d2h:PPU'
DEVICE = {'host_physical_id': 0, 'pci_bdf': '00000001:c9:00.0',
          'uuid': '011ef111-4522-040e-0000-0000c08f6579',
          'host_device_node': '/dev/alixpu_ppu0', 'container_node': '/dev/alixpu_ppu0', 'device_minor': 1}


class TheadTransferContractTests(unittest.TestCase):
    def test_chip_scoped_cases_share_direction_contracts(self):
        self.assertIn(H2D, CASES)
        self.assertIn(D2H, CASES)
        self.assertIn('interconnect-h2d:P800', CASES)
        self.assertEqual(contract(H2D), contract('interconnect-h2d:P800'))
        self.assertEqual(contract(H2D)['metric'], 'transfer-bandwidth')
        self.assertEqual(contract(D2H)['metric'], 'd2h-bandwidth')
        with self.assertRaises(transfer_contract.ConfigurationError):
            contract('interconnect-h2d:UNKNOWN')

    def test_h2d_chip_assets_carry_the_bounded_contract(self):
        assets = resolve_case_assets(BASE, H2D, 'thead')
        self.assertEqual((assets['vendor'], assets['chip'], assets['selector']), ('thead', 'PPU', 'thead/PPU'))
        self.assertEqual([item['relative_path'] for item in assets['requirement_files']],
                         ['benchmarks/interconnect-h2d/thead/PPU/runtime_requirements.json'])
        self.assertTrue(assets['entrypoint']['relative_path'].endswith('interconnect-h2d/main.py'))
        merged = assets['merged_config']
        for key, expected in (('PAYLOAD_BYTES', 512 * 2 ** 20), ('DIRECTION', 'h2d'), ('PIN_MEMORY', False),
                              ('NON_BLOCKING', False), ('REUSE_DESTINATION', True), ('DTYPE', 'float32'),
                              ('DIST_BACKEND', 'gloo'), ('IMPLEMENTATION', 'native-xpytorch'), ('MODE', 'measured'),
                              ('SEED', 519)):
            self.assertEqual(merged.get(key), expected, key)
        validate_config(merged, H2D)
        self.assertEqual(assets['requirements']['process_scope'], {'nnodes': 1, 'nproc_per_node': 1})

    def test_d2h_chip_assets_carry_the_bounded_contract(self):
        merged = resolve_case_assets(BASE, D2H, 'thead')['merged_config']
        self.assertEqual(merged['DIRECTION'], 'd2h')
        validate_config(merged, D2H)

    def test_measured_variants_stay_bounded(self):
        for direction in ('h2d', 'd2h'):
            directory = BASE / ('benchmarks/interconnect-' + direction + '/thead/PPU')
            base = read_config(directory / 'case_config.yaml')
            for variant in ('measured-pinned-blocking', 'measured-pinned-nonblocking',
                            'measured-pageable-nonblocking'):
                with self.subTest(direction=direction, variant=variant):
                    override = read_config(directory / ('case_config.' + variant + '.yaml'))
                    self.assertEqual(override.get('MODE'), 'measured', variant)
                    validate_config({**base, **override}, 'interconnect-' + direction + ':PPU')
            calibration = read_config(directory / 'case_config.calibration.yaml')
            self.assertEqual(calibration['MODE'], 'calibration')
            validate_config({**base, **calibration}, 'interconnect-' + direction + ':PPU')

    def test_registries_cover_both_transfer_selectors(self):
        for selector in (H2D, D2H):
            self.assertIn(selector, thead_provider.PERFORMANCE_CASES)
            self.assertIn(selector, container_probe.PERFORMANCE_CASES)

    def test_provider_accepts_bounded_transfer_request(self):
        provider = thead_provider.TheadProvider()
        assets = resolve_case_assets(BASE, H2D, 'thead')

        def request(case=H2D):
            return type('R', (), {'case': case, 'allow_privileged_root': False,
                                  'context': type('C', (), {'timeout': 300, 'result_root': None})(),
                                  'result_dir': None, 'nproc_per_node': 1})()
        provider.validate_benchmark(request(), assets)
        with self.assertRaises(transfer_contract.ConfigurationError):
            provider.validate_benchmark(request(), {**assets, 'merged_config':
                                                     {**assets['merged_config'], 'PIN_MEMORY': 'yes'}})


class TheadTransferExecutionTests(unittest.TestCase):
    def run_copy(self, case, direction, env_case=None):
        # CPU-mocked driver; the hand-built binding keeps validate_metric's
        # placement check consistent with str(torch.device('cpu')).
        binding = {'framework_device_name': 'cpu', 'vendor': 'thead', 'chip': 'PPU'}
        directory = BASE / 'benchmarks/interconnect-h2d/thead/PPU'
        config = read_config(directory / 'case_config.yaml')
        config.update(PAYLOAD_BYTES=4096, ITERS=4, WARMUP=1, DIRECTION=direction)
        context = {'case': case, 'run_id': 'fixture', 'case_assets_sha256': 'case-hash'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'context.json').write_text(json.dumps({'case_assets_sha256': 'case-hash'}))
            driver = Mock()
            driver.device.return_value = torch.device('cpu')
            driver.memory_info.return_value = (2 ** 40, 2 ** 40)
            driver.evidence.return_value = {'binding': binding, 'bindings': [binding],
                                            'run_id': 'fixture', 'context_sha256': 'context'}
            calls = []
            driver.synchronize.side_effect = lambda: calls.append('sync')

            def tick():
                calls.append('timer')
                return calls.count('timer') * 10 ** 9

            env = {'FLAGPERF_BENCHMARK_OUTPUT': str(root), 'FLAGPERF_HOST_CONTEXT': str(root / 'context.json'),
                   'FLAGPERF_BENCHMARK_CASE': env_case or case}
            with patch.dict(os.environ, env), \
                 patch.object(transfer_driver.time, 'perf_counter_ns', side_effect=tick), \
                 patch.object(transfer_driver, 'benchmark_measurement_start', side_effect=lambda: calls.append('start')), \
                 patch.object(transfer_driver, 'benchmark_measurement_finish', side_effect=lambda token: calls.append('finish')):
                value = transfer_driver.run_verified_transfer(driver, config, 0, 1, 0, direction)
                metric = json.loads((root / 'metric-rank-0.json').read_text())
                correctness = json.loads((root / 'correctness-rank-0.json').read_text())
        return value, metric, correctness, context, binding, config

    def test_verified_h2d_chain_accepts_the_ppu_selector(self):
        value, metric, correctness, context, binding, config = self.run_copy(H2D, 'h2d')
        gb, gib = value
        self.assertGreater(gb, 0)
        validate_metric(metric, correctness, context, 'context', binding, config)
        self.assertEqual((metric['metric'], metric['unit'], metric['binding']),
                         ('transfer-bandwidth', 'GB/s', binding))
        self.assertEqual([record['phase'] for record in correctness['checks']],
                         ['sentinel', 'changed-input', 'before', 'after'])

    def test_verified_d2h_chain_accepts_the_ppu_selector(self):
        value, metric, correctness, context, binding, config = self.run_copy(D2H, 'd2h')
        validate_metric(metric, correctness, context, 'context', binding, config)
        self.assertEqual(metric['metric'], 'd2h-bandwidth')

    def test_case_selector_guard_rejects_direction_mismatch(self):
        with patch.dict(os.environ, {'FLAGPERF_BENCHMARK_CASE': H2D}):
            with self.assertRaises(RuntimeError):
                transfer_driver.run_verified_transfer(Mock(), {'DIRECTION': 'd2h'}, 0, 1, 0, 'd2h')


if __name__ == '__main__':
    unittest.main()
