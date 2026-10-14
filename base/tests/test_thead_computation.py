"""CPU-only contracts for the thead PPU computation-FP32 bounded case."""
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
from benchmarks.case_assets import read_config, resolve_case_assets
from benchmarks.computation_contract import CASES, contract, validate_config
from drivers import fp32
from executors.bounded_benchmark import validate_metric
from base.vendors.protocol import ConfigurationError
from base.vendors.thead import provider as thead_provider
from base.vendors.thead import container_probe
from base.vendors.thead.provider import TheadProvider, binding_records

PPU_UUID = '011ef111-4522-040e-0000-0000c08f6579'
DEVICE = {'host_physical_id': 0, 'pci_bdf': '00000001:c9:00.0', 'uuid': PPU_UUID,
          'host_device_node': '/dev/alixpu_ppu0', 'container_node': '/dev/alixpu_ppu0', 'device_minor': 1}
CASE = 'computation-FP32:PPU'


def thead_case_config(mode=None):
    directory = BASE / 'benchmarks/computation-FP32/thead/PPU'
    config = read_config(directory / 'case_config.yaml')
    if mode:
        config.update(read_config(directory / ('case_config.' + mode + '.yaml')))
    return config


class TheadComputationContractTests(unittest.TestCase):
    def test_ppu_selector_shares_the_p800_precision_contract(self):
        self.assertIn(CASE, CASES)
        self.assertIn('computation-FP32:P800', CASES)
        self.assertEqual(contract(CASE), contract('computation-FP32:P800'))
        for selector in ('computation-FP32:UNKNOWN', 'computation-INT9:PPU', 'computation-FP32'):
            with self.subTest(selector=selector), self.assertRaises(ConfigurationError):
                contract(selector)

    def test_case_assets_layer_without_foreign_fallback(self):
        assets = resolve_case_assets(BASE, CASE, 'thead')
        self.assertEqual((assets['vendor'], assets['chip'], assets['selector']), ('thead', 'PPU', 'thead/PPU'))
        self.assertEqual(sorted(assets['configurations']), ['chip', 'generic'])
        self.assertEqual(assets['environments'], [])
        self.assertEqual([item['relative_path'] for item in assets['requirement_files']],
                         ['benchmarks/computation-FP32/thead/PPU/runtime_requirements.json'])
        self.assertTrue(assets['entrypoint']['relative_path'].endswith('computation-FP32/main.py'))
        merged = assets['merged_config']
        for key, expected in (('DIST_BACKEND', 'gloo'), ('DTYPE', 'float32'), ('IMPLEMENTATION', 'native-xpytorch'),
                              ('MODE', 'qualification'), ('SEED', 519), ('M', 8192), ('N', 8192), ('K', 8192),
                              ('WARMUP', 100), ('OPERATOR', 'torch.mm'), ('FAULT_MODE', 'none')):
            self.assertEqual(merged.get(key), expected, key)
        self.assertEqual(merged['ATOL'], 0.0001)
        self.assertEqual(merged['RTOL'], 0.0001)
        self.assertEqual(type(merged['ITERS']), int)
        validate_config(merged, CASE)
        self.assertEqual(assets['requirements']['process_scope'], {'nnodes': 1, 'nproc_per_node': 1})
        self.assertEqual(assets['requirements']['runtime_profile'], 'torch210_ppu_candidate')

    def test_generic_layer_is_overridden_not_inherited(self):
        merged = resolve_case_assets(BASE, CASE, 'thead')['merged_config']
        self.assertNotEqual(merged['DIST_BACKEND'], 'mpi')
        self.assertNotEqual(merged['ITERS'], 10000)
        self.assertNotEqual(merged['M'], 4096)

    def test_calibration_override_stays_bounded(self):
        config = thead_case_config('calibration')
        self.assertEqual(config['MODE'], 'calibration')
        validate_config({**thead_case_config(), **config}, CASE)

    def test_provider_registration_and_bounded_request_checks(self):
        provider = TheadProvider()
        assets = resolve_case_assets(BASE, CASE, 'thead')

        def request(nproc=None, timeout=300, privileged=False, case=CASE):
            return type('R', (), {'case': case, 'allow_privileged_root': privileged,
                                  'context': type('C', (), {'timeout': timeout, 'result_root': None})(),
                                  'result_dir': None, 'nproc_per_node': nproc})()
        provider.validate_benchmark(request(), assets)
        provider.validate_benchmark(request(nproc=1), assets)
        for kwargs in ({'nproc': 2}, {'timeout': 20}, {'privileged': True},
                       {'case': 'interconnect-MPI_intraserver:PPU'}, {'case': 'computation-FP8:PPU'}):
            with self.subTest(**kwargs), self.assertRaises(ConfigurationError):
                provider.validate_benchmark(request(**kwargs), assets)
        broken = {'requirements': assets['requirements'],
                  'merged_config': {**assets['merged_config'], 'ITERS': 20001}}
        with self.assertRaises(ConfigurationError):
            provider.validate_benchmark(request(), broken)

    def test_probe_and_provider_registries_match(self):
        registered = ('computation-FP32:PPU', 'computation-FP16:PPU', 'computation-BF16:PPU',
                      'computation-INT8:PPU', 'computation-FP64:PPU', 'computation-TF32:PPU',
                      'interconnect-h2d:PPU', 'interconnect-d2h:PPU')
        self.assertEqual(container_probe.PERFORMANCE_CASES, thead_provider.PERFORMANCE_CASES)
        self.assertEqual(container_probe.PERFORMANCE_CASES, registered)


class TheadComputationExecutionTests(unittest.TestCase):
    def test_verified_fp32_chain_accepts_the_ppu_case_selector(self):
        binding = binding_records([DEVICE], [PPU_UUID])[0]
        config = thead_case_config()
        config.update(M=32, N=32, K=32, ITERS=4, WARMUP=1)
        context = {'case': CASE, 'run_id': 'fixture', 'case_assets_sha256': 'case-hash'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'context.json').write_text(json.dumps({'case_assets_sha256': 'case-hash'}))
            driver = Mock()
            driver.device.return_value = torch.device('cpu')
            driver.evidence.return_value = {'binding': binding, 'bindings': [binding],
                                            'run_id': 'fixture', 'context_sha256': 'context'}
            calls = []
            driver.synchronize.side_effect = lambda: calls.append('sync')

            def tick():
                calls.append('timer')
                return calls.count('timer') * 10**9

            env = {'FLAGPERF_BENCHMARK_OUTPUT': str(root), 'FLAGPERF_HOST_CONTEXT': str(root / 'context.json'),
                   'FLAGPERF_BENCHMARK_CASE': CASE}
            with patch.dict(os.environ, env), patch.object(fp32.time, 'perf_counter_ns', side_effect=tick), \
                 patch.object(fp32, 'benchmark_measurement_start', side_effect=lambda: calls.append('start')), \
                 patch.object(fp32, 'benchmark_measurement_finish', side_effect=lambda token: calls.append('finish')):
                value = fp32.run_verified_fp32(driver, config, 0, 1, 0)
            self.assertEqual(calls[-6:], ['sync', 'start', 'timer', 'sync', 'timer', 'finish'])
            self.assertEqual(value, 2 * 32**3 * 4 / 1e12)
            metric = json.loads((root / 'metric-rank-0.json').read_text())
            correctness = json.loads((root / 'correctness-rank-0.json').read_text())
            validate_metric(metric, correctness, context, 'context', binding, config)
            self.assertEqual((metric['unit'], metric['metric'], metric['binding']), ('TFLOPS', 'computation-FP32', binding))
            self.assertEqual(correctness['status'], 'passed')

    def test_case_selector_guard_rejects_precision_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'context.json').write_text(json.dumps({'case_assets_sha256': 'case-hash'}))
            driver = Mock()
            driver.device.return_value = torch.device('cpu')
            driver.evidence.return_value = {'run_id': 'fixture', 'context_sha256': 'context'}
            env = {'FLAGPERF_BENCHMARK_OUTPUT': str(root), 'FLAGPERF_HOST_CONTEXT': str(root / 'context.json'),
                   'FLAGPERF_BENCHMARK_CASE': 'computation-FP16:PPU'}
            with patch.dict(os.environ, env):
                with self.assertRaises(RuntimeError):
                    fp32.run_verified_fp32(driver, thead_case_config(), 0, 1, 0)


class TheadMainRoutingTests(unittest.TestCase):
    def test_main_routes_thread_selector_to_verified_fp32(self):
        import importlib.util
        from argparse import Namespace
        spec = importlib.util.spec_from_file_location(
            'computation_fp32_main', BASE / 'benchmarks/computation-FP32/main.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from drivers import thead as thead_driver
        case_config = Namespace(**thead_case_config())
        routed = []

        def fake_run(driver, values, rank, world_size, local_rank):
            routed.append((driver is thead_driver, values['M'], rank, world_size, local_rank))
            return 12.5

        with patch.object(module, 'set_ieee_float32') as set_ieee, \
             patch('drivers.fp32.run_verified_fp32', side_effect=fake_run):
            result = module.main(Namespace(vendor='thead/PPU'), case_config, 0, 1, 0)
        self.assertEqual(result, 12.5)
        self.assertEqual(routed, [(True, case_config.M, 0, 1, 0)])
        set_ieee.assert_called_once_with('thead/PPU')



class TheadDay5ComputationAssetTests(unittest.TestCase):
    def test_new_precisions_share_contracts_across_chips(self):
        for precision in ('FP16', 'BF16', 'FP64', 'TF32'):
            with self.subTest(precision=precision):
                case = 'computation-' + precision + ':PPU'
                self.assertIn(case, CASES)
                self.assertEqual(contract(case), contract('computation-' + precision + ':P800'))

    def test_int8_contract_is_chip_specific(self):
        p800 = contract('computation-INT8:P800')
        ppu = contract('computation-INT8:PPU')
        self.assertEqual((ppu['operator'], ppu['output_dtype'], ppu['atol'], ppu['rtol']),
                         ('torch._int_mm', 'int32', 0.0, 0.0))
        self.assertEqual((p800['operator'], p800['output_dtype'], p800['atol']),
                         ('xtorch_ops.gemm_I8_I8_bf16_nt', 'bfloat16', 0.02))
        # torch._int_mm refuses M <= 16 on the device; the P800 xtorch kernel
        # keeps its historical small cases.
        self.assertEqual(ppu['shapes'], [[32, 32, 32], [32, 32, 48], [32, 16, 16], [32, 2048, 16], [32, 1024, 32]])
        self.assertEqual(p800['shapes'], [[32, 32, 32], [16, 32, 48], [16, 16, 16], [16, 2048, 16], [16, 1024, 32]])

    def test_day5_chip_assets_validate(self):
        expectations = {
            'FP16': {'DTYPE': 'float16', 'ATOL': 0.001, 'OPERATOR': 'torch.mm'},
            'BF16': {'DTYPE': 'bfloat16', 'ATOL': 0.008, 'OPERATOR': 'torch.mm'},
            'INT8': {'DTYPE': 'int8', 'ATOL': 0, 'OPERATOR': 'torch._int_mm'},
            'FP64': {'DTYPE': 'float64', 'ATOL': 1e-12, 'OPERATOR': 'torch.mm'},
            'TF32': {'DTYPE': 'float32', 'ATOL': 0.008, 'OPERATOR': 'torch.mm'},
        }
        for precision, expected in expectations.items():
            case = 'computation-' + precision + ':PPU'
            with self.subTest(precision=precision):
                assets = resolve_case_assets(BASE, case, 'thead')
                merged = assets['merged_config']
                for key, value in expected.items():
                    self.assertEqual(merged.get(key), value, key)
                self.assertEqual(merged['RTOL'], expected['ATOL'])
                self.assertEqual(merged['MODE'], 'qualification')
                self.assertEqual(merged['DIST_BACKEND'], 'gloo')
                validate_config(merged, case)
                self.assertEqual(assets['requirements']['process_scope'], {'nnodes': 1, 'nproc_per_node': 1})
                if precision == 'INT8':
                    self.assertTrue(assets['entrypoint']['relative_path'].endswith('computation-INT8/thead/PPU/main.py'))
                else:
                    self.assertTrue(assets['entrypoint']['relative_path'].endswith('computation-' + precision + '/main.py'))

    def test_day5_calibration_overrides_stay_bounded(self):
        for precision in ('FP16', 'BF16', 'INT8', 'FP64', 'TF32'):
            directory = BASE / ('benchmarks/computation-' + precision + '/thead/PPU')
            with self.subTest(precision=precision):
                override = read_config(directory / 'case_config.calibration.yaml')
                self.assertEqual(override['MODE'], 'calibration')
                self.assertLessEqual(override['ITERS'], 128)
                validate_config({**read_config(directory / 'case_config.yaml'), **override},
                                'computation-' + precision + ':PPU')

    def test_fp8_stays_an_evidenced_skip(self):
        requirements = json.loads((BASE / 'benchmarks/computation-FP8/thead/PPU/runtime_requirements.json').read_text())
        self.assertIs(requirements['supported'], False)
        self.assertTrue(requirements['unsupported_reason'])
        assets = resolve_case_assets(BASE, 'computation-FP8:PPU', 'thead')
        self.assertIs(assets['requirements']['supported'], False)


if __name__ == '__main__':
    unittest.main()
