import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
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
from benchmarks.computation_contract import contract, validate_config
from drivers import correctness, fp32
from executors.benchmark import BenchmarkRunRequest, BenchmarkExecutor, add_cli_arguments
from executors.bounded_benchmark import validate_metric
from base.vendors.protocol import ConfigurationError
from test_kunlunxin_fp32 import rank_records, DEVICE, UID
from base.vendors.kunlunxin.provider import binding_records


def config_for(precision, mode='smoke'):
    directory = BASE / ('benchmarks/computation-' + precision + '/kunlunxin/P800')
    config = read_config(directory / 'case_config.yaml')
    if mode != 'qualification':
        config.update(read_config(directory / ('case_config.' + mode + '.yaml')))
    return config


class ComputationContractTests(unittest.TestCase):
    def test_precision_contract_rejects_dtype_operator_and_tolerance_drift(self):
        for precision in ('FP16', 'BF16', 'INT8'):
            config = config_for(precision)
            case = 'computation-' + precision + ':P800'
            validate_config(config, case)
            for change in ({'DTYPE': 'float32'}, {'OPERATOR': 'cast-mm'}, {'ATOL': 1.0},
                           {'RTOL': float('nan')}, {'ITERS': 200001}, {'M': 8192},
                           {'FAULT_MODE': 'cpu-wait'}, {'DIST_BACKEND': 'nccl'}):
                with self.subTest(precision=precision, change=change), self.assertRaises(ConfigurationError):
                    validate_config({**config, **change}, case)
        with self.assertRaises(ConfigurationError):
            contract('computation-UNKNOWN:P800')

    def test_fp32_bounds_are_not_relaxed(self):
        with self.assertRaises(ConfigurationError):
            validate_config({**config_for('FP32'), 'ITERS': 20001}, 'computation-FP32:P800')

    def test_new_cases_dry_run_without_device_side_effects(self):
        for precision in ('FP16', 'BF16', 'INT8'):
            parser = argparse.ArgumentParser()
            add_cli_arguments(parser)
            args = parser.parse_args(['--config', str(BASE / 'configs/kunlunxin_p800_xpytorch29.yaml'),
                                      '--case', 'computation-' + precision + ':P800', '--physical-device-ids', '5',
                                      '--timeout', '300', '--dry-run'])
            with patch('subprocess.run', side_effect=AssertionError('side effect')), redirect_stdout(io.StringIO()):
                self.assertEqual(BenchmarkExecutor().execute(BenchmarkRunRequest.from_namespace(args)), 0)

    def test_low_precision_metric_rejects_cross_case_and_incomplete_numerics(self):
        for precision in ('FP16', 'BF16'):
            config = config_for(precision)
            spec = contract('computation-' + precision + ':P800')
            context = {'case': 'computation-' + precision + ':P800', 'run_id': 'fixture', 'case_assets_sha256': 'case'}
            binding = binding_records([DEVICE], [UID])[0]
            metric, record = rank_records(context, 'hash', binding, config)
            fields = {'dtype': spec['dtype'], 'operator': 'torch.mm',
                      'reference_inputs': 'CPU inputs quantized to target dtype before FP64 reference'}
            metric.update(metric=spec['metric'], output_dtype=spec['output_dtype'], **fields)
            record.update(**fields)
            numerical = record['small_cases'][0]
            record['small_cases'] = [{**numerical, 'shape': shape} for shape in
                                    [[32, 32, 32], [17, 29, 11], [16, 16, 16], [4, 2048, 4], [2, 1024, 2]]]
            validate_metric(metric, record, context, 'hash', binding, config)
            for change in ({'metric': 'computation-FP32'}, {'dtype': 'float32'}, {'unit': 'TOPS'},
                           {'operator': 'cast-mm'}, {'reference_inputs': 'unquantized CPU inputs'}):
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    validate_metric({**metric, **change}, record, context, 'hash', binding, config)
            broken = deepcopy(record)
            broken['small_cases'][-1]['shape'] = [32, 32, 32]
            with self.assertRaises(RuntimeError):
                validate_metric(metric, broken, context, 'hash', binding, config)


class ComputationNumericalTests(unittest.TestCase):
    def test_reference_uses_quantized_inputs_and_budget_detects_corruption(self):
        left, right = correctness.input_pair(17, 29, 11, 519)
        for precision in ('FP16', 'BF16'):
            spec = contract('computation-' + precision + ':P800')
            dtype = getattr(torch, spec['dtype'])
            quantized_left, quantized_right = left.to(dtype), right.to(dtype)
            reference = quantized_left.double() @ quantized_right.double()
            self.assertFalse(torch.equal(reference, left.double() @ right.double()))
            rounded = reference.to(dtype)
            self.assertTrue(correctness.compare(rounded, reference, atol=spec['atol'], rtol=spec['rtol'])['passed'])
            for corruption in (0.25, float('inf'), float('nan')):
                changed = rounded.clone()
                changed[0, 0] += corruption
                self.assertFalse(correctness.compare(changed, reference, atol=spec['atol'], rtol=spec['rtol'])['passed'])

    def test_new_cases_execute_full_correctness_with_synchronized_timer(self):
        for precision in ('FP16', 'BF16'):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = config_for(precision)
                (root / 'context.json').write_text(json.dumps({'case_assets_sha256': 'case-hash'}))
                driver = Mock()
                driver.device.return_value = torch.device('cpu')
                driver.evidence.return_value = {'run_id': 'fixture', 'context_sha256': 'context'}
                calls = []
                driver.synchronize.side_effect = lambda: calls.append('sync')
                def tick():
                    calls.append('timer')
                    return calls.count('timer') * 10**9
                env = {'FLAGPERF_BENCHMARK_OUTPUT': str(root), 'FLAGPERF_HOST_CONTEXT': str(root / 'context.json')}
                with patch.dict(os.environ, env), patch.object(fp32.time, 'perf_counter_ns', side_effect=tick), \
                     patch.object(fp32, 'benchmark_measurement_start', side_effect=lambda: calls.append('start')), \
                     patch.object(fp32, 'benchmark_measurement_finish', side_effect=lambda token: calls.append('finish')):
                    value = fp32.run_verified_computation(driver, config, 0, 1, 0, precision)
                self.assertEqual(calls[-6:], ['sync', 'start', 'timer', 'sync', 'timer', 'finish'])
                self.assertEqual(value, 2 * 32**3 * 4 / 1e12)
                record = json.loads((root / 'correctness-rank-0.json').read_text())
                self.assertEqual(record['status'], 'passed')
                self.assertEqual(len(record['small_cases']), 5)
                self.assertEqual(record['dtype'], contract('computation-' + precision + ':P800')['dtype'])

    def test_low_precision_placement_rejects_cpu_and_promoted_output(self):
        with self.assertRaises(RuntimeError):
            fp32.require_placement([torch.ones(2, dtype=torch.float16)], torch.device('cuda:0'), torch.float16)
        with self.assertRaises(RuntimeError):
            fp32.require_placement([torch.ones(2)], torch.device('cpu'), torch.float16)

    def test_int8_exact_reference_detects_overflow_and_single_bit_error(self):
        left = torch.full((16, 1024), -128, dtype=torch.int8)
        right = torch.full((1024, 16), 127, dtype=torch.int8)
        reference = left.long() @ right.long()
        actual = torch._int_mm(left, right)
        self.assertTrue(correctness.compare(actual, reference, atol=0, rtol=0)['passed'])
        self.assertFalse(correctness.compare(actual.to(torch.int8), reference, atol=0, rtol=0)['passed'])
        actual[0, 0] += 1
        self.assertFalse(correctness.compare(actual, reference, atol=0, rtol=0)['passed'])

    def test_int8_full_case_outputs_tops_and_int32(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'context.json').write_text(json.dumps({'case_assets_sha256': 'case-hash'}))
            driver = Mock()
            driver.device.return_value = torch.device('cpu')
            driver.evidence.return_value = {'run_id': 'fixture', 'context_sha256': 'context'}
            env = {'FLAGPERF_BENCHMARK_OUTPUT': str(root), 'FLAGPERF_HOST_CONTEXT': str(root / 'context.json')}
            with patch.dict(os.environ, env), patch.object(fp32, 'benchmark_measurement_start'), \
                 patch.object(fp32, 'benchmark_measurement_finish'):
                fp32.run_verified_computation(driver, config_for('INT8'), 0, 1, 0, 'INT8')
            metric = json.loads((root / 'metric-rank-0.json').read_text())
            self.assertEqual((metric['unit'], metric['dtype'], metric['output_dtype']), ('TOPS', 'int8', 'int32'))
            self.assertEqual(json.loads((root / 'correctness-rank-0.json').read_text())['reference_dtype'], 'int64')


if __name__ == '__main__':
    unittest.main()
