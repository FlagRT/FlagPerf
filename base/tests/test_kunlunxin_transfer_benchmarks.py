import math
from copy import deepcopy
from pathlib import Path
import sys
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from benchmarks.transfer_contract import bandwidth, validate_config, validate_metric
from executors.bounded_benchmark import validate_stdout
from base.vendors.protocol import ConfigurationError


class TransferContractTests(unittest.TestCase):
    def test_invalid_elapsed_rejected(self):
        for elapsed in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                bandwidth(4096, 10, elapsed)

    def test_all_eight_modes_and_strict_boolean_contract(self):
        for direction in ('h2d', 'd2h'):
            for pinned in (False, True):
                for nonblocking in (False, True):
                    validate_config({**self.config(direction), 'PIN_MEMORY': pinned, 'NON_BLOCKING': nonblocking},
                                    'interconnect-' + direction + ':P800')
        for key in ('PIN_MEMORY', 'NON_BLOCKING', 'REUSE_DESTINATION'):
            with self.assertRaises(ConfigurationError):
                validate_config({**self.config(), key: 1}, 'interconnect-h2d:P800')

    def test_stdout_requires_exact_consistent_unit_pair(self):
        raw = {'value': 1.073741824, 'value_gib_s': 1.0}
        gb = "[FlagPerf Result]Rank 0's transfer-bandwidth=1.073741824GB/s\n"
        gib = "[FlagPerf Result]Rank 0's transfer-bandwidth=1.0GiB/s\n"
        validate_stdout(gb + gib, raw, 'interconnect-h2d:P800')
        for text in (gb, gib, gb + gb, gb + gib + gib, gb + gib.replace('1.0GiB', '2.0GiB'),
                     gb + gib.replace('Rank 0', 'Rank 1')):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                validate_stdout(text, raw, 'interconnect-h2d:P800')

    def evidence(self):
        config = self.config()
        binding = {'framework_device_name': 'cuda:0'}
        context = {'case': 'interconnect-h2d:P800', 'run_id': 'fixture', 'case_assets_sha256': 'assets'}
        common = dict(schema_version=1, status='passed', run_id='fixture', context_sha256='context', binding=binding,
                      rank=0, direction='h2d', host_memory='pageable', non_blocking=False, dtype='float32',
                      api='Tensor.copy_', payload_bytes=config['PAYLOAD_BYTES'])
        gb, gib = bandwidth(config['PAYLOAD_BYTES'], config['ITERS'], 2.0)
        metric = dict(common, world_size=1, metric='transfer-bandwidth', unit='GB/s', case_assets_sha256='assets',
                      iterations=config['ITERS'], warmup=config['WARMUP'], mode='smoke', implementation='native-xpytorch',
                      started_monotonic_ns=1, finished_monotonic_ns=2000000001, elapsed_seconds=2.0,
                      total_bytes=config['PAYLOAD_BYTES'] * config['ITERS'], value=gb, value_gib_s=gib)
        placement = dict(dtype='float32', contiguous=True, numel=config['PAYLOAD_BYTES']//4)
        correctness = dict(common, destination_reused=True, source_reused=True, host_is_pinned=False,
                           source=dict(placement, device='cpu'), destination=dict(placement, device='cuda:0'),
                           memory_budget=dict(host_bytes=1024, device_bytes=1024, host_available=102400, device_available=102400),
                           checks=[dict(phase=phase, passed=True, mismatch_count=0, elements=config['PAYLOAD_BYTES']//4)
                                   for phase in ('sentinel', 'changed-input', 'before', 'after')])
        return metric, correctness, context, 'context', binding, config

    def test_metric_identity_semantics_content_and_formula(self):
        evidence = self.evidence()
        validate_metric(*evidence)
        for index, key, value in ((0, 'value_gib_s', 999), (0, 'iterations', 5), (0, 'direction', 'd2h'),
                                  (0, 'context_sha256', 'stale'), (0, 'total_bytes', 1), (0, 'elapsed_seconds', 0),
                                  (1, 'destination_reused', False), (1, 'host_is_pinned', True), (1, 'checks', []),
                                  (1, 'memory_budget', {}), (1, 'destination', {})):
            corrupted = deepcopy(evidence)
            corrupted[index][key] = value
            with self.subTest(key=key), self.assertRaises((RuntimeError, ValueError)):
                validate_metric(*corrupted)

    def test_copy_loop_only_reuses_destination(self):
        from benchmarks.drivers.transfer import copy_loop
        from unittest.mock import Mock
        destination, source = Mock(), object()
        copy_loop(destination, source, 3, True)
        self.assertEqual(destination.copy_.call_count, 3)
        self.assertTrue(all(call.args == (source,) and call.kwargs == {'non_blocking': True}
                            for call in destination.copy_.call_args_list))

    def config(self, direction='h2d'):
        return {'PAYLOAD_BYTES': 4 * 1024 * 1024, 'WARMUP': 10, 'ITERS': 1000, 'SEED': 519,
                'DIRECTION': direction, 'PIN_MEMORY': False, 'NON_BLOCKING': False,
                'REUSE_DESTINATION': True, 'DTYPE': 'float32', 'DIST_BACKEND': 'gloo',
                'IMPLEMENTATION': 'native-xpytorch', 'MODE': 'smoke', 'FAULT_MODE': 'none'}

    def test_units_are_derived_from_one_way_bytes(self):
        gb, gib = bandwidth(4 * 1024 * 1024, 100, 2.0)
        self.assertAlmostEqual(gb, 0.2097152)
        self.assertAlmostEqual(gib, 0.1953125)
        self.assertAlmostEqual(gb / gib, 2**30 / 10**9)

    def test_semantics_are_explicit_and_bounded(self):
        validate_config(self.config(), 'interconnect-h2d:P800')
        with self.assertRaises(ConfigurationError):
            validate_config({**self.config(), 'REUSE_DESTINATION': False}, 'interconnect-h2d:P800')
        with self.assertRaises(ConfigurationError):
            validate_config({**self.config(), 'DIRECTION': 'd2h'}, 'interconnect-h2d:P800')
        with self.assertRaises(ConfigurationError):
            validate_config({**self.config(), 'PAYLOAD_BYTES': 4 * 2**30 + 4096}, 'interconnect-h2d:P800')


if __name__ == '__main__':
    unittest.main()
