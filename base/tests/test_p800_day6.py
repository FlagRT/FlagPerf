"""CPU-only Day 6 contract tests: formulas, granularity and fail-closed gates."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from benchmarks import day6_contract as contract
from base.vendors.protocol import ConfigurationError
from executors.benchmark import HIGH_RISK_CASES

CONTEXT = {'run_id': 'run-day6-test', 'case_assets_sha256': 'a' * 64}
CONTEXT_HASH = hashlib.sha256(json.dumps(CONTEXT, sort_keys=True).encode()).hexdigest()
BINDING = 'binding-day6-test'

VALID = {
    'main_memory-bandwidth:P800': {'Melements': 2, 'WARMUP': 2, 'ITERS': 50, 'SEED': 519,
                                   'DTYPE': 'float32', 'DIST_BACKEND': 'gloo',
                                   'IMPLEMENTATION': 'native-xpytorch', 'MODE': 'qualification'},
    'main_memory-capacity:P800': {'INITSIZE': 64, 'MIN_MIB': 1, 'SEED': 519, 'DTYPE': 'float32',
                                  'DIST_BACKEND': 'gloo', 'IMPLEMENTATION': 'native-xpytorch',
                                  'MODE': 'capacity', 'BOUND_REQUEST_BY_FREE_MEMORY': True,
                                  'POST_TEST_WAIT_SECONDS': 0},
    'interconnect-MPI_intraserver:P800': {'Melements': 4, 'WARMUP': 1, 'ITERS': 100, 'SEED': 519,
                                          'DTYPE': 'float32', 'DIST_BACKEND': 'cpu:gloo,cuda:flagcx',
                                          'IMPLEMENTATION': 'native-xpytorch', 'MODE': 'qualification'},
    'interconnect-P2P_intraserver:P800': {'Melements': 4, 'WARMUP': 1, 'ITERS': 100, 'SEED': 519,
                                          'DTYPE': 'float32', 'DIST_BACKEND': 'cpu:gloo,cuda:flagcx',
                                          'IMPLEMENTATION': 'native-xpytorch', 'MODE': 'qualification'},
}

ELAPSED = 1000.0


def metric_record(case, rank=0):
    config = VALID[case]
    world_size = contract.expected_ranks(case)
    started = 10 ** 12
    record = {'schema_version': 1, 'status': 'passed', 'run_id': CONTEXT['run_id'],
              'context_sha256': CONTEXT_HASH, 'rank': rank, 'binding': BINDING,
              'world_size': world_size, 'case_assets_sha256': CONTEXT['case_assets_sha256'],
              'mode': config['MODE'], 'implementation': config['IMPLEMENTATION'],
              'metric': contract.METRIC_NAMES[case],
              'started_monotonic_ns': started, 'finished_monotonic_ns': started + int(ELAPSED * 10 ** 9),
              'elapsed_seconds': ELAPSED}
    for field, key in (('ITERS', 'iterations'), ('WARMUP', 'warmup')):
        if field in config:
            record[key] = config[field]
    payload = contract.payload_bytes(config['Melements']) if 'Melements' in config else 0
    if case == 'main_memory-bandwidth:P800':
        gb, gib = contract.memory_bandwidth(payload, config['ITERS'], ELAPSED)
        record.update(unit='GB/s', value=gb, value_gib_s=gib, total_bytes=2 * payload * config['ITERS'])
    elif case == 'main_memory-capacity:P800':
        held = 4096
        record.update(unit='GB', value=held * (1 << 20) / 1e9, value_gib_s=held / 1024, held_mib=held)
    elif case == 'interconnect-MPI_intraserver:P800':
        algbw, algbw_gib, busbw, busbw_gib = contract.allreduce_bandwidth(payload, ELAPSED, world_size)
        record.update(unit='GB/s', value=algbw, value_gib_s=algbw_gib,
                      busbw_gb_s=busbw, busbw_gib_s=busbw_gib, message_bytes=payload)
    else:
        gb, gib = contract.p2p_one_way_bandwidth(payload, config['ITERS'], ELAPSED)
        record.update(unit='GB/s', value=gb, value_gib_s=gib,
                      total_bytes=payload * config['ITERS'], direction='rank0-to-rank1')
    return record


def correctness_record(case, rank=0):
    record = {'schema_version': 1, 'status': 'passed', 'run_id': CONTEXT['run_id'],
              'context_sha256': CONTEXT_HASH, 'rank': rank, 'binding': BINDING}
    if case == 'main_memory-capacity:P800':
        record.update(released=True, release_verified=True)
    return record


class Day6Formulas(unittest.TestCase):
    def test_payload_bytes_counts_mebielements(self):
        self.assertEqual(contract.payload_bytes(1024), 4 << 30)   # 4 GiB of float32
        self.assertEqual(contract.payload_bytes(1), 4 << 20)

    def test_memory_bandwidth_counts_read_and_write(self):
        gb, gib = contract.memory_bandwidth(4 << 30, 100, 2.0)
        self.assertAlmostEqual(gb, 2 * (4 << 30) * 100 / 2.0 / 1e9)
        self.assertAlmostEqual(gib, 400.0)  # 400 GiB/s, not 800

    def test_allreduce_busbw_factor_never_doubles(self):
        payload = 64 << 20
        for world_size, factor in ((2, 1.0), (4, 1.5), (8, 1.75)):
            algbw, _gib, busbw, _bgib = contract.allreduce_bandwidth(payload, 1.0, world_size)
            self.assertAlmostEqual(busbw / algbw, factor)
        algbw, _g, busbw, _b = contract.allreduce_bandwidth(payload, 1.0, 2)
        self.assertAlmostEqual(busbw, algbw)  # ws=2: busbw == algbw, no extra x2

    def test_p2p_is_one_way_only(self):
        gb, gib = contract.p2p_one_way_bandwidth(1 << 20, 10, 0.01)
        self.assertAlmostEqual(gb, (1 << 20) * 10 / 0.01 / 1e9)
        self.assertAlmostEqual(gib, (1 << 20) * 10 / 0.01 / 2 ** 30)

    def test_expected_ranks_split_memory_and_communication(self):
        for case in contract.MEMORY_CASES:
            self.assertEqual(contract.expected_ranks(case), 1)
        for case in contract.COMMUNICATION_CASES:
            self.assertEqual(contract.expected_ranks(case), 2)


class Day6ConfigValidation(unittest.TestCase):
    def test_valid_configs_accepted(self):
        for case, config in VALID.items():
            contract.validate_config(dict(config), case)

    def test_rejects_wrong_dtype_backend_and_implementation(self):
        bad = dict(VALID['main_memory-bandwidth:P800'], DTYPE='float16')
        with self.assertRaises(ConfigurationError):
            contract.validate_config(bad, 'main_memory-bandwidth:P800')
        bad = dict(VALID['interconnect-P2P_intraserver:P800'], DIST_BACKEND='gloo')
        with self.assertRaises(ConfigurationError):
            contract.validate_config(bad, 'interconnect-P2P_intraserver:P800')
        bad = dict(VALID['main_memory-capacity:P800'], IMPLEMENTATION='cuda')
        with self.assertRaises(ConfigurationError):
            contract.validate_config(bad, 'main_memory-capacity:P800')

    def test_rejects_out_of_range_fields(self):
        bad = dict(VALID['main_memory-bandwidth:P800'], Melements=0)
        with self.assertRaises(ConfigurationError):
            contract.validate_config(bad, 'main_memory-bandwidth:P800')
        bad = dict(VALID['interconnect-MPI_intraserver:P800'], ITERS=200001)
        with self.assertRaises(ConfigurationError):
            contract.validate_config(bad, 'interconnect-MPI_intraserver:P800')

    def test_capacity_must_bound_requests_and_skip_wait(self):
        bad = dict(VALID['main_memory-capacity:P800'], BOUND_REQUEST_BY_FREE_MEMORY=False)
        with self.assertRaises(ConfigurationError):
            contract.validate_config(bad, 'main_memory-capacity:P800')
        bad = dict(VALID['main_memory-capacity:P800'], POST_TEST_WAIT_SECONDS=5)
        with self.assertRaises(ConfigurationError):
            contract.validate_config(bad, 'main_memory-capacity:P800')

    def test_capacity_is_the_only_high_risk_case(self):
        self.assertEqual(HIGH_RISK_CASES, {'main_memory-capacity'})


class Day6Interserver(unittest.TestCase):
    def test_interserver_cases_marked_unsupported_before_docker(self):
        for case in contract.INTERSERVER_CASES:
            self.assertNotIn(case, contract.CASES)
            path = BASE / 'benchmarks' / case.split(':')[0] / 'kunlunxin/P800/runtime_requirements.json'
            record = json.loads(path.read_text())
            self.assertIs(record['supported'], False)
            self.assertTrue(record['unsupported_reason'].strip())
            self.assertEqual(record['process_scope']['nnodes'], 2)


class Day6ArtifactValidation(unittest.TestCase):
    def validate(self, case):
        return contract.validate_metric(metric_record(case, 0), correctness_record(case, 0),
                                        CONTEXT, CONTEXT_HASH, BINDING, VALID[case], case)

    def test_each_case_validates_with_recomputed_numbers(self):
        for case in contract.CASES:
            metric = self.validate(case)
            self.assertEqual(metric['metric'], contract.METRIC_NAMES[case])

    def test_two_rank_cases_validate_both_ranks(self):
        for case in contract.COMMUNICATION_CASES:
            for rank in (0, 1):
                contract.validate_metric(metric_record(case, rank), correctness_record(case, rank),
                                         CONTEXT, CONTEXT_HASH, BINDING, VALID[case], case)

    def test_tampered_values_are_rejected(self):
        with self.assertRaises(RuntimeError):
            metric = metric_record('main_memory-bandwidth:P800')
            metric['value'] *= 1.01
            contract.validate_metric(metric, correctness_record('main_memory-bandwidth:P800'),
                                     CONTEXT, CONTEXT_HASH, BINDING, VALID['main_memory-bandwidth:P800'],
                                     'main_memory-bandwidth:P800')
        with self.assertRaises(RuntimeError):
            metric = metric_record('interconnect-MPI_intraserver:P800')
            metric['busbw_gb_s'] = metric['value'] * 2  # the forbidden extra x2
            contract.validate_metric(metric, correctness_record('interconnect-MPI_intraserver:P800'),
                                     CONTEXT, CONTEXT_HASH, BINDING,
                                     VALID['interconnect-MPI_intraserver:P800'],
                                     'interconnect-MPI_intraserver:P800')
        with self.assertRaises(RuntimeError):
            metric = metric_record('interconnect-P2P_intraserver:P800')
            metric['total_bytes'] *= 2
            contract.validate_metric(metric, correctness_record('interconnect-P2P_intraserver:P800'),
                                     CONTEXT, CONTEXT_HASH, BINDING,
                                     VALID['interconnect-P2P_intraserver:P800'],
                                     'interconnect-P2P_intraserver:P800')

    def test_capacity_requires_verified_release_and_granularity(self):
        case = 'main_memory-capacity:P800'
        correctness = correctness_record(case)
        correctness['released'] = False
        with self.assertRaises(RuntimeError):
            contract.validate_metric(metric_record(case), correctness,
                                     CONTEXT, CONTEXT_HASH, BINDING, VALID[case], case)
        metric = metric_record(case)
        metric['held_mib'] = 0
        with self.assertRaises(RuntimeError):
            contract.validate_metric(metric, correctness_record(case),
                                     CONTEXT, CONTEXT_HASH, BINDING, VALID[case], case)

    def test_rank_outside_world_size_rejected(self):
        metric = metric_record('interconnect-MPI_intraserver:P800', rank=2)
        with self.assertRaises(RuntimeError):
            contract.validate_metric(metric, correctness_record('interconnect-MPI_intraserver:P800', 2),
                                     CONTEXT, CONTEXT_HASH, BINDING,
                                     VALID['interconnect-MPI_intraserver:P800'],
                                     'interconnect-MPI_intraserver:P800')

    def test_stdout_pairs_cover_every_published_number(self):
        for case in contract.CASES:
            pairs = contract.expected_stdout_pairs(case, metric_record(case))
            self.assertTrue(pairs)
            for name, unit, value in pairs:
                self.assertIsInstance(value, float)
                self.assertTrue(unit in ('GB/s', 'GiB/s', 'GB', 'GiB'))


class Day6ReportRegeneration(unittest.TestCase):
    def test_report_regenerates_from_archived_smoke_artifacts(self):
        matches = list((BASE / 'vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day6')
                       .glob('**/memory-smoke-a01-card5/summary.json'))
        if not matches:
            self.skipTest('archived smoke run not present')
        source = matches[0].parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            shutil.copytree(source, root)
            from generate_benchmark_report import generate_benchmark_report
            metadata = generate_benchmark_report(root)
            self.assertEqual(metadata['status'], 'passed')
            regenerated = root / metadata['path']
            import hashlib as _hashlib
            self.assertEqual(_hashlib.sha256(regenerated.read_bytes()).hexdigest(),
                             metadata['sha256'])
            text = regenerated.read_text()
            self.assertIn('device-memory-bandwidth', text)


if __name__ == '__main__':
    unittest.main()
