"""CPU-only thead vendor, monitor and driver-routing contracts (pre-registration)."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from base.vendors.protocol import ConfigurationError
from base.vendors.thead import provider as thead_provider
from base.vendors.thead.provider import TheadProvider, binding_records
from base.vendors.registry import PROVIDERS, get_provider
from monitoring.thead_usage import UsageMonitor, parse_card, parse_q_sections

FIXTURE = BASE / 'tests/fixtures/ppu-q-16cards.txt'
U0 = '011ef111-4522-040e-0000-0000c08f6579'
U1 = '011ef111-47c0-0010-0000-0000a05fba7d'


class FakeCommands:
    def __init__(self, stdout):
        self.stdout = stdout
        self.calls = []
        self.origin = time.monotonic()

    def run(self, arguments, timeout=None, **kwargs):
        self.calls.append(list(arguments))
        return {'arguments': list(arguments), 'stdout': self.stdout, 'stderr': '', 'returncode': 0,
                'started_monotonic_s': self.origin, 'finished_monotonic_s': self.origin + 1.0}


def device_record(slot, uuid_text, bdf):
    return {'host_physical_id': slot, 'pci_bdf': thead_provider.normalize_bdf(bdf), 'uuid': uuid_text,
            'host_device_node': '/dev/alixpu_ppu%d' % slot, 'container_node': '/dev/alixpu_ppu%d' % slot,
            'device_minor': slot + 1}


def targets_pair():
    return [{'device_id': 'thead/' + U0, 'physical_id': 0, 'pci_bdf': '00000001:c9:00.0', 'uuid': U0},
            {'device_id': 'thead/' + U1, 'physical_id': 1, 'pci_bdf': '00000001:c8:00.0', 'uuid': U1}]


class TheadMonitorTests(unittest.TestCase):
    def test_full_query_fixture_yields_sixteen_consistent_cards(self):
        raw = FIXTURE.read_text()
        sections = parse_q_sections(raw)
        self.assertEqual(len(sections), 16)
        minors = []
        for bdf, section in sections.items():
            card = parse_card(section, bdf)
            minors.append(card['device_minor'])
            self.assertEqual(card['host_device_node'], '/dev/alixpu_ppu%d' % (card['device_minor'] - 1))
            self.assertEqual(card['total_memory_mib'], 98304)
            self.assertLessEqual(card['used_memory_mib'], card['total_memory_mib'])
        self.assertEqual(sorted(minors), list(range(1, 17)))
        first = parse_card(sections['00000001:c9:00.0'], '00000001:c9:00.0')
        self.assertEqual(first['uuid'], U0)
        self.assertEqual(first['health_status'], 'observed-clean')
        self.assertFalse(first['processes_empty'])
        self.assertEqual(first['temperature_c'], 33.0)

    def test_collect_once_parses_one_query_per_round(self):
        raw = FIXTURE.read_text()
        commands = FakeCommands(raw)
        monitor = UsageMonitor(targets_pair(), commands=commands)
        monitor.collect_once()
        self.assertEqual(commands.calls, [['ppu-smi', '-q']])
        self.assertEqual([s['valid'] for s in monitor.samples], [True, True])
        self.assertEqual(monitor.samples[0]['values']['used_memory_mib'], 191)
        self.assertEqual(monitor.samples[0]['values']['utilization_percent'], 0)
        self.assertEqual(len(monitor.raw_records), 1)

    def test_stale_target_identity_invalidates_only_that_target(self):
        raw = FIXTURE.read_text()
        stale = [dict(targets_pair()[0], uuid='0' * 36), targets_pair()[1]]
        monitor = UsageMonitor(stale, commands=FakeCommands(raw))
        monitor.collect_once()
        self.assertFalse(monitor.samples[0]['valid'])
        self.assertIn('identity', monitor.samples[0]['error'])
        self.assertTrue(monitor.samples[1]['valid'])

    def test_finish_records_schema_two_summary(self):
        raw = FIXTURE.read_text()
        monitor = UsageMonitor(targets_pair(), commands=FakeCommands(raw))
        windows = [{'role': 'measurement', 'device_id': t['device_id'],
                    'started_offset_s': 0.0, 'finished_offset_s': 2.0} for t in targets_pair()]
        for _ in range(10):
            monitor.collect_once()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = monitor.finish(root, root / 'monitor', windows, [])
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['vendor'], 'thead')
            self.assertEqual(result['collector'], 'ppu-smi single -q full parse')
            self.assertEqual(result['schema_version'], 2)
            self.assertEqual(result['primary_sample_counts_by_target'],
                             {t['device_id']: 10 for t in targets_pair()})
            self.assertEqual(result['total_sample_commands'], 10)
            self.assertTrue((root / 'monitor/samples.jsonl').exists())
            self.assertTrue((root / 'monitor/summary.json').exists())

    def test_neutral_report_renderer_renders_thread_telemetry_without_vendor_branch(self):
        from benchmark_report_schema import monitor_svg, render_monitor_report
        raw = FIXTURE.read_text()
        monitor = UsageMonitor(targets_pair(), commands=FakeCommands(raw))
        for _ in range(3):
            monitor.collect_once()
        windows = [{'role': 'measurement', 'device_id': t['device_id'],
                    'started_offset_s': 0.0, 'finished_offset_s': 2.0} for t in targets_pair()]
        policy = TheadProvider().monitor_policy(True)
        summary = {'schema_version': 2, 'vendor': 'thead', 'status': 'passed',
                   'collector': 'ppu-smi single -q full parse',
                   'primary_sample_counts_by_target': {t['device_id']: 3 for t in targets_pair()}}
        rendered = render_monitor_report(Path('.'), summary,
                                         {'policy': policy, 'workload_windows': windows,
                                          'vendor': 'thead', 'status': 'passed'},
                                         monitor.samples, None)
        for expected in ('thead/' + U0, 'thead/' + U1, 'Device utilization', 'Allocated device memory'):
            self.assertIn(expected, rendered)
        self.assertNotIn('npu', rendered)
        svg = monitor_svg(monitor.samples, {'policy': policy, 'workload_windows': windows})
        self.assertIsNotNone(svg)
        self.assertIn('thead/' + U0, svg)


class TheadBindingTests(unittest.TestCase):
    def test_binding_records_join_uuids_without_guessing_order(self):
        devices = [device_record(0, U0, '00000001:C9:00.0'), device_record(1, U1, '00000001:C8:00.0')]
        records = binding_records(devices, [U1, U0])
        self.assertEqual([r['framework_local_rank'] for r in records], [0, 1])
        self.assertEqual([r['framework_logical_id'] for r in records], [1, 0])
        self.assertEqual([r['framework_device_name'] for r in records], ['cuda:1', 'cuda:0'])
        self.assertEqual([r['resource_key'] for r in records], ['thead/' + U0, 'thead/' + U1])
        self.assertEqual([r['vendor'] for r in records], ['thead', 'thead'])
        self.assertEqual(records[0]['container_device_node'], '/dev/alixpu_ppu0')

    def test_binding_records_reject_ambiguous_or_drifting_evidence(self):
        devices = [device_record(0, U0, '00000001:C9:00.0'), device_record(1, U1, '00000001:C8:00.0')]
        for observed in ([U0, U0], [U0], [U0, 'ffffffff-0000-0000-0000-000000000000'], []):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                binding_records(devices, observed)
        drifted = [device_record(0, U0, '00000001:C9:00.0'), dict(device_record(1, U1, '00000001:C8:00.0'), device_minor=7)]
        with self.assertRaises(ValueError):
            binding_records(drifted, [U0, U1])


class TheadProviderTests(unittest.TestCase):
    def load_config(self):
        return json.loads((BASE / 'configs/thead_ppu_torch210.yaml').read_text())

    def test_config_contract_accepts_draft_and_rejects_drift(self):
        config = self.load_config()
        TheadProvider().validate_config(config)
        mutations = [{'chip': 'P800'}, {'shm_size': '128m'}, {'required_devices': ['/dev/alixpu_ctl']},
                     {'runtime_environment': {'USE_FLAGGEMS': '0'}}, {'host_mounts': ['/data']},
                     {'requires_privileged_root': True}]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ConfigurationError):
                TheadProvider().validate_config({**config, **mutation})

    def test_monitor_contract_and_targets(self):
        provider = TheadProvider()
        policy = provider.monitor_policy(True)
        self.assertEqual(policy['collector'], 'ppu-smi single -q full parse')
        self.assertEqual(policy['target_resource'], 'ppu-device')
        self.assertEqual(policy['required_samples_per_target'], 10)
        host = {'devices': [device_record(0, U0, '00000001:C9:00.0'), device_record(1, U1, '00000001:C8:00.0')]}
        targets = provider.monitor_targets(host)
        self.assertEqual([t['device_id'] for t in targets], ['thead/' + U0, 'thead/' + U1])
        monitor = provider.create_monitor(targets)
        self.assertIsInstance(monitor, UsageMonitor)
        container_policy = provider.container_policy(self.load_config())
        self.assertEqual(container_policy['cap_drop'], ['ALL'])
        self.assertFalse(container_policy['privileged_root'])

    def test_day3_selection_contract_requires_physical_ids(self):
        provider = TheadProvider()
        recorded = []

        class FakeContext:
            physical_device_ids = [1]

            @classmethod
            def validate(cls, require_selection=False):
                recorded.append(require_selection)

        provider.validate_selection(FakeContext)
        self.assertEqual(recorded, [True])
        legacy = type('LegacyContext', (), {'physical_device_ids': None, 'validate': lambda self, require_selection=False: None})()
        with self.assertRaises(ConfigurationError):
            provider.validate_selection(legacy)

    def test_day4_computation_case_registration_and_residual_fail_closed(self):
        from benchmarks.case_assets import resolve_case_assets
        provider = TheadProvider()
        assets = resolve_case_assets(BASE, 'computation-FP32:PPU', 'thead')

        def request(case, nproc=None):
            return type('R', (), {'case': case, 'allow_privileged_root': False,
                                  'context': type('C', (), {'timeout': 300, 'result_root': None})(),
                                  'result_dir': None, 'nproc_per_node': nproc})()
        provider.validate_benchmark(request('computation-FP32:PPU'), assets)
        provider.validate_benchmark(request('computation-FP32:PPU', nproc=1), assets)
        for case, nproc in (('interconnect-MPI_intraserver:PPU', 1), ('computation-FP16:PPU', 1),
                            ('computation-FP32:PPU', 2)):
            with self.subTest(case=case, nproc=nproc), self.assertRaises(ConfigurationError):
                provider.validate_benchmark(request(case, nproc), assets)

    def test_day3_container_methods_produce_bounded_contract(self):
        provider = TheadProvider()
        self.assertEqual(provider.code_identity_files(),
                         ('configs/thead_ppu_torch210.yaml', 'vendors/thead/runtime_bootstrap.sh'))
        self.assertTrue((BASE / 'configs/thead_ppu_torch210.yaml').is_file())
        self.assertTrue((BASE / 'vendors/thead/runtime_bootstrap.sh').is_file())
        host = {'devices': [device_record(0, U0, '00000001:C9:00.0')]}
        lease = provider.lease_spec(host)
        self.assertEqual(lease['resource_keys'], ['thead/' + U0])
        self.assertEqual([str(p) for p in lease['compatibility_paths']], ['/tmp/flagperf-ppu-' + U0 + '.lock'])
        with tempfile.TemporaryDirectory() as tmp:
            arguments, expected = provider.probe_spec(BASE, Path(tmp) / 'result',
                                                      self.load_config(), host, 'sha256:' + '0' * 64)
        self.assertIn('--network=thead-internal', arguments)
        self.assertIn('--ipc=private', arguments)
        self.assertIn('--cap-drop=ALL', arguments)
        self.assertIn('--read-only', arguments)
        self.assertIn('--shm-size=16g', arguments)
        self.assertIn('--group-add', arguments)
        self.assertIn('--env', arguments)
        self.assertIn('CUDA_VISIBLE_DEVICES=0', arguments)
        self.assertIn('FLAGPERF_CVD=0', arguments)
        self.assertIn('--device', arguments)
        self.assertIn('/dev/alixpu_ppu0:/dev/alixpu_ppu0:rwm', arguments)
        self.assertIn('/dev/alixpu:/dev/alixpu:rwm', arguments)
        self.assertIn('--entrypoint', arguments)
        self.assertIn('/workspace/FlagPerf/base/vendors/thead/runtime_bootstrap.sh', arguments)
        self.assertEqual(expected['image_id'], 'sha256:' + '0' * 64)
        self.assertEqual(expected['network'], 'thead-internal')
        self.assertEqual(expected['devices'], [('/dev/alixpu_ppu0', '/dev/alixpu_ppu0', 'rwm'),
                                               ('/dev/alixpu', '/dev/alixpu', 'rwm')])

    def test_day3_resolve_bindings_round_trip_and_rejections(self):
        provider = TheadProvider()
        devices = [device_record(0, U0, '00000001:C9:00.0')]
        expected = binding_records(devices, [U0])
        result = {'schema_version': 1, 'context_sha256': 'a' * 64,
                  'observed_uuids': [U0], 'bindings': expected}
        bindings = provider.resolve_bindings(host={'devices': devices}, context_hash='a' * 64, result=result)
        self.assertEqual([b.record() for b in bindings], expected)
        for drifted in ({'schema_version': 2, 'context_sha256': 'a' * 64, 'observed_uuids': [U0], 'bindings': expected},
                        {'schema_version': 1, 'context_sha256': 'b' * 64, 'observed_uuids': [U0], 'bindings': expected},
                        {'schema_version': 1, 'context_sha256': 'a' * 64, 'observed_uuids': [U0],
                         'bindings': [dict(expected[0], resource_key='thead/other')]}):
            with self.subTest(drifted=drifted['schema_version']), self.assertRaises(RuntimeError):
                provider.resolve_bindings({'devices': devices}, 'a' * 64, drifted)
        with self.assertRaises(ConfigurationError):
            provider.bindings({}, [0])
        with self.assertRaises(ConfigurationError):
            provider.preflight()

    def test_registry_registers_thread_with_day3_contract(self):
        self.assertIs(get_provider('thead'), PROVIDERS['thead'])
        self.assertEqual(PROVIDERS['thead'].name, 'thead')
        self.assertTrue(PROVIDERS['thead'].supports_preflight)
        self.assertTrue(PROVIDERS['thead'].supports_bounded_benchmark)
        with self.assertRaises(ConfigurationError):
            get_provider('unknown')


class TheadAssetTests(unittest.TestCase):
    def test_config_and_mpi_requirements_declare_thead_runtime(self):
        config = json.loads((BASE / 'configs/thead_ppu_torch210.yaml').read_text())
        self.assertEqual(config['vendor'], 'thead')
        self.assertEqual(config['chip'], 'PPU')
        self.assertEqual(config['image'], 'flagtree-ppu-3.6-dev:202608')
        self.assertEqual(config['expected_device_ids'], list(range(16)))
        requirements = json.loads(
            (BASE / 'benchmarks/interconnect-MPI_intraserver/thead/PPU/runtime_requirements.json').read_text())
        self.assertEqual(requirements['backend'], 'cpu:gloo,cuda:nccl')
        self.assertEqual(requirements['process_scope']['max_nproc_per_node'], 16)
        self.assertEqual(requirements['runtime_profile'], config['runtime_profile'])

    def test_driver_branches_route_to_thead_adapter(self):
        script = """
import sys
sys.path.insert(0, BASE_PATH)
sys.path.insert(0, BENCH_PATH)
from unittest.mock import patch
from drivers import utils, thead as thead_driver
with patch.object(thead_driver, 'initialize') as init:
    utils.bootstrap_vendor('thead')
    init.assert_called_once_with()
with patch.object(thead_driver, 'device', return_value='cuda:3') as device:
    assert utils.accelerator_device('thead', 3) == 'cuda:3'
    device.assert_called_once_with(3)
with patch.object(thead_driver, 'set_float32', return_value='set') as set_float32:
    assert utils.set_ieee_float32('thead') == 'set'
import torch
torch.backends.cuda.matmul.allow_tf32 = False
utils.unset_ieee_float32('thead')
assert torch.backends.cuda.matmul.allow_tf32 is True
with patch.object(thead_driver, 'synchronize') as sync:
    utils.host_device_sync('thead')
    sync.assert_called_once_with()
with patch.object(utils.torch.distributed, 'barrier') as barrier:
    utils.multi_device_sync('thead')
    barrier.assert_called_once_with()
try:
    utils._vendor_name('unknown')
except ValueError:
    pass
else:
    raise AssertionError('unknown vendor accepted')
""".replace('BASE_PATH', repr(str(BASE))).replace('BENCH_PATH', repr(str(BASE / 'benchmarks')))
        import subprocess
        proc = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == '__main__':
    unittest.main()
