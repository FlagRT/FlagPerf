"""CPU-only thead host preflight, runtime lock, probe-context and worker-vendor contracts."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from base.vendors.thead import container_probe, preflight, runtime

FIXTURE = BASE / 'tests/fixtures/ppu-q-16cards.txt'
U1 = '011ef111-47c0-0010-0000-0000a05fba7d'
DRIVER = runtime.locked_host_observation()['driver']


def fake_check_nodes(devices, control='/dev/alixpu'):
    # The CPU test environment has no PPU device nodes; record the observed
    # day-1 major so the summary contract stays exercised.
    for device in devices:
        device['device_major'] = 236


def uuid_for(slot):
    return '00000000-0000-0000-0000-%012d' % slot


def bdf_for(slot):
    return '00000001:%02X:00.0' % (0xC9 + slot)


def card_section(slot, *, used=18, util=0, processes_none=True, uuid_text=None, minor=None):
    lines = ['PPU %s' % bdf_for(slot),
             '    Product Name                            : PPU-ZW810E',
             '    Serial Number                           : SERIAL%02d' % slot,
             '    PPU UUID                                : GPU-%s' % (uuid_text or uuid_for(slot)),
             '    Minor Number                            : %d' % (minor if minor is not None else slot + 1),
             '    Memory Usage',
             '        Total                               : 98304 MiB',
             '        Used                                : %d MiB' % used,
             '        Free                                : %d MiB' % (98304 - used),
             '    PPU Utilization',
             '        Ppu                                 : %d %%' % util]
    if processes_none:
        lines.append('    Processes                                : None')
    else:
        lines += ['    Processes', '        Process ID                          : 999999']
    return '\n'.join(lines) + '\n'


HEADER = ('==============PPUSMI LOG==============\n\n'
          'Timestamp                                   : Wed Oct 14 10:52:03 2026\n'
          'Driver Version                              : %s\n\n'
          'Attached PPUs                               : 16\n\n' % DRIVER)


def full_dump(**card_one):
    sections = [card_section(slot, **card_one) if slot == 1 else card_section(slot) for slot in range(16)]
    return HEADER + '\n' + '\n'.join(sections)


def single_dump(slot, **kwargs):
    return HEADER + '\n' + card_section(slot, **kwargs)


class ScriptedCommands:
    def __init__(self, script):
        self.script = script
        self.calls = []

    def run(self, arguments, timeout=None, evidence=None, privileged=False):
        self.calls.append(list(arguments))
        stdout, stderr, returncode = self.script(list(arguments))
        return {'arguments': list(arguments), 'stdout': stdout, 'stderr': stderr, 'returncode': returncode,
                'started_monotonic_s': 1.0, 'finished_monotonic_s': 2.0}

    def checked(self, arguments, timeout=None, evidence=None):
        result = self.run(arguments, timeout=timeout, evidence=evidence)
        if result['returncode']:
            raise RuntimeError('command failed: ' + ' '.join(arguments))
        return result


def idle_script(card_one=None, requested=(1,)):
    card_one = dict(card_one or {})

    def script(arguments):
        if arguments[:2] == ['ppu-smi', '-q']:
            return full_dump(**card_one), '', 0
        if arguments[:2] == ['ppu-smi', '-i']:
            slot = int(arguments[2])
            return (single_dump(slot, **card_one), '', 0) if slot in requested else ('', 'no such PPU', 1)
        if arguments[0] == 'fuser':
            return '', '', 1
        return '', 'unexpected command', 127
    return script


class TheadInventoryTests(unittest.TestCase):
    def test_machine_inventory_reads_sixteen_unique_cards_with_serials(self):
        rows = preflight.machine_inventory(FIXTURE.read_text())
        self.assertEqual([row['host_physical_id'] for row in rows], list(range(16)))
        for field in ('pci_bdf', 'uuid', 'serial'):
            self.assertEqual(len({row[field] for row in rows}), 16)
        self.assertEqual(rows[1]['device_minor'], 2)
        self.assertEqual(rows[1]['host_device_node'], '/dev/alixpu_ppu1')

    def test_machine_inventory_rejects_duplicate_sections(self):
        raw = FIXTURE.read_text()
        doubled = raw + card_section(1, uuid_text=uuid_for(1), minor=2)
        with self.assertRaises(ValueError):
            preflight.machine_inventory(doubled)

    def test_query_record_extracts_driver_and_identity(self):
        info = preflight.query_record(single_dump(1), bdf_for(1))
        self.assertEqual(info['driver'], DRIVER)
        self.assertEqual(info['uuid'], uuid_for(1))
        self.assertEqual(info['device_minor'], 2)
        self.assertEqual(info['serial'], 'SERIAL01')
        with self.assertRaises(ValueError):
            preflight.query_record(single_dump(2), bdf_for(1))

    def test_readonly_smi_arguments_whitelist(self):
        for arguments in (['-q'], [], ['-i', '3', '-q'], ['-i', '03', '-q']):
            self.assertTrue(preflight.readonly_smi_arguments(arguments), arguments)
        for arguments in (['-i', '3'], ['-q', '-w'], ['-i', '3', '-q', '-x'], ['-r']):
            self.assertFalse(preflight.readonly_smi_arguments(arguments), arguments)

    def test_inspect_handles_classifies_observers_strictly(self):
        device = {'host_physical_id': 1, 'host_device_node': '/dev/alixpu_ppu1'}
        with tempfile.TemporaryDirectory() as tmp:
            free = ScriptedCommands(lambda a: ('', '', 1))
            self.assertEqual(preflight.inspect_handles(device, free, Path(tmp)), [])
            observer = ScriptedCommands(lambda a: self._observer_script(a))
            found = preflight.inspect_handles(device, observer, Path(tmp), allow_observers=True)
            self.assertEqual(found, [{'pid': 4242, 'executable': '/usr/local/bin/ppu-smi',
                                      'arguments': ['-q'], 'classification': 'read-only-observer'}])
            with self.assertRaises(RuntimeError):
                preflight.inspect_handles(device, observer, Path(tmp) / 'strict', allow_observers=False)
            foreign = ScriptedCommands(lambda a: self._observer_script(a, exe='/usr/bin/blackbox'))
            with self.assertRaises(RuntimeError):
                preflight.inspect_handles(device, foreign, Path(tmp) / 'foreign', allow_observers=True)
            agent = ScriptedCommands(lambda a: self._observer_script(a, exe='/usr/bin/teamagent', argv='teamagent\0'))
            with patch.dict(preflight.KNOWN_READONLY_MONITORS, {'teamagent': '/usr/bin/teamagent'}):
                with patch.dict(os.environ, {'THEAD_KNOWN_READONLY_MONITORS': 'teamagent'}):
                    found = preflight.inspect_handles(device, agent, Path(tmp) / 'agent', allow_observers=True)
            self.assertEqual(found[0]['classification'], 'read-only-monitor-agent')
            self.assertEqual(found[0]['agent'], 'teamagent')
            with patch.dict(preflight.KNOWN_READONLY_MONITORS, {'teamagent': '/usr/bin/teamagent'}):
                with self.assertRaises(RuntimeError):
                    preflight.inspect_handles(device, agent, Path(tmp) / 'unauthorized', allow_observers=True)

    @staticmethod
    def _observer_script(arguments, exe='/usr/local/bin/ppu-smi', argv='ppu-smi\0-q\0'):
        if arguments[0] == 'fuser':
            return '4242\n', '/dev/alixpu_ppu1:', 0
        if arguments[:3] == ['readlink', '-e', '/proc/4242/exe']:
            return exe + '\n', '', 0
        if arguments[:3] == ['cat', '/proc/4242/cmdline']:
            return argv, '', 0
        return '', 'unexpected command', 127

    def test_same_identity_accepts_equal_and_rejects_drift(self):
        rows = preflight.machine_inventory(FIXTURE.read_text())
        fields = ('host_physical_id', 'uuid', 'pci_bdf', 'host_device_node', 'container_node', 'device_minor', 'device_major')
        records = [{k: row[k] for k in fields if k in row} for row in rows]
        for record in records:
            record['container_node'] = record['host_device_node']
            record['device_major'] = 236
        first = {'devices': records}
        preflight.same_identity(first, first)
        drifted = {'devices': [dict(first['devices'][0], uuid=uuid_for(99))] + first['devices'][1:]}
        with self.assertRaises(RuntimeError):
            preflight.same_identity(first, drifted)


class TheadInspectHostTests(unittest.TestCase):
    def run_inspect_host(self, script, requested=(1,), **environ):
        config = {'expected_device_ids': list(range(16)), 'allow_readonly_smi_handles': False}
        with tempfile.TemporaryDirectory() as tmp:
            saved = {key: os.environ.get(key) for key in ('THEAD_ALLOW_FOREIGN_HANDLES', 'THEAD_KNOWN_READONLY_MONITORS')}
            os.environ.update({key: value for key, value in environ.items() if value is not None})
            try:
                with patch.object(preflight, 'check_nodes', fake_check_nodes):
                    return preflight.inspect_host(config, list(requested), ScriptedCommands(script), Path(tmp) / 'host')
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_idle_card_passes_with_full_evidence_contract(self):
        summary = self.run_inspect_host(idle_script())
        self.assertEqual(summary['schema_version'], 1)
        self.assertEqual(summary['status'], 'passed')
        self.assertEqual(summary['selection'], {'source': 'physical-device-ids', 'requested_ids': [1]})
        self.assertFalse(summary['foreign_handles_authorized'])
        self.assertFalse(summary['foreign_occupancy_observed'])
        self.assertEqual(summary['authorized_monitors'], [])
        self.assertEqual(summary['readonly_observers'], {'1': []})
        device = summary['devices'][0]
        self.assertEqual(device['host_physical_id'], 1)
        self.assertEqual(device['uuid'], uuid_for(1))
        self.assertEqual(device['container_node'], '/dev/alixpu_ppu1')
        self.assertEqual(device['device_major'], 236)
        self.assertEqual(device['driver'], DRIVER)
        self.assertEqual(device['total_memory_mib'], 98304)

    def test_occupied_cards_fail_closed_by_process_utilization_and_memory(self):
        for occupied_by in ({'processes_none': False}, {'util': 5}, {'used': 30000}):
            with self.subTest(occupied_by=sorted(occupied_by)):
                with self.assertRaises(RuntimeError):
                    self.run_inspect_host(idle_script(occupied_by))

    def test_foreign_handles_require_explicit_environment_authorization(self):
        summary = self.run_inspect_host(idle_script({'used': 30000}), THEAD_ALLOW_FOREIGN_HANDLES='1')
        self.assertTrue(summary['foreign_handles_authorized'])
        self.assertTrue(summary['foreign_occupancy_observed'])
        self.assertEqual(summary['devices'][0]['foreign_occupancy_observed'], True)
        self.assertEqual(summary['readonly_observers'],
                         {'1': [{'classification': 'foreign-handles-authorized',
                                 'inspection': 'skipped-by-explicit-user-authorization'}]})

    def test_identity_drift_between_inventory_and_query_fails(self):
        def script(arguments):
            if arguments[:2] == ['ppu-smi', '-q']:
                return full_dump(), '', 0
            if arguments[:2] == ['ppu-smi', '-i']:
                return single_dump(1, uuid_text='ffffffff-0000-0000-0000-000000000001'), '', 0
            return '', '', 1
        with self.assertRaises(RuntimeError):
            self.run_inspect_host(script)

    def test_inventory_smaller_than_configured_fails(self):
        fifteen = HEADER + '\n' + '\n'.join(card_section(slot) for slot in range(1, 16))
        with self.assertRaises(RuntimeError):
            self.run_inspect_host(lambda a: (fifteen, '', 0) if a[:2] == ['ppu-smi', '-q'] else ('', '', 1))


class TheadRuntimeLockTests(unittest.TestCase):
    def load_manifest(self):
        return json.loads((runtime.PROFILE / 'image-manifest.json').read_text())

    def test_repository_manifest_is_candidate_and_validates(self):
        manifest = self.load_manifest()
        self.assertIs(manifest['validated'], False)
        runtime.validate_manifest(manifest)
        runtime.validate_manifest(manifest, [{'Id': manifest['image_id'], 'Architecture': 'amd64',
                                              'RepoDigests': manifest['repo_digests']}])

    def test_observed_identity_drift_fails(self):
        manifest = self.load_manifest()
        for observed in ([{'Id': manifest['image_id'], 'Architecture': 'amd64', 'RepoDigests': ['repo@sha256:' + '0' * 64]}],
                         [{'Id': 'sha256:' + 'f' * 64, 'Architecture': 'amd64', 'RepoDigests': []}],
                         [{'Id': manifest['image_id'], 'Architecture': 'arm64', 'RepoDigests': []}],
                         [{'Id': manifest['image_id'], 'Architecture': 'amd64', 'RepoDigests': []},
                          {'Id': manifest['image_id'], 'Architecture': 'amd64', 'RepoDigests': []}]):
            with self.subTest(observed=observed[0]['Architecture'] if len(observed) == 1 else 'two-records'), \
                 self.assertRaises(ValueError):
                runtime.validate_manifest(manifest, observed)

    def test_state_machine_rejects_undefined_and_unbacked_states(self):
        manifest = self.load_manifest()
        for mutation in ({'release_stage': 'candidate', 'validated': True},
                         {'release_stage': 'validated', 'validated': False},
                         {'release_stage': 'released', 'validated': False},
                         {'image_id': 'flagtree-ppu-3.6-dev:202608'},
                         {'platform': 'linux/arm64'},
                         {'hardware_model': 'PPU-ZW810X'}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runtime.validate_manifest({**manifest, **mutation})
        validated = {**manifest, 'release_stage': 'validated', 'validated': True,
                     'validation_record': {'path': 'validation-summary.json', 'sha256': '0' * 64}}
        with self.assertRaises((ValueError, OSError)):
            runtime.validate_manifest(validated)

    def test_locked_host_observation_matches_day2_fixture_driver(self):
        import re
        locked = runtime.locked_host_observation()
        fixture_driver = re.search(r'^Driver Version[ \t]*:[ \t]*(\S+)[ \t]*$', FIXTURE.read_text(), re.M).group(1)
        self.assertEqual(locked['driver'], fixture_driver)
        self.assertEqual(DRIVER, fixture_driver)


class TheadProbeContextTests(unittest.TestCase):
    def valid_context(self, **overrides):
        context = {'schema_version': 1, 'kind': 'benchmark-preflight', 'vendor': 'thead',
                   'run_id': 'preflight-unittest', 'probe_mode': 'identity', 'timeout': 120,
                   'allow_candidate_runtime': True,
                   'reservation_start': '2026-10-13T00:00:00+00:00',
                   'reservation_end': '2027-01-01T00:00:00+00:00',
                   'reservation_reference': 'unittest',
                   'host': {'devices': [{'host_physical_id': 0, 'pci_bdf': '00000001:c9:00.0',
                                         'uuid': U1, 'host_device_node': '/dev/alixpu_ppu0',
                                         'container_node': '/dev/alixpu_ppu0', 'device_minor': 1,
                                         'device_major': 236}]},
                   'nproc_per_node': 1}
        context.update(overrides)
        return context

    def validate(self, context, environ=None, nodes=None, now=None):
        return container_probe.validate_context(
            context, environ if environ is not None else {'CUDA_VISIBLE_DEVICES': '0'},
            nodes if nodes is not None else ['/dev/alixpu', '/dev/alixpu_ppu0'],
            now or datetime(2026, 10, 14, tzinfo=timezone.utc))

    def test_valid_identity_context_passes(self):
        devices = self.validate(self.valid_context())
        self.assertEqual(devices[0]['uuid'], U1)

    def test_candidate_runtime_requires_explicit_authorization(self):
        with self.assertRaises(RuntimeError):
            self.validate(self.valid_context(allow_candidate_runtime=False))

    def test_wrong_vendor_or_mode_or_environment_fails(self):
        with self.assertRaises(RuntimeError):
            self.validate(self.valid_context(vendor='kunlunxin'))
        with self.assertRaises(RuntimeError):
            self.validate(self.valid_context(probe_mode='capability-fp32'))
        with self.assertRaises(RuntimeError):
            self.validate(self.valid_context(), environ={'CUDA_VISIBLE_DEVICES': '0', 'XPU_VISIBLE_DEVICES': '0'})
        with self.assertRaises(RuntimeError):
            self.validate(self.valid_context(), environ={'CUDA_VISIBLE_DEVICES': '0,1'})

    def test_reservation_window_is_enforced_with_timezone(self):
        with self.assertRaises(RuntimeError):
            self.validate(self.valid_context(), now=datetime(2030, 1, 1, tzinfo=timezone.utc))
        with self.assertRaises(RuntimeError):
            self.validate(self.valid_context(reservation_end='2027-01-01T00:00:00'))
        with self.assertRaises(RuntimeError):
            self.validate(self.valid_context(reservation_start='2026-10-15T00:00:00+00:00'))

    def test_unexpected_device_nodes_and_unregistered_case_fail(self):
        with self.assertRaises(RuntimeError):
            self.validate(self.valid_context(), nodes=['/dev/alixpu', '/dev/alixpu_ppu0', '/dev/alixpu_ctl'])
        benchmark = self.valid_context(kind='benchmark', probe_mode='performance',
                                       case='interconnect-MPI_intraserver:PPU', case_assets_sha256='0' * 64)
        with self.assertRaises(RuntimeError):
            self.validate(benchmark)

    def test_day4_registered_computation_case_passes_probe_gate(self):
        benchmark = self.valid_context(kind='benchmark', probe_mode='performance',
                                       case='computation-FP32:PPU', case_assets_sha256='0' * 64)
        self.assertEqual(self.validate(benchmark)[0]['uuid'], U1)


class WorkerVendorParameterizationTests(unittest.TestCase):
    def test_worker_passes_context_vendor_to_asset_verification(self):
        import benchmark_worker
        sentinel = RuntimeError('stop before execution')
        for context, expected in (({'kind': 'benchmark', 'nproc_per_node': 1,
                                    'case': 'computation-FP32:PPU'}, 'kunlunxin'),
                                  ({'kind': 'benchmark', 'nproc_per_node': 1,
                                    'case': 'computation-FP32:PPU', 'vendor': 'thead'}, 'thead')):
            with self.subTest(vendor=context.get('vendor', 'kunlunxin')):
                with patch.object(benchmark_worker, 'verify_worker_assets', side_effect=sentinel) as verify:
                    with self.assertRaises(RuntimeError):
                        benchmark_worker.run_bound_case(context, Path('ctx/host-context.json'), Path('.'))
                self.assertEqual(verify.call_args[0][3], expected)

    def test_worker_rank_ceiling_stays_one_to_eight_until_day6(self):
        import benchmark_worker
        with patch.object(benchmark_worker, 'verify_worker_assets') as verify:
            with self.assertRaises(RuntimeError):
                benchmark_worker.run_bound_case({'kind': 'benchmark', 'nproc_per_node': 9, 'case': 'x'},
                                                Path('ctx/host-context.json'), Path('.'))
            verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
