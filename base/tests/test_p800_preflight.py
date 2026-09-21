"""CPU fixtures only: synthetic identities; query layout derived from PR0 evidence."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from executors.common import DeviceLease, DeviceLeaseError, ConfigurationError
from executors.lifecycle import HostCommands, ManagedContainer
from executors import preflight as executor
from base.vendors.kunlunxin import preflight as host
from base.vendors.kunlunxin.provider import KunlunxinProvider, binding_records
from base.vendors.kunlunxin.container_probe import validate_context
from base.vendors.registry import get_provider
from monitoring.kunlunxin_usage import UsageMonitor

UID = '00000000-0000-0000-0000-000000000001'
QUERY = (BASE / 'tests/fixtures/p800-query.txt').read_text()
CONFIG = BASE / 'configs/kunlunxin_p800_xpytorch29.yaml'
DEVICE = dict(host_physical_id=1, uuid=UID, pci_bdf='0000:11:00.0', device_minor=3,
              device_major=240, host_device_node='/dev/xpu3', container_node='/dev/xpu3')


def machine(index=1, **updates):
    fields = ['0'] * 32
    fields[0], fields[1], fields[3] = f'0000:{16+index:02x}:00.0', str(index), f'FIXTURE-{index}'
    fields[18], fields[21] = '98304', 'P800 OAM'
    for key, value in updates.items():
        fields[int(key)] = str(value)
    return shlex.join(fields)


def result(stdout='', code=0, stderr=''):
    now = time.monotonic()
    return dict(stdout=stdout, stderr=stderr, returncode=code, timed_out=False,
                started_monotonic_s=now, finished_monotonic_s=now+.01)


def arguments(**updates):
    parser = argparse.ArgumentParser()
    executor.add_cli_arguments(parser)
    args = parser.parse_args(['--config', str(CONFIG), '--physical-device-ids', '1'])
    for key, value in updates.items():
        setattr(args, key, value)
    return args


class ParserTests(unittest.TestCase):
    def test_inventory_variable_count_and_quoted_model(self):
        for count in (1, 8, 12):
            self.assertEqual(len(host.machine_inventory('\n'.join(machine(i) for i in range(count)))), count)

    def test_machine_rejects_schema_ranges_model_and_duplicates(self):
        for raw in ('', machine()+' extra', shlex.join(shlex.split(machine())[:-1]),
                    machine(**{'17': -1}), machine(**{'19': 101}), machine(**{'18': 0}),
                    machine(**{'17': 'NaN'}), machine(**{'21': 'R300p'}), machine()+'\n'+machine(),
                    machine()+'\n'+machine(2, **{'0': '0000:11:00.0'}),
                    machine()+'\n'+machine(2, **{'3': 'FIXTURE-1'})):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                host.machine_inventory(raw)

    def test_query_units_identity_versions_and_two_digit_minor(self):
        parsed = host.query_record(QUERY, DEVICE['pci_bdf'])
        self.assertEqual((parsed['uuid'], parsed['total_memory_mib'], parsed['driver']), (UID, 98304, '5.0.21.47'))
        changed = QUERY.replace('Minor Number                          : 3', 'Minor Number                          : 13')
        self.assertEqual(host.query_record(changed, DEVICE['pci_bdf'])['host_device_node'], '/dev/xpu13')

    def test_query_rejects_missing_ambiguous_units_and_identity(self):
        for raw in (QUERY.replace('98304 MiB', '98304 MB'), QUERY.replace('0 %', '101 %'),
                    QUERY.replace('XPU UUID', 'Missing UUID'), QUERY + '\nDriver Version : 1\n',
                    QUERY.replace('Processes', 'Missing')):
            with self.subTest(raw=raw[-60:]), self.assertRaises(ValueError):
                host.query_record(raw, DEVICE['pci_bdf'])
        with self.assertRaises(ValueError):
            host.query_record(QUERY, '0000:12:00.0')

    def inspect_fixture(self, query=QUERY, raw=None, handles=None, config=None):
        commands = Mock()
        commands.checked.side_effect = [result(raw or machine()), result(query)]
        commands.run.return_value = handles or result(code=1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(host, 'check_nodes'):
            return host.inspect_host(config or {'expected_device_ids': [1]}, [1], commands, Path(tmp)/'host')

    def test_host_idle_accepted(self):
        self.assertEqual(self.inspect_fixture()['devices'][0]['uuid'], UID)

    def test_host_busy_handles_driver_inventory_fail_closed(self):
        for kwargs in ({'raw': machine(**{'17': 1})}, {'query': QUERY.replace('0 %', '1 %')},
                       {'query': QUERY.replace('    Processes                             : None', '    Processes\n        Process ID                        : 123')},
                       {'query': QUERY.replace('5.0.21.47', '5.0.21.48')},
                       {'handles': result('42', 0)}, {'handles': result(code=1, stderr='permission denied')},
                       {'config': {'expected_device_ids': [0, 1]}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.inspect_fixture(**kwargs)

    def test_nodes_reject_symlink_wrong_minor_and_regular_file(self):
        for mode, minor in ((stat.S_IFLNK, 3), (stat.S_IFREG, 3), (stat.S_IFCHR, 2)):
            with patch.object(Path, 'lstat', return_value=SimpleNamespace(st_mode=mode, st_rdev=os.makedev(240, minor))):
                with self.assertRaises(RuntimeError):
                    host.check_nodes([deepcopy(DEVICE)])

    def test_lock_recheck_identity_drift(self):
        original = {'devices': [DEVICE]}
        host.same_identity(original, deepcopy(original))
        changed = deepcopy(original); changed['devices'][0]['device_major'] += 1
        with self.assertRaises(RuntimeError):
            host.same_identity(original, changed)


class ContractTests(unittest.TestCase):
    def test_registered_provider_and_deferred_dry_run_without_side_effects(self):
        self.assertIsInstance(get_provider('kunlunxin'), KunlunxinProvider)
        with patch('subprocess.run', side_effect=AssertionError('subprocess forbidden')), patch.object(Path, 'mkdir', side_effect=AssertionError('write forbidden')):
            static, _, _ = executor.plan(arguments(dry_run=True))
        self.assertIn('deferred', static['runtime_bindings'])
        self.assertEqual(static['measurement_status'], 'not-run')

    def test_aliases_multiple_and_invalid_selections_rejected(self):
        for kwargs in ({'physical_device_ids': None, 'npu_ids': '1'}, {'physical_device_ids': None, 'device_ids': '1'},
                       {'physical_device_ids': '1,2'}, {'physical_device_ids': '1,1'}, {'physical_device_ids': '-1'},
                       {'physical_device_ids': '8'}, {'privilege_command': 'sudo bash -c'}, {'timeout': 181}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                executor.plan(arguments(**kwargs))

    def test_candidate_and_reservation_gate_before_files_or_commands(self):
        with patch.object(Path, 'mkdir', side_effect=AssertionError('write forbidden')), patch('subprocess.run', side_effect=AssertionError('subprocess forbidden')):
            with self.assertRaises(ConfigurationError):
                executor.execute(arguments())
            with self.assertRaises(ConfigurationError):
                executor.execute(arguments(allow_candidate_runtime=True))

    def test_uuid_join_preserves_requested_rank_not_logical_order(self):
        second = dict(DEVICE, host_physical_id=6, uuid='00000000-0000-0000-0000-000000000002',
                      pci_bdf='0000:12:00.0', host_device_node='/dev/xpu13', container_node='/dev/xpu13', device_minor=13)
        records = binding_records([second, DEVICE], [UID, second['uuid']])
        self.assertEqual([r['framework_logical_id'] for r in records], [1, 0])
        self.assertEqual([r['framework_local_rank'] for r in records], [0, 1])
        for observed in ([UID], [UID, UID], [UID, second['uuid'], 'unexpected']):
            with self.assertRaises(ValueError):
                binding_records([second, DEVICE], observed)

    def test_resolved_binding_rejects_stale_hash_and_tampering(self):
        valid = dict(schema_version=1, context_sha256='abc', observed_uuids=[UID], bindings=binding_records([DEVICE], [UID]))
        provider = KunlunxinProvider()
        self.assertEqual(provider.resolve_bindings({'devices': [DEVICE]}, 'abc', valid)[0].framework_logical_id, 0)
        for record in (dict(valid, context_sha256='stale'), dict(valid, bindings=[])):
            with self.assertRaises(RuntimeError):
                provider.resolve_bindings({'devices': [DEVICE]}, 'abc', record)

    def test_container_context_rejects_nodes_environment_and_expired_window(self):
        now = datetime.now(timezone.utc)
        ctx = dict(schema_version=1, kind='benchmark-preflight', probe_mode='identity', allow_candidate_runtime=True,
                   run_id='fixture', timeout=120, reservation_reference='synthetic',
                   reservation_start=(now-timedelta(seconds=1)).isoformat(),
                   reservation_end=(now+timedelta(minutes=5)).isoformat(), host={'devices': [DEVICE]})
        env = {'CUDA_VISIBLE_DEVICES': '0', 'USE_FLAGGEMS': '0'}
        nodes = ['/dev/xpu3', '/dev/xpuctrl']
        self.assertEqual(validate_context(ctx, env, nodes), [DEVICE])
        zulu = dict(ctx, reservation_start=ctx['reservation_start'].replace('+00:00', 'Z'))
        self.assertEqual(validate_context(zulu, env, nodes), [DEVICE])
        for c, e, n in ((ctx, env, nodes+['/dev/xpu4']), (ctx, dict(env, CUDA_VISIBLE_DEVICES='1'), nodes),
                        (ctx, dict(env, XPU_EVENT_KL3_ENABLE='1'), nodes), (dict(ctx, allow_candidate_runtime=False), env, nodes),
                        (dict(ctx, reservation_end=(now-timedelta(seconds=1)).isoformat()), env, nodes)):
            with self.assertRaises(RuntimeError):
                validate_context(c, e, n)

    def test_p800_unimplemented_precision_rejected_without_device_access(self):
        from run import main
        with patch('subprocess.run', side_effect=AssertionError('device access forbidden')):
            code = main(['benchmark', 'run', '--config', str(CONFIG), '--physical-device-ids', '1',
                         '--case', 'computation-FP16:P800', '--dry-run'])
        self.assertEqual(code, 2)

    def test_supervisor_spawns_gated_no_site_child_and_propagates_failure(self):
        from base.vendors.kunlunxin import container_probe as probe
        manifest = json.loads((probe.runtime.ROOT/'image-manifest.json').read_text())
        ctx = dict(run_id='fixture', timeout=120, image_identity={
            'Id': manifest['image_id'], 'Architecture': 'amd64', 'RepoDigests': manifest['repo_digests']})
        audit = {'python': '3.10.18', 'inventory_errors': [], 'packages': {}}
        original_stat = Path.lstat
        def node_stat(path, *args, **kwargs):
            if str(path) in ('/dev/xpu3', '/dev/xpuctrl'):
                return SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(240, 3))
            return original_stat(path, *args, **kwargs)
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); context = root/'context.json'; context.write_text(json.dumps(ctx))
                with patch.object(sys, 'argv', ['probe', '--context', str(context), '--output', str(root)]), \
                     patch.object(sys, 'flags', SimpleNamespace(no_site=1)), \
                     patch.object(sys, 'executable', str(probe.runtime.PREFIX/'bin/python')), \
                     patch.object(probe, 'validate_context', return_value=[DEVICE]), \
                     patch.object(Path, 'lstat', node_stat), \
                     patch.object(probe.runtime, 'static_audit', return_value=audit), \
                     patch.object(probe.runtime, 'validate_packages', return_value=[]), \
                     patch.object(probe.subprocess, 'run', return_value=SimpleNamespace(returncode=exit_code)) as run:
                    self.assertEqual(probe.main(), exit_code)
                    child = run.call_args.args[0]
                    self.assertEqual(child[1], '-S'); self.assertIn('--worker', child)
                    self.assertGreater(run.call_args.kwargs['timeout'], ctx['timeout'])
                self.assertTrue((root/'runtime-audit.json').exists())
                self.assertFalse((root/'probe.json').exists())  # parent cannot fabricate worker success


class LeaseTests(unittest.TestCase):
    def test_pr0_flock_interlocks_both_directions_and_keeps_inode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); legacy = root/'pr0.lock'; legacy.touch(); inode = legacy.stat().st_ino
            def lease():
                return DeviceLease([], root=root/'new', resource_keys=['kunlunxin/'+UID], compatibility_paths=[legacy], run_id='fixture', kind='fixture')
            with legacy.open('r+') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(DeviceLeaseError): lease().acquire()
            with lease():
                script = 'import fcntl,sys; f=open(sys.argv[1],"r+"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
                child = subprocess.run([sys.executable, '-c', script, str(legacy)], capture_output=True)
                self.assertNotEqual(child.returncode, 0)
            with lease(): pass
            self.assertEqual(legacy.stat().st_ino, inode)

    def test_symlink_hardlink_permission_and_partial_failure_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); target = root/'target'; target.touch(); bad = root/'zbad'; bad.symlink_to(target)
            def lease(paths=()):
                return DeviceLease([], root=root/'a', resource_keys=['kunlunxin/'+UID], compatibility_paths=paths, run_id='fixture', kind='fixture')
            with self.assertRaises(OSError): lease([bad]).acquire()
            with lease(): pass
            bad.unlink(); os.link(target, bad)
            with self.assertRaises(DeviceLeaseError): lease([bad]).acquire()
            with patch('executors.common.os.open', side_effect=PermissionError('fixture user denied')):
                with self.assertRaises(PermissionError): lease().acquire()
            with lease(): pass


class LifecycleTests(unittest.TestCase):
    def test_host_timeout_and_prefix_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp, patch('subprocess.run', side_effect=subprocess.TimeoutExpired(['fixture'], 1, b'partial', b'error')) as run:
            record = HostCommands('sudo -n').run(['fuser', '/dev/fixture'], privileged=True, timeout=1, evidence=Path(tmp)/'command.json')
            self.assertEqual(run.call_args.args[0], ['sudo', '-n', 'fuser', '/dev/fixture'])
            self.assertTrue(record['timed_out']); self.assertEqual(record['stdout'], 'partial')
            self.assertTrue((Path(tmp)/'command.json').is_file())

    def fixture(self, root):
        commands = Mock()
        cid = 'a'*64
        record = {'Id': cid, 'Config': {'Labels': {'flagperf.run_id': 'fixture'}}, 'Image': 'sha256:'+'b'*64,
                  'HostConfig': {'IpcMode': 'private', 'Privileged': False, 'Devices': [], 'NetworkMode': 'none', 'ReadonlyRootfs': True,
                                 'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges']}, 'Mounts': [],
                  'State': {'Running': False, 'Status': 'exited', 'ExitCode': 0}}
        container = ManagedContainer(commands, root, 'fixture-name', 'fixture')
        expected = dict(image_id=record['Image'], devices=[], mounts=[])
        return container, commands, record, expected

    def test_create_inspects_immutable_identity_and_permissions(self):
        mutations = [lambda r: None, lambda r: r.update(Image='wrong'), lambda r: r['HostConfig'].update(Privileged=True),
                     lambda r: r['Config']['Labels'].update({'flagperf.run_id': 'other'}),
                     lambda r: r['HostConfig'].update(Devices=[dict(PathOnHost='/dev/extra', PathInContainer='/dev/extra', CgroupPermissions='rwm')]),
                     lambda r: r['HostConfig'].update(NetworkMode='host'), lambda r: r['HostConfig'].update(ReadonlyRootfs=False),
                     lambda r: r['HostConfig'].update(SecurityOpt=[]),
                     lambda r: r['HostConfig'].update(SecurityOpt=['no-new-privileges=false']),
                     lambda r: r['HostConfig'].update(IpcMode='host'),
                     lambda r: r['HostConfig'].update(CapAdd=['SYS_ADMIN'])]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                container, commands, record, expected = self.fixture(Path(tmp)); mutate(record)
                commands.checked.side_effect = [result('a'*64), result(json.dumps([record]))]
                if index:
                    with self.assertRaises(RuntimeError): container.create([], expected)
                else: container.create([], expected)
                self.assertFalse(any('start' in c.args[0] for c in commands.checked.call_args_list))

    def test_wait_uses_remaining_probe_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            container, commands, record, _ = self.fixture(Path(tmp))
            commands.checked.return_value = result(json.dumps([record]))
            self.assertEqual(container.wait(time.monotonic()+.1), 0)
            self.assertLessEqual(commands.checked.call_args.kwargs['timeout'], .1)
            with self.assertRaises(TimeoutError): container.wait(time.monotonic()-1)

    def test_cleanup_ownership_mismatch_never_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            container, commands, record, _ = self.fixture(Path(tmp)); container.create_attempted = True
            record['Config']['Labels']['flagperf.run_id'] = 'someone-else'
            commands.run.return_value = result('a'*64); commands.checked.return_value = result(json.dumps([record]))
            cleanup = container.cleanup()
            self.assertIsNone(cleanup['container_absent'])
            self.assertFalse(any('rm' in call.args[0] for call in commands.run.call_args_list))

    def test_cleanup_timedout_create_recovers_only_owned_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            container, commands, record, _ = self.fixture(Path(tmp)); container.create_attempted = True
            record['State']['Running'] = True
            commands.checked.return_value = result(json.dumps([record]))
            commands.run.side_effect = [result('a'*64), result('log'), result(), result(), result()]
            cleanup = container.cleanup()
            self.assertTrue(cleanup['container_absent'])
            self.assertEqual(cleanup['container_id'], 'a'*64)
            self.assertTrue(any('stop' in call.args[0] for call in commands.run.call_args_list))

    def test_cleanup_daemon_errors_are_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            container, commands, _, _ = self.fixture(Path(tmp)); container.create_attempted = True
            commands.run.return_value = result(code=124); commands.checked.side_effect = RuntimeError('daemon unavailable')
            self.assertIsNone(container.cleanup()['container_absent'])


class MonitorTests(unittest.TestCase):
    def monitor(self):
        commands = Mock(); commands.run.side_effect = [result(machine()), result(QUERY)]
        return UsageMonitor(KunlunxinProvider().monitor_targets({'devices': [DEVICE]}), commands=commands)

    def test_telemetry_uuid_and_units(self):
        monitor = self.monitor(); monitor.collect_once()
        self.assertTrue(monitor.samples[0]['valid'])
        self.assertEqual(monitor.samples[0]['values']['total_memory_mib'], 98304)

    def test_active_process_table_is_valid_telemetry_but_not_idle(self):
        active = QUERY.replace('    Processes                             : None', '    Processes\n        Process ID                        : 123')
        self.assertFalse(host.query_record(active, DEVICE['pci_bdf'])['processes_empty'])
        monitor = self.monitor(); monitor.commands.run.side_effect = [result(machine()), result(active)]
        monitor.collect_once()
        self.assertTrue(monitor.samples[0]['valid'])

    def test_failed_or_wrong_identity_samples_are_invalid_not_zero(self):
        for query in (result(code=124), result(QUERY.replace(UID, '00000000-0000-0000-0000-000000000002'))):
            monitor = self.monitor(); monitor.commands.run.side_effect = [result(machine()), query]; monitor.collect_once()
            self.assertFalse(monitor.samples[0]['valid']); self.assertEqual(monitor.samples[0]['values'], {})

    def test_window_counts_partial_missing_wrong_role_and_invalid(self):
        for mode in ('passed', 'missing', 'wrong-role', 'invalid', 'short'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                monitor = self.monitor(); uid = monitor.targets[0]['device_id']
                monitor.samples = [dict(device_id=uid, started_offset_s=i, finished_offset_s=i+.1, valid=True, values={}) for i in range(12)]
                windows = [dict(role='probe-observation', device_id=uid, started_offset_s=0, finished_offset_s=12)]
                if mode == 'missing': windows=[]
                if mode == 'wrong-role': windows[0]['role']='measurement'
                if mode == 'invalid': monitor.samples[5]['valid']=False
                if mode == 'short': windows[0]['finished_offset_s']=3
                output = monitor.finish(Path(tmp), Path(tmp)/'monitor', windows, [], primary_role='probe-observation')
                self.assertEqual(output['status'], 'passed' if mode == 'passed' else 'partial')


class ExecutionTests(unittest.TestCase):
    def run_fixture(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; provider=KunlunxinProvider(); leases=[]; calls=[]
            def make_lease(ids, **kwargs):
                kwargs.update(root=Path(tmp)/'locks', compatibility_paths=[Path(tmp)/'compat.lock'])
                lease=DeviceLease(ids, **kwargs); leases.append(lease); return lease
            def inspect(config, selected, commands, directory):
                calls.append(directory.name)
                if directory.name != 'host-preflight': self.assertTrue(leases[0]._streams)
                if mode == 'identity-drift' and directory.name == 'locked-preflight':
                    return {'devices': [dict(DEVICE, device_major=241)]}
                return {'devices': [deepcopy(DEVICE)]}
            def wait(deadline):
                if mode == 'timeout': raise TimeoutError('fixture timeout')
                if mode == 'interrupt': raise KeyboardInterrupt('fixture interrupt')
                ctxpath=root/'control/host-context.json'; ctx=json.loads(ctxpath.read_text()); h=hashlib.sha256(ctxpath.read_bytes()).hexdigest()
                bindings=binding_records([DEVICE], [UID]); out=root/'artifacts'
                (out/'runtime-bindings.json').write_text(json.dumps(dict(schema_version=1, run_id=ctx['run_id'], context_sha256=h, observed_uuids=[UID], bindings=bindings)))
                (out/'probe.json').write_text(json.dumps(dict(status='passed', run_id=ctx['run_id'], context_sha256=h, binding=bindings[0])))
                return 0
            container=Mock(); container.wait.side_effect=wait
            if mode == 'create-failed': container.create.side_effect=RuntimeError('fixture create failure')
            if mode == 'start-failed': container.start.side_effect=RuntimeError('fixture start failure')
            container.cleanup.return_value={'container_absent': mode != 'cleanup-failed', 'status': 'failed' if mode == 'cleanup-failed' else 'passed'}
            if mode == 'cleanup-exception': container.cleanup.side_effect=OSError('fixture disk error')
            manifest=json.loads((BASE/'vendors/kunlunxin/xpytorch_2.9_p800_candidate/image-manifest.json').read_text())
            image={'Id':manifest['image_id'], 'Architecture':'amd64', 'RepoDigests':manifest['repo_digests']}
            commands=Mock(); commands.checked.side_effect=lambda argv, **kw: result(json.dumps([image]) if argv[0]=='docker' else 'fixture')
            commands.deadline=time.monotonic()+60
            monitor=Mock(); monitor.finish.return_value={'status':'partial'}
            args=arguments(allow_candidate_runtime=True, result_dir=root, monitor='on' if mode=='partial' else 'off',
                           reservation_end=(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat(), reservation_reference='synthetic fixture')
            with patch.object(executor,'HostCommands',return_value=commands), patch.object(executor,'ManagedContainer',return_value=container), \
                 patch.object(executor,'DeviceLease',side_effect=make_lease), patch.object(KunlunxinProvider,'inspect_host',side_effect=inspect), \
                 patch.object(KunlunxinProvider,'create_monitor',return_value=monitor), redirect_stdout(io.StringIO()):
                code=executor.execute(args)
            summary=json.loads((root/'summary.json').read_text())
            self.assertFalse(leases[0]._streams)
            if mode not in ('cleanup-failed', 'cleanup-exception'):
                self.assertEqual(calls[-1], 'host-postflight')
            else:
                self.assertNotIn('host-postflight', calls); self.assertTrue(summary['recovery_required'])
            self.assertEqual(summary['measurement_status'],'not-run')
            self.assertFalse((root/'benchmark-result.json').exists())
            before=(root/'summary.json').read_bytes(); report=(root/'report.md').read_bytes()
            executor.render_report(root)
            self.assertEqual((root/'summary.json').read_bytes(),before); self.assertEqual((root/'report.md').read_bytes(),report)
            return code, summary

    def test_normal_postflight_before_release_and_deterministic_report(self):
        code, summary=self.run_fixture('normal'); self.assertEqual(code,0); self.assertEqual(summary['status'],'passed')

    def test_create_and_start_failure_cleanup(self):
        for mode in ('create-failed', 'start-failed'):
            with self.subTest(mode=mode):
                code, summary = self.run_fixture(mode)
                self.assertEqual(code, 1)
                self.assertEqual(summary['cleanup_status'], 'passed')

    def test_partial_monitor_is_not_pass(self):
        code, summary=self.run_fixture('partial'); self.assertEqual(code,2); self.assertEqual(summary['status'],'partial')

    def test_timeout_and_interrupt_cleanup_and_release(self):
        for mode in ('timeout','interrupt'):
            with self.subTest(mode=mode):
                code, summary=self.run_fixture(mode); self.assertEqual(code,1); self.assertEqual(summary['cleanup_status'],'passed')

    def test_lock_identity_change_blocks_container_start(self):
        code, summary=self.run_fixture('identity-drift'); self.assertEqual(code,1); self.assertEqual(summary['failure_stage'],'locked-preflight')

    def test_cleanup_unknown_or_exception_releases_lock_but_requires_recovery(self):
        for mode in ('cleanup-failed','cleanup-exception'):
            with self.subTest(mode=mode):
                code, summary=self.run_fixture(mode); self.assertEqual(code,1); self.assertTrue(summary['recovery_required'])


if __name__ == '__main__':
    unittest.main()
