# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Exercise real shared flock behavior with temporary paths and fake devices."""
from contextlib import contextmanager, ExitStack, redirect_stdout, redirect_stderr
import copy
import fcntl
import io
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime import cli, diagnosis
from vendors.kunlunxin.p800 import Adapter
from vendors.kunlunxin import host


def arguments(*extra):
    return cli.parser().parse_args(['run', '--vendor', 'kunlunxin', '--runtime-profile',
                                   'p800-xpytorch29', '--device-ids', '4,7', '--diagnostics', 'off', *extra])


def observed(devices):
    return {'status': 'passed', 'devices': [
        {'host_physical_id': d, 'uuid': str(uuid.UUID(int=d+1)), 'device_minor': 5 if d == 4 else 2,
         'host_device_node': '/dev/xpu5' if d == 4 else '/dev/xpu2',
         'container_node': '/dev/xpu5' if d == 4 else '/dev/xpu2', 'device_major': 123,
         'pci_bdf': f'0000:{d:02x}:00.0', 'serial': f'fixture-{d}'} for d in devices]}


@contextmanager
def fake_host(root, observation=None):
    common, lifecycle, vendor = host.base_modules()
    compatibility = root / 'legacy-locks';compatibility.mkdir()
    original_spec = vendor.KunlunxinProvider.lease_spec
    def spec(provider, record):
        result = original_spec(provider, record)
        return {**result, 'compatibility_paths': [compatibility / p.name for p in result['compatibility_paths']]}
    def inspect(config, requested, commands, directory):
        if observation:
            return observation(config, requested, commands, directory)
        return copy.deepcopy(observed(requested))
    with patch.object(common, 'DEFAULT_LEASE_ROOT', root / 'base-locks'), \
         patch.object(host, 'check_process_visibility', return_value={'method': 'fixture'}), \
         patch.object(vendor.KunlunxinProvider, 'lease_spec', spec), \
         patch.object(vendor.KunlunxinProvider, 'inspect_host', side_effect=inspect) as inspect_mock, \
         patch.dict('os.environ', {'P800_ALLOW_FOREIGN_HANDLES': '0'}):
        yield common, inspect_mock


class P800HostTests(unittest.TestCase):
    def test_incomplete_process_visibility_cannot_pass_idle_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp);(proc/'123/fd').mkdir(parents=True)
            (proc/'mounts').write_text('proc /proc proc rw,hidepid=0 0 0\n')
            self.assertEqual(host.check_process_visibility(proc)['inspected_processes'], 1)
            with patch.object(host.os, 'open', side_effect=PermissionError('fixture')):
                with self.assertRaisesRegex(RuntimeError, 'cannot inspect'): host.check_process_visibility(proc)
            (proc/'mounts').write_text('proc /proc proc rw,hidepid=2 0 0\n')
            with self.assertRaisesRegex(RuntimeError, 'visibility'): host.check_process_visibility(proc)

    def test_physical_order_nodes_and_pending_framework_identity(self):
        with tempfile.TemporaryDirectory() as tmp, fake_host(Path(tmp)):
            root = Path(tmp);adapter = Adapter();args = arguments('--device-ids', '7,4', '--case', 'mm', '--dtype', 'FP32', 'INT32')
            _, plan = cli.resolve(args)
            lease = adapter.create_lease(root, [7, 4], 'fixture')
            with self.assertRaises(RuntimeError): adapter.docker_options(4, args)
            lease.acquire()
            try:
                record = adapter.preflight(root, [7, 4])
                self.assertEqual(record['host'], socket.gethostname())
                self.assertEqual(record['execution_context'], 'host')
                adapter.bind_tasks(plan['tasks'], record)
                self.assertTrue(any(not t['applicable'] for t in plan['tasks']))
                for task in plan['tasks']:
                    self.assertIsNone(task['worker_device'])
                    self.assertIsNone(task['device_binding']['framework_logical_id'])
                    self.assertEqual(task['runtime_config']['physical_uuid'], str(uuid.UUID(int=task['device_id']+1)))
                opts = adapter.docker_options(4, args)
                self.assertIn('/dev/xpu5:/dev/xpu5:rwm', opts)
                self.assertIn('/dev/xpuctrl:/dev/xpuctrl:rwm', opts)
                self.assertFalse(any('/dev/xpu2' in v for v in opts))
                with self.assertRaises(ValueError): adapter.docker_options(1, args)
                self.assertEqual(lease.device_ids, ())
                self.assertEqual(set(lease.resource_keys), {'kunlunxin/' + str(uuid.UUID(int=d+1)) for d in (4, 7)})
            finally:
                lease.release()
            with self.assertRaises(RuntimeError): adapter.preflight(root, [7, 4])

    def test_uuid_join_observes_runtime_order_and_rejects_bad_sets(self):
        devices = observed([7, 4])['devices'];uuids = [d['uuid'] for d in reversed(devices)]
        joined = host.resolve_framework_bindings(devices, uuids)
        self.assertEqual([d['framework_logical_id'] for d in joined], [1, 0])
        self.assertEqual([d['host_physical_id'] for d in joined], [7, 4])
        for bad in ([], uuids[:1], [uuids[0], uuids[0]], [uuids[0], str(uuid.UUID(int=99))]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                host.resolve_framework_bindings(devices, bad)

    def test_base_toolkit_and_operation_block_each_other(self):
        with tempfile.TemporaryDirectory() as tmp, fake_host(Path(tmp)) as (common, _):
            root = Path(tmp);adapter = Adapter()
            lease = adapter.create_lease(root, [4, 7], 'operation-fixture')
            for kind in ('benchmark', 'toolkit'):
                other = common.DeviceLease([], root=common.DEFAULT_LEASE_ROOT, run_id=kind, kind=kind,
                                           resource_keys=lease.resource_keys, compatibility_paths=lease.compatibility_paths)
                lease.acquire()
                with self.assertRaises(common.DeviceLeaseError): other.acquire()
                lease.release();other.acquire()
                with self.assertRaises(common.DeviceLeaseError): lease.acquire()
                other.release();lease.acquire();lease.release()
            # Historical Toolkit flock path also conflicts without a resource key.
            with lease.compatibility_paths[0].open('r+') as legacy:
                fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(common.DeviceLeaseError): lease.acquire()
                fcntl.flock(legacy, fcntl.LOCK_UN)
                lease.acquire()
                with self.assertRaises(BlockingIOError): fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lease.release()

    def test_failed_multi_lock_acquisition_releases_earlier_locks(self):
        with tempfile.TemporaryDirectory() as tmp, fake_host(Path(tmp)) as (common, _):
            root = Path(tmp);lease = Adapter().create_lease(root, [4, 7], 'operation-fixture')
            last = sorted(lease.compatibility_paths)[-1]
            with last.open('a+') as blocking:
                fcntl.flock(blocking, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(common.DeviceLeaseError): lease.acquire()
                self.assertFalse(lease._streams)
                probe = common.DeviceLease([], root=common.DEFAULT_LEASE_ROOT, run_id='probe', kind='toolkit',
                                           resource_keys=lease.resource_keys,
                                           compatibility_paths=[p for p in lease.compatibility_paths if p != last])
                probe.acquire();probe.release()

    def test_lock_recheck_rejects_identity_drift_and_busy_device(self):
        for change in ('uuid', 'node', 'busy'):
            def inspect(config, requested, commands, directory):
                record = observed(requested)
                if directory.name == 'preflight':
                    if change == 'busy': raise RuntimeError('selected device is occupied')
                    if change == 'uuid': record['devices'][0]['uuid'] = str(uuid.UUID(int=99))
                    if change == 'node': record['devices'][0].update(host_device_node='/dev/xpu6', container_node='/dev/xpu6', device_minor=6)
                return record
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp, fake_host(Path(tmp), inspect):
                root = Path(tmp);adapter = Adapter();lease = adapter.create_lease(root, [4, 7], 'fixture')
                with self.assertRaises(RuntimeError): adapter.preflight(root, [4, 7])
                lease.acquire()
                try:
                    with self.assertRaises(RuntimeError): adapter.preflight(root, [4, 7])
                    self.assertIsNone(adapter.host_session.current)
                finally: lease.release()

    def test_foreign_handle_override_is_never_inherited(self):
        with tempfile.TemporaryDirectory() as tmp, fake_host(Path(tmp)) as (_, inspect_mock), \
             patch.dict('os.environ', {'P800_ALLOW_FOREIGN_HANDLES': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'does not accept'):
                Adapter().create_lease(Path(tmp), [4], 'fixture')
            inspect_mock.assert_not_called()

    def test_local_lease_does_not_discover_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Adapter();args = arguments('--execution', 'local', '--device-ids', '9')
            with patch.object(adapter, 'create_lease', side_effect=AssertionError('host')), \
                 patch.object(adapter, 'lease_root', side_effect=AssertionError('physical lease')), \
                 patch.object(adapter, 'local_lease_root', return_value=Path(tmp)):
                lease = cli.make_lease(adapter, Path(tmp), args, [9], 'local-fixture')
                self.assertIs(type(lease), cli.Lease)
                lease.acquire();lease.release()
                self.assertTrue((Path(tmp) / 'logical-device-9.lock').exists())


class P800LifecycleTests(unittest.TestCase):
    def test_replay_cleanup_interruption_preserves_release_and_signal_handler(self):
        for failure in ('cleanup', 'postflight', 'release'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, fake_host(Path(tmp)) as (common, _):
                root = Path(tmp);adapter = Adapter();saved_handler = signal.getsignal(signal.SIGTERM)
                args = cli.parser().parse_args(['diagnose', '--source-task', str(root/'source-task'), '--device-ids', '4'])
                _, plan = cli.resolve(arguments('--device-ids', '4', '--case', 'abs'))
                data = {'checks': []};events = []
                original_preflight = adapter.preflight
                original_release = common.DeviceLease.release
                def preflight(*args, **kwargs):
                    if kwargs.get('label') == 'postflight':
                        events.append('postflight')
                        self.assertTrue(adapter.host_session.lease._streams)
                        if failure == 'postflight': raise KeyboardInterrupt('postflight interrupted')
                    return original_preflight(*args, **kwargs)
                def close():
                    self.assertTrue(adapter.host_session.lease._streams)
                    if failure == 'cleanup': raise KeyboardInterrupt('cleanup interrupted')
                def release(lease):
                    original_release(lease)
                    if failure == 'release': raise KeyboardInterrupt('release interrupted')
                with patch.object(diagnosis, 'get_vendor', return_value=adapter), \
                     patch.object(adapter, 'ensure_execution_ready'), \
                     patch.object(adapter, 'preflight', side_effect=preflight), \
                     patch.object(cli.Workers, 'close', side_effect=close), \
                     patch.object(common.DeviceLease, 'release', release), \
                     patch.object(diagnosis.subprocess, 'check_output', return_value=json.dumps([{'Id':adapter.lock['image_id']}])), \
                     patch.object(diagnosis.diagnostics, 'run_check', side_effect=RuntimeError('phase failed')):
                    with self.assertRaisesRegex(RuntimeError, 'phase failed'):
                        diagnosis.replay(root, args, plan['tasks'][0], {'actual_image_id':adapter.lock['image_id']}, data)
                self.assertFalse(adapter.host_session.lease._streams)
                self.assertEqual(events, ['postflight'])
                self.assertEqual(signal.getsignal(signal.SIGTERM), saved_handler)
                self.assertIn('postflight_error' if failure == 'postflight' else 'cleanup_error', data)

    def test_local_cpu_process_is_reaped_without_other_processes(self):
        proc = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'], start_new_session=True)
        try:
            result = host.cleanup_process(proc)
            self.assertTrue(result['process_group_absent'])
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None: proc.kill();proc.wait(timeout=5)

    def test_local_cleanup_checks_surviving_group_after_leader_exit(self):
        proc = Mock(pid=12345, poll=Mock(return_value=0))
        running = True
        def killpg(pid, sig):
            nonlocal running
            if not running: raise ProcessLookupError()
            if sig == signal.SIGTERM: running = False
        with patch.object(host.os, 'killpg', side_effect=killpg) as kill:
            self.assertTrue(host.cleanup_process(proc)['process_group_absent'])
        kill.assert_any_call(12345, signal.SIGTERM)

    def exercise(self, tmp, failure=None, local=False):
        root = Path(tmp);events = [];saved_handler = signal.getsignal(signal.SIGTERM)
        def inspect(config, requested, commands, directory):
            events.append(directory.name)
            if failure == 'postflight' and directory.name == 'postflight': raise RuntimeError('postflight failed')
            if failure == 'recovery' and directory.name.startswith('health-after'): raise RuntimeError('device busy')
            return observed(requested)
        with fake_host(root, inspect) as (common, _):
            args = arguments('--device-ids', '4', '--case', 'abs', '--case', 'add', '--result-root', str(root/'results'),
                             *(['--execution', 'local'] if local else []))
            adapter, plan = cli.resolve(args)
            def phase(work, task, mode):
                events.append((task['case'], mode))
                if not local:
                    self.assertTrue(adapter.host_session.lease._streams)
                if mode == 'reference' and failure == 'launch': raise RuntimeError('container launch failed')
                if mode == 'probe' and failure in ('watchdog', 'cleanup', 'recovery'):
                    raise RuntimeError('watchdog exceeded')
                if mode == 'probe' and failure == 'interrupt': raise KeyboardInterrupt('fixture interrupt')
                if mode == 'probe' and failure == 'sigterm': signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                if mode == 'probe':
                    cli.write(work/'probe.json', {});cli.write(work/'correctness.json', {'status': 'passed'})
                    (work/'probe.log').write_text('')
                if mode == 'measure':
                    cli.write(work/'measurement.json', {'correctness': {'status':'passed'}})
                    (work/'measure.log').write_text('')
                if mode == 'memory': cli.write(work/'memory.json', {'status':'not-supported'})
            def close(*args, **kwargs):
                events.append('cleanup')
                if not local: self.assertTrue(adapter.host_session.lease._streams)
                if failure == 'cleanup': raise cli.CleanupError('unconfirmed owned container')
            with ExitStack() as stack:
                for manager in [patch.object(adapter, 'ensure_execution_ready'), patch.object(cli, 'snapshot'),
                                patch.object(cli, 'report'), patch.object(cli, 'seal'),
                                patch.object(adapter, 'local_lease_root', return_value=root/'local-locks'),
                                patch.object(adapter, 'route', return_value={'status':'passed'}),
                                patch.object(cli.Workers, 'phase', side_effect=phase),
                                patch.object(cli.Workers, 'close', side_effect=close),
                                patch.object(cli.subprocess, 'check_output', return_value=json.dumps([{'Id':adapter.lock['image_id']}])),
                                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())]:
                    stack.enter_context(manager)
                result = cli.execute(args, adapter, plan)
            summary = cli.read(next((root/'results').glob('*/summary.json')))
            self.assertEqual(signal.getsignal(signal.SIGTERM), saved_handler)
            if not local:
                self.assertFalse(adapter.host_session.lease._streams)
                self.assertTrue(summary['lease']['released'])
                self.assertEqual(events[-1], 'postflight')
                # Reacquisition proves no lock was stranded after any failure.
                adapter.host_session.lease.acquire();adapter.host_session.lease.release()
            return result, summary, events

    def test_success_checks_health_before_releasing_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, summary, events = self.exercise(tmp)
            self.assertEqual(result, 0)
            self.assertEqual(events[:2], ['prelease', 'preflight'])
            self.assertEqual(summary['preflight']['execution_context'], 'host')
            self.assertTrue(all(t['device_binding']['framework_logical_id'] is None for t in summary['tasks']))

    def test_launch_watchdog_interrupt_and_postflight_failures(self):
        for failure in ('launch', 'watchdog', 'interrupt', 'sigterm', 'postflight'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                result, summary, events = self.exercise(tmp, failure)
                self.assertEqual(result, 1)
                self.assertIn('cleanup', events)
                if failure == 'postflight': self.assertIn('postflight_error', summary)
                if failure in ('interrupt', 'sigterm'):
                    self.assertEqual([t['status'] for t in summary['tasks']], ['interrupted', 'not-run'])

    def test_cleanup_or_recovery_failure_stops_next_task_and_diagnostics(self):
        for failure in ('cleanup', 'recovery'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                result, summary, events = self.exercise(tmp, failure)
                self.assertEqual(result, 1)
                self.assertEqual(summary['tasks'][1]['status'], 'not-run')
                self.assertTrue(summary['tasks'][0]['diagnostic_execution_blocked'])
                self.assertFalse(any(isinstance(e, tuple) and e[0] == 'add' for e in events))
                if failure == 'cleanup': self.assertIn('cleanup_error', summary)

    def test_local_reports_external_management_without_host_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, summary, events = self.exercise(tmp, local=True)
            self.assertEqual(result, 0)
            self.assertEqual(summary['preflight']['status'], 'externally-managed')
            self.assertNotIn('host', summary['preflight'])
            self.assertNotIn('actual_image_id', summary)
            self.assertNotIn('prelease', events)

    def test_container_launch_timeout_is_owned_and_retained_for_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments();adapter = Adapter();pool = cli.Workers(Path(tmp), adapter, args, 'fixture-image', 'fixture-run')
            with patch.object(cli.subprocess, 'run', side_effect=subprocess.TimeoutExpired('docker', 60)) as launch:
                with self.assertRaises(subprocess.TimeoutExpired):
                    pool.start({'device_id':4, 'oplib':'nativetorch'}, 'reference')
            self.assertIn('flagperf.run_id=fixture-run', launch.call_args.args[0])
            self.assertEqual(len(pool.active), 1)
            worker = next(iter(pool.active.values()))
            with patch.object(adapter, 'cleanup_worker', return_value={'container_absent': False}) as cleanup:
                with self.assertRaises(cli.CleanupError): pool.close()
                cleanup.assert_called_once_with(worker, 'fixture-run')
                self.assertEqual(len(pool.active), 1)
            with patch.object(adapter, 'cleanup_worker', return_value={'container_absent': True}): pool.close()
            self.assertFalse(pool.active)

    def test_worker_death_and_watchdog_are_detected(self):
        for kind in ('container-dead', 'process-dead', 'watchdog'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp);work = root/'case';work.mkdir();control=root/'control';control.mkdir()
                args = arguments('--watchdog', '0.01' if kind == 'watchdog' else '300')
                pool = cli.Workers(root, Adapter(), args, 'image', 'fixture')
                worker = {'control':control, 'name':'owned-fixture', 'proc':Mock(poll=Mock(return_value=1)) if kind == 'process-dead' else None}
                with patch.object(pool, 'start', return_value=worker), patch.object(cli.time, 'monotonic', side_effect=iter(range(100))), \
                     patch.object(cli.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,stdout='false',stderr='')):
                    with self.assertRaises(RuntimeError): pool.phase(work, {'device_id':4}, 'probe')

    def test_cleanup_checks_label_and_never_deletes_another_run(self):
        common, lifecycle, _ = host.base_modules()
        for owner, expected in [('other-run', False), ('fixture-run', True)]:
            with self.subTest(owner=owner), tempfile.TemporaryDirectory() as tmp:
                calls=[];lookups=0;cid='a'*64
                def command(self, argv, **kwargs):
                    nonlocal lookups
                    calls.append(argv)
                    if argv[1:3] == ['ps', '-a']:
                        lookups+=1;out=cid if lookups==1 else ''
                    elif argv[1] == 'inspect':
                        out=json.dumps([{'Id':cid,'Config':{'Labels':{'flagperf.run_id':owner}},'State':{'Running':False}}])
                    else: out=''
                    return {'returncode':0,'stdout':out,'stderr':''}
                with patch.object(lifecycle.HostCommands, 'run', command):
                    record=host.cleanup_worker({'name':'owned-fixture','control':Path(tmp)}, 'fixture-run')
                self.assertEqual(record.get('container_absent') is True, expected)
                self.assertEqual(any(argv[1]=='rm' for argv in calls), expected)


if __name__ == '__main__':
    unittest.main()
