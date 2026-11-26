# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Training observation alignment: monitor wiring and the result-report read path.

The monitor tests need a sidecar process and file locking, so they run on
POSIX only; the report test exercises the same reading pipeline inference
uses over a synthetic training-interval store.
"""
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flagprof.store import Writer, create_run, finalize


def training_span(step, phase, corr, ts, dur, **extra):
    attrs = dict(training_schema='training/1', phase=phase, step=step,
                 outcome=dict(status='complete'))
    attrs.update(extra)
    return dict(t='span', ph='x', cat='user', name='training.'+phase, corr=corr,
                ts=ts, dur=dur, attrs=attrs)


@unittest.skipUnless(os.name == 'posix', 'monitor sidecar and store locking require POSIX')
class TrainingMonitorWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        inference = str(Path(__import__('flagprof').__file__).resolve().parent.parent)
        self.env = patch.dict(os.environ, dict(
            FLAGPROF_TRAIN_MONITOR='on', PYTHONPATH=inference+os.pathsep+os.environ.get('PYTHONPATH', '')))
        self.env.start(); self.addCleanup(self.env.stop)
        os.environ.pop('FLAGPROF_MONITOR_REFERENCE', None)

    def test_single_process_session_and_diagnose_config(self):
        from flagprof.training.__main__ import monitor_start
        monitoring = monitor_start(self.root, 'thead', 1, 0)
        try:
            self.assertIsNotNone(monitoring)
            self.assertEqual(monitoring['topology'], 'local')
            self.assertEqual(monitoring['config'], dict(monitor='on', monitor_topology='local'))
            self.assertNotIn('FLAGPROF_MONITOR_REFERENCE', os.environ)
            self.assertEqual(monitoring['reference'],
                             str((self.root/'monitor-ref.json').resolve()))
            reference = json.loads((self.root/'monitor-ref.json').read_text())
            self.assertEqual(reference.get('topology'), 'local')
            self.assertEqual(reference['pre_sample'], 'available')
            self.assertTrue((self.root/'monitor').is_dir())
            manifests = list((self.root/'monitor').glob('*/manifest.json'))
            self.assertEqual(len(manifests), 1, 'monitor session run missing')
            inventory = json.loads((manifests[0].parent/'inventory.json').read_text())
            self.assertEqual(inventory['device_telemetry'], 'unavailable')
            self.assertEqual(inventory['source'], 'host-procfs')
            self.assertEqual(json.loads(manifests[0].read_text()).get('identity', {}).get('kind'), 'monitor')
        finally:
            monitoring['owner'].close()
        closed = json.loads((self.root/'monitor-ref.json').read_text())
        self.assertIsNotNone(closed.get('reclaim'), 'monitor close left no reclaim receipt')

    def test_inference_and_training_use_host_sidecar_outside_checkout(self):
        from flagprof.monitor.lifecycle import Monitor
        from flagprof.training.__main__ import monitor_start
        cwd = Path.cwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.root)
        with patch.dict(os.environ):
            os.environ.pop('PYTHONPATH', None)
            for vendor in ('thead', 'kunlunxin'):
                root = self.root/vendor
                root.mkdir()
                monitor = Monitor(root, root/'default.json', vendor, dict(monitor='on'))
                try:
                    receipt = json.loads(monitor.path.read_text())
                    self.assertEqual((receipt['topology'], receipt['pre_sample']), ('local', 'available'))
                finally:
                    monitor.close()
                obs = monitor_start(root, vendor, 1, 0)
                try:
                    self.assertEqual((obs['topology'], obs['diagnose']), ('local', 'off'))
                finally:
                    obs['close']()

    def test_vendor_without_device_telemetry_keeps_diagnose_off(self):
        from flagprof.training.__main__ import monitor_start
        with patch.dict(os.environ, dict(FLAGPROF_TRAIN_DIAGNOSE='on')):
            monitoring = monitor_start(self.root, 'thead', 1, 0)
            try:
                self.assertEqual(monitoring['topology'], 'local')
                self.assertEqual(monitoring['diagnose'], 'off')
                self.assertEqual(monitoring['supervisor'], None)
            finally:
                monitoring['owner'].close()

    def test_coverage_reuses_agent_monitor_and_records_diagnosis_availability(self):
        from unittest.mock import Mock
        from flagprof.probes.config import resolve
        from flagprof.training.__main__ import monitor_start, record_monitor_status
        with patch('flagprof.monitor.subscriptions.request', return_value=dict(session_id='watch')), \
             patch('subprocess.Popen') as start:
            monitoring = monitor_start(self.root, 'ascend', 1, 0, preset='coverage')
            try:
                start.assert_not_called()
                _, config = resolve('coverage', monitoring['config'])
                self.assertEqual((config['monitor_topology'], config['diagnose']), ('agent', 'off'))
                receipt = self.root/'rank-0-ref.json'
                key = dict(pid=13)
                data = json.loads(Path(monitoring['reference']).read_text())
                data.update(worker_proc_key=key, subscription_status='registered', target_id='rank-0')
                receipt.write_text(json.dumps(data))
                handle = Mock(monitor_enabled=True, monitor_reference=receipt, state=dict(process=key))
                record_monitor_status(handle, monitoring, 'npu:0')
                self.assertEqual((handle.state['monitor_topology'], handle.state['monitor_diagnose']), ('agent', 'off'))
                self.assertIn('coverage', handle.state['monitor_diagnose_reason'])
            finally:
                monitoring['close']()

    def test_failed_watch_never_enables_agent_diagnosis(self):
        from unittest.mock import Mock
        from flagprof.training.__main__ import monitor_start
        child = Mock()
        child.poll.return_value = 1
        with patch('flagprof.monitor.subscriptions.request', side_effect=ConnectionRefusedError), \
             patch('subprocess.Popen', return_value=child):
            monitoring = monitor_start(self.root, 'ascend', 1, 0)
        self.assertEqual(monitoring['topology'], 'unavailable')
        self.assertEqual(monitoring['config'], dict(monitor='off', diagnose='off'))
        self.assertIsNone(monitoring['reference'])
        self.assertIn('watch supervisor failed', monitoring['errors'][0])

    def test_monitor_broadcast_failure_stops_only_owned_supervisor(self):
        import sys
        from types import ModuleType
        from unittest.mock import Mock
        from flagprof.training.__main__ import monitor_start
        dist = Mock()
        dist.broadcast_object_list.side_effect = RuntimeError('broadcast failed')
        torch = ModuleType('torch')
        torch.distributed = dist
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                child = Mock()
                child.poll.return_value = None
                replies = [dict(session_id='monitor')] * 2
                if not reuse:
                    replies.insert(0, ConnectionRefusedError())
                with patch.dict(sys.modules, {'torch': torch, 'torch.distributed': dist}), \
                     patch('flagprof.monitor.subscriptions.request', side_effect=replies), \
                     patch('subprocess.Popen', return_value=child) as start:
                    with self.assertRaisesRegex(RuntimeError, 'broadcast failed'):
                        monitor_start(self.root, 'ascend', 2, 0, control=object(),
                                      reference=self.root/('ref-'+str(reuse)+'.json'))
                if reuse:
                    start.assert_not_called()
                    child.terminate.assert_not_called()
                else:
                    child.terminate.assert_called_once()
                    child.wait.assert_called_once()

    def test_target_receipt_determines_rank_monitor_state(self):
        from unittest.mock import Mock
        from flagprof.training.__main__ import record_monitor_status
        key = {'pid': 13}
        decision = dict(topology='agent', diagnose='on', errors=[])
        receipt = self.root/'monitor-ref.json'
        for subscription, diagnostic, expected in (
                ('capacity', None, ('unavailable', 'off')),
                ('registered', 'registration_unavailable', ('agent', 'off')),
                ('registered', None, ('agent', 'on'))):
            with self.subTest(subscription=subscription, diagnostic=diagnostic):
                receipt.write_text(json.dumps(dict(worker_proc_key=key, subscription_status=subscription,
                                                   target_id='worker-A')))
                handle = Mock(monitor_enabled=True, monitor_reference=receipt,
                              state=dict(process=key, diagnostic_status=diagnostic))
                record_monitor_status(handle, decision, 'npu:1')
                self.assertEqual((handle.state['monitor_topology'], handle.state['monitor_diagnose']), expected)
        receipt.unlink()
        handle = Mock(monitor_enabled=True, monitor_reference=receipt, state=dict(process=key))
        record_monitor_status(handle, decision, 'npu:1')
        self.assertEqual(handle.state['monitor_topology'], 'unavailable')

    def test_attach_failure_reclaims_owner_and_restores_environment(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from flagprof.training.__main__ import observation_start
        close = Mock()
        monitoring = dict(close=close, reference=None, config={})
        with patch.dict(os.environ, dict(FLAGPROF_TRAIN_OBSERVE=str(self.root))), \
             patch('flagprof.training.__main__.monitor_start', return_value=monitoring), \
             patch('flagprof.runtime.attach', side_effect=RuntimeError('attachment failed')):
            os.environ.pop('XPU_ENABLE_PROFILER_TRACING', None)
            with self.assertRaisesRegex(RuntimeError, 'attachment failed'):
                observation_start(SimpleNamespace(task='external', device='npu'))
            self.assertNotIn('XPU_ENABLE_PROFILER_TRACING', os.environ)
        close.assert_called_once()

    def test_disabled_monitor_returns_none_without_side_effects(self):
        from flagprof.training.__main__ import monitor_start
        with patch.dict(os.environ, dict(FLAGPROF_TRAIN_MONITOR='off')):
            self.assertIsNone(monitor_start(self.root, 'thead', 1, 0))
        self.assertIsNone(os.environ.get('FLAGPROF_MONITOR_REFERENCE'))
        self.assertFalse((self.root/'monitor-ref.json').exists())


class ObservationPlatformTests(unittest.TestCase):
    def test_shared_cuda_runtime_selects_physical_vendor_or_requires_override(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from flagprof.training.__main__ import observation_platform
        cuda = Mock()
        cuda.is_available.return_value = True
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)), patch.dict(os.environ, clear=True), \
             patch('flagprof.vendors.shutil.which', return_value=None):
            for name, vendor in (('Kunlun P800 OAM', 'kunlunxin'), ('PPU-ZW810E', 'thead'),
                                 ('NVIDIA H100', 'nvidia')):
                cuda.get_device_name.return_value = name
                self.assertEqual(observation_platform('cuda:3', None), (vendor, 'cuda:3'))
                cuda.get_device_name.assert_called_with(3)
            cuda.get_device_name.return_value = 'GPU'
            with self.assertRaisesRegex(ValueError, 'FLAGPROF_TRAIN_OBSERVE_VENDOR'):
                observation_platform('cuda:3', None)
            os.environ['XPU_VISIBLE_DEVICES'] = '5,2,4,1'
            self.assertEqual(observation_platform('cuda:3', None)[0], 'kunlunxin')
            os.environ['FLAGPROF_TRAIN_OBSERVE_VENDOR'] = 'thead'
            self.assertEqual(observation_platform('cuda:3', None)[0], 'thead')

    def test_p800_prerequisites_precede_torch_and_restore_environment(self):
        from flagprof.training import __main__ as cli
        seen = []
        def observe(args):
            seen.append((os.environ['XPU_ENABLE_PROFILER_TRACING'], os.environ['XPU_CUPTI_ENABLE_DEVICE']))
        settings = dict(FLAGPROF_TRAIN_OBSERVE='store', FLAGPROF_TRAIN_OBSERVE_VENDOR='kunlunxin',
                        XPU_CUPTI_ENABLE_DEVICE='7', LOCAL_RANK='2')
        with patch.dict(os.environ, settings, clear=True), patch.object(cli, 'execute', side_effect=observe):
            opts = ['--device', 'cuda:3', '--outputs', 'unused']
            cli.main(opts)
            cli.main(opts + ['--ddp'])
            self.assertEqual(seen, [('1', '3'), ('1', '2')])
            self.assertEqual(os.environ['XPU_CUPTI_ENABLE_DEVICE'], '7')
            self.assertNotIn('XPU_ENABLE_PROFILER_TRACING', os.environ)

    def test_final_capture_preserves_global_rank_and_binds_local_rank(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from types import SimpleNamespace
        from flagprof.offline.capture import capture_target, record_final
        cp = SimpleNamespace(returncode=7, timed_out=False)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, FLAGPROF_LAUNCH_WORKER='1', LOCAL_RANK='1'), \
             patch('flagprof.offline.process.run_owned', return_value=cp) as launch, redirect_stdout(StringIO()):
            code = record_final([sys.executable, '-c', 'pass'], store=tmp, vendor='kunlunxin', rank=4)
            self.assertEqual(code, 7)
            self.assertEqual(launch.call_args.kwargs['env']['XPU_CUPTI_ENABLE_DEVICE'], '1')
            config, = Path(tmp).glob('capture-*/record-config.json')
            self.assertEqual(json.loads(config.read_text())['rank'], 4)
            with patch('flagprof.vendors.kunlunxin.execution.initialize', side_effect=RuntimeError('bind')) as bind:
                with self.assertRaisesRegex(RuntimeError, 'bind'):
                    capture_target(tmp, 'kunlunxin', 'perf', ['-c', 'pass'], rank=4)
                bind.assert_called_once_with(1)

    def test_distributed_perf_and_modules_use_managed_capture_signature(self):
        from flagprof.training import distributed
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, RANK='4', LOCAL_RANK='1', WORLD_SIZE='8'), \
             patch('flagprof.vendors.kunlunxin.execution.initialize', return_value='cuda:1'), \
             patch('flagprof.offline.capture.record_final', autospec=True, return_value=0) as capture:
            for preset in ('perf', 'modules'):
                root = Path(tmp)/preset
                rc = distributed.main(['--store', str(root), '--vendor', 'kunlunxin', '--preset', preset,
                                       '--', '--ddp'])
                self.assertEqual(rc, 0)
                self.assertEqual(capture.call_args.kwargs['store'], root/'rank-4'/'store')
                self.assertEqual(capture.call_args.kwargs['rank'], 4)
                self.assertEqual(json.loads((root/'rank-4'/'worker.json').read_text())['status'], 'complete')


class TrainingReportAlignmentTests(unittest.TestCase):
    def test_rank_index_links_open_from_relative_output_directory(self):
        from urllib.parse import unquote
        import re
        from flagprof.training.group import build
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            root = Path(tmp)/'stores #1'
            run = create_run(root/'rank-0')
            finalize(run)
            report = run.parent/(run.name+'-report')
            report.mkdir()
            (report/'report.html').write_text('ready')
            (run/'oracle.json').write_text('{}')
            out = Path(tmp)/'reports'
            build(root, out)
            links = re.findall(r'\]\(([^)]+)\)', (out/'group.md').read_text())
            self.assertEqual({(out/unquote(link)).resolve() for link in links},
                             {p.resolve() for p in (report/'report.html', run/'manifest.json', run/'oracle.json')})
            self.assertFalse(any(Path(link).is_absolute() for link in links))

    def test_report_renders_training_panel_from_interval_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = create_run(root/'rank-0'/'run-1')
            with Writer(run) as writer:
                corr = 0
                corr += 1
                writer.emit(training_span(0, 'data', corr, 500, 100))
                for step in range(2):
                    corr += 1
                    writer.emit(training_span(step, 'step', corr, 1000+step*10**7, 10**7, accumulation=2))
                    for micro in range(2):
                        corr += 1
                        writer.emit(training_span(step, 'forward', corr, 1000+step*10**7+micro*100, 90, microstep=micro))
                        corr += 1
                        writer.emit(training_span(step, 'backward', corr, 1000+step*10**7+micro*100+100, 90, microstep=micro))
                    corr += 1
                    writer.emit(training_span(step, 'zero_grad', corr, 900+step*10**7, 10))
                    corr += 1
                    writer.emit(training_span(step, 'optimizer', corr, 950+step*10**7, 40))
            finalize(run)
            (run/'oracle.json').write_text(json.dumps(dict(
                rows=[dict(step=0, key='layer', kind='activation', basis='digest', verdict='exact')],
                first_divergence=None, coverage=dict(partial=True))))
            from flagprof.report import generate
            result = generate(run)
            out = Path(result['output'])
            self.assertEqual(result['panels'].get('training'), 'ok')
            self.assertTrue((out/'report.html').exists())
            self.assertTrue((out/'report.md').exists())
            document = json.loads((out/'result.json').read_text())
            first = document['results']['training.oracle.first_divergence']
            self.assertEqual(first['status'], 'partial')
            self.assertTrue(first['coverage']['partial'])
            self.assertIn('仅覆盖已判定', (out/'report.md').read_text())
            panel = document['results'].get('training')
            self.assertIsNotNone(panel, 'training panel missing from result report')
            self.assertEqual(panel['status'], 'ok')
            steps = [row['step'] for row in panel['data']]
            self.assertEqual(sorted(steps), [0, 1])


if __name__ == '__main__':
    unittest.main()
