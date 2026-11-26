# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Training observation alignment: monitor wiring and the result-report read path.

The monitor tests need a sidecar process and file locking, so they run on
POSIX only; the report test exercises the same reading pipeline inference
uses over a synthetic training-interval store.
"""
import json
import os
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

    def test_single_process_session_wiring_is_consistent(self):
        """thead attempts the agent session first; when the telemetry library
        is unavailable it falls back to the local sidecar. Either way the
        topology, reference file and diagnose flag must agree."""
        from flagprof.training.__main__ import monitor_start
        monitoring = monitor_start(self.root, 'thead', 1, 0)
        try:
            self.assertIsNotNone(monitoring)
            topology = monitoring['topology']
            self.assertIn(topology, ('local', 'agent'))
            if topology == 'agent':
                self.assertEqual(monitoring['config']['monitor_topology'], 'agent')
                self.assertEqual(monitoring['diagnose'], 'on')
                self.assertIsNotNone(monitoring['supervisor'])
                reference = self.root/'monitor-ref.json'
            else:
                self.assertEqual(monitoring['config'], dict(monitor='on', monitor_topology='local'))
                self.assertEqual(monitoring['diagnose'], 'off')
                self.assertIsNone(monitoring['supervisor'])
                reference = self.root/'monitor-ref.local.json'
            self.assertEqual(monitoring['reference'], str(reference.resolve()))
            payload = json.loads(reference.read_text())
            self.assertEqual(payload.get('topology'), topology)
            self.assertTrue((self.root/'monitor').is_dir(), 'local monitor run missing')
            manifests = list((self.root/'monitor').glob('*/manifest.json'))
            self.assertEqual(len(manifests), 1, 'monitor session run missing')
            self.assertEqual(json.loads(manifests[0].read_text()).get('identity', {}).get('kind'), 'monitor')
        finally:
            monitoring['owner'].close()
        # The close receipt lives beside whichever reference the session used.
        closed = json.loads(Path(monitoring['reference']).read_text())
        self.assertIsNotNone(closed.get('reclaim'), 'monitor close left no reclaim receipt')

    def test_local_fallback_keeps_diagnose_off(self):
        """Force only the agent supervisor to fail; the local sidecar must
        take over with the .local.json reference and diagnosis honestly off."""
        import subprocess
        from unittest.mock import Mock
        from flagprof.training.__main__ import monitor_start
        real_popen = subprocess.Popen
        state = {'calls': 0}

        def popen(*args, **kwargs):
            state['calls'] += 1
            if state['calls'] == 1:  # the agent supervisor launch
                child = Mock()
                child.poll.return_value = 1
                return child
            return real_popen(*args, **kwargs)

        # Pre-seed the supervisor's own failure reason: the fallback error must
        # surface it instead of a generic message.
        (self.root/'session.json').write_text(json.dumps(dict(
            status='failed', worker_error='ReadError: NVML-compatible library unavailable')))
        with patch.dict(os.environ, dict(FLAGPROF_TRAIN_DIAGNOSE='on')), \
             patch('flagprof.monitor.subscriptions.request', side_effect=ConnectionRefusedError), \
             patch('subprocess.Popen', side_effect=popen):
            monitoring = monitor_start(self.root, 'thead', 1, 0)
        try:
            self.assertEqual(monitoring['topology'], 'local')
            self.assertEqual(monitoring['diagnose'], 'off')
            self.assertIsNone(monitoring['supervisor'])
            self.assertTrue(monitoring['reference'].endswith('monitor-ref.local.json'))
            self.assertTrue(monitoring['errors'], 'fallback must record the agent failure')
            self.assertIn('NVML-compatible library unavailable', monitoring['errors'][0],
                          'fallback error must name the underlying cause')
        finally:
            monitoring['owner'].close()

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


class TrainingReportAlignmentTests(unittest.TestCase):
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
