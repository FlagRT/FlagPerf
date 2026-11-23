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

    def test_single_process_session_and_diagnose_config(self):
        from flagprof.training.__main__ import monitor_start
        monitoring = monitor_start(self.root, 'thead', 1, 0)
        try:
            self.assertIsNotNone(monitoring)
            self.assertEqual(monitoring['topology'], 'local')
            self.assertEqual(monitoring['config'], dict(monitor='on', monitor_topology='local'))
            self.assertEqual(os.environ['FLAGPROF_MONITOR_REFERENCE'],
                             str((self.root/'monitor-ref.json').resolve()))
            reference = json.loads((self.root/'monitor-ref.json').read_text())
            self.assertEqual(reference.get('topology'), 'local')
            self.assertIn(reference.get('pre_sample'), ('available', 'no_pre_sample'))
            self.assertTrue((self.root/'monitor').is_dir())
            manifests = list((self.root/'monitor').glob('*/manifest.json'))
            self.assertEqual(len(manifests), 1, 'monitor session run missing')
            self.assertEqual(json.loads(manifests[0].read_text()).get('identity', {}).get('kind'), 'monitor')
        finally:
            monitoring['owner'].close()
        closed = json.loads((self.root/'monitor-ref.json').read_text())
        self.assertIsNotNone(closed.get('reclaim'), 'monitor close left no reclaim receipt')

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
            from flagprof.report import generate
            result = generate(run)
            out = Path(result['output'])
            self.assertEqual(result['panels'].get('training'), 'ok')
            self.assertTrue((out/'report.html').exists())
            self.assertTrue((out/'report.md').exists())
            document = json.loads((out/'result.json').read_text())
            panel = document['results'].get('training')
            self.assertIsNotNone(panel, 'training panel missing from result report')
            self.assertEqual(panel['status'], 'ok')
            steps = [row['step'] for row in panel['data']]
            self.assertEqual(sorted(steps), [0, 1])


if __name__ == '__main__':
    unittest.main()
