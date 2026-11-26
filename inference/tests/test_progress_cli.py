# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Workflow progress stays outside machine output, including nested producers."""
import json
import os
from contextlib import contextmanager
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from flagprof.workflow.executor import Execution


class WorkflowProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root/'simple.json'

    def command(self, flags, program):
        self.config.write_text(json.dumps({
            'schema_version': 1, 'profile': {'preset': 'none'},
            'target': {'type': 'command', 'command': [sys.executable, '-c', program]},
        }))
        return [sys.executable, '-B', '-m', 'flagprof', 'run', str(self.config),
                '--out', str(self.root/'job'), *flags]

    def environment(self):
        return dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))

    def invoke(self, *flags, program='print("target stdout")'):
        return subprocess.run(self.command(flags, program), cwd=self.root, env=self.environment(),
                              capture_output=True, text=True, timeout=30)

    def test_plain_child_phase_reaches_parent_without_corrupting_stdout(self):
        script = ("from flagprof.progress import phase\n"
                  "with phase('child-work'):\n    print('target stdout')")
        result = self.invoke('--progress', 'plain', '--json', program=script)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary['exit_code'], 0)
        step = summary['steps'][0]
        self.assertIn('child-work', result.stderr)
        self.assertIn(step['log'], result.stderr)
        self.assertIn('工作流', result.stderr)
        self.assertEqual(Path(step['log']).read_text().strip(), 'target stdout')
        self.assertNotIn('child-work', Path(step['stderr']).read_text())
        self.assertNotIn('[运行]', result.stdout)

    def test_preview_and_machine_defaults_stay_quiet(self):
        for flags in (('--json', '--check'), ('--json', '--dry-run', '--progress', 'plain'),
                      ('--json', '--check', '--progress', 'off')):
            with self.subTest(flags=flags):
                result = self.invoke(*flags)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)['status'], 'ready')
                self.assertEqual(result.stderr, '')
                self.assertFalse((self.root/'job').exists())
        result = self.invoke('--json')
        self.assertEqual(result.stderr, '')
        self.assertEqual(json.loads(result.stdout)['status'], 'complete')

    def test_nonzero_exit_retains_command_log_and_receipt(self):
        result = self.invoke('--json', '--progress', 'plain', program='raise SystemExit(17)')
        self.assertEqual(result.returncode, 17)
        summary = json.loads(result.stdout)
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['steps'][0]['exit_code'], 17)
        self.assertTrue(Path(summary['steps'][0]['stderr']).is_file())
        self.assertEqual(json.loads((self.root/'job/summary.json').read_text()), summary)

    @unittest.skipUnless(hasattr(os, 'killpg'), 'process-group interruption requires POSIX')
    def test_terminal_interrupt_reaps_target_and_keeps_summary(self):
        marker = self.root/'target.pid'
        program = ("import os, time\nfrom pathlib import Path\n"
                   "Path('target.pid').write_text(str(os.getpid()))\ntime.sleep(60)")
        with subprocess.Popen(self.command(['--progress', 'plain', '--json'], program),
                              cwd=self.root, env=self.environment(), start_new_session=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as cli:
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and cli.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(marker.exists())
                os.killpg(cli.pid, signal.SIGINT)
                stdout, stderr = cli.communicate(timeout=10)
            finally:
                if cli.poll() is None:
                    os.killpg(cli.pid, signal.SIGKILL)
            self.assertEqual(cli.returncode, 130, stderr)
        self.assertEqual(json.loads(stdout)['exit_code'], 130)
        self.assertEqual(json.loads((self.root/'job/summary.json').read_text())['status'], 'failed')
        with self.assertRaises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)

    def test_dynamic_rank_steps_expand_denominator_before_work(self):
        step = dict(id='compare', action='compare', variant='comparison')
        plan = dict(steps=[step], directory=str(self.root), runners=[], skipped={},
                    output_type='report', component_owner='existing', vendor=None)
        state = Execution({'mode':'single'}, self.config, {}, self.root, plan)
        tracker = Mock()
        rank_steps = []
        @contextmanager
        def progress(name, **options):
            if name == '工作流':
                self.assertEqual(options['total'], 1)
                yield tracker
            else:
                rank_steps.append((options['rank'], options['variant']))
                yield Mock()
        actual = [dict(step, rank=i, id=f'compare-rank-{i}') for i in (0,1)]
        with patch('flagprof.workflow.executor.phase', progress), \
                patch('flagprof.workflow.executor.comparison_steps', return_value=actual), \
                patch.object(state, 'initialize'), patch.object(state, 'command', side_effect=[0,17]):
            self.assertEqual(state.run({})['exit_code'], 17)
        tracker.update.assert_any_call(0,total=2)
        tracker.update.assert_any_call(1)
        self.assertNotIn(((2,),{}), tracker.update.call_args_list)
        self.assertEqual(rank_steps, [(0,'comparison'),(1,'comparison')])
