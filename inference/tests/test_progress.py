# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""User-visible progress behaviour at an interactive terminal and in parallel jobs."""
from contextlib import ExitStack, redirect_stderr
import io
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from flagprof.progress import finish_display, phase, session


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTests(unittest.TestCase):
    def test_auto_uses_tty_and_preserves_partial_count_on_interrupt(self):
        terminal = Terminal()
        with redirect_stderr(terminal), self.assertRaises(KeyboardInterrupt):
            with session(), phase('requests', total=10, unit='items') as step:
                step.update(3)
                raise KeyboardInterrupt
        output = terminal.getvalue()
        self.assertIn('\r', output)
        self.assertIn('3/10 items (30%)', output)
        self.assertNotIn('100%', output)
        self.assertIn('退出 130', output)

    def test_unknown_work_shows_elapsed_without_percent(self):
        log = Terminal()
        with redirect_stderr(log), session('plain'), phase('vendor'):
            pass
        self.assertNotIn('\r', log.getvalue())
        self.assertNotIn('%', log.getvalue())
        self.assertIn('阶段', log.getvalue())
        self.assertIn('总计', log.getvalue())

    def test_active_line_is_cleared_before_cli_prints_its_receipt(self):
        terminal = Terminal()
        with redirect_stderr(terminal), session(), phase('import'):
            finish_display()
            print('receipt', file=terminal)
        self.assertIn('\r\033[2Kreceipt\n', terminal.getvalue())

    def test_off_in_another_thread_is_independent(self):
        def quiet_job():
            with session('off'), phase('silent-job'):
                pass
        log = io.StringIO()
        with redirect_stderr(log), session('plain'), phase('visible-job'):
            worker = threading.Thread(target=quiet_job)
            worker.start()
            worker.join()
        self.assertIn('visible-job', log.getvalue())
        self.assertNotIn('silent-job', log.getvalue())

    def test_concurrent_processes_emit_before_exit_and_clean_up(self):
        script = """from flagprof.progress import phase, session
import sys
with session('plain'), phase(sys.argv[1], total=5, unit='items') as step:
    step.update(2)
    input()
"""
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            env = dict(os.environ, TMPDIR=temp, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
            jobs = {}
            for label in ('first-job', 'second-job'):
                job = stack.enter_context(subprocess.Popen(
                    [sys.executable, '-B', '-c', script, label], env=env,
                    stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE))
                stack.callback(job.kill)
                jobs[label] = job
            for label, job in jobs.items():
                live = b''
                deadline = time.monotonic() + 8
                while b'2/5 items' not in live:
                    timeout = max(0, deadline - time.monotonic())
                    self.assertTrue(select.select([job.stderr], [], [], timeout)[0], live)
                    data = os.read(job.stderr.fileno(), 4096)
                    self.assertTrue(data, live)
                    live += data
                self.assertIsNone(job.poll())
                stdout, remaining = job.communicate(input=b'\n', timeout=8)
                self.assertEqual(job.returncode, 0, remaining)
                self.assertEqual(stdout, b'')
                self.assertIn(label.encode(), live)
                other = next(k for k in jobs if k != label)
                self.assertNotIn(other.encode(), live + remaining)
            self.assertEqual(list(Path(temp).glob('flagprof-progress-*')), [])
