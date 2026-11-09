# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CLI-level tests for flagprof.runner __main__ arg contract (subprocess)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _build_package(tmp):
    pkg = Path(tmp) / 'ctr'
    subprocess.run([sys.executable, '-B', str(ROOT / 'targets' / 'make_ctr_package.py'),
                    '--samples', '8', '--seed', '531', '--out', str(pkg)],
                   check=True, capture_output=True)
    return pkg


class RunnerCliTests(unittest.TestCase):
    def _run(self, argv):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        return subprocess.run([sys.executable, '-B', '-m', 'flagprof.runner'] + argv,
                              capture_output=True, text=True, env=env, cwd=str(ROOT))

    def test_ctr_task_runs_without_model_positional(self):
        # regression: model positional was required -> argparse killed --task ctr
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _build_package(tmp)
            out = Path(tmp) / 'out'
            proc = self._run(['--task', 'ctr', '--package', str(pkg),
                              '--arch', 'dlrm', '--seed', '531', '--device', 'cpu',
                              '--dtype', 'float32', '--forwards', '1',
                              '--outputs', str(out)])
            self.assertEqual(proc.returncode, 0, proc.stderr[-600:])
            doc = json.loads((out / 'runner.json').read_text())
            self.assertEqual(doc['task'], 'ctr')

    def test_ctr_task_rejects_model_positional(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run(['--task', 'ctr', 'some-model-dir', '--package', 'p',
                              '--arch', 'dlrm', '--outputs', str(Path(tmp) / 'o')])
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('takes no model', proc.stderr)

    def test_embedding_task_errors_cleanly_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run(['--task', 'embedding', '--inputs', 'x.jsonl',
                              '--outputs', str(Path(tmp) / 'o')])
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('hf:<directory>', proc.stderr)


if __name__ == '__main__':
    unittest.main()
