# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Batch driver tests: config contract, command building, CPU battery e2e."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import batch_driver  # noqa: E402


class ConfigTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('yaml'), 'pyyaml required for batch_driver')
    def test_load_local_yaml(self):
        rows = batch_driver.load_config(str(ROOT / 'tools' / 'models_local.yaml'))
        self.assertEqual([r['id'] for r in rows],
                         ['ctr-dlrm', 'ctr-din', 'blocked-example'])
        self.assertEqual(rows[2]['status'], 'blocked')

    @unittest.skipUnless(importlib.util.find_spec('yaml'), 'pyyaml required for batch_driver')
    def test_rejects_duplicate_and_missing(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.yaml'
            p.write_text(yaml.dump({'models': [
                {'id': 'x', 'task': 'ctr'}, {'id': 'x', 'task': 'ctr'}]}))
            with self.assertRaises(ValueError):
                batch_driver.load_config(str(p))
            p.write_text(yaml.dump({'models': [{'id': 'y'}]}))
            with self.assertRaises(ValueError):
                batch_driver.load_config(str(p))


class CmdTests(unittest.TestCase):
    def test_score_digest_both_structures(self):
        import hashlib
        self.assertEqual(batch_driver._score_digest({'score_batches_sha256': 'x'}), 'x')
        flat = {'runs': [{'outputs': [{'sha256': 'a'}, {'sha256': 'b'}]}]}
        self.assertEqual(batch_driver._score_digest(flat),
                         hashlib.sha256(b'a|b').hexdigest())
        wrapped = {'runs': [{'run': {'outputs': [{'sha256': 'a'}]}}]}
        self.assertEqual(batch_driver._score_digest(wrapped),
                         hashlib.sha256(b'a').hexdigest())
        self.assertIsNone(batch_driver._score_digest({'runs': []}))

    def test_flags_passthrough_is_boolean(self):
        row = {'id': 'j', 'task': 'embedding', 'model': 'hf:/m', 'pooling': 'mean',
               'inputs': 'i.jsonl', 'batch_size': 4, 'flags': ['trust-remote-code']}
        cmd, _ = batch_driver.build_cmd(row, 1, 'out')
        self.assertEqual(cmd.count('--trust-remote-code'), 1)
        self.assertEqual(cmd[cmd.index('--trust-remote-code') + 1], '--batch-size')

    def test_ctr_cmd_shape(self):
        row = dict(id='ctr-dlrm', task='ctr', package='/w/packages/ctr-v1',
                   arch='dlrm', seed=531)
        cmd, out = batch_driver.build_cmd(row, 2, '/out', device='cuda:0')
        self.assertEqual(cmd[:4], ['-B', '-m', 'flagprof.runner',
                                   '--task'])
        self.assertIn('--arch', cmd)
        self.assertEqual(cmd[cmd.index('--arch') + 1], 'dlrm')
        self.assertEqual(cmd[cmd.index('--seed') + 1], '531')
        self.assertTrue(out.endswith('ctr-dlrm-run2'))

    def test_rerank_cmd_shape(self):
        row = dict(id='rk', task='rerank', model='hf:/w/models/r',
                   package='/w/packages/p', score='sequence_logit',
                   batch_size=4)
        cmd, _ = batch_driver.build_cmd(row, 1, '/out')
        self.assertEqual(cmd[cmd.index('--score') + 1], 'sequence_logit')
        self.assertEqual(cmd[cmd.index('--batch-size') + 1], '4')
        self.assertIn('hf:/w/models/r', cmd)

    def test_attach_cmd_shape(self):
        row = dict(id='jina', task='attach',
                   target='/w/FlagPerf/inference/targets/target_embedder.py',
                   args={'model': 'hf:/w/models/jina', 'pooling': 'mean',
                         'max-length': 256})
        cmd, out = batch_driver.build_cmd(row, 1, '/out', dtype='bfloat16')
        self.assertEqual(cmd[1], row['target'])
        self.assertEqual(cmd[cmd.index('--pooling') + 1], 'mean')
        self.assertEqual(cmd[-4:], ['--dtype', 'bfloat16', '--outputs', out])


class BatteryE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pkg = Path('/tmp/localtest-ctr-v1')
        if not cls.pkg.exists():
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            subprocess.run(
                [sys.executable, '-B', str(ROOT / 'targets' / 'make_ctr_package.py'),
                 '--samples', '8', '--seed', '531', '--out', str(cls.pkg)],
                check=True, capture_output=True, env=env)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.pkg, ignore_errors=True)

    def test_battery_ctr_cpu_passed(self):
        row = dict(id='ctr-dlrm', task='ctr', package=str(self.pkg),
                   arch='dlrm', seed=531, dtype='float32', forwards=1)
        with tempfile.TemporaryDirectory() as tmp:
            e = batch_driver.run_battery(row, tmp, device='cpu',
                                         runner_cwd=str(ROOT))
            self.assertEqual(e['status'], 'passed', e['runs'])
            self.assertTrue(e['aa_sha_equal'])
            self.assertTrue(e['aa_ident_equal'])
            self.assertEqual(e['oracle_identity']['arch'], 'dlrm')

    def test_battery_marks_failed_and_summary(self):
        row = dict(id='ctr-bad', task='ctr', package=str(self.pkg),
                   arch='no-such-arch', seed=531, dtype='float32', forwards=1)
        with tempfile.TemporaryDirectory() as tmp:
            e = batch_driver.run_battery(row, tmp, device='cpu',
                                         runner_cwd=str(ROOT))
            self.assertEqual(e['status'], 'failed')
            s = batch_driver.summarize([e])
            self.assertEqual((s['failed'], s['passed']), (1, 0))
            md = batch_driver.render_markdown(s, 'T')
            self.assertIn('ctr-bad | ctr | failed', md)


if __name__ == '__main__':
    unittest.main()
