# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Runner promotion (rerank/ctr tasks) CPU tests: canonical module contract,
shim equivalence and attach/runner structural parity."""
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'targets'))

from flagprof.runner import ctr as ctr_mod
from flagprof.runner import rerank as rerank_mod
from flagprof.runner.rerank_common import load_package as canonical_load_package
import rerank_common  # targets shim


class RunnerCtrTests(unittest.TestCase):
    def _build_package(self, tmp):
        targets_dir = str(Path(__file__).resolve().parent.parent / 'targets')
        if targets_dir not in sys.path:
            sys.path.insert(0, targets_dir)
        import make_ctr_package
        out = Path(tmp) / 'ctr'
        argv_backup = sys.argv
        sys.argv = ['make_ctr_package.py', '--samples', '8', '--seed', '531', '--out', str(out)]
        try:
            make_ctr_package.main()
        finally:
            sys.argv = argv_backup
        return out

    def test_run_ctr_cpu_deterministic_and_identity(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = self._build_package(tmp)
            kw = dict(package=str(pkg), arch='dlrm', seed=531, device='cpu',
                      dtype='float32', forwards=1)
            a = ctr_mod.run_ctr(outputs=str(Path(tmp) / 'a'), **kw)
            b = ctr_mod.run_ctr(outputs=str(Path(tmp) / 'b'), **kw)
            self.assertEqual(a['score_batches_sha256'], b['score_batches_sha256'])
            self.assertEqual(a['oracle_identity'], b['oracle_identity'])
            ident = a['oracle_identity']
            self.assertEqual(ident['kind'], 'ctr-template-v1')
            self.assertEqual(ident['arch'], 'dlrm')
            self.assertEqual(len(ident['weight_sha256']), 64)
            self.assertEqual(ident['seed'], 531)
            self.assertEqual(len(a['runs']), 2)  # 8 samples / batch 4
            self.assertIn('samples', a['runs'][0])
            self.assertIn('outputs', a['runs'][0]['run'])

    def test_run_ctr_rejects_arch_outside_contract(self):
        targets_dir = str(Path(__file__).resolve().parent.parent / 'targets')
        if targets_dir not in sys.path:
            sys.path.insert(0, targets_dir)
        import make_ctr_package
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'ctr'
            argv_backup = sys.argv
            sys.argv = ['make_ctr_package.py', '--samples', '8', '--seed', '531',
                        '--archs', 'dlrm', '--out', str(out)]
            try:
                make_ctr_package.main()
            finally:
                sys.argv = argv_backup
            with self.assertRaises(ValueError):
                ctr_mod.run_ctr(package=str(out), arch='din', seed=531, device='cpu',
                                dtype='float32', forwards=1,
                                outputs=str(Path(tmp) / 'x'))

    def test_attach_main_exposed_for_thin_target(self):
        self.assertTrue(callable(ctr_mod.attach_main))
        self.assertEqual(ctr_mod.ARCH_CHOICES,
                         ['dlrm', 'din', 'widedeep', 'deepfm', 'dcn', 'esmm',
                          'mmoe', 'ple', 'bst', 'dien'])

    def test_thin_targets_delegate_to_canonical(self):
        import template_ctr  # targets shim module
        import target_reranker  # targets shim module
        from flagprof.runner.rerank_common import sha as canonical_sha
        self.assertIs(template_ctr.attach_main, ctr_mod.attach_main)
        self.assertIs(target_reranker.attach_main, rerank_mod.attach_main)
        self.assertIs(rerank_common.load_package, canonical_load_package)
        self.assertIs(rerank_common.sha, canonical_sha)


class RunnerRerankTests(unittest.TestCase):
    def _build_package(self, tmp):
        import torch
        from rerank_common import sha
        root = Path(tmp) / 'rk'
        (root / 'prepared').mkdir(parents=True)
        rows = [dict(sample_id='a', input_ids=torch.tensor([1, 2, 3], dtype=torch.int64)),
                dict(sample_id='b', input_ids=torch.tensor([4, 5], dtype=torch.int64))]
        torch.save(rows, root / 'prepared' / 'inputs.pt')
        (root / 'prepared' / 'samples.json').write_text(json.dumps(dict(samples=['a', 'b'])))
        manifest = dict(files={
            'prepared/inputs.pt': dict(sha256=sha(root / 'prepared' / 'inputs.pt')),
            'prepared/samples.json': dict(sha256=sha(root / 'prepared' / 'samples.json')),
        }, oracle_id='t', score_semantics='sequence_logit', template_sha='',
            contract=dict(batch_size=2, max_length=64))
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return root

    def test_canonical_loader_matches_targets_shim(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = self._build_package(tmp)
            a, ma = canonical_load_package(str(pkg), batch_size=2)
            b, mb = rerank_common.load_package(str(pkg), batch_size=2)
            self.assertEqual(ma, mb)
            for (ids_a, ta), (ids_b, tb) in zip(a, b):
                self.assertEqual(ids_a, ids_b)
                self.assertTrue(torch.equal(ta['input_ids'], tb['input_ids']))
                self.assertTrue(torch.equal(ta['attention_mask'], tb['attention_mask']))

    def test_run_rerank_rejects_semantics_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = self._build_package(tmp)
            _, manifest = rerank_common.load_package(str(pkg))
            # package declares sequence_logit; causal request must be a hard error
            with self.assertRaises(ValueError):
                rerank_mod.validate_package_semantics(manifest, 'causal_yesno')
            rerank_mod.validate_package_semantics(manifest, 'sequence_logit')  # no raise
        self.assertTrue(callable(rerank_mod.run_rerank))
        self.assertTrue(callable(rerank_mod.attach_main))

    def test_model_files_identity_requires_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                rerank_mod.model_files_identity(tmp)


if __name__ == '__main__':
    unittest.main()
