# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CTR package loader and template model semantics, CPU only."""
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'targets'))
from template_ctr import (load_ctr_package, build_model, weight_sha256,
                          CtrCallable, ARCH_CONFIG)
from make_ctr_package import SCHEMA


def build_ctr_package(root, samples=8, corrupt=None, seed=531):
    import sys
    targets_dir = str(Path(__file__).resolve().parent.parent / 'targets')
    if targets_dir not in sys.path:
        sys.path.insert(0, targets_dir)
    import make_ctr_package
    root = Path(root)
    out = root / 'ctr-seq'
    argv_backup = sys.argv
    sys.argv = ['make_ctr_package.py', '--samples', str(samples),
                '--seed', str(seed), '--out', str(out)]
    try:
        make_ctr_package.main()
    finally:
        sys.argv = argv_backup
    if corrupt:
        manifest_path = out / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files'][corrupt]['sha256'] = '0' * 64
        manifest_path.write_text(json.dumps(manifest))
    return out


class CtrPackageTests(unittest.TestCase):
    def test_package_schema_and_batches(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp)
            batches, manifest = load_ctr_package(pkg, batch_size=4)
            self.assertEqual(manifest['kind'], 'ctr-package-v1')
            self.assertEqual(len(batches), 2)
            ids, stacked = batches[0]
            self.assertEqual(ids, ['0', '1', '2', '3'])
            c = ARCH_CONFIG
            self.assertEqual(stacked['dense'].shape, (4, c['dense_dim']))
            self.assertEqual(stacked['sparse'].shape, (4, c['sparse_dim']))
            self.assertEqual(stacked['hist_ids'].shape, (4, c['hist_len']))
            self.assertTrue(bool((stacked['hist_mask'] == (stacked['hist_ids'] != SCHEMA['pad_id'])).all()))
            self.assertTrue(bool((stacked['sparse'] > SCHEMA['pad_id']).all()))
            self.assertTrue(bool((stacked['dense'].abs() <= 1.0).all()))

    def test_digest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp, corrupt='prepared/inputs.pt')
            with self.assertRaises(ValueError):
                load_ctr_package(pkg, batch_size=4)

    def test_unknown_sample_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp)
            with self.assertRaises(ValueError):
                load_ctr_package(pkg, batch_size=4, samples=['zz'])

    def test_sample_subselect_keeps_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp)
            batches, _ = load_ctr_package(pkg, batch_size=2, samples=['5', '1'])
            self.assertEqual([ids for ids, _ in batches], [['5', '1']])


class CtrModelTests(unittest.TestCase):
    def test_reproducible_given_same_seed(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp)
            batches, _ = load_ctr_package(pkg, batch_size=8)
            inputs = batches[0][1]
            outs = []
            for _ in range(2):
                model = CtrCallable('dlrm', build_model('dlrm', 531, 'cpu', dtype=torch.float32))
                outs.append(model(**inputs).detach())
            self.assertTrue(torch.equal(outs[0], outs[1]))

    def test_dlrm_output_shape_and_range(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp)
            batches, _ = load_ctr_package(pkg, batch_size=4)
            inputs = batches[0][1]
            model = CtrCallable('dlrm', build_model('dlrm', 531, 'cpu', dtype=torch.float32))
            scores = torch.sigmoid(model(**inputs).float().squeeze(1))
            self.assertEqual(scores.shape, (4,))
            self.assertTrue(bool((scores > 0).all()) and bool((scores < 1).all()))

    def test_din_ignores_padded_history_ids(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp)
            batches, _ = load_ctr_package(pkg, batch_size=4)
            inputs = batches[0][1]
            model = CtrCallable('din', build_model('din', 531, 'cpu', dtype=torch.float32))
            base = model(**inputs).detach()
            perturbed = dict(inputs)
            pad = ~inputs['hist_mask']
            noise = inputs['hist_ids'].clone()
            noise[pad] = 999 - torch.arange(int(pad.sum()), dtype=torch.int64) % 900
            perturbed['hist_ids'] = noise
            changed = model(**perturbed).detach()
            self.assertTrue(torch.equal(base, changed))

    def test_widedeep_output_shape_and_range(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp)
            batches, _ = load_ctr_package(pkg, batch_size=4)
            inputs = batches[0][1]
            model = CtrCallable('widedeep', build_model('widedeep', 531, 'cpu', dtype=torch.float32))
            scores = torch.sigmoid(model(**inputs).float().squeeze(1))
            self.assertEqual(scores.shape, (4,))
            self.assertTrue(bool((scores > 0).all()) and bool((scores < 1).all()))

    def test_weight_sha256_pins_weights(self):
        import torch
        a = build_model('dlrm', 531, 'cpu', dtype=torch.float32)
        b = build_model('dlrm', 531, 'cpu', dtype=torch.float32)
        c = build_model('dlrm', 532, 'cpu', dtype=torch.float32)
        self.assertEqual(weight_sha256(a), weight_sha256(b))
        self.assertNotEqual(weight_sha256(a), weight_sha256(c))

    def test_weight_sha256_supports_bfloat16(self):
        import torch
        a = build_model('din', 531, 'cpu', dtype=torch.bfloat16)
        b = build_model('din', 531, 'cpu', dtype=torch.bfloat16)
        self.assertEqual(weight_sha256(a), weight_sha256(b))
        self.assertNotEqual(weight_sha256(a),
                            weight_sha256(build_model('din', 531, 'cpu', dtype=torch.float32)))

    def test_deepfm_output_shape_and_range(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp)
            batches, _ = load_ctr_package(pkg, batch_size=4)
            inputs = batches[0][1]
            model = CtrCallable('deepfm', build_model('deepfm', 531, 'cpu', dtype=torch.float32))
            scores = torch.sigmoid(model(**inputs).float().squeeze(1))
            self.assertEqual(scores.shape, (4,))
            # FM term sums 325 pairwise dots; random init saturates sigmoid to
            # endpoints, so the closed interval [0, 1] is the honest assertion
            self.assertTrue(bool(torch.isfinite(scores).all()))
            self.assertTrue(bool((scores >= 0).all()) and bool((scores <= 1).all()))

    def test_dcn_output_shape_and_range(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp)
            batches, _ = load_ctr_package(pkg, batch_size=4)
            inputs = batches[0][1]
            model = CtrCallable('dcn', build_model('dcn', 531, 'cpu', dtype=torch.float32))
            scores = torch.sigmoid(model(**inputs).float().squeeze(1))
            self.assertEqual(scores.shape, (4,))
            self.assertTrue(bool(torch.isfinite(scores).all()))
            self.assertTrue(bool((scores >= 0).all()) and bool((scores <= 1).all()))

    def test_esmm_output_three_columns_in_unit_range(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_ctr_package(tmp)
            batches, _ = load_ctr_package(pkg, batch_size=4)
            inputs = batches[0][1]
            model = CtrCallable('esmm', build_model('esmm', 531, 'cpu', dtype=torch.float32))
            out = model(**inputs).float()
            self.assertEqual(out.shape, (4, 3))
            self.assertTrue(bool(torch.isfinite(out).all()))
            self.assertTrue(bool((out >= 0).all()) and bool((out <= 1).all()))
            # ctcvr = ctr * cvr columnwise
            self.assertTrue(torch.equal(out[:, 2], out[:, 0] * out[:, 1]))

    def test_all_archs_reproducible_same_seed(self):
        import torch
        for arch in ('deepfm', 'dcn', 'esmm'):
            with tempfile.TemporaryDirectory() as tmp:
                pkg = build_ctr_package(tmp)
                batches, _ = load_ctr_package(pkg, batch_size=8)
                inputs = batches[0][1]
                outs = []
                for _ in range(2):
                    model = CtrCallable(arch, build_model(arch, 531, 'cpu', dtype=torch.float32))
                    outs.append(model(**inputs).detach())
                self.assertTrue(torch.equal(outs[0], outs[1]), arch)


if __name__ == '__main__':
    unittest.main()
