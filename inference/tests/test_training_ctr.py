# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""CTR training task: derived labels, per-arch losses and engine integration."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'targets'))
from flagprof.training.ctr import (TASK_ARCHS, build_loss_fn, label_bit, loss_definition,
                                   prepare_ctr)
from flagprof.training.engine import train
from flagprof.training.state import fingerprint


def build_ctr_package(root, samples=8, seed=531):
    import make_ctr_package
    out = Path(root) / 'ctr-seq'
    argv_backup = sys.argv
    sys.argv = ['make_ctr_package.py', '--samples', str(samples),
                '--seed', str(seed), '--out', str(out)]
    try:
        make_ctr_package.main()
    finally:
        sys.argv = argv_backup
    return out


class CtrLabelTests(unittest.TestCase):
    def test_label_bit_is_deterministic_salted_and_balanced(self):
        ids = [str(i) for i in range(64)]
        click = [label_bit('click', i) for i in ids]
        self.assertEqual(click, [label_bit('click', i) for i in ids])
        self.assertTrue(all(v in (0, 1) for v in click))
        cvr = [label_bit('cvr', i) for i in ids]
        self.assertNotEqual(click, cvr)  # salts give independent streams
        self.assertTrue(16 <= sum(click) <= 48)  # roughly balanced, not degenerate

    def test_loss_definition_covers_every_task_arch(self):
        covered = {arch for arch in TASK_ARCHS if loss_definition(arch)}
        self.assertEqual(covered, set(TASK_ARCHS))
        with self.assertRaisesRegex(ValueError, 'no training loss'):
            loss_definition('widedeep-v3')


@unittest.skipIf(torch is None, 'training semantics require torch')
class CtrTaskTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.package = build_ctr_package(self.root)

    def prepare(self, arch, **options):
        return prepare_ctr(arch=arch, package=str(self.package), seed=531,
                           batch_size=options.pop('batch_size', 4), **options)

    def test_validation_rejects_bad_configuration(self):
        with self.assertRaisesRegex(ValueError, 'unknown CTR architecture'):
            self.prepare('xgboost')
        with self.assertRaisesRegex(ValueError, 'batch-size >= 2'):
            self.prepare('dlrm', batch_size=1)
        with self.assertRaisesRegex(ValueError, 'multiple of batch-size'):
            self.prepare('dlrm', batch_size=3)

    def test_labels_user_idx_and_rank_split(self):
        net, batches, loss_fn, protocol = self.prepare('dlrm')
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0]['click_label'].shape, (4,))
        self.assertTrue(set(batches[0]['click_label'].tolist()) <= {0.0, 1.0})
        self.assertNotIn('cvr_label', batches[0])
        self.assertNotIn('user_idx', batches[0])
        self.assertEqual(batches[0]['dense'].shape, (4, 13))
        _, esmm_batches, _, _ = self.prepare('esmm')
        self.assertIn('cvr_label', esmm_batches[0])
        _, lg_batches, _, _ = self.prepare('lightgcn')
        self.assertEqual(lg_batches[0]['user_idx'].dtype, torch.long)
        self.assertEqual(lg_batches[0]['user_idx'].tolist(), [0, 1, 2, 3])
        left = self.prepare('dlrm', rank=0, world_size=2)[1]
        right = self.prepare('dlrm', rank=1, world_size=2)[1]
        self.assertEqual([len(left), len(right)], [1, 1])
        self.assertFalse(torch.equal(left[0]['dense'], right[0]['dense']))
        with self.assertRaisesRegex(ValueError, 'no microbatches'):
            self.prepare('dlrm', rank=2, world_size=3)
        self.assertEqual(protocol['arch'], 'dlrm')
        self.assertTrue(all(k in protocol for k in ('ctr_package_oracle_id', 'ctr_inputs_sha256',
                                                    'ctr_config_sha256', 'ctr_weight_sha256',
                                                    'labels', 'loss')))
        from flagprof.runner.ctr import build_model, weight_sha256
        reference = build_model('dlrm', 531, 'cpu', package=str(self.package))
        self.assertEqual(protocol['ctr_weight_sha256'], weight_sha256(reference))

    def test_every_arch_loss_is_finite_scalar_with_finite_gradients(self):
        for arch in TASK_ARCHS:
            with self.subTest(arch=arch):
                net, batches, loss_fn, _ = self.prepare(arch)
                net.train()
                loss = loss_fn(net, batches[0])
                self.assertEqual(loss.ndim, 0)
                self.assertTrue(bool(torch.isfinite(loss)))
                net.zero_grad(set_to_none=True)
                loss.backward()
                grads = [p.grad for p in net.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))

    def test_logit_loss_matches_manual_bce_with_logits(self):
        net, batches, loss_fn, _ = self.prepare('dlrm')
        inputs = {k: v for k, v in batches[0].items() if not k.endswith('_label')}
        with torch.no_grad():
            expected = torch.nn.functional.binary_cross_entropy_with_logits(
                net(**inputs).reshape(-1), batches[0]['click_label'])
        self.assertAlmostEqual(float(loss_fn(net, batches[0])), float(expected), places=6)

    def test_multi_task_loss_consumes_both_labels(self):
        net, batches, loss_fn, _ = self.prepare('esmm')
        base = loss_fn(net, batches[0])
        flipped = dict(batches[0])
        flipped['click_label'] = 1 - flipped['click_label']
        self.assertNotAlmostEqual(float(base), float(loss_fn(net, flipped)))

    def test_engine_matches_uninstrumented_reference(self):
        net, batches, loss_fn, protocol = self.prepare('deepfm')
        reference = copy.deepcopy(net)
        reference.train()  # the engine trains in train mode; mirror the protocol
        optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        losses = []
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(reference, batch)
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        result = train(net, torch.optim.SGD(net.parameters(), lr=.01), batches, loss_fn,
                       outputs=self.root/'observed', steps=2,
                       protocol=dict(protocol), device='cpu')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(fingerprint(reference.state_dict()), fingerprint(net.state_dict()))
        self.assertEqual(losses, [row['loss'] for row in result['steps']])
        self.assertNotEqual(result['protocol']['initial_model_sha256'],
                            result['final_model_sha256'])

    def test_compare_two_identical_ctr_runs_pass(self):
        for name in ('left', 'right'):
            net, batches, loss_fn, protocol = self.prepare('dien')
            train(net, torch.optim.AdamW(net.parameters(), lr=.01), batches, loss_fn,
                  outputs=self.root/name, steps=2, protocol=dict(protocol), device='cpu')
        from flagprof.training.compare import compare
        result = compare(self.root/'left', self.root/'right')
        self.assertEqual(result['status'], 'pass')

    def test_cli_accepts_ctr_options(self):
        from flagprof.training.__main__ import parser
        arguments = parser().parse_args(['--task', 'ctr', '--ctr-arch', 'dlrm',
                                         '--package', str(self.package),
                                         '--outputs', str(self.root/'cli')])
        self.assertEqual(arguments.task, 'ctr')
        self.assertEqual(arguments.ctr_arch, 'dlrm')
        self.assertEqual(arguments.package, str(self.package))


if __name__ == '__main__':
    unittest.main()
