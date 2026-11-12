# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Embedding training task: corpus, pooling, InfoNCE loss and engine integration."""
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'targets'))
from flagprof.training.embedding import (POOLS, loss_definition,
                                         pool_description, prepare_embedding)
from flagprof.training.engine import train
from flagprof.training.state import fingerprint


def build_corpus(root, groups=6, negs=2, seed=531):
    import make_embedding_corpus
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = root / 'groups.jsonl'
    argv_backup = sys.argv
    sys.argv = ['make_embedding_corpus.py', '--groups', str(groups), '--negs', str(negs),
                '--seed', str(seed), '--out', str(out)]
    try:
        make_embedding_corpus.main()
    finally:
        sys.argv = argv_backup
    return out


if torch is not None:
    class TinyEncoder(torch.nn.Module):
        def __init__(self, vocab=2048, dim=8):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, dim)

        def forward(self, input_ids, attention_mask):
            return types.SimpleNamespace(last_hidden_state=self.emb(input_ids))


class EmbeddingContractTests(unittest.TestCase):
    def test_pool_and_loss_definitions(self):
        for pool in POOLS:
            self.assertTrue(pool_description(pool))
            self.assertTrue(loss_definition(pool, .05))
        with self.assertRaisesRegex(ValueError, 'unknown pooling'):
            pool_description('max')

    def test_corpus_is_deterministic_and_well_formed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = build_corpus(root / 'a')
            second = build_corpus(root / 'b')
            self.assertEqual(first.read_text(), second.read_text())
            rows = [json.loads(line) for line in first.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(len(row['negs']) == 2 for row in rows))
            self.assertTrue(all(len(row['anchor']) >= 1 and len(row['pos']) >= 1 for row in rows))
            self.assertTrue(all(all(i >= 1 for i in seq) for row in rows
                                for seq in [row['anchor'], row['pos'], *row['negs']]))
            third = build_corpus(root / 'c', seed=532)
            self.assertNotEqual(first.read_text(), third.read_text())


@unittest.skipIf(torch is None, 'training semantics require torch')
class EmbeddingTaskTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.corpus = build_corpus(self.root)

    def prepare(self, pooling='mean', **options):
        net = TinyEncoder()
        return prepare_embedding(model=net, inputs=str(self.corpus), pooling=pooling,
                                 temperature=.05,
                                 batch_size=options.pop('batch_size', 3), **options)

    def test_validation_rejects_bad_configuration(self):
        with self.assertRaisesRegex(ValueError, 'pooling must be one of'):
            self.prepare(pooling='max')
        with self.assertRaisesRegex(ValueError, 'multiple of batch-size'):
            self.prepare(batch_size=4)
        bad = self.root / 'bad.jsonl'
        bad.write_text(json.dumps(dict(anchor=[1], pos=[2], negs=[[3]])) + '\n'
                       + json.dumps(dict(anchor=[1], pos=[2], negs=[[3], [4]])) + '\n')
        with self.assertRaisesRegex(ValueError, 'same negative count'):
            prepare_embedding(model=TinyEncoder(), inputs=str(bad), pooling='mean',
                              temperature=.05, batch_size=1)

    def test_batches_pad_per_microbatch_and_rank_split_interleaves(self):
        net, batches, loss_fn, protocol = self.prepare()
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0]['anchor_ids'].shape, (3, batches[0]['anchor_ids'].shape[1]))
        self.assertEqual(batches[0]['cand_ids'].shape[0], 3 * 3)  # pos + 2 negs per group
        self.assertTrue(bool((batches[0]['anchor_mask'].sum(1) >= 1).all()))
        self.assertEqual(protocol['pooling'], 'mean')
        self.assertIn('embedding-loss/2', protocol['loss_precision'])
        self.assertEqual(protocol['groups_per_rank'], 6)
        self.assertIn('loss', protocol)
        left = self.prepare(rank=0, world_size=2)[1]
        right = self.prepare(rank=1, world_size=2)[1]
        self.assertEqual([len(left), len(right)], [1, 1])
        self.assertFalse(torch.equal(left[0]['anchor_ids'], right[0]['anchor_ids']))
        with self.assertRaisesRegex(ValueError, 'no groups'):
            self.prepare(rank=6, world_size=7)

    def test_info_nce_loss_matches_manual_reference(self):
        net, batches, loss_fn, _ = self.prepare()
        net.train()
        batch = batches[0]

        def masked_mean(hidden, mask):
            weight = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weight).sum(1) / weight.sum(1)

        anchor = torch.nn.functional.normalize(
            masked_mean(net(input_ids=batch['anchor_ids'],
                            attention_mask=batch['anchor_mask']).last_hidden_state,
                        batch['anchor_mask']), dim=-1)
        cand = torch.nn.functional.normalize(
            masked_mean(net(input_ids=batch['cand_ids'],
                            attention_mask=batch['cand_mask']).last_hidden_state,
                        batch['cand_mask']), dim=-1)
        stride = cand.shape[0] // anchor.shape[0]
        expected = torch.nn.functional.cross_entropy(
            anchor @ cand.t() / .05, torch.arange(anchor.shape[0]) * stride)
        self.assertAlmostEqual(float(loss_fn(net, batch)), float(expected), places=5)

    def test_pooling_variants_change_the_loss(self):
        losses = {}
        for pool in POOLS:
            net, batches, loss_fn, _ = self.prepare(pooling=pool)
            net.train()
            losses[pool] = float(loss_fn(net, batches[0]))
            self.assertTrue(torch.isfinite(torch.tensor(losses[pool])))
        self.assertEqual(len(set(losses.values())), len(POOLS))

    def test_engine_matches_uninstrumented_reference(self):
        net, batches, loss_fn, protocol = self.prepare()
        reference = copy.deepcopy(net)
        reference.train()
        optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        for batch in batches[:2]:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(reference, batch).backward()
            optimizer.step()
        result = train(net, torch.optim.SGD(net.parameters(), lr=.01), batches, loss_fn,
                       outputs=self.root/'observed', steps=2,
                       protocol=dict(protocol), device='cpu')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(fingerprint(reference.state_dict()), fingerprint(net.state_dict()))

    def test_cli_accepts_embedding_options(self):
        from flagprof.training.__main__ import parser
        arguments = parser().parse_args(['--task', 'embedding', '--model', '/models/emb',
                                         '--inputs', str(self.corpus), '--pooling', 'last',
                                         '--temperature', '0.07', '--outputs', str(self.root/'cli')])
        self.assertEqual(arguments.task, 'embedding')
        self.assertEqual(arguments.pooling, 'last')
        self.assertEqual(arguments.temperature, 0.07)


if __name__ == '__main__':
    unittest.main()
