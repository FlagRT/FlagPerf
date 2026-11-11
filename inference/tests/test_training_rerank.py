# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Rerank training task: derived relevance bits, both scoring semantics, engine integration."""
import copy
import hashlib
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flagprof.training.rerank import build_loss_fn, loss_definition, prepare_rerank, relevance_bit
from flagprof.training.engine import train
from flagprof.training.state import fingerprint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_package(root, rows=8, score='causal_yesno'):
    """Minimal sealed rerank-package-v1 with the same layout the runner loader verifies."""
    import torch
    package = Path(root) / 'rerank-seq'
    (package / 'prepared').mkdir(parents=True, exist_ok=True)
    prepared = [dict(sample_id='r%d' % index,
                     input_ids=torch.tensor([3 + index % 7, 5 + index % 5, 7, 2 + index % 3]),
                     ) for index in range(rows)]
    torch.save(prepared, package / 'prepared' / 'inputs.pt')
    (package / 'prepared' / 'samples.json').write_text(json.dumps(
        [row['sample_id'] for row in prepared]))
    manifest = dict(oracle_id='test-oracle', score_semantics=score,
                    template_sha='t' * 16, contract=dict(batch_size=4, score=score),
                    files={'prepared/inputs.pt': {'sha256': sha(package / 'prepared' / 'inputs.pt')},
                           'prepared/samples.json': {'sha256': sha(package / 'prepared' / 'samples.json')}})
    (package / 'manifest.json').write_text(json.dumps(manifest))
    return package


if torch is not None:
    class TinyCausalLM(torch.nn.Module):
        def __init__(self, vocab=2048, dim=8):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, dim)
            self.head = torch.nn.Linear(dim, vocab)

        def forward(self, input_ids, attention_mask):
            return types.SimpleNamespace(logits=self.head(self.emb(input_ids)))

    class TinySeqCls(torch.nn.Module):
        def __init__(self, vocab=2048, dim=8):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, dim)
            self.head = torch.nn.Linear(dim, 1)

        def forward(self, input_ids, attention_mask):
            return types.SimpleNamespace(logits=self.head(self.emb(input_ids)[:, 0]))


class RerankContractTests(unittest.TestCase):
    def test_relevance_bit_is_deterministic_salted_and_balanced(self):
        ids = ['r%d' % i for i in range(64)]
        bits = [relevance_bit(i) for i in ids]
        self.assertEqual(bits, [relevance_bit(i) for i in ids])
        self.assertTrue(all(v in (0, 1) for v in bits))
        self.assertTrue(16 <= sum(bits) <= 48)
        self.assertEqual(loss_definition('causal_yesno').find('no, yes') > 0, True)
        self.assertTrue(loss_definition('sequence_logit'))

    def test_prepare_rejects_bad_score_without_loader_side_effects(self):
        with self.assertRaisesRegex(ValueError, 'score must be one of'):
            prepare_rerank(model=object(), package='nowhere', score='pairwise',
                           batch_size=2)


@unittest.skipIf(torch is None, 'training semantics require torch')
class RerankTaskTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def prepare(self, score='causal_yesno', model=None, **options):
        if model is None:
            model = TinyCausalLM() if score == 'causal_yesno' else TinySeqCls()
        return prepare_rerank(model=model, package=str(build_package(self.root, score=score)),
                              score=score, batch_size=options.pop('batch_size', 4),
                              yes_id=options.pop('yes_id', 30), no_id=options.pop('no_id', 31),
                              **options)

    def test_targets_ranks_and_protocol(self):
        net, batches, loss_fn, protocol = self.prepare()
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0]['rerank_target'].shape, (4,))
        self.assertTrue(set(batches[0]['rerank_target'].tolist()) <= {0.0, 1.0})
        left = self.prepare(rank=0, world_size=2)[1]
        right = self.prepare(rank=1, world_size=2)[1]
        self.assertEqual([len(left), len(right)], [1, 1])
        self.assertFalse(torch.equal(left[0]['input_ids'], right[0]['input_ids']))
        with self.assertRaisesRegex(ValueError, 'no microbatches'):
            self.prepare(rank=2, world_size=3)
        self.assertEqual(protocol['score_semantics'], 'causal_yesno')
        self.assertTrue(all(k in protocol for k in ('rerank_package_oracle_id', 'rerank_inputs_sha256',
                                                    'rerank_weight_sha256', 'labels', 'loss')))
        from flagprof.runner.ctr import weight_sha256
        self.assertEqual(protocol['rerank_weight_sha256'], weight_sha256(net))

    def test_causal_yesno_loss_matches_manual(self):
        net, batches, loss_fn, _ = self.prepare()
        net.train()
        batch = batches[0]
        logits = net(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        mask = batch['attention_mask']
        positions = torch.arange(logits.shape[1]).expand_as(mask)
        last = positions.masked_fill(mask == 0, -1).max(dim=1).values
        picked = logits[torch.arange(logits.shape[0]), last][:, [31, 30]]
        expected = torch.nn.functional.cross_entropy(picked, batch['rerank_target'].long())
        self.assertAlmostEqual(float(loss_fn(net, batch)), float(expected), places=5)

    def test_sequence_logit_loss_matches_manual(self):
        net, batches, loss_fn, protocol = self.prepare(score='sequence_logit')
        net.train()
        batch = batches[0]
        logits = net(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0], batch['rerank_target'])
        self.assertEqual(protocol['score_semantics'], 'sequence_logit')
        self.assertAlmostEqual(float(loss_fn(net, batch)), float(expected), places=5)

    def test_causal_yesno_requires_token_ids(self):
        with self.assertRaisesRegex(ValueError, 'yes/no token ids'):
            prepare_rerank(model=TinyCausalLM(), package=str(build_package(self.root)),
                           score='causal_yesno', batch_size=4)

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

    def test_cli_accepts_rerank_options(self):
        from flagprof.training.__main__ import parser
        arguments = parser().parse_args(['--task', 'rerank', '--model', '/models/rer',
                                         '--package', '/pkg', '--rerank-score', 'sequence_logit',
                                         '--outputs', str(self.root/'cli')])
        self.assertEqual(arguments.task, 'rerank')
        self.assertEqual(arguments.rerank_score, 'sequence_logit')


if __name__ == '__main__':
    unittest.main()
