# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Rerank package v1 loader and score extraction, CPU only."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'targets'))
from rerank_common import load_package, causal_yes_scores, sequence_logit_scores, sha


def build_package(root, rows, batch_size=2, corrupt=None):
    import torch
    root = Path(root)
    (root / 'prepared').mkdir(parents=True)
    torch.save([dict(sample_id=r['id'], input_ids=torch.tensor(r['ids'], dtype=torch.int64)) for r in rows],
               root / 'prepared' / 'inputs.pt')
    (root / 'prepared' / 'samples.json').write_text(json.dumps(dict(samples=[r['id'] for r in rows])))
    manifest = dict(files={
        'prepared/inputs.pt': dict(sha256=sha(root / 'prepared' / 'inputs.pt')),
        'prepared/samples.json': dict(sha256=sha(root / 'prepared' / 'samples.json')),
    }, oracle_id='t', score_semantics='sequence_logit', template_sha='',
        contract=dict(batch_size=batch_size, max_length=64))
    if corrupt:
        manifest['files'][corrupt]['sha256'] = '0' * 64
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch CPU runtime')
class RerankPackageTests(unittest.TestCase):
    def test_load_selects_pads_and_masks_right(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_package(tmp, [dict(id='a', ids=[1, 2, 3]), dict(id='b', ids=[4, 5]),
                                      dict(id='c', ids=[6, 7, 8, 9])])
            batches, manifest = load_package(pkg, batch_size=2)
            self.assertEqual(manifest['score_semantics'], 'sequence_logit')
            self.assertEqual(len(batches), 2)
            ids, inputs = batches[0]
            self.assertEqual(ids, ['a', 'b'])
            self.assertEqual(inputs['input_ids'].tolist(), [[1, 2, 3], [4, 5, 0]])
            self.assertEqual(inputs['attention_mask'].tolist(), [[1, 1, 1], [1, 1, 0]])
            ids1, inputs1 = batches[1]
            self.assertEqual(ids1, ['c'])
            self.assertEqual(inputs1['input_ids'].tolist(), [[6, 7, 8, 9]])

    def test_digest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_package(tmp, [dict(id='a', ids=[1])], corrupt='prepared/inputs.pt')
            with self.assertRaises(ValueError):
                load_package(pkg)

    def test_unknown_sample_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = build_package(tmp, [dict(id='a', ids=[1])])
            with self.assertRaises(ValueError):
                load_package(pkg, samples=['zzz'])

    def test_causal_yes_scores_use_last_valid_position(self):
        import torch
        vocab = 5
        yes_id, no_id = 1, 2
        logits = torch.zeros(2, 4, vocab)
        logits[0, 2, no_id] = 0.0
        logits[0, 2, yes_id] = 0.0
        logits[1, 3, no_id] = 0.0
        logits[1, 3, yes_id] = 2.0
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        scores = causal_yes_scores(logits, mask, yes_id, no_id)
        self.assertAlmostEqual(scores[0].item(), 0.5, places=6)
        self.assertAlmostEqual(scores[1].item(), 1.0 / (1.0 + pow(2.718281828, -2.0)) * 1.0, places=5)

    def test_sequence_logit_takes_first_column(self):
        import torch
        logits = torch.tensor([[1.5, 9.0], [-2.0, 9.0]])
        scores = sequence_logit_scores(logits)
        self.assertTrue(torch.equal(scores, torch.tensor([1.5, -2.0])))


if __name__ == '__main__':
    unittest.main()
