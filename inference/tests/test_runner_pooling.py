# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Runner embedding pooling semantics: cls option, padding invariance, prior behavior."""
import unittest

from flagprof.runner import pool_embedding


class PoolingTests(unittest.TestCase):
    def test_cls_picks_index_zero_and_is_unit_norm(self):
        import torch
        hidden = torch.randn(2, 5, 3)
        mask = torch.ones(2, 5, dtype=torch.long)
        out = pool_embedding(hidden, mask, cls=True)
        self.assertTrue(torch.equal(out, torch.nn.functional.normalize(hidden[:, 0].float(), p=2, dim=1)))
        self.assertTrue(torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5))

    def test_cls_is_invariant_to_padding_side(self):
        import torch
        hidden = torch.randn(1, 5, 3)
        left = torch.tensor([[1, 1, 1, 0, 0]])
        right = torch.tensor([[0, 0, 1, 1, 1]])
        self.assertTrue(torch.equal(pool_embedding(hidden, left, cls=True),
                                    pool_embedding(hidden, right, cls=True)))

    def test_last_and_mean_behavior_unchanged(self):
        import torch
        hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [5.0, 7.0]]])
        mask = torch.tensor([[1, 1, 0]])
        last = pool_embedding(hidden, mask)
        self.assertTrue(torch.equal(last, torch.nn.functional.normalize(hidden[:, 1].float(), p=2, dim=1)))
        mean = pool_embedding(hidden, mask, mean=True)
        expected = torch.nn.functional.normalize(hidden[0, :2].mean(dim=0).float(), p=2, dim=0)
        self.assertTrue(torch.allclose(mean[0], expected, atol=1e-6))

    def test_cls_coexists_with_mean_flag(self):
        import torch
        hidden = torch.randn(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.long)
        cls = pool_embedding(hidden, mask, mean=True, cls=True)
        self.assertTrue(torch.equal(cls, torch.nn.functional.normalize(hidden[:, 0].float(), p=2, dim=1)))


if __name__ == '__main__':
    unittest.main()
