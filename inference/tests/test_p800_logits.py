# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Actual low-margin rounding regression; no device or model weights required."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch CPU')
class P800LogitsTests(unittest.TestCase):
    def test_explicit_policy_preserves_margin_and_default(self):
        import torch
        from flagprof.vendors.kunlunxin.onerec import configure_logits
        head = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[24.5, 0.0], [24.5, 1.0]]))
        model = SimpleNamespace(lm_head=head, config=SimpleNamespace(tie_word_embeddings=False))
        x = torch.tensor([[1.0, 0.03125]], dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'identity.json'
            path.write_text(json.dumps(dict(devices=[dict(model='P800')])))
            with patch.dict(os.environ, {'FLAGPROF_P800_ONEREC_LOGITS':'model'}):
                self.assertIsNone(configure_logits(model))
                baseline = head(x)
                self.assertEqual(baseline.argmax(-1).item(), 0)
                self.assertEqual(baseline[0, 0], baseline[0, 1])
            with patch.dict(os.environ, {'FLAGPROF_P800_ONEREC_LOGITS':'fp32', 'FLAGPROF_P800_IDENTITY':str(path)}):
                metadata = configure_logits(model)
                actual = head(x)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.argmax(-1).item(), 1)
                self.assertEqual(float((actual[0, 1]-actual[0, 0]).detach()), 0.03125)
                self.assertEqual(metadata['original_weight_dtype'], 'torch.bfloat16')
                torch.testing.assert_close(actual, torch.nn.functional.linear(x.float(),
                    torch.tensor([[24.5, 0.0], [24.5, 1.0]])), rtol=0, atol=0)

    def test_missing_identity_cannot_enable_policy(self):
        from flagprof.vendors.kunlunxin.onerec import configure_logits
        with patch.dict(os.environ, {'FLAGPROF_P800_ONEREC_LOGITS':'fp32'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'IDENTITY'):
                configure_logits(None)
