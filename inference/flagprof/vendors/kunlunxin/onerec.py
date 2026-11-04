# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit P800 OneRec logits policy; leaves the backbone and greedy rule intact."""
import json
import os
from pathlib import Path


def configure_logits(model):
    """Retain FP32 logits before argmax, only when the caller explicitly opts in.

    Casting a BF16 output afterwards cannot recover a rounded-away margin.
    Weight dtype conversion preserves DTensor placements and TP collectives;
    the pre-hook casts only the head input, after the backbone has completed.
    This is a different numerical execution policy, not an accuracy waiver.
    """
    policy = os.environ.get('FLAGPROF_P800_ONEREC_LOGITS', 'model')
    if policy not in ('model', 'fp32'):
        raise ValueError('FLAGPROF_P800_ONEREC_LOGITS must be model or fp32')
    if policy == 'model':
        return None
    path = os.environ.get('FLAGPROF_P800_IDENTITY')
    if not path:
        raise ValueError('P800 logits policy requires FLAGPROF_P800_IDENTITY')
    devices = json.loads(Path(path).read_text())['devices']
    if not devices or any(d.get('model') != 'P800' for d in devices):
        raise ValueError('P800 logits policy requires explicit P800 identity')
    import torch
    head = model.lm_head
    original = str(head.weight.dtype)
    if original not in ('torch.bfloat16', 'torch.float32'):
        raise ValueError('P800 candidate logits policy supports BF16 or FP32 weights')
    if getattr(model.config, 'tie_word_embeddings', False):
        raise ValueError('P800 logits policy requires an untied vocabulary head')
    head.to(dtype=torch.float32)
    def input_dtype(module, args):
        return (args[0].to(dtype=torch.float32), *args[1:])
    head.register_forward_pre_hook(input_dtype)
    return dict(policy='p800-onerec-fp32-logits/1', original_weight_dtype=original,
                weight_dtype=str(head.weight.dtype), input_dtype='torch.float32',
                output_dtype='torch.float32', selection='unchanged torch.argmax',
                qualification='candidate; device generation not yet validated')
