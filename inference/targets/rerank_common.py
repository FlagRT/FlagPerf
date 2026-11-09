# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Rerank package v1: sealed query-doc inputs shared by reranker attach targets.

Layout:
  manifest.json           files[].sha256, oracle_id, score_semantics,
                          template_sha, contract{batch_size}
  prepared/inputs.pt      list of rows dict(sample_id, input_ids int64 var-len)
  prepared/samples.json   {"samples": [...]}

The loader verifies digests, selects samples, right-pads batches and returns
(inputs, manifest) so targets can bind exact oracle identity.
"""
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_package(package, *, batch_size=None, samples=None):
    import torch
    root = Path(package)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name in ('prepared/inputs.pt', 'prepared/samples.json'):
        if sha(root / name) != manifest['files'][name]['sha256']:
            raise ValueError('rerank package SHA mismatch: ' + name)
    rows = torch.load(root / 'prepared/inputs.pt', map_location='cpu', weights_only=True)
    index = {row['sample_id']: row for row in rows}
    selected = list(samples) if samples else [row['sample_id'] for row in rows]
    if len(selected) != len(set(selected)) or any(s not in index for s in selected):
        raise ValueError('unknown or duplicate sample ID')
    size = batch_size or manifest['contract']['batch_size']
    batches = []
    for offset in range(0, len(selected), size):
        chunk = [index[s] for s in selected[offset:offset + size]]
        width = max(int(row['input_ids'].numel()) for row in chunk)
        ids = torch.zeros((len(chunk), width), dtype=chunk[0]['input_ids'].dtype)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, row in enumerate(chunk):
            n = int(row['input_ids'].numel())
            ids[i, :n] = row['input_ids']
            mask[i, :n] = 1
        batches.append(([row['sample_id'] for row in chunk],
                        {'input_ids': ids, 'attention_mask': mask}))
    return batches, manifest


def causal_yes_scores(logits, attention_mask, yes_id, no_id):
    """Yes-probability per row from causal logits at each last valid position."""
    import torch
    positions = torch.arange(logits.shape[1], device=logits.device).expand_as(attention_mask)
    last = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    picked = logits[torch.arange(logits.shape[0], device=logits.device), last][:, [no_id, yes_id]]
    return torch.softmax(picked.float(), dim=1)[:, 1]


def sequence_logit_scores(logits):
    """Raw relevance logit per row from a sequence-classification head."""
    return logits[:, 0].float()
