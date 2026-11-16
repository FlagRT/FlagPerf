# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Shared model, text input and embedding semantics; optional imports are local."""
import json
from pathlib import Path


def load_model(path, dtype, attention='eager', **options):
    import torch
    from transformers import AutoModel
    return AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                    torch_dtype=getattr(torch, dtype),
                                    attn_implementation=attention, **options).eval()


def load_tokenizer(path, padding_side='left'):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                         padding_side=padding_side)


def text_batch(path, tokenizer, batch_size=4, max_length=256):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    texts = [r if isinstance(r, str) else r.get('text', r.get('query', r.get('input')))
             for r in rows[:batch_size]]
    if len(texts) != batch_size or not all(isinstance(t, str) for t in texts):
        raise ValueError('inputs need batch-size text rows')
    return dict(tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                          return_tensors='pt'))


def embedding_outputs(hidden, mask, *, mean=False, cls=False):
    import torch
    from . import pool_embedding
    pooled = pool_embedding(hidden, mask, mean=mean, cls=cls, normalize=False)
    return pooled, torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
