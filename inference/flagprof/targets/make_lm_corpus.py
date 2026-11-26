# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Deterministic synthetic pretokenized LM corpus; no tokenizer, no network.

Every row is a fixed-length ``input_ids`` list of seeded random token IDs with
an all-ones ``attention_mask``; labels follow the standard next-token shift in
the training loader. Keep ``--vocab`` below the target model's vocabulary so
IDs stay in range. This verifies the causal-lm fine-tuning path, not any
language quality.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=int, default=16, help='number of JSONL rows')
    p.add_argument('--seq-len', type=int, default=64, help='tokens per row; at least two')
    p.add_argument('--vocab', type=int, default=65536, help='exclusive upper bound for token IDs')
    p.add_argument('--seed', type=int, default=531)
    p.add_argument('--out', required=True, help='output JSONL path')
    args = p.parse_args()
    if args.rows <= 0 or args.seq_len < 2 or args.vocab < 2:
        raise SystemExit('rows must be positive; seq-len and vocab need at least two')
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{'input_ids': [rng.randrange(args.vocab) for _ in range(args.seq_len)],
             'attention_mask': [1] * args.seq_len} for _ in range(args.rows)]
    payload = ''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows).encode('utf-8')
    temporary = out.with_name(out.name + '.tmp')
    temporary.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    temporary.replace(out)
    print(json.dumps({'rows': args.rows, 'seq_len': args.seq_len, 'vocab': args.vocab,
                      'seed': args.seed, 'out': str(out), 'sha256': digest}))


if __name__ == '__main__':
    main()
