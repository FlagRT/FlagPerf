# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Generate a deterministic pretokenized triplet-group corpus for the embedding task.

Rows: {"anchor": [ids], "pos": [ids], "negs": [[ids] x K]}. Token ids are drawn
from a seeded generator per group index (no tokenizer, no text, no network);
the consumer pins the corpus by the engine input digest recorded in the
training protocol. Output line count equals --groups.
"""
import argparse
import hashlib
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--groups', type=int, required=True)
    ap.add_argument('--negs', type=int, default=3)
    ap.add_argument('--seed', type=int, default=531)
    ap.add_argument('--len-min', type=int, default=8)
    ap.add_argument('--len-max', type=int, default=32)
    ap.add_argument('--vocab', type=int, default=1024)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.groups < 1 or args.negs < 1:
        raise SystemExit('groups and negs must be positive')
    if not (1 <= args.len_min <= args.len_max):
        raise SystemExit('require 1 <= len-min <= len-max')
    if args.vocab < 2:
        raise SystemExit('vocab must be at least 2 (ids must avoid zero, the pad id)')
    digest = hashlib.sha256()

    import random
    with open(args.out, 'w', encoding='utf-8') as stream:
        for index in range(args.groups):
            rng = random.Random(str((args.seed, index)))
            anchor = [rng.randint(1, args.vocab - 1) for _ in range(rng.randint(args.len_min, args.len_max))]
            pos = anchor[:max(1, len(anchor) * 3 // 4)] + [rng.randint(1, args.vocab - 1)]
            negs = []
            for ni in range(args.negs):
                neg_rng = random.Random(str((args.seed, index, 'neg', ni)))
                negs.append([neg_rng.randint(1, args.vocab - 1)
                             for _ in range(neg_rng.randint(args.len_min, args.len_max))])
            row = dict(anchor=anchor, pos=pos, negs=negs)
            line = json.dumps(row, separators=(',', ':'))
            digest.update(line.encode())
            digest.update(b'\n')
            stream.write(line + '\n')
    print(json.dumps(dict(groups=args.groups, negs=args.negs, seed=args.seed,
                          out=args.out, corpus_sha256=digest.hexdigest())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
