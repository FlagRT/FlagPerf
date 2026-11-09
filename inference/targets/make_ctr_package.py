# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Build a sealed CTR input package v1.

Schema (fixed, Criteo-canonical shape):
  dense     float32 [13]   continuous features
  sparse    int64   [26]   categorical ids, vocab 1000 each (ids in [1, 999])
  hist_ids  int64   [20]   behavior sequence for attention archs (0 = pad)
  hist_mask bool    [20]   True where hist_ids != 0

All randomness comes from torch.Generator seeded with --seed (default 531),
so the package is byte-reproducible from the manifest alone. The manifest
pins the schema sha (canonical JSON of the schema spec), the inputs sha256
and the arch list the package supports.
"""
import argparse
import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_common import sha

SCHEMA = dict(dense_dim=13, sparse_dim=26, vocab=1000, hist_len=20, pad_id=0)
SCHEMA_SPEC = dict(
    fields=dict(dense=dict(dtype='float32', shape=[SCHEMA['dense_dim']]),
                sparse=dict(dtype='int64', shape=[SCHEMA['sparse_dim']]),
                hist_ids=dict(dtype='int64', shape=[SCHEMA['hist_len']]),
                hist_mask=dict(dtype='bool', shape=[SCHEMA['hist_len']])),
    id_range=[SCHEMA['pad_id'] + 1, SCHEMA['vocab'] - 1],
    pad_id=SCHEMA['pad_id'], dense_normalized=True)


def build_sample(generator):
    import torch
    dense = torch.rand(SCHEMA['dense_dim'], generator=generator) * 2.0 - 1.0
    sparse = torch.randint(SCHEMA['pad_id'] + 1, SCHEMA['vocab'], (SCHEMA['sparse_dim'],),
                           generator=generator, dtype=torch.int64)
    lengths = torch.randint(5, SCHEMA['hist_len'] + 1, (1,), generator=generator)
    ids = torch.randint(SCHEMA['pad_id'] + 1, SCHEMA['vocab'], (SCHEMA['hist_len'],),
                        generator=generator, dtype=torch.int64)
    ids[lengths[0]:] = SCHEMA['pad_id']
    mask = ids != SCHEMA['pad_id']
    return dict(dense=dense, sparse=sparse, hist_ids=ids, hist_mask=mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--seed', type=int, default=531)
    p.add_argument('--archs', default='din,dlrm,widedeep,deepfm,dcn,esmm,mmoe,ple,bst,dien')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    import torch
    generator = torch.Generator().manual_seed(args.seed)
    prepared = [build_sample(generator) for _ in range(args.samples)]
    out = Path(args.out)
    (out / 'prepared').mkdir(parents=True, exist_ok=True)
    torch.save(prepared, out / 'prepared' / 'inputs.pt')
    (out / 'prepared' / 'samples.json').write_text(json.dumps(
        dict(samples=[str(i) for i in range(args.samples)]), sort_keys=True, indent=1) + '\n')
    schema_sha = hashlib.sha256(json.dumps(SCHEMA_SPEC, sort_keys=True).encode()).hexdigest()
    inputs_sha = sha(out / 'prepared' / 'inputs.pt')
    oracle_id = hashlib.sha256(json.dumps(dict(
        kind='ctr-package-v1', schema_sha256=schema_sha, inputs_sha256=inputs_sha,
        seed=args.seed, samples=args.samples), sort_keys=True).encode()).hexdigest()
    manifest = dict(files={'prepared/inputs.pt': dict(sha256=inputs_sha),
                           'prepared/samples.json': dict(sha256=sha(out / 'prepared' / 'samples.json'))},
                    oracle_id=oracle_id, kind='ctr-package-v1', schema_sha256=schema_sha,
                    schema=SCHEMA_SPEC, seed=args.seed,
                    contract=dict(archs=args.archs.split(',')))
    (out / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=1) + '\n')
    print(json.dumps(dict(out=str(out), samples=args.samples, seed=args.seed,
                          schema_sha256=schema_sha[:16], oracle_id=oracle_id)))


if __name__ == '__main__':
    main()
