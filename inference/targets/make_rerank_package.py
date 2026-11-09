# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Build a sealed rerank package v1 from a pairs JSONL and a tokenizer.

Pairs JSONL rows: {"id": str, "query": str, "doc": str} (extra fields kept).
--score causal_yesno requires --template, a text file whose literal {query} and
{doc} placeholders are substituted before tokenization; its digest is recorded
as template_sha and must be reproduced by every run consuming the package.
"""
import argparse
import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_common import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True)
    p.add_argument('--tokenizer', required=True, help='model dir holding tokenizer files')
    p.add_argument('--score', choices=['causal_yesno', 'sequence_logit'], required=True)
    p.add_argument('--template')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.score == 'causal_yesno' and not args.template:
        p.error('--template required for causal_yesno')
    template_text = Path(args.template).read_text() if args.template else None
    rows = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(r.get('id'), str) or not isinstance(r.get('query'), str)
                       or not isinstance(r.get('doc'), str) for r in rows):
        raise ValueError('pairs need id/query/doc string rows')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    prepared = []
    for row in rows:
        if args.score == 'causal_yesno':
            text = template_text.replace('{query}', row['query']).replace('{doc}', row['doc'])
            ids = tokenizer(text, truncation=True, max_length=args.max_length, add_special_tokens=False)['input_ids']
        else:
            ids = tokenizer(row['query'], row['doc'], truncation='longest_first',
                            max_length=args.max_length)['input_ids']
        import torch
        prepared.append(dict(sample_id=row['id'], input_ids=torch.tensor(ids, dtype=torch.int64)))
    out = Path(args.out)
    (out / 'prepared').mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(prepared, out / 'prepared' / 'inputs.pt')
    samples = dict(samples=[row['id'] for row in rows])
    (out / 'prepared' / 'samples.json').write_text(json.dumps(samples, sort_keys=True, indent=1) + '\n')
    template_sha = sha(args.template) if args.template else ''
    inputs_sha = sha(out / 'prepared' / 'inputs.pt')
    oracle_id = hashlib.sha256(json.dumps(dict(score_semantics=args.score, template_sha=template_sha,
        inputs_sha256=inputs_sha, batch_size=args.batch_size), sort_keys=True).encode()).hexdigest()
    manifest = dict(files={
        'prepared/inputs.pt': dict(sha256=inputs_sha),
        'prepared/samples.json': dict(sha256=sha(out / 'prepared' / 'samples.json')),
    }, oracle_id=oracle_id, score_semantics=args.score, template_sha=template_sha,
        contract=dict(batch_size=args.batch_size, max_length=args.max_length,
                      tokenizer_dir=str(args.tokenizer)))
    (out / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=1) + '\n')
    print(json.dumps(dict(out=str(out), samples=len(rows), score=args.score, oracle_id=oracle_id)))


if __name__ == '__main__':
    main()
