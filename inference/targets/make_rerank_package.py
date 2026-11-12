# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Build a sealed rerank package v1 from a pairs JSONL and a tokenizer.

Pairs JSONL rows: {"id": str, "query": str, "doc": str}.

--score sequence_logit
    Tokenizes (query, doc) as a pair, truncation longest_first. No template.

--score causal_yesno
    Mirrors the model-card process_inputs order: body text is truncated to
    max_length minus the prefix/suffix token budget, THEN prefix/body/suffix
    token ids are concatenated (add_special_tokens=False everywhere). The body
    template carries {instruction}/{query}/{doc} placeholders; --instruction
    substitutes {instruction} (default: the model-card web-search instruction).
    template_sha pins body+prefix+suffix bytes together.
"""
import argparse
import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_common import sha

DEFAULT_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True)
    p.add_argument('--tokenizer', required=True, help='model dir holding tokenizer files')
    p.add_argument('--score', choices=['causal_yesno', 'sequence_logit'], required=True)
    p.add_argument('--body')
    p.add_argument('--prefix-file')
    p.add_argument('--suffix-file')
    p.add_argument('--instruction', default=DEFAULT_INSTRUCTION)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.score == 'causal_yesno' and not (args.body and args.prefix_file and args.suffix_file):
        p.error('--body --prefix-file --suffix-file required for causal_yesno')
    body_text = Path(args.body).read_text() if args.body else None
    prefix_text = Path(args.prefix_file).read_text() if args.prefix_file else None
    suffix_text = Path(args.suffix_file).read_text() if args.suffix_file else None
    rows = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(r.get('id'), str) or not isinstance(r.get('query'), str)
                       or not isinstance(r.get('doc'), str) for r in rows):
        raise ValueError('pairs need id/query/doc string rows')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    import torch
    prepared = []
    if args.score == 'causal_yesno':
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        budget = args.max_length - len(prefix_ids) - len(suffix_ids)
        if budget <= 0:
            raise ValueError('max_length smaller than prefix+suffix budget')
    for row in rows:
        if args.score == 'causal_yesno':
            body = (body_text.replace('{instruction}', args.instruction)
                    .replace('{query}', row['query']).replace('{doc}', row['doc']))
            middle = tokenizer.encode(body, add_special_tokens=False, truncation=True, max_length=budget)
            ids = prefix_ids + middle + suffix_ids
        else:
            ids = tokenizer(row['query'], row['doc'], truncation='longest_first',
                            max_length=args.max_length)['input_ids']
        prepared.append(dict(sample_id=row['id'], input_ids=torch.tensor(ids, dtype=torch.int64)))
    out = Path(args.out)
    (out / 'prepared').mkdir(parents=True, exist_ok=True)
    torch.save(prepared, out / 'prepared' / 'inputs.pt')
    samples = dict(samples=[row['id'] for row in rows])
    (out / 'prepared' / 'samples.json').write_text(json.dumps(samples, sort_keys=True, indent=1) + '\n')
    template_sha = hashlib.sha256(''.join(filter(None, (body_text, prefix_text, suffix_text))).encode()).hexdigest()
    inputs_sha = sha(out / 'prepared' / 'inputs.pt')
    oracle_id = hashlib.sha256(json.dumps(dict(score_semantics=args.score, template_sha=template_sha,
        inputs_sha256=inputs_sha, batch_size=args.batch_size, instruction=args.instruction if args.score == 'causal_yesno' else None),
        sort_keys=True).encode()).hexdigest()
    manifest = dict(files={
        'prepared/inputs.pt': dict(sha256=inputs_sha),
        'prepared/samples.json': dict(sha256=sha(out / 'prepared' / 'samples.json')),
    }, oracle_id=oracle_id, score_semantics=args.score, template_sha=template_sha,
        contract=dict(batch_size=args.batch_size, max_length=args.max_length,
                      tokenizer_dir=str(args.tokenizer),
                      instruction=args.instruction if args.score == 'causal_yesno' else None))
    (out / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=1) + '\n')
    print(json.dumps(dict(out=str(out), samples=len(rows), score=args.score, oracle_id=oracle_id)))


if __name__ == '__main__':
    main()
