# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Vendor-agnostic reranker attach target: sealed package -> per-pair scores.

Run under ``flagprof record`` (attachment auto-detected) or standalone (no
collection). Two scoring semantics:

  causal_yesno      AutoModelForCausalLM, logits at each last valid position,
                    softmax over [no, yes] token pair -> yes probability.
  sequence_logit    AutoModelForSequenceClassification, raw relevance logit.

Works for any HF reranker of either class without trust_remote_code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_common import load_package, causal_yes_scores, sequence_logit_scores


def model_files_identity(model_path):
    files = {}
    for path in sorted(Path(model_path).iterdir()):
        if path.suffix in ('.safetensors', '.bin'):
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b''):
                    digest.update(chunk)
            files[path.name] = digest.hexdigest()
    if not files:
        raise ValueError('identity needs safetensors/bin weights under ' + str(model_path))
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', required=True)
    p.add_argument('--model', required=True, help='hf:<directory> holding weights + tokenizer')
    p.add_argument('--score', choices=['causal_yesno', 'sequence_logit'], required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--outputs', required=True)
    p.add_argument('--samples')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--forwards', type=int, default=3)
    args = p.parse_args()
    if not args.model.startswith('hf:'):
        p.error('hf:<directory> required')
    model_path = args.model[3:]

    import torch
    torch.manual_seed(531)
    from flagprof import current_attachment
    from flagprof.runner import run_prepared
    from flagprof.runner.__main__ import model_files_identity
    handle = current_attachment()
    sync = torch.cuda.synchronize if args.device.startswith('cuda') else (torch.npu.synchronize if args.device.startswith('npu') else (lambda: None))
    if args.device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(args.device)

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    cls = AutoModelForCausalLM if args.score == 'causal_yesno' else AutoModelForSequenceClassification
    model = cls.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
        torch_dtype=getattr(torch, args.dtype), attn_implementation='eager').to(args.device).eval()
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    if handle:
        handle.register_model(model)
    batches, manifest = load_package(args.package, batch_size=args.batch_size,
        samples=args.samples.split(',') if args.samples else None)
    if manifest['score_semantics'] != args.score:
        p.error('package score_semantics ' + manifest['score_semantics'] + ' conflicts with --score')
    yes_id = no_id = None
    if args.score == 'causal_yesno':
        yes_id = tokenizer.convert_tokens_to_ids('yes')
        no_id = tokenizer.convert_tokens_to_ids('no')
        if yes_id == tokenizer.unk_token_id or no_id == tokenizer.unk_token_id:
            raise ValueError('tokenizer lacks literal yes/no tokens')
    identity = dict(oracle_id=manifest['oracle_id'],
        inputs_sha256=manifest['files']['prepared/inputs.pt']['sha256'],
        score_semantics=args.score, template_sha=manifest['template_sha'],
        model_files=model_files_identity(model_path), dtype=args.dtype, device=args.device,
        samples=[s for ids_, _ in batches for s in ids_])
    out = Path(args.outputs)
    run_rows = []
    import inspect
    for batch_index, (sample_ids, cpu) in enumerate(batches):
        inputs = {key: value.to(args.device) for key, value in cpu.items()}
        inputs.update(return_dict=True)
        if 'use_cache' in inspect.signature(model.forward).parameters:
            inputs['use_cache'] = False
        if handle:
            handle.bind_tensor_inputs(sample_ids, inputs, identity=identity)

        def transform(output, _mask=inputs['attention_mask']):
            if args.score == 'causal_yesno':
                result = causal_yes_scores(output.logits, _mask, yes_id, no_id)
            else:
                result = sequence_logit_scores(output.logits)
            if handle:
                handle.record_tensor('scores', result)
            return result

        folder = out / str(batch_index) if len(batches) > 1 else out
        run_rows.append(dict(samples=sample_ids,
            run=run_prepared(model, inputs, warmup=args.warmup, forwards=args.forwards,
                             outputs=folder, sync=sync, transform=transform)))
    score_sha = hashlib.sha256(json.dumps([r['run']['outputs'] for r in run_rows], sort_keys=True).encode()).hexdigest()
    result = dict(model=model_path, score_semantics=args.score, yes_token_id=yes_id, no_token_id=no_id,
        semantics=('{causal last-valid-position softmax([no,yes]) -> yes prob}' if args.score == 'causal_yesno'
                   else '{sequence-classification raw logit}') + '; use_cache=False; eager',
        oracle_identity=identity, batch_size=args.batch_size, score_batches_sha256=score_sha, runs=run_rows)
    (out / 'target-result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), attachment=bool(handle),
        score_batches_sha256=score_sha)))


if __name__ == '__main__':
    main()
