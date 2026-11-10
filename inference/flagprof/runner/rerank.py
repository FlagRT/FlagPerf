# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Canonical reranker execution: sealed rerank-package-v1 -> per-pair scores.

This module is the single implementation shared by

  * the attach target (targets/target_reranker.py -> attach_main, writes
    target-result.json), and
  * the runner task (flagprof.runner --task rerank -> execute, writes
    runner.json).

Two scoring semantics (package manifest pins which):

  causal_yesno      AutoModelForCausalLM, logits at each last valid position,
                    softmax over [no, yes] token pair -> yes probability.
  sequence_logit    AutoModelForSequenceClassification, raw relevance logit.

Works for any HF reranker of either class. Models shipping custom auto_map
code load through the controlled vendor path (flagprof/runner/vendor.py)
when trust_remote_code is set: directory-as-package import, local auto_map
entry required, every consumed code file sha256-pinned into oracle identity.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from .rerank_common import causal_yes_scores, load_package, sequence_logit_scores


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


def validate_package_semantics(manifest, score):
    """Package/flag semantics conflict is a hard error before any model load."""
    if manifest['score_semantics'] != score:
        raise ValueError('package score_semantics ' + manifest['score_semantics'] +
                         ' conflicts with requested ' + score)


def run_rerank(*, package, model_path, score, device, dtype, outputs,
               samples=None, batch_size=None, warmup=0, forwards=3,
               trust_remote_code=False):
    """Canonical rerank execution; returns the target-style result dict."""
    import torch
    from flagprof import current_attachment
    from flagprof.runner import run_prepared
    handle = current_attachment()
    sync = torch.cuda.synchronize if device.startswith('cuda') else (torch.npu.synchronize if device.startswith('npu') else (lambda: None))
    if device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(device)

    torch.manual_seed(531)
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    vendor_code = None
    if trust_remote_code:
        from .vendor import code_files_identity
        vendor_code = code_files_identity(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    cls = AutoModelForCausalLM if score == 'causal_yesno' else AutoModelForSequenceClassification
    if vendor_code:
        # Controlled vendor-code load (see flagprof/runner/vendor.py):
        # directory-as-package, dynamic-module cache never engaged; the
        # auto_map entry for this score's class family must be local.
        from .vendor import load_with_vendor_code
        model = load_with_vendor_code(model_path, getattr(torch, dtype),
            auto_key='AutoModelForCausalLM' if score == 'causal_yesno' else 'AutoModelForSequenceClassification').to(device).eval()
    else:
        model = cls.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, dtype), attn_implementation='eager').to(device).eval()
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    if handle:
        handle.register_model(model)
    batches, manifest = load_package(package, batch_size=batch_size, samples=samples)
    validate_package_semantics(manifest, score)
    yes_id = no_id = None
    if score == 'causal_yesno':
        yes_id = tokenizer.convert_tokens_to_ids('yes')
        no_id = tokenizer.convert_tokens_to_ids('no')
        if yes_id == tokenizer.unk_token_id or no_id == tokenizer.unk_token_id:
            raise ValueError('tokenizer lacks literal yes/no tokens')
    identity = dict(oracle_id=manifest['oracle_id'],
        inputs_sha256=manifest['files']['prepared/inputs.pt']['sha256'],
        score_semantics=score, template_sha=manifest['template_sha'],
        model_files=model_files_identity(model_path), dtype=dtype, device=device,
        samples=[s for ids_, _ in batches for s in ids_])
    if vendor_code:
        identity['code_files'] = vendor_code

    out = Path(outputs)
    run_rows = []
    for batch_index, (sample_ids, cpu) in enumerate(batches):
        inputs = {key: value.to(device) for key, value in cpu.items()}
        inputs.update(return_dict=True)
        if 'use_cache' in inspect.signature(model.forward).parameters:
            inputs['use_cache'] = False
        if handle:
            handle.bind_tensor_inputs(sample_ids, inputs, identity=identity)

        def transform(output, _mask=inputs['attention_mask']):
            if score == 'causal_yesno':
                result = causal_yes_scores(output.logits, _mask, yes_id, no_id)
            else:
                result = sequence_logit_scores(output.logits)
            if handle:
                handle.record_tensor('scores', result)
            return result

        folder = out / str(batch_index) if len(batches) > 1 else out
        run_rows.append(dict(samples=sample_ids,
            run=run_prepared(model, inputs, warmup=warmup, forwards=forwards,
                             outputs=folder, sync=sync, transform=transform)))
    score_sha = hashlib.sha256(json.dumps([r['run']['outputs'] for r in run_rows],
        sort_keys=True).encode()).hexdigest()
    result = dict(model=model_path, score_semantics=score, yes_token_id=yes_id,
        no_token_id=no_id,
        semantics=('{causal last-valid-position softmax([no,yes]) -> yes prob}' if score == 'causal_yesno'
                   else '{sequence-classification raw logit}')
                  + ('; vendor code (use_flash_attn=False)' if vendor_code else '; use_cache=False; eager'),
        oracle_identity=identity, batch_size=batch_size, score_batches_sha256=score_sha,
        vendor_code=bool(vendor_code),
        runs=run_rows, attachment=bool(handle))
    return result


def execute(args):
    """flagprof.runner --task rerank: same canonical run, runner.json surface."""
    import time
    from .fixed import software
    if not args.model.startswith('hf:'):
        raise ValueError('hf:<directory> required for --task rerank')
    start = time.perf_counter()
    result = run_rerank(package=args.package, model_path=args.model[3:],
                        score=args.score, device=args.device, dtype=args.dtype,
                        outputs=args.outputs,
                        samples=args.samples.split(',') if args.samples else None,
                        batch_size=args.batch_size, warmup=args.warmup,
                        forwards=args.forwards, trust_remote_code=args.trust_remote_code)
    out = Path(args.outputs)
    runner_result = dict(model=result['model'], task='rerank', package=args.package,
        load_s=time.perf_counter() - start, score_semantics=result['score_semantics'],
        semantics=result['semantics'], oracle_identity=result['oracle_identity'],
        batch_size=result['batch_size'], score_batches_sha256=result['score_batches_sha256'],
        vendor_code=result['vendor_code'],
        software=software(), instrumentation='attachment' if result['attachment'] else 'none',
        runs=[r['run'] for r in result['runs']])
    (out / 'runner.json').write_text(json.dumps(runner_result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), attachment=result['attachment'],
        score_batches_sha256=result['score_batches_sha256'])))


def attach_main():
    """Attach CLI (targets/target_reranker.py delegates here); target-result.json
    surface, byte-identical contract."""
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
    result = run_rerank(package=args.package, model_path=args.model[3:],
                        score=args.score, device=args.device, dtype=args.dtype,
                        outputs=args.outputs,
                        samples=args.samples.split(',') if args.samples else None,
                        batch_size=args.batch_size, warmup=args.warmup,
                        forwards=args.forwards)
    out = Path(args.outputs)
    payload = dict(model=result['model'], score_semantics=result['score_semantics'],
        yes_token_id=result['yes_token_id'], no_token_id=result['no_token_id'],
        semantics=result['semantics'], oracle_identity=result['oracle_identity'],
        batch_size=result['batch_size'], score_batches_sha256=result['score_batches_sha256'],
        runs=result['runs'])
    (out / 'target-result.json').write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), attachment=result['attachment'],
        score_batches_sha256=result['score_batches_sha256'])))
