# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Vendor-agnostic embedding attach target: texts -> pooled sentence vectors.

Standard HF models load via AutoModel with trust_remote_code=False. Models
whose config.json auto_map points at remote custom code (e.g. jina
embeddings v3 -> XLMRobertaLoRA) load through a local package bypass: the
model directory itself is imported as a Python package so the vendor's
modeling files and their two-level relative imports resolve from disk, never
from the transformers dynamic-module cache. Every code file consumed this
way is sha256-pinned into the run identity (code_files).

The vendor flash-attention path is force-disabled (config.use_flash_attn
False when present) so execution uses the native attention implementation,
consistent with the eager-attention protocol used elsewhere in this suite.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


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


def code_files_identity(model_path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(model_path).glob('*.py'))}


def load_with_vendor_code(model_path, dtype):
    """Import the model dir as a package and build the auto_map model class."""
    import importlib
    import importlib.util
    root = Path(model_path)
    config = json.loads((root / 'config.json').read_text())
    auto_map = config.get('auto_map') or {}
    if 'AutoModel' not in auto_map or '--' in str(auto_map['AutoModel']):
        raise ValueError('vendor-code load needs a local auto_map AutoModel entry')
    module_name, _, class_name = auto_map['AutoModel'].rpartition('.')
    pkg_name = 'flagprof_vendor_' + root.name.replace('-', '_').replace('.', '_')
    spec = importlib.util.spec_from_file_location(
        pkg_name, root / (module_name + '.py'), submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = package
    spec.loader.exec_module(package)
    model_cls = getattr(package, class_name)
    config_cls = None
    config_map = auto_map.get('AutoConfig')
    if config_map and '--' not in str(config_map):
        config_module, _, config_class = config_map.rpartition('.')
        config_cls = getattr(importlib.import_module(pkg_name + '.' + config_module), config_class)
    if hasattr(model_cls, 'from_pretrained'):
        cfg = config_cls.from_pretrained(str(root), local_files_only=True) if config_cls else None
        if cfg is not None and hasattr(cfg, 'use_flash_attn'):
            cfg.use_flash_attn = False
        load_kwargs = dict(local_files_only=True, torch_dtype=dtype)
        if cfg is not None:
            model = model_cls.from_pretrained(str(root), config=cfg, **load_kwargs)
        else:
            model = model_cls.from_pretrained(str(root), **load_kwargs)
    else:
        cfg = config_cls() if config_cls else None
        if cfg is not None and hasattr(cfg, 'use_flash_attn'):
            cfg.use_flash_attn = False
        model = model_cls(cfg) if cfg is not None else model_cls()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='hf:<directory> holding weights (+ optional vendor *.py)')
    p.add_argument('--inputs', required=True, help='JSONL text rows (text/query/input key or bare strings)')
    p.add_argument('--pooling', choices=['mean', 'cls', 'last'], default='mean')
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--outputs', required=True)
    p.add_argument('--samples')
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--forwards', type=int, default=3)
    args = p.parse_args()
    if not args.model.startswith('hf:'):
        p.error('hf:<directory> required')

    import torch
    torch.manual_seed(531)
    from flagprof import current_attachment
    from flagprof.runner import run_prepared, pool_embedding
    from flagprof.runner.__main__ import POOLING_SEMANTICS
    handle = current_attachment()
    sync = torch.cuda.synchronize if args.device.startswith('cuda') else (torch.npu.synchronize if args.device.startswith('npu') else (lambda: None))
    if args.device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(args.device)

    model_path = args.model[3:]
    rows = [json.loads(line) for line in Path(args.inputs).read_text().splitlines() if line.strip()]
    texts = [(r if isinstance(r, str) else r.get('text', r.get('query', r.get('input')))
              ) for r in rows[:args.batch_size]]
    if len(texts) != args.batch_size or not all(isinstance(t, str) for t in texts):
        raise ValueError('inputs need batch-size text rows')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    vendor_code = code_files_identity(model_path)
    if vendor_code and any(name.startswith('modeling_') for name in vendor_code):
        model = load_with_vendor_code(model_path, getattr(torch, args.dtype))
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, args.dtype), attn_implementation='eager')
    model = model.to(args.device).eval()
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    if handle:
        handle.register_model(model)

    encoded = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
    inputs = {key: value.to(args.device) for key, value in encoded.items()}
    inputs.update(return_dict=True)
    identity = dict(inputs_sha256=hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
        model_files=model_files_identity(model_path),
        code_files=vendor_code or None,
        adapter_names=getattr(model.config, 'adapter_names', None),
        pooling=POOLING_SEMANTICS[args.pooling],
        batch_size=args.batch_size, max_length=args.max_length,
        dtype=args.dtype, device=args.device,
        samples=[str(i) for i in range(args.batch_size)])

    out = Path(args.outputs)
    import inspect
    sample_ids = identity['samples']

    def forward_row():
        call = {key: value for key, value in inputs.items()}
        if 'use_cache' in inspect.signature(model.forward).parameters:
            call['use_cache'] = False
        return call

    def transform(output):
        pooled = pool_embedding(output.last_hidden_state, inputs['attention_mask'],
                                mean=args.pooling == 'mean', cls=args.pooling == 'cls', normalize=False)
        result = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        if handle:
            handle.record_tensor('pooled', pooled)
            handle.record_tensor('output', result)
        return result

    call = forward_row()
    if handle:
        handle.bind_tensor_inputs(sample_ids, call, identity=identity)
    run_payload = run_prepared(model, call, warmup=args.warmup, forwards=args.forwards,
                               outputs=out, sync=sync, transform=transform)
    embed_sha = hashlib.sha256(json.dumps(run_payload['outputs'], sort_keys=True).encode()).hexdigest()
    result = dict(model=model_path, pooling=args.pooling,
        semantics=POOLING_SEMANTICS[args.pooling] + '; use_flash_attn=False' if vendor_code else POOLING_SEMANTICS[args.pooling] + '; use_cache=False; eager',
        vendor_code=bool(vendor_code), oracle_identity=identity,
        batch_size=args.batch_size, embed_batches_sha256=embed_sha,
        runs=[dict(samples=sample_ids, run=run_payload)])
    (out / 'target-result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), attachment=bool(handle), vendor_code=bool(vendor_code),
        embed_batches_sha256=embed_sha)))


if __name__ == '__main__':
    main()
