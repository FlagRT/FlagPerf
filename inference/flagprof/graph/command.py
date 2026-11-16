# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CLI configuration and independent export worker (no optional imports here)."""
import json
from pathlib import Path
import subprocess
import sys


def add_parser(verbs):
    p = verbs.add_parser('export', help='export a fixed-shape CPU model graph independently')
    p.add_argument('model', help='hf:<local Qwen3-Embedding directory>')
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--inputs', help='fixed text JSONL')
    inputs.add_argument('--inputs-from', help='existing fixed-input reference package')
    p.add_argument('--batch-index', type=int, default=0)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--dtype', choices=('bfloat16','float32'), default='bfloat16')
    p.add_argument('--pooling', choices=('last','mean'), default='last')
    p.add_argument('--padding-side', choices=('left','right'), default='left')
    p.add_argument('--formats', default='fx,onnx')
    p.add_argument('--bundle', choices=('structure','full'), default='structure')
    p.add_argument('--verify-onnx', action='store_true', help='validate FP32 outputs using CPU ONNX Runtime')
    p.add_argument('--out', required=True, help='new output directory')
    p.add_argument('--timeout', type=float, default=1800, help='worker timeout in seconds (default 1800)')


def configuration(args):
    config = vars(args).copy()
    config.pop('verb', None)
    config['formats'] = config['formats'].split(',')
    if not args.model.startswith('hf:') or not args.model[3:]:
        raise ValueError('model must be hf:<local directory>')
    if args.batch_index < 0 or (args.batch_size is not None and args.batch_size <= 0) or min(args.max_length, args.timeout) <= 0 or not __import__('math').isfinite(args.timeout):
        raise ValueError('batch index must be nonnegative and sizes/timeout positive')
    if args.inputs and args.batch_index:
        raise ValueError('batch-index selects a batch from inputs-from only')
    if not config['formats'] or len(set(config['formats'])) != len(config['formats']) or set(config['formats'])-{'fx','onnx'}:
        raise ValueError('formats must be fx, onnx or fx,onnx')
    if args.verify_onnx and (args.dtype != 'float32' or 'onnx' not in config['formats']):
        raise ValueError('verify-onnx requires float32 and the onnx format')
    return config


def run(args):
    from flagprof import current_attachment
    if current_attachment() is not None:
        raise ValueError('export must run independently of record/attach')
    config = configuration(args)
    root = Path(args.out)
    if root.exists():
        raise ValueError('export destination already exists')
    try:
        result = subprocess.run([sys.executable, '-B', '-m', 'flagprof.graph.worker', json.dumps(config)],
                                timeout=args.timeout)
        if result.returncode < 0:
            interrupted(root, config, 'worker terminated by signal ' + str(-result.returncode))
        return result.returncode if result.returncode >= 0 else 1
    except subprocess.TimeoutExpired:
        interrupted(root, config, 'export worker timeout after ' + str(args.timeout) + ' seconds')
        return 124


def interrupted(root, config, message):
    from .data import SCHEMA, write_json
    root.mkdir(parents=True, exist_ok=True)
    path = root/'manifest.json'
    value = json.loads(path.read_text()) if path.exists() else dict(schema=SCHEMA, formats={}, bundle=config['bundle'])
    value.update(status='failed', error=message)
    # Never leave a completed-looking page after a worker interruption.
    from .core import finish
    finish(root, value)
    write_json(root/'worker-failure.json', dict(error=message))


def prepare(config):
    import torch
    from . import ExportSpec
    from .data import file_info
    from flagprof.runner.embedding import load_model, load_tokenizer, text_batch, embedding_outputs
    from flagprof.runner.fixed import load
    model_path = Path(config['model'][3:])
    model = load_model(str(model_path), config['dtype'], 'eager').cpu()
    if getattr(model.config, 'model_type', None) != 'qwen3':
        raise ValueError('the built-in graph entrypoint currently supports Qwen3 embedding; use ExportSpec for other models')
    model.config.use_cache = False
    if config['inputs_from']:
        batches, source = load(config['inputs_from'], batch_size=config['batch_size'], model=str(model_path))
        if config['batch_index'] >= len(batches):
            raise ValueError('batch-index outside the prepared input package')
        samples, values = batches[config['batch_index']]
    else:
        size = config['batch_size'] or 4
        tokenizer = load_tokenizer(str(model_path), config['padding_side'])
        values = text_batch(config['inputs'], tokenizer, size, config['max_length'])
        samples = [str(i) for i in range(size)]
        source = dict(text_input=file_info(config['inputs']))
    if set(values) != {'input_ids','attention_mask'}:
        raise ValueError('Qwen export requires exactly input_ids and attention_mask')
    if values['input_ids'].shape != values['attention_mask'].shape or not bool((values['attention_mask'].sum(1)>0).all()):
        raise ValueError('Qwen inputs require matching shapes and a valid token per sample')
    model_files = {p.relative_to(model_path).as_posix():file_info(p)['sha256'] for p in sorted(model_path.rglob('*'))
                   if p.is_file() and p.suffix in ('.json','.safetensors','.model','.txt')}
    identity = dict(model='Qwen3-Embedding', model_files=model_files, source=source, samples=samples,
                    batch_index=config['batch_index'], dtype=config['dtype'], attention='eager',
                    pooling=config['pooling'], padding_side=config['padding_side'] if config['inputs'] else 'prepared',
                    use_cache=False, components='off',output_semantics='pooled; FP32 L2 normalized output')
    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
        def forward(self, input_ids, attention_mask):
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=False, return_dict=True).last_hidden_state
            return embedding_outputs(hidden, attention_mask, mean=config['pooling']=='mean')
    return ExportSpec(Embedding().eval(), (values['input_ids'], values['attention_mask']),
                      ('input_ids','attention_mask'), ('pooled','output'), identity=identity)
