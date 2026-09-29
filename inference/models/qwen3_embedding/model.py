# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Qwen3 adapter: fixed inputs, last-valid-token pooling, explicit normalization."""
from pathlib import Path
import json
import torch
from transformers import AutoModel, AutoTokenizer


def load_model_cpu(cfg):
    model = AutoModel.from_pretrained(cfg['model']['path'], local_files_only=True,
        trust_remote_code=False, dtype=getattr(torch, cfg['model']['dtype']),
        attn_implementation='eager').eval()
    if model.config.model_type != 'qwen3' or model.config.hidden_size != 1024 or len(model.layers) != 28:
        raise ValueError('model does not match the Qwen3-Embedding-0.6B adapter')
    return model


def load_model(cfg, device):
    if cfg['runtime'].get('parallelism') == 'tp':
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh('npu',(len(cfg['runtime']['devices']),),mesh_dim_names=('tp',))
        model = AutoModel.from_pretrained(cfg['model']['path'],local_files_only=True,
            trust_remote_code=False,dtype=getattr(torch,cfg['model']['dtype']),
            attn_implementation='eager',tp_plan='auto',device_mesh=mesh).eval()
        if model.config.model_type != 'qwen3' or model.config.hidden_size != 1024 or len(model.layers) != 28:
            raise ValueError('model does not match Qwen3-Embedding-0.6B')
        return model
    return load_model_cpu(cfg).to(device)


def tokenize(cfg):
    samples = []
    for line in Path(cfg['inputs']['path']).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get('id'),str) or not row['id'] or not isinstance(row.get('text'),str):
            raise ValueError('each JSONL row needs a nonempty string id and string text')
        samples.append(row)
    if not samples or len({x['id'] for x in samples}) != len(samples):
        raise ValueError('input must be nonempty with unique sample IDs')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['path'], local_files_only=True,
        trust_remote_code=False, padding_side=cfg['inputs']['padding_side'])
    batches, metadata = [], []
    size = cfg['inputs']['batch_size']
    # Tokenize once without truncation; prepare/pad those exact IDs afterwards.
    raw = tokenizer([r['text'] for r in samples], add_special_tokens=True, truncation=False,
                    padding=False)['input_ids']
    maximum = cfg['inputs']['max_length']
    for row, ids in zip(samples, raw):
        if not ids:
            # An explicit EOS is the defined representation of empty token sequences.
            if tokenizer.eos_token_id is None:
                raise ValueError('empty token sequence and no EOS token')
            ids.append(tokenizer.eos_token_id)
        metadata.append(dict(row, original_tokens=len(ids), truncated=len(ids)>maximum))
    for start in range(0, len(samples), size):
        ids = [tokens[:maximum] for tokens in raw[start:start+size]]
        tokens = tokenizer.pad({'input_ids':ids}, padding=True, return_attention_mask=True,
                               return_tensors='pt')
        batches.append({'ids':[s['id'] for s in samples[start:start+size]],
                        'inputs':dict(tokens)})
    return batches, metadata


def pool(hidden, mask):
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(mask == 0, -1).max(dim=1).values
    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
    embedding = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
    return pooled, embedding


def selected_layers(model, names):
    modules = dict(model.named_modules())
    selected = [f'layers.{i}' for i in range(len(model.layers))] if names == ['all'] else names
    if len(set(selected)) != len(selected):
        raise ValueError('duplicate layer names')
    for name in selected:
        if name not in modules or not name:
            raise ValueError(f'unknown model module: {name}')
    return {name:modules[name] for name in selected}


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, return_dict=True).last_hidden_state
        return pool(hidden, attention_mask)
