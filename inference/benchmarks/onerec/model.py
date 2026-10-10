"""Offline Qwen3 loading and request-local greedy generation; vendor independent."""
import json
from pathlib import Path

from runtime.execution import ExecutionSession, ExecutionStep


def tokenize(cfg):
    from transformers import AutoTokenizer
    rows = [json.loads(line) for line in Path(cfg['inputs']['path']).read_text().splitlines() if line.strip()]
    if (not rows or any(not isinstance(r, dict) or not isinstance(r.get('id'), str) or not r['id']
                       or not isinstance(r.get('text'), str) for r in rows)
            or len({r['id'] for r in rows}) != len(rows)):
        raise ValueError('OneRec inputs require unique string IDs and text')
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['path'], local_files_only=True,
        trust_remote_code=False, padding_side=cfg['inputs']['padding_side'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('OneRec tokenizer has no padding token')
    raw = tokenizer([r['text'] for r in rows], add_special_tokens=True, padding=False,
                    truncation=False)['input_ids']
    maximum, size = cfg['inputs']['max_length'], cfg['inputs']['batch_size']
    samples, batches = [], []
    for row, ids in zip(rows, raw):
        if not ids:
            if tokenizer.eos_token_id is None:
                raise ValueError('empty text requires an explicit EOS representation')
            ids.append(tokenizer.eos_token_id)
        samples.append({**row, 'original_tokens': len(ids), 'truncated': len(ids) > maximum})
    for start in range(0, len(rows), size):
        inputs = tokenizer.pad({'input_ids': [t[:maximum] for t in raw[start:start+size]]},
                               padding=True, return_attention_mask=True, return_tensors='pt')
        batches.append({'ids': [r['id'] for r in rows[start:start+size]], 'inputs': dict(inputs)})
    return batches, samples


def load_cpu(cfg):
    import torch
    from transformers import AutoModelForCausalLM
    return configure(AutoModelForCausalLM.from_pretrained(cfg['model']['path'], local_files_only=True,
        trust_remote_code=False, dtype=getattr(torch, cfg['model']['dtype']),
        attn_implementation='eager').eval(), cfg)


def configure(model, cfg):
    if (model.config.model_type != 'qwen3' or model.config.hidden_size != 4096
            or model.config.num_hidden_layers != 36 or model.config.vocab_size != 176384):
        raise ValueError('weights do not match OneRec-8B architecture')
    model._flagperf_generation = dict(cfg['model']['options'])
    return model.requires_grad_(False)


def tensor_parallel_plan(architecture):
    base = architecture.base_model_tp_plan
    if not isinstance(base, dict) or not base:
        raise ValueError('OneRec requires an explicit supported Qwen3 block TP plan')
    return {**{'model.'+name:style for name,style in base.items()}, 'lm_head':'colwise_gather_output'}


class GreedyExecution(ExecutionSession):
    def __init__(self, model, inputs, mode):
        self.model, self.inputs, self.mode = model, inputs, mode
        self.maximum = model._flagperf_generation['max_new_tokens']
        self.fixed = mode == 'performance'
        self.ids = inputs['input_ids']
        self.mask = inputs['attention_mask'].clone()
        self.cache = None
        self.index = 0
        self.done = False
        self.active = [True] * self.ids.shape[0]
        self.stop_reasons = [None] * self.ids.shape[0]
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.pad = model.generation_config.pad_token_id
        if self.pad is None:
            self.pad = model.config.pad_token_id
        if self.pad is None:
            self.pad = next((v for v in self.eos if v is not None), 0)

    @property
    def context(self):
        return {'phase': 'prefill' if self.index == 0 else 'decode', 'generation_step': self.index,
                'mode': self.mode, 'active_samples': list(self.active),
                'input_shape': list(self.ids.shape),
                'kv_length_before': 0 if self.cache is None else self.cache.get_seq_length()}

    def advance(self):
        import torch
        if self.done:
            raise StopIteration
        context = self.context
        region = self.mask.clone() if self.index == 0 else torch.tensor(
            self.active, device=self.ids.device, dtype=self.mask.dtype).unsqueeze(1)
        positions = self.mask.long().cumsum(-1)-1
        positions.masked_fill_(self.mask == 0, 0)
        positions = positions if self.index == 0 else positions[:, -1:]
        out = self.model.model(input_ids=self.ids, attention_mask=self.mask, position_ids=positions,
                               past_key_values=self.cache, use_cache=True, return_dict=True)
        self.cache = out.past_key_values
        hidden = out.last_hidden_state
        if self.index == 0:
            indices = torch.arange(self.mask.shape[1], device=self.mask.device).expand_as(self.mask)
            indices = indices.masked_fill(self.mask == 0, -1).max(1).values
            head = hidden[torch.arange(hidden.shape[0], device=hidden.device), indices].unsqueeze(1)
        else:
            head = hidden
        logits = self.model.lm_head(head)
        chosen = logits[:, -1].argmax(-1)
        # A small control transfer is part of the generation host envelope, never hidden.
        selected = chosen.tolist()
        for i, token in enumerate(selected):
            if self.active[i] and not self.fixed and token in self.eos:
                self.active[i] = False
                self.stop_reasons[i] = 'eos'
        self.index += 1
        self.done = self.index >= self.maximum or not any(self.active)
        if self.done:
            self.stop_reasons = [r or 'max_new_tokens' for r in self.stop_reasons]
        else:
            alive = torch.tensor(self.active, device=chosen.device, dtype=torch.bool)
            self.ids = chosen.masked_fill(~alive, self.pad).unsqueeze(1)
            self.mask = torch.cat([self.mask, alive.to(self.mask.dtype).unsqueeze(1)], dim=1)
        return ExecutionStep({'hidden': hidden}, context, chosen,
                             {'attention_mask': region, 'lm_head_mask': torch.tensor(
                                 context['active_samples'], device=chosen.device, dtype=self.mask.dtype).unsqueeze(1)})

    def cache_snapshot(self):
        if self.cache is None:
            return {'status': 'not_initialized', 'storage_bytes': 0, 'layers': []}
        seen, size, layers = set(), 0, []
        for index, layer in enumerate(self.cache.layers):
            entry = {'layer': index, 'tensors': {}}
            for name in ('keys', 'values'):
                tensor = getattr(layer, name, None)
                if tensor is None:
                    continue
                storage = tensor.untyped_storage()
                key = (str(tensor.device), storage.data_ptr())
                if key not in seen:
                    seen.add(key); size += storage.nbytes()
                entry['tensors'][name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                           'storage_bytes': storage.nbytes()}
            layers.append(entry)
        return {'status': 'measured', 'storage_bytes': size, 'sequence_length': self.cache.get_seq_length(),
                'scope': 'rank-local deduplicated live KV storage; includes padded batch positions', 'layers': layers}

    def close(self):
        self.cache = None
        self.model = self.inputs = self.ids = self.mask = None
