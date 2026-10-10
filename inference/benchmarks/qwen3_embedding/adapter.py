"""Qwen3 semantics, including compatibility with sealed schema-1 A100 packages."""
from pathlib import Path, PurePosixPath

from runtime.common import read_json, file_hash
from runtime.workload import BoundarySpec, ModelAdapter

MODEL_BOUNDARIES = ['pooled', 'embedding']
SEMANTICS = 'qwen3-last-valid-token-fp32-l2-no-cache-eager-v1'
TIMING = 'resident-input-synchronized-forward-pool-normalize-host-envelope-v1'


class Qwen3EmbeddingAdapter(ModelAdapter):
    semantics = SEMANTICS
    timing = TIMING
    parallel_strategy = 'transformers_qwen3_auto'
    throughput_unit = 'valid_input_tokens'
    reference_dtype = 'bfloat16'
    transfer_key = 'output_embedding'
    transfer_objects = ['embedding']
    transfer_scope = 'first prepared batch, extra forward before warmup; only embedding copy timed'

    def report_descriptions(self):
        return {'model_description': '28 层、hidden size 1024、1024 维 FP32 embedding',
                'input_kind': '文本', 'input_description': 'input_ids、attention_mask',
                'forward_description': '前向、最后有效 token pooling、FP32 归一化',
                'token_description': 'attention_mask 非零数；不含 padding，不是生成 token',
                'cache_description': 'use_cache=False；KV cache 显存不适用',
                'transfer_key': self.transfer_key, 'transfer_objects': self.transfer_objects}

    def validate_config(self, cfg):
        if cfg['model']['dtype'] not in ('bfloat16', 'float16', 'float32') or cfg['model']['attention'] != 'eager':
            raise ValueError('dtype must be bfloat16/float16/float32; attention must be eager')
        for group in ('model', 'inputs'):
            if not isinstance(cfg[group].get('options', {}), dict) or cfg[group].get('options'):
                raise ValueError(f'{group}.options: Qwen3 adapter accepts no additional options')
        if type(cfg['inputs']['max_length']) is not int or cfg['inputs']['max_length'] < 1:
            raise ValueError('inputs.max_length must be a positive integer')
        if cfg['inputs']['padding_side'] not in ('left', 'right'):
            raise ValueError('padding_side must be left or right')

    def validate_assets(self, cfg):
        if not cfg['model'].get('path'):
            raise ValueError('set model.path or pass --model-path with a local Qwen3-Embedding directory')
        root = Path(cfg['model']['path'])
        for name in ('config.json', 'tokenizer_config.json'):
            with (root / name).open('rb') as stream:
                stream.read(16)
        if (root / 'model.safetensors').is_file():
            with (root / 'model.safetensors').open('rb') as stream:
                stream.read(16)
        else:
            index = read_json(root / 'model.safetensors.index.json')
            weights = index.get('weight_map')
            if not isinstance(weights, dict) or not weights:
                raise ValueError('empty safetensors shard index')
            for name in set(weights.values()):
                if (not isinstance(name, str) or not name.endswith('.safetensors') or '\\' in name
                        or PurePosixPath(name).is_absolute() or any(x in ('', '.', '..') for x in name.split('/'))):
                    raise ValueError('unsafe safetensors shard: ' + str(name))
                target = root / name
                if not target.is_file() or not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
                    raise ValueError('missing or unsafe safetensors shard: ' + name)
                with target.open('rb') as stream:
                    stream.read(16)
        if not Path(cfg['inputs']['path']).is_file():
            raise ValueError('input JSONL path does not exist; no fallback')

    def asset_fingerprints(self, cfg):
        root = Path(cfg['model']['path'])
        return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*'))
                if p.is_file() and p.suffix in ('.json', '.safetensors', '.model', '.txt')}

    def prepare_inputs(self, cfg):
        from benchmarks.qwen3_embedding.model import tokenize
        return tokenize(cfg)

    def load_cpu(self, cfg):
        from benchmarks.qwen3_embedding.model import load_model_cpu
        return load_model_cpu(cfg)

    def load(self, cfg, device, parallel=None):
        from benchmarks.qwen3_embedding.model import load_model
        return load_model(cfg, device, parallel=parallel)

    def load_model(self, cfg, device, session, parallel=None):
        if parallel is not None:
            return super().load_model(cfg, device, session, parallel)
        return self.load_via_cpu(cfg, device, session)

    def metadata(self, cfg, model=None):
        result = {**super().metadata(cfg, model), **self.report_descriptions()}
        blocks = self.block_names(cfg)
        result['boundary_relations'] = {'kind': 'sequential', 'edges': [list(pair) for pair in zip(blocks, blocks[1:])]}
        return result

    def forward(self, model, inputs):
        from benchmarks.qwen3_embedding.model import pool
        hidden = model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        pooled, embedding = pool(hidden, inputs['attention_mask'])
        return {'pooled': pooled, 'embedding': embedding}

    def performance_output(self, outputs):
        return outputs['embedding']

    def output_transfer(self, outputs):
        return outputs['embedding'] if isinstance(outputs, dict) else outputs

    def selected_modules(self, model, names):
        from benchmarks.qwen3_embedding.model import selected_layers
        return selected_layers(model, names)

    def block_names(self, cfg):
        architecture = read_json(Path(cfg['model']['path']) / 'config.json')
        return [f'layers.{i}' for i in range(architecture['num_hidden_layers'])]

    def instrumentation(self, cfg):
        return self.block_names(cfg)

    def boundary_specs(self, cfg, model=None, names=None):
        architecture = model.config if model is not None else read_json(Path(cfg['model']['path']) / 'config.json')
        hidden = architecture.hidden_size if model is not None else architecture['hidden_size']
        blocks = [f'layers.{i}' for i in range(len(model.layers))] if model is not None else self.block_names(cfg)
        selected = blocks if names in (None, ['all']) else names
        modules = dict(model.named_modules()) if model is not None else {}
        output = [BoundarySpec(n, 'model', sample_shape=[hidden]) for n in MODEL_BOUNDARIES]
        for name in selected:
            # Nested modules can expose TP-local shards; only complete blocks are oracle boundaries.
            path = ()
            if name in blocks:
                # Transformers versions can return a tensor or a tuple for a block.
                path = ()
            output.append(BoundarySpec(name, 'layer', valid_region={'input': 'attention_mask', 'axis': 0},
                layout='replicated' if name in blocks else 'rank_local',
                sample_shape=['valid_tokens', hidden if name in blocks else None], output_path=path))
        return output

    def module_value(self, output, spec):
        if isinstance(output, (tuple, list)):
            output = output[0]
        return super().module_value(output, spec)

    def batch_metadata(self, batch, index=None):
        return {**({'batch_index': index} if index is not None else {}),
                'sample_ids': batch['ids'], 'input_shape': list(batch['inputs']['input_ids'].shape),
                'valid_tokens': batch['inputs']['attention_mask'].sum(dim=1).tolist()}

    def sample_counts(self, batch):
        return len(batch['ids']), int(batch['inputs']['attention_mask'].sum())

    def valid_regions(self, batch):
        return {'attention_mask': batch['inputs']['attention_mask']}

    def artifact_extras(self, batch):
        return {'mask': batch['inputs']['attention_mask']}

    def forward_inputs(self, cfg, batches, model=None):
        value = super().forward_inputs(cfg, batches, model)
        architecture = model.config if model is not None else read_json(Path(cfg['model']['path']) / 'config.json')
        value['hidden_size'] = architecture.hidden_size if model is not None else architecture['hidden_size']
        # Historical order is retained for readers and immutable reference compatibility.
        value['boundaries'] = self.block_names(cfg) + MODEL_BOUNDARIES
        return value

    def communication_roles(self, model):
        return {name: ('attention_output' if name.endswith('self_attn.o_proj') else 'mlp_output')
                for name, _ in model.named_modules()
                if name.endswith(('self_attn.o_proj', 'mlp.down_proj'))}

    def expected_collectives(self, model):
        return 2 * len(model.layers)

    def parallel_metadata(self, cfg, model):
        shards = {name: list(t.shape) for name, t in model.named_parameters() if name.startswith('layers.')}
        return {'plan': model._tp_plan, 'parameter_shapes': shards,
                'layer0_parameter_shapes': {k: v for k, v in shards.items() if k.startswith('layers.0.')}}

    def oracle_layout(self, cfg, model):
        return {'tp_plan': model._tp_plan, 'boundary_contract': 'complete Qwen3 block outputs after rowwise reductions; no local shard concatenation'}

    def reference_contract(self, cfg, files=None):
        # Empty extension dictionaries are omitted so old packages retain their exact contract.
        sections = {g: {k: v for k, v in cfg[g].items() if k != 'path' and not (k == 'options' and not v)}
                    for g in ('model', 'inputs')}
        return {'semantics': self.semantics, **sections,
                'model_files': self.asset_fingerprints(cfg) if files is None else files,
                'input_sha256': file_hash(cfg['inputs']['path']), 'seed': cfg['runtime']['seed']}

    def reference_boundaries(self, cfg, command='accuracy', metadata=None):
        blocks = ([n for n in metadata['boundaries'] if n not in MODEL_BOUNDARIES]
                  if metadata is not None else self.block_names(cfg))
        levels = ['model', 'layer'] if command == 'preview' else cfg['accuracy']['levels']
        names = ['all'] if command == 'preview' else cfg['accuracy']['layers']
        layers = blocks if names == ['all'] else names
        if len(set(layers)) != len(layers):
            raise ValueError('duplicate oracle boundary selection')
        if cfg['runtime'].get('parallelism') == 'tp' and 'layer' in levels and not set(layers) <= set(blocks):
            raise ValueError('oracle TP comparison supports complete Transformer blocks only; no shard reconstruction')
        return (MODEL_BOUNDARIES if 'model' in levels else []) + (layers if 'layer' in levels else [])

    def validate_batches(self, batches, samples):
        import torch
        super().validate_batches(batches, samples)
        for batch in batches:
            tokens, mask = batch['inputs']['input_ids'], batch['inputs']['attention_mask']
            if tokens.dtype != torch.int64 or mask.dtype != torch.int64 or tokens.ndim != 2 or tokens.shape != mask.shape:
                raise ValueError('invalid token/mask dtype or shape')
            if len(batch['ids']) != tokens.shape[0] or not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.sum(1).gt(0).all()):
                raise ValueError('invalid oracle batch IDs or mask')
            if any(t.shape != tokens.shape for t in batch['inputs'].values()):
                raise ValueError('invalid oracle model inputs')

    def validate_reference_output(self, value, batch, boundaries, hidden_size=None):
        import torch
        if value['ids'] != batch['ids'] or not torch.equal(value['mask'], batch['inputs']['attention_mask']):
            raise ValueError('candidate IDs/order/mask differ from oracle inputs')
        if not set(boundaries) <= set(value['outputs']):
            raise ValueError('missing candidate boundary')
        size, seq = batch['inputs']['input_ids'].shape
        for name in boundaries:
            t = value['outputs'][name]
            if not isinstance(t, torch.Tensor) or t.device.type != 'cpu':
                raise ValueError('output must be a CPU tensor')
            expected = (size, hidden_size) if name in MODEL_BOUNDARIES else (size, seq, hidden_size)
            if hidden_size and tuple(t.shape) != expected:
                raise ValueError('incomplete/sharded oracle boundary: ' + name)

    def legacy_specs(self, names):
        return {n: BoundarySpec(n, 'model' if n in MODEL_BOUNDARIES else 'layer',
                    valid_region=None if n in MODEL_BOUNDARIES else {'input': 'attention_mask', 'axis': 0}).metadata()
                for n in names}

    def prepared_specs(self, expected, names):
        layers = [n for n in expected['boundaries'] if n not in MODEL_BOUNDARIES]
        selected = layers if names == ['all'] else names
        specs = self.legacy_specs(MODEL_BOUNDARIES + selected)
        for name, spec in specs.items():
            spec['sample_shape'] = ([expected['hidden_size']] if spec['level'] == 'model'
                                    else ['valid_tokens', expected['hidden_size'] if name in layers else None])
            if name not in layers and spec['level'] == 'layer':
                spec['layout'] = 'rank_local'
        return list(specs.values())

    def legacy_regions(self, batch):
        return {'attention_mask': batch['mask']}

    def legacy_metadata(self, identity):
        return {'schema_version': 1, 'model_adapter': identity['model']['name'], 'semantics': self.semantics,
                'timing': self.timing, 'throughput_unit': self.throughput_unit,
                'boundaries': list(self.legacy_specs(MODEL_BOUNDARIES).values()), **self.report_descriptions()}

    def export_spec(self, cfg, batch):
        from benchmarks.qwen3_embedding.model import ExportModel
        return ExportModel(self.load_cpu(cfg)).eval(), (batch['input_ids'], batch['attention_mask']), \
               ['input_ids', 'attention_mask'], MODEL_BOUNDARIES
