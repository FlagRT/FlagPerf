"""Reusable sealed assets and numeric outputs; model mathematics stay in adapters."""
from pathlib import Path

from runtime.common import read_json, file_hash
from runtime.workload import ModelAdapter


def read_samples(path, keys, label):
    import json
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if (not rows or any(not isinstance(r, dict) or set(r) != set(keys)
                       or not isinstance(r['id'], str) or not r['id'] for r in rows)):
        raise ValueError(label + ' requires nonempty sample IDs and declared features')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate ' + label + ' sample IDs')
    return rows


def categorical_ids(values, cardinality):
    return isinstance(values, list) and all(type(x) is int and 0 <= x < cardinality for x in values)


class RecommendationAdapter(ModelAdapter):
    reference_dtype = 'float32'
    throughput_unit = 'samples'
    transfer_objects = ['probability']

    def normalize_config(self, cfg):
        cfg['model'].pop('attention', None)
        cfg['inputs'].pop('max_length', None)
        cfg['inputs'].pop('padding_side', None)
        cfg['model'].setdefault('options', {}).setdefault('compare_probability', False)
        return cfg

    def validate_config(self, cfg):
        if cfg['model']['dtype'] != 'float32':
            raise ValueError(self.label + ' supports float32 inference only')
        options = cfg['model'].get('options', {})
        if (set(options) != {'compare_probability'} or type(options['compare_probability']) is not bool
                or cfg['inputs'].get('options')):
            raise ValueError(self.label + ' model.options accepts boolean compare_probability only; architecture is sealed in config.json')
        if cfg['accuracy']['task_score']:
            raise ValueError(self.label + ' synthetic inputs have no task labels')

    def validate_assets(self, cfg):
        path = cfg['model'].get('path')
        required = ('config.json', 'model.safetensors', 'assets.json')
        if not path or any(not (Path(path) / name).is_file() for name in required):
            raise ValueError(self.label + ' requires sealed local assets; run python3 -m benchmarks.' +
                             self.asset_model + '.prepare_assets --output ASSET_DIRECTORY, then set model.path')
        config = self.architecture(cfg['model']['path'])
        if cfg['runtime'].get('parallelism') == 'tp':
            self.validate_tp_architecture(config, len(cfg['runtime']['devices']))
        root = Path(cfg['model']['path'])
        manifest = read_json(root / 'assets.json')
        if manifest.get('schema_version') != 1 or manifest.get('model') != self.asset_model:
            raise ValueError('invalid ' + self.label + ' asset manifest')
        if set(manifest.get('files', {})) != {'config.json', 'model.safetensors'}:
            raise ValueError('incomplete ' + self.label + ' sealed assets')
        for name, sha in manifest['files'].items():
            if (root / name).is_symlink() or file_hash(root / name) != sha:
                raise ValueError(self.label + ' asset digest mismatch: ' + name)
        if not Path(cfg['inputs']['path']).is_file():
            raise ValueError(self.label + ' input JSONL is missing')

    def asset_fingerprints(self, cfg):
        return {n: file_hash(Path(cfg['model']['path']) / n)
                for n in ('assets.json', 'config.json', 'model.safetensors')}

    def load_cpu(self, cfg):
        from safetensors.torch import load_file
        model = self.build_model(self.architecture(cfg['model']['path']))
        model.load_state_dict(load_file(str(Path(cfg['model']['path']) / 'model.safetensors'), device='cpu'), strict=True)
        return model.eval().requires_grad_(False)

    def load(self, cfg, device, parallel=None):
        model = self.load_cpu(cfg)
        if parallel is not None:
            model = self.shard_model(model, self.architecture(cfg['model']['path']), parallel)
        return model.to(device)

    def load_model(self, cfg, device, session, parallel=None):
        return super().load_model(cfg, device, session, parallel) if parallel else self.load_via_cpu(cfg, device, session)

    def comparison_outputs(self, cfg, outputs):
        names = ['hidden', 'probability'] if cfg['model']['options']['compare_probability'] else ['hidden']
        return {name: outputs[name] for name in names}

    def output_transfer(self, outputs):
        return {'probability': outputs['probability']}

    def validate_outputs(self, outputs):
        import torch
        p = outputs['probability']
        if p.ndim != 1 or p.dtype != torch.float32 or not bool((torch.isfinite(p) & (p >= 0) & (p <= 1)).all()):
            raise ValueError(self.label + ' probability must be finite FP32 [B] in [0,1]')
        h = outputs['hidden']
        if h.ndim != 2 or h.shape[0] != p.shape[0] or h.dtype != torch.float32 or not bool(torch.isfinite(h).all()):
            raise ValueError(self.label + ' hidden must be finite FP32 [B,H] matching probability samples')

    def communication_roles(self, model):
        return {name: ('embedding_output' if style == 'embedding_dim_all_gather' else 'mlp_output')
                for name, style in getattr(model, '_tp_plan', {}).items()}

    def expected_collectives(self, model):
        return len(model._tp_plan) if hasattr(model, '_tp_plan') else None

    def parallel_metadata(self, cfg, model):
        return {'plan': model._tp_plan, 'rank': model._tp_rank, 'degree': model._tp_degree,
                'parameter_shapes': {n: list(t.shape) for n, t in model.named_parameters()},
                'output_layout': 'replicated complete embedding, MLP hidden and probability on each rank'}

    def oracle_layout(self, cfg, model):
        return {'tp_plan': model._tp_plan,
                'boundary_contract': 'complete hidden after reductions; compare each rank without shard concatenation'}

    def reference_contract(self, cfg, files=None):
        value = {'semantics': self.semantics,
                 **{g: {k: v for k, v in cfg[g].items() if k != 'path' and not (k == 'options' and not v)}
                    for g in ('model', 'inputs')},
                 'model_files': self.asset_fingerprints(cfg) if files is None else files,
                 'input_sha256': file_hash(cfg['inputs']['path']), 'seed': cfg['runtime']['seed']}
        layout = self.input_layout(cfg)
        if layout is not None:
            value['input_layout'] = layout
        return value
