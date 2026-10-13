"""Dense and ragged categorical recommendation adapter; no vendor-specific execution."""
import math

from benchmarks.dlrm.config import architecture, interaction_width, validate_tp_architecture
from benchmarks.recommendation.adapter import RecommendationAdapter, read_samples, categorical_ids
from runtime.workload import BoundarySpec

MODULES = ['bottom_mlp', 'embeddings', 'interaction', 'top_mlp']


def pack_samples(rows):
    import torch
    inputs = {'dense': torch.tensor([r['dense'] for r in rows], dtype=torch.float32)}
    for field in range(len(rows[0]['sparse'])):
        values, offsets = [], [0]
        for row in rows:
            values.extend(row['sparse'][field])
            offsets.append(len(values))
        inputs[f'values_{field}'] = torch.tensor(values, dtype=torch.int64)
        inputs[f'offsets_{field}'] = torch.tensor(offsets, dtype=torch.int64)
    return inputs


class DLRMAdapter(RecommendationAdapter):
    label = 'DLRM'
    asset_model = 'dlrm'
    architecture = staticmethod(architecture)
    validate_tp_architecture = staticmethod(validate_tp_architecture)
    semantics = 'classic-dlrm-sum-bags-fp32-head-input-hidden-v1'
    timing = 'resident-dense-ragged-synchronized-forward-host-envelope-v1'
    parallel_strategy = 'recommendation_embedding_dim_gather_uneven_mlp_input_reduce_v1'

    @staticmethod
    def build_model(config):
        from benchmarks.dlrm.model import DLRM
        return DLRM(**config)

    @staticmethod
    def shard_model(model, config, parallel):
        from benchmarks.dlrm.parallel import shard
        return shard(model, config, parallel)

    def input_layout(self, cfg):
        config = architecture(cfg['model']['path'])
        specs = {'dense': {'kind': 'batch', 'dtype': 'torch.float32', 'sample_shape': [config['dense_dim']]}}
        for i, cardinality in enumerate(config['field_dims']):
            specs[f'values_{i}'] = {'kind': 'ragged_values', 'dtype': 'torch.int64',
                                  'offsets': f'offsets_{i}', 'cardinality': cardinality}
            specs[f'offsets_{i}'] = {'kind': 'ragged_offsets', 'dtype': 'torch.int64', 'values': f'values_{i}'}
        return {'schema_version': 1, 'tensors': specs}

    def prepare_inputs(self, cfg):
        config = architecture(cfg['model']['path'])
        rows = read_samples(cfg['inputs']['path'], {'id', 'dense', 'sparse'}, self.label)
        for row in rows:
            if (not isinstance(row['dense'], list) or len(row['dense']) != config['dense_dim']
                    or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3.4028234663852886e38
                           for x in row['dense'])
                    or not isinstance(row['sparse'], list) or len(row['sparse']) != len(config['field_dims'])
                    or any(not categorical_ids(ids, cardinality) for ids, cardinality
                           in zip(row['sparse'], config['field_dims']))):
                raise ValueError('DLRM requires finite FP32 dense features and legal sparse bags in field order')
        size = cfg['inputs']['batch_size']
        batches = [{'ids': [r['id'] for r in rows[i:i+size]], 'inputs': pack_samples(rows[i:i+size])}
                   for i in range(0, len(rows), size)]
        from runtime.input_layout import validate_tensors
        for batch in batches:
            validate_tensors(batch['inputs'], len(batch['ids']), self.input_layout(cfg))
        return batches, rows

    def validate_batches(self, batches, samples):
        import torch
        super().validate_batches(batches, samples)
        expected = {r['id']: r for r in samples}
        for batch in batches:
            wanted = pack_samples([expected[s] for s in batch['ids']])
            if set(wanted) != set(batch['inputs']) or any(
                    batch['inputs'][n].dtype != t.dtype or not torch.equal(batch['inputs'][n], t)
                    for n, t in wanted.items()):
                raise ValueError('DLRM archive differs from sealed samples')

    def batch_metadata(self, batch, index=None):
        stats = []
        for i in range((len(batch['inputs']) - 1) // 2):
            offsets = batch['inputs'][f'offsets_{i}']
            lengths = (offsets[1:] - offsets[:-1]).tolist()
            stats.append({'field': i, 'indices': batch['inputs'][f'values_{i}'].numel(),
                          'bag_lengths': lengths, 'min': min(lengths), 'max': max(lengths),
                          'empty_bags': lengths.count(0)})
        return {**super().batch_metadata(batch, index), 'sparse_fields': stats}

    def forward(self, model, inputs):
        return model(**inputs)

    @staticmethod
    def accuracy_layers(config):
        return ['bottom_mlp', 'embeddings', 'interaction'] + [f'top_mlp.{2*i+1}' for i in range(len(config['top_mlp_dims']))]

    def selected_modules(self, model, names):
        choices = dict(model.named_modules())
        chosen = MODULES if names == ['all'] else names
        if not chosen or len(set(chosen)) != len(chosen) or any(n not in choices or not n for n in chosen):
            raise ValueError('unknown or duplicate DLRM module selection')
        return {n: choices[n] for n in chosen}

    def instrumentation(self, cfg):
        names = cfg['performance']['layers']
        return list(MODULES) if names == ['all'] else names

    def boundary_specs(self, cfg, model=None, names=None):
        config = architecture(cfg['model']['path'])
        defaults = self.accuracy_layers(config)
        complete = cfg.get('_command') in ('preview', 'performance', 'performance-baseline', 'export')
        requested = cfg['accuracy']['layers'] if names is None else names
        chosen = defaults if requested == ['all'] else requested
        if complete and (names is None or names == ['all']):
            chosen = list(dict.fromkeys([*MODULES, *defaults, *[n for section in ('accuracy', 'performance')
                                      for n in cfg[section]['layers'] if n != 'all']]))
        shapes = {'bottom_mlp': [config['embed_dim']],
                  'embeddings': [len(config['field_dims']), config['embed_dim']],
                  'interaction': [interaction_width(config)], 'top_mlp': [1]}
        shapes.update({f'embeddings.{i}': [config['embed_dim']] for i in range(len(config['field_dims']))})
        for block, key in [('bottom_mlp', 'bottom_mlp_dims'), ('top_mlp', 'top_mlp_dims')]:
            for i, width in enumerate(config[key]):
                shapes.update({f'{block}.{2*i}': [width], f'{block}.{2*i+1}': [width]})
        shapes[f"top_mlp.{2*len(config['top_mlp_dims'])}"] = [1]
        if not chosen or len(set(chosen)) != len(chosen) or any(n not in shapes for n in chosen):
            raise ValueError('unknown or duplicate DLRM module selection')
        if model is not None:
            self.selected_modules(model, chosen)
        outputs = [BoundarySpec('hidden', 'model', sample_shape=[config['top_mlp_dims'][-1]])]
        if cfg['model']['options']['compare_probability']:
            outputs.append(BoundarySpec('probability', 'model', sample_shape=[]))
        return outputs + [BoundarySpec(n, 'layer', sample_shape=shapes[n],
                                       output_path=(0,) if n == 'top_mlp' else ()) for n in chosen]

    def forward_inputs(self, cfg, batches, model=None):
        value = super().forward_inputs(cfg, batches, model)
        complete = cfg.get('_command') in ('preview', 'performance', 'performance-baseline', 'export')
        value['default_layer_boundaries'] = ([n for n, s in value['boundary_specs'].items() if s['level'] == 'layer']
                                             if complete else self.accuracy_layers(architecture(cfg['model']['path'])))
        return value

    def reference_boundaries(self, cfg, command='accuracy', metadata=None):
        if metadata and metadata.get('boundary_specs'):
            return super().reference_boundaries(cfg, command, metadata)
        levels = ['model', 'layer'] if command == 'preview' else cfg['accuracy']['levels']
        names = ['all'] if command == 'preview' else cfg['accuracy']['layers']
        chosen = self.accuracy_layers(architecture(cfg['model']['path'])) if names == ['all'] else names
        return [s.name for s in self.boundary_specs(cfg, names=chosen) if s.level in levels]

    def export_spec(self, cfg, batch):
        import torch
        inner, keys = self.load_cpu(cfg), list(batch)
        class Export(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner
            def forward(self, *inputs):
                return (self.inner(**dict(zip(keys, inputs)))['probability'],)
        return Export().eval(), tuple(batch[n] for n in keys), keys, ['probability']

    def metadata(self, cfg, model=None):
        config = architecture(cfg['model']['path'])
        layers = self.accuracy_layers(config)[3:]
        edges = [['bottom_mlp', 'interaction'], ['embeddings', 'interaction'], ['interaction', layers[0]],
                 *[[a, b] for a, b in zip(layers, layers[1:])], [layers[-1], 'hidden'],
                 [layers[-1], 'top_mlp'], ['top_mlp', 'probability']]
        return {**super().metadata(cfg, model), 'boundary_relations': {'kind': 'branch', 'edges': edges},
                'model_description': f"经典 DLRM FP32；{config['dense_dim']} 稠密特征、{len(config['field_dims'])} 稀疏字段；{config['interaction']} interaction",
                'input_kind': '稠密与变长离散特征',
                'input_description': f"dense: float32[B,{config['dense_dim']}]；每字段 values: int64[N]、offsets: int64[B+1]；sum 聚合，空袋为零",
                'forward_description': 'bottom MLP 与 EmbeddingBag 汇合，经 interaction、top MLP、sigmoid',
                'numeric_boundary_description': 'top MLP 最后隐藏层、标量输出头前的 hidden；probability 仅在显式开启时差分',
                'token_description': '不适用；吞吐以 samples/s 表示', 'cache_description': '无 KV cache；不适用',
                'metric_notes': ['合成资产用于数值差分与性能分析，不提供 CTR/AUC',
                                 '模型级 hidden 不包含输出头误差；显式 probability 差分结合幅度误差阅读']}
