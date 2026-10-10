"""Categorical DeepFM inference: head-input hidden numerics and complete CTR forward."""
from pathlib import Path

from runtime.common import read_json
from runtime.workload import BoundarySpec
from benchmarks.recommendation.adapter import RecommendationAdapter, read_samples, categorical_ids

DEFAULTS = {'field_dims': [1000,2000,500,300,50], 'embed_dim': 16,
            'mlp_dims': [400,400,400], 'dropout': 0.2}
MODULES = ['linear','embedding','fm','mlp']


def architecture(path):
    value = read_json(Path(path)/'config.json')
    if set(value) != set(DEFAULTS): raise ValueError('invalid DeepFM architecture keys')
    if (not isinstance(value['field_dims'], list) or not value['field_dims'] or
            any(type(n) is not int or n < 1 for n in value['field_dims']) or
            type(value['embed_dim']) is not int or value['embed_dim'] < 1 or
            not isinstance(value['mlp_dims'], list) or not value['mlp_dims'] or
            any(type(n) is not int or n < 1 for n in value['mlp_dims']) or
            type(value['dropout']) not in (int,float) or not 0 <= value['dropout'] < 1):
        raise ValueError('invalid DeepFM architecture dimensions/dropout')
    return value


class DeepFMAdapter(RecommendationAdapter):
    label = 'DeepFM'
    asset_model = 'deepfm'
    architecture = staticmethod(architecture)

    @staticmethod
    def build_model(config):
        from benchmarks.deepfm.model import DeepFM
        return DeepFM(**config)

    @staticmethod
    def validate_tp_architecture(config, degree):
        from benchmarks.deepfm.config import validate_tp_architecture
        validate_tp_architecture(config, degree)

    @staticmethod
    def shard_model(model, config, parallel):
        from benchmarks.deepfm.parallel import shard
        return shard(model, config, parallel)

    semantics = 'deepfm-eval-fp32-head-input-hidden-v2'
    timing = 'resident-categorical-synchronized-forward-host-envelope-v1'
    reference_dtype = 'float32'
    parallel_strategy = 'deepfm_embedding_gather_mlp_input_reduce_v1'
    throughput_unit = 'samples'
    transfer_objects = ['probability']
    boundary_relations = {'kind': 'branch', 'edges': [['embedding','fm'],['embedding','mlp'],
                            ['linear','probability'],['fm','probability'],['mlp','probability']]}


    def prepare_inputs(self, cfg):
        import torch
        dims = architecture(cfg['model']['path'])['field_dims']
        rows = read_samples(cfg['inputs']['path'], {'id', 'features'}, self.label)
        if not rows or any(set(row) != {'id','features'} or not isinstance(row['id'],str) or not row['id'] or
                           not isinstance(row['features'],list) or len(row['features']) != len(dims) or
                           any(not categorical_ids([x], d) for x,d in zip(row['features'],dims)) for row in rows):
            raise ValueError('DeepFM requires unique IDs and legal categorical features in field order')
        if len({r['id'] for r in rows}) != len(rows): raise ValueError('duplicate DeepFM sample IDs')
        size = cfg['inputs']['batch_size']
        batches = [{'ids':[r['id'] for r in rows[i:i+size]],
                    'inputs':{'x':torch.tensor([r['features'] for r in rows[i:i+size]],dtype=torch.int64)}}
                   for i in range(0,len(rows),size)]
        return batches,rows

    def validate_batches(self, batches, samples):
        import torch
        super().validate_batches(batches,samples)
        for batch in batches:
            if set(batch['inputs']) != {'x'}: raise ValueError('DeepFM input must be x')
            x = batch['inputs']['x']
            if x.dtype != torch.int64 or x.ndim != 2 or x.shape[0] != len(batch['ids']):
                raise ValueError('invalid DeepFM fixed input shape/dtype')
        expected = {r['id']:r['features'] for r in samples}
        if any(batch['inputs']['x'].tolist() != [expected[s] for s in batch['ids']] for batch in batches):
            raise ValueError('DeepFM archive differs from sealed samples')


    def forward(self, model, inputs): return model(**inputs, return_hidden=True)


    def selected_modules(self, model, names):
        choices = dict(model.named_modules())
        selected = MODULES if names == ['all'] else names
        if not selected or len(set(selected)) != len(selected) or any(n not in choices or not n for n in selected):
            raise ValueError('unknown or duplicate DeepFM module selection')
        return {n:choices[n] for n in selected}

    def instrumentation(self, cfg):
        names = cfg['performance']['layers']
        return list(MODULES) if names == ['all'] else names

    def boundary_specs(self, cfg, model=None, names=None):
        config = architecture(cfg['model']['path'])
        defaults = self.accuracy_layers(config)
        if names is None:
            complete = cfg.get('_command') in ('preview','performance','performance-baseline','export')
            chosen = list(dict.fromkeys([*MODULES,*defaults])) if complete else defaults if cfg['accuracy']['layers'] == ['all'] else cfg['accuracy']['layers']
            if complete:
                chosen = list(dict.fromkeys([*chosen,*[n for section in ('accuracy','performance') for n in cfg[section]['layers'] if n != 'all']]))
        else:
            chosen = defaults if names == ['all'] else names
            if names == ['all'] and (cfg.get('_command') in ('preview','performance','performance-baseline','export')):
                chosen = list(dict.fromkeys([*MODULES,*defaults,*[n for section in ('accuracy','performance') for n in cfg[section]['layers'] if n != 'all']]))
        shapes = {'linear':[1],'embedding':[len(config['field_dims']),config['embed_dim']], 'fm':[1],'mlp':[1]}
        shapes.update({'linear.fc':[len(config['field_dims']),1],
                       'embedding.embedding':shapes['embedding'], 'mlp.mlp':[1]})
        for index,width in enumerate(config['mlp_dims']):
            shapes.update({f'mlp.mlp.{4*index+offset}':[width] for offset in range(4)})
        shapes[f"mlp.mlp.{4*len(config['mlp_dims'])}"] = [1]
        if not chosen or len(set(chosen)) != len(chosen) or any(n not in shapes for n in chosen):
            raise ValueError('unknown or duplicate DeepFM module selection')
        if model is not None: self.selected_modules(model,chosen)
        outputs = [BoundarySpec('hidden','model',sample_shape=[config['mlp_dims'][-1]])]
        if cfg['model']['options']['compare_probability']:
            outputs.append(BoundarySpec('probability','model',sample_shape=[]))
        return outputs + [
                BoundarySpec(n,'layer',sample_shape=shapes[n],output_path=(0,) if n in ('mlp','mlp.mlp') else ()) for n in chosen]

    @staticmethod
    def accuracy_layers(config):
        return ['embedding'] + [f'mlp.mlp.{4*index+3}' for index in range(len(config['mlp_dims']))]

    def forward_inputs(self, cfg, batches, model=None):
        value = super().forward_inputs(cfg, batches, model)
        complete = cfg.get('_command') in ('preview','performance','performance-baseline','export')
        value['default_layer_boundaries'] = ([n for n,s in value['boundary_specs'].items() if s['level']=='layer']
                                             if complete else self.accuracy_layers(architecture(cfg['model']['path'])))
        return value

    def reference_boundaries(self, cfg, command='accuracy', metadata=None):
        if metadata and metadata.get('boundary_specs'):
            return super().reference_boundaries(cfg, command, metadata)
        levels = ['model','layer'] if command == 'preview' else cfg['accuracy']['levels']
        names = ['all'] if command == 'preview' else cfg['accuracy']['layers']
        chosen = self.accuracy_layers(architecture(cfg['model']['path'])) if names == ['all'] else names
        return [s.name for s in self.boundary_specs(cfg,names=chosen) if s.level in levels]


    def export_spec(self, cfg, batch):
        import torch
        inner = self.load_cpu(cfg)
        class Export(torch.nn.Module):
            def __init__(self): super().__init__(); self.inner = inner
            def forward(self,x): return (self.inner(x),)
        return Export().eval(), (batch['x'],), ['x'], ['probability']

    def metadata(self, cfg, model=None):
        config = architecture(cfg['model']['path'])
        hidden_layers = self.accuracy_layers(config)[1:]
        edges = [['embedding','fm'],['embedding','mlp'],['embedding',hidden_layers[0]],
                 *[[a,b] for a,b in zip(hidden_layers,hidden_layers[1:])],
                 [hidden_layers[-1],'hidden'],[hidden_layers[-1],'mlp'],
                 ['linear','probability'],['fm','probability'],['mlp','probability']]
        return {**super().metadata(cfg,model),
                'boundary_relations':{'kind':'branch','edges':edges},
                'model_description':'DeepFM FP32；categorical wide + FM + MLP；eval 模式',
                'input_kind':'离散特征', 'input_description':f"x: int64[B,{len(config['field_dims'])}]；各字段 ID 按词表范围验证",
                'forward_description':'linear、FM、MLP 分支求和后 sigmoid',
                'numeric_boundary_description':'MLP 最后隐藏层、标量输出头前的 hidden；probability 仅在显式开启时差分',
                'token_description':'不适用；吞吐以 samples/s 表示',
                'cache_description':'无 KV cache；不适用',
                'metric_notes':['模型级默认比较 head 前 hidden；完整前向仍计算 probability',
                                '显式比较 probability 时：正标量的 cosine 通常为 1，应同时看绝对误差']}
