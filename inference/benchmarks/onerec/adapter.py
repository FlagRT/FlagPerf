"""OneRec input, boundary, generation, TP and reference contracts."""
from pathlib import Path, PurePosixPath
import json
import math
from runtime.common import read_json, file_hash
from runtime.workload import ModelAdapter, BoundarySpec


class OneRecAdapter(ModelAdapter):
    semantics = 'onerec-greedy-rmsnorm-hidden-prefix-through-first-token-divergence-v1'
    timing = 'resident-input-empty-kv-fixed-greedy-generation-synchronized-host-envelope-v1'
    execution_kind = 'autoregressive'
    parallel_strategy = 'transformers_qwen3_causallm_gathered_head_v1'
    throughput_unit = 'generated_tokens'
    reference_dtype = 'bfloat16'
    transfer_key = 'output'
    transfer_objects = ['final_hidden', 'generated_ids']
    transfer_scope = 'first batch extra generation; final hidden and generated token IDs copy only'

    def normalize_config(self, cfg):
        cfg['model']['options'].setdefault('max_new_tokens', 32)
        return cfg

    def validate_config(self, cfg):
        if cfg['model']['dtype'] != 'bfloat16' or cfg['model']['attention'] != 'eager':
            raise ValueError('OneRec supports BF16 eager attention')
        options = cfg['model'].get('options', {})
        if set(options) != {'max_new_tokens'} or type(options['max_new_tokens']) is not int or options['max_new_tokens'] < 1:
            raise ValueError('OneRec model.options requires positive max_new_tokens only')
        if cfg['inputs'].get('options') or cfg['inputs']['padding_side'] not in ('left', 'right'):
            raise ValueError('OneRec uses plain text and explicit left/right padding')
        if type(cfg['inputs']['max_length']) is not int or cfg['inputs']['max_length'] < 1:
            raise ValueError('OneRec max_length must be positive')
        if cfg['inputs']['max_length']+options['max_new_tokens'] > 40960:
            raise ValueError('OneRec prompt plus generation exceeds context capacity')
        if cfg['accuracy']['task_score']:
            raise ValueError('OneRec task scoring is not implemented')

    def validate_assets(self, cfg):
        if not cfg['model'].get('path'):
            raise ValueError('set model.path or pass --model-path with a local OneRec directory')
        root = Path(cfg['model']['path'])
        architecture = read_json(root/'config.json')
        if (architecture.get('architectures') != ['Qwen3ForCausalLM'] or
                [architecture.get(n) for n in ('hidden_size','num_hidden_layers','vocab_size','num_attention_heads','num_key_value_heads','head_dim','intermediate_size')] != [4096,36,176384,32,8,128,12288]
                or architecture.get('tie_word_embeddings') is not False or architecture.get('attention_bias') is not False):
            raise ValueError('OneRec-8B architecture mismatch')
        for n in ('tokenizer_config.json','generation_config.json'):
            read_json(root/n)
        weights = read_json(root/'model.safetensors.index.json').get('weight_map', {})
        if not weights or 'lm_head.weight' not in weights:
            raise ValueError('OneRec requires indexed complete CausalLM weights')
        expected={'model.embed_tokens.weight':[176384,4096],'model.norm.weight':[4096],'lm_head.weight':[176384,4096]}
        for index in range(36):
            shapes={'self_attn.q_proj':[4096,4096],'self_attn.k_proj':[1024,4096],'self_attn.v_proj':[1024,4096],
                    'self_attn.o_proj':[4096,4096],'self_attn.q_norm':[128],'self_attn.k_norm':[128],
                    'mlp.gate_proj':[12288,4096],'mlp.up_proj':[12288,4096],'mlp.down_proj':[4096,12288],
                    'input_layernorm':[4096],'post_attention_layernorm':[4096]}
            expected.update({f'model.layers.{index}.{n}.weight':s for n,s in shapes.items()})
        if set(weights)!=set(expected):
            raise ValueError('incomplete or unexpected OneRec weight index; random initialization is not allowed')
        for name in set(weights.values()):
            if (not isinstance(name, str) or '\\' in name or PurePosixPath(name).is_absolute()
                    or any(p in ('', '.', '..') for p in name.split('/')) or not name.endswith('.safetensors')
                    or (root/name).is_symlink() or not (root/name).is_file()
                    or not (root/name).resolve().is_relative_to(root.resolve())):
                raise ValueError('unsafe or missing OneRec weight shard')
            size=(root/name).stat().st_size
            with (root/name).open('rb') as stream:
                length=int.from_bytes(stream.read(8),'little')
                if not 1<=length<=16*2**20 or length+8>size:
                    raise ValueError('invalid OneRec safetensors header')
                header=json.loads(stream.read(length))
            entries={n:v for n,v in header.items() if n!='__metadata__'}
            if set(entries)!={n for n,s in weights.items() if s==name}:
                raise ValueError('OneRec shard header differs from index')
            for tensor,value in entries.items():
                offsets=value.get('data_offsets')
                if (value.get('dtype')!='BF16' or value.get('shape')!=expected[tensor]
                        or not isinstance(offsets,list) or len(offsets)!=2 or any(type(o) is not int for o in offsets)
                        or not 0<=offsets[0]<=offsets[1]<=size-8-length
                        or offsets[1]-offsets[0]!=math.prod(expected[tensor])*2):
                    raise ValueError('OneRec weight tensor shape/dtype/storage mismatch: '+tensor)
        if not Path(cfg['inputs']['path']).is_file():
            raise ValueError('OneRec input file is missing')

    def asset_fingerprints(self, cfg):
        root = Path(cfg['model']['path'])
        return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*'))
                if p.is_file() and '.cache' not in p.relative_to(root).parts
                and p.suffix in ('.json', '.safetensors', '.txt', '.model')}

    def prepare_inputs(self, cfg):
        from benchmarks.onerec.model import tokenize
        return tokenize(cfg)

    def load_cpu(self, cfg):
        from benchmarks.onerec.model import load_cpu
        return load_cpu(cfg)

    def load(self, cfg, device, parallel=None):
        if parallel is None:
            return self.load_cpu(cfg).to(device)
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig
        from benchmarks.onerec.model import configure, tensor_parallel_plan
        architecture = AutoConfig.from_pretrained(cfg['model']['path'],local_files_only=True,trust_remote_code=False)
        mesh = parallel.device_mesh()
        model = AutoModelForCausalLM.from_pretrained(cfg['model']['path'], local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager',
            tp_plan=tensor_parallel_plan(architecture), device_mesh=mesh).eval()
        if (model._tp_plan.get('lm_head') != 'colwise_gather_output'
                or list(model.lm_head.weight.shape) != [176384//mesh.size(),4096]):
            raise ValueError('OneRec TP loader did not shard the vocabulary head as declared')
        return configure(model, cfg)

    def load_model(self, cfg, device, session, parallel=None):
        return super().load_model(cfg, device, session, parallel) if parallel else self.load_via_cpu(cfg, device, session)

    def open_execution(self, model, inputs, mode='accuracy'):
        from benchmarks.onerec.model import GreedyExecution
        return GreedyExecution(model, inputs, mode)

    def execution_steps(self, cfg, mode='performance'):
        return cfg['model']['options']['max_new_tokens']

    def forward(self, model, inputs):
        from runtime.execution import run_execution
        return run_execution(self, model, inputs, 'performance')[0]

    def block_names(self, cfg):
        return [f'model.layers.{i}' for i in range(36)]

    def selected_modules(self, model, names):
        chosen = [f'model.layers.{i}' for i in range(len(model.model.layers))]+['model.norm','lm_head'] if names == ['all'] else names
        modules = dict(model.named_modules())
        if not chosen or len(chosen) != len(set(chosen)) or any(n not in modules or not n for n in chosen):
            raise ValueError('unknown or duplicate OneRec modules')
        return {n: modules[n] for n in chosen}

    def boundary_specs(self, cfg, model=None, names=None):
        defaults = self.block_names(cfg)+['model.norm','lm_head']
        chosen = defaults if names in (None, ['all']) else names
        if names is None:
            extras = [n for group in ('accuracy','performance') for n in cfg.get(group,{}).get('layers',[]) if n != 'all']
            chosen = list(dict.fromkeys([*defaults,*extras]))
        specs = [BoundarySpec('hidden','model',valid_region={'input':'attention_mask','axis':0},
                             sample_shape=['valid_tokens',4096],phase_shapes={'prefill':['valid_tokens',4096],'decode':[1,4096]})]
        for name in chosen:
            length = 1 if name == 'lm_head' else 'valid_tokens'
            tail = [176384] if name == 'lm_head' else [4096] if name in defaults else [None]
            if name.endswith(('.self_attn.q_norm','.self_attn.k_norm')):
                tail = [None, model.config.head_dim if model is not None else 128]
            prompt = [length,*tail]
            specs.append(BoundarySpec(name,'layer',
                valid_region={'input':'lm_head_mask' if name == 'lm_head' else 'attention_mask','axis':0},
                layout='replicated' if name in defaults else 'rank_local',sample_shape=prompt,
                phase_shapes={'prefill':prompt,'decode':[1,*tail]}))
        return specs

    def module_value(self, output, spec):
        if hasattr(output,'last_hidden_state'):
            output = output.last_hidden_state
        elif isinstance(output,(tuple,list)):
            output = output[0]
        return super().module_value(output, spec)

    def batch_metadata(self, batch, index=None):
        return {**super().batch_metadata(batch, index),
                'valid_tokens': batch['inputs']['attention_mask'].sum(1).tolist()}

    def sample_counts(self, batch):
        return len(batch['ids']), int(batch['inputs']['attention_mask'].sum())

    def valid_regions(self, batch):
        return {'attention_mask': batch['inputs']['attention_mask']}

    def forward_inputs(self, cfg, batches, model=None):
        return {**super().forward_inputs(cfg, batches, model),
                'execution_kind': self.execution_kind, 'generation': self.metadata(cfg, model)['generation'],
                'default_layer_boundaries':self.block_names(cfg)+['model.norm','lm_head'],
                'output_dtype':'torch.'+cfg['model']['dtype']}

    def validate_batches(self, batches, samples):
        import torch
        super().validate_batches(batches, samples)
        for batch in batches:
            if set(batch['inputs']) != {'input_ids','attention_mask'}:
                raise ValueError('OneRec inputs must be tokens and mask')
            t, m = batch['inputs']['input_ids'], batch['inputs']['attention_mask']
            if (t.dtype != torch.int64 or m.dtype != torch.int64 or t.ndim != 2 or t.shape != m.shape
                    or t.shape[0] != len(batch['ids']) or not bool(((m == 0)|(m == 1)).all())
                    or not bool(m.sum(1).gt(0).all()) or not bool(((t >= 0)&(t < 176384)).all())):
                raise ValueError('invalid OneRec token/mask archive')

    def communication_roles(self, model):
        return {n: ('lm_head_output' if n == 'lm_head' else 'attention_output' if n.endswith('self_attn.o_proj') else 'mlp_output')
                for n, _ in model.named_modules() if n == 'lm_head' or n.endswith(('self_attn.o_proj','mlp.down_proj'))}

    def expected_collectives(self, model):
        if model._tp_plan.get('lm_head') != 'colwise_gather_output':
            raise ValueError('OneRec TP communication requires the declared gathered vocabulary head')
        return (2*len(model.model.layers)+1)*model._flagperf_generation['max_new_tokens']

    def parallel_metadata(self, cfg, model):
        return {'plan': model._tp_plan, 'parameter_shapes': {n:list(t.shape) for n,t in model.named_parameters()},
                'output_layout': 'complete hidden and gathered vocabulary logits on each rank'}

    def oracle_layout(self, cfg, model):
        return {'tp_plan':model._tp_plan, 'boundary_contract':'replicated complete blocks and final RMSNorm hidden; no shard concatenation'}

    def reference_contract(self, cfg, files=None):
        return {'semantics':self.semantics,
                **{g:{k:v for k,v in cfg[g].items() if k != 'path'} for g in ('model','inputs')},
                'model_files': self.asset_fingerprints(cfg) if files is None else files,
                'input_sha256':file_hash(cfg['inputs']['path']), 'seed':cfg['runtime']['seed']}

    def metadata(self, cfg, model=None):
        value = super().metadata(cfg, model)
        blocks = self.block_names(cfg)
        generation = model.generation_config.to_dict() if model is not None else read_json(Path(cfg['model']['path'])/'generation_config.json')
        eos = generation.get('eos_token_id')
        eos = eos if isinstance(eos,list) else [] if eos is None else [eos]
        return {**value, 'schema_version':2, 'execution_kind':self.execution_kind,
                'default_layer_boundaries':blocks+['model.norm','lm_head'],
                'generation':{'max_new_tokens':cfg['model']['options']['max_new_tokens'], 'decoding':'greedy',
                              'accuracy_stop':'eos_or_max_new_tokens', 'performance_stop':'fixed_max_new_tokens',
                              'preview_modes':['accuracy','performance'], 'eos_token_ids':eos, 'vocab_size':176384},
                'boundary_relations':{'kind':'sequential','edges':[list(p) for p in zip(blocks,blocks[1:])]},
                'model_description':'OneRec-8B；36 blocks，4096 hidden；BF16 greedy generation',
                'numeric_boundary_description':'LM head 前、最终 RMSNorm 后的 hidden；prefill 的有效提示位置和 decode 的当前有效位置',
                'input_kind':'文本', 'input_description':'OneRec tokenizer 封存的 input_ids、attention_mask；原始文本',
                'forward_description':'空 KV cache prefill、LM head greedy 选 token、逐步 decode 与同步',
                'token_description':'输入与生成 token 分别计数；prefill 产生第一个生成 token',
                'cache_description':'use_cache=True；每请求新建 DynamicCache，结束/异常释放；KV storage 按 rank 实测'}
