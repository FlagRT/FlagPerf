"""Sealed workload protocol: offline consumption requires no model implementation."""
from pathlib import Path
from runtime.common import read_json
from runtime.workload import sample_view


def validate_workload(value, contract=None):
    if not isinstance(value,dict) or value.get('schema_version') not in (1,2):
        raise ValueError('unsupported sealed workload metadata')
    specs = value.get('boundaries')
    from runtime.input_layout import validate_layout
    validate_layout(value.get('input_layout'))
    if contract and value.get('input_layout') != contract.get('input_layout'):
        raise ValueError('sealed input layout differs from reference contract')
    if not isinstance(specs,list) or not specs or len({s['name'] for s in specs}) != len(specs):
        raise ValueError('invalid sealed boundary coverage')
    for s in specs:
        if (not isinstance(s.get('name'),str) or not s['name'] or s.get('level') not in ('model','layer') or
                type(s.get('sample_axis')) is not int or s['sample_axis'] < 0 or
                s.get('layout') not in ('replicated','rank_local','sharded') or
                (s.get('sample_shape') is not None and not isinstance(s['sample_shape'],list))):
            raise ValueError('invalid sealed boundary specification')
        if any(d is not None and d != 'valid_tokens' and (type(d) is not int or d <= 0) for d in s.get('sample_shape') or []):
            raise ValueError('invalid sealed sample shape')
        if 'phase_shapes' in s and (not isinstance(s['phase_shapes'],dict) or set(s['phase_shapes'])!={'prefill','decode'}
                or any(not isinstance(shape,list) or not shape or any(d is not None and d!='valid_tokens' and
                    (type(d) is not int or d<1) for d in shape) for shape in s['phase_shapes'].values())):
            raise ValueError('invalid sealed execution phase shapes')
        r = s.get('valid_region')
        if r is not None and (not isinstance(r,dict) or set(r) != {'input','axis'} or
                              not isinstance(r['input'],str) or type(r['axis']) is not int or r['axis'] < 0):
            raise ValueError('invalid sealed valid region')
    if contract and (value.get('model_adapter') != contract['model']['name'] or
                     value.get('semantics') != contract['semantics'] or
                     ('timing' in contract and value.get('timing') != contract['timing'])):
        raise ValueError('sealed workload differs from reference contract')
    if value.get('execution_kind') == 'autoregressive':
        g=value.get('generation',{})
        if (value['schema_version']!=2 or type(g.get('max_new_tokens')) is not int or g['max_new_tokens']<1
                or g.get('decoding')!='greedy' or g.get('accuracy_stop')!='eos_or_max_new_tokens'
                or g.get('performance_stop')!='fixed_max_new_tokens'
                or type(g.get('vocab_size')) is not int or g['vocab_size']<1
                or not isinstance(g.get('eos_token_ids'),list)
                or any(type(t) is not int or not 0<=t<g['vocab_size'] for t in g['eos_token_ids'])):
            raise ValueError('invalid sealed generation contract')
        if contract and g['max_new_tokens']!=contract['model']['options']['max_new_tokens']:
            raise ValueError('sealed generation length differs from contract')
    return {s['name']:s for s in specs}


def package_specs(root, manifest):
    if manifest.get('protocol_schema_version', manifest['schema_version']) in (2,3):
        workload = read_json(Path(root)/'prepared/workload.json')
        if manifest.get('workload') != workload: raise ValueError('sealed workload metadata mismatch')
        return validate_workload(workload,manifest['contract'])
    # Explicit read-only compatibility; schema 1 belongs only to Qwen3.
    from runtime.adapters import create
    if manifest['contract']['model']['name'] != 'qwen3_embedding_0.6b':
        raise ValueError('schema 1 is only supported for legacy Qwen3')
    return create('models','qwen3_embedding_0.6b').legacy_specs(manifest['boundaries'])


def validate_archive(batches, samples, layout=None):
    from runtime.input_layout import validate_tensors
    if not isinstance(batches,list) or not batches: raise ValueError('empty fixed input archive')
    ids = []
    for b in batches:
        if not isinstance(b.get('ids'),list) or not b['ids'] or not isinstance(b.get('inputs'),dict) or not b['inputs']:
            raise ValueError('invalid fixed input batch')
        validate_tensors(b['inputs'], len(b['ids']), layout)
        ids.extend(b['ids'])
    if ids != [s['id'] for s in samples] or len(ids) != len(set(ids)):
        raise ValueError('fixed input sample order/coverage mismatch')


def batch_regions(batch, specs):
    return {s['valid_region']['input']:batch['inputs'][s['valid_region']['input']]
            for s in specs.values() if s.get('valid_region')}


def validate_outputs(value, batch, names, specs=None):
    import torch
    specs = specs or value.get('boundary_specs')
    if not specs: raise ValueError('output lacks explicit boundary specifications')
    if value.get('ids') != batch['ids'] or not set(names) <= set(value.get('outputs',{})):
        raise ValueError('candidate IDs/order or boundary coverage mismatch')
    regions = batch_regions(batch,specs)
    observed = value.get('valid_regions',{})
    if any(k not in observed or not torch.equal(t,observed[k]) for k,t in regions.items()):
        raise ValueError('candidate valid region/mask differs from sealed inputs')
    for n in names:
        s,t = specs[n],value['outputs'][n]
        if (not isinstance(t,torch.Tensor) or t.device.type != 'cpu' or
                s['sample_axis'] >= t.ndim or t.shape[s['sample_axis']] != len(batch['ids'])):
            raise ValueError('invalid output tensor/sample axis: '+n)
        if value.get('boundary_specs',{}).get(n) != s:
            raise ValueError('candidate boundary specification differs: '+n)
        for i in range(len(batch['ids'])):
            row = sample_view(t,i,s,regions)
            shape = s.get('sample_shape')
            if shape is not None and (len(shape) != row.ndim or any(
                    type(d) is int and d != actual for d,actual in zip(shape,row.shape))):
                raise ValueError('incomplete/sharded boundary shape: '+n)
