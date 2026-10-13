"""Sealed input layouts; validation never imports a model or a device plugin."""


def validate_layout(layout):
    if layout is None:
        return None
    if (not isinstance(layout, dict) or set(layout) != {'schema_version', 'tensors'}
            or layout['schema_version'] != 1 or not isinstance(layout['tensors'], dict)
            or not layout['tensors']):
        raise ValueError('invalid sealed input layout')
    specs = layout['tensors']
    for name, spec in specs.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise ValueError('invalid input tensor specification')
        kind = spec.get('kind')
        keys = {'batch': {'kind', 'dtype', 'sample_shape'},
                'ragged_values': {'kind', 'dtype', 'offsets', 'cardinality'},
                'ragged_offsets': {'kind', 'dtype', 'values'}}
        if kind not in keys or set(spec) != keys[kind]:
            raise ValueError('invalid input tensor layout: ' + name)
        if kind == 'batch':
            if (spec['dtype'] not in ('torch.float32', 'torch.float16', 'torch.bfloat16', 'torch.int64', 'torch.bool')
                    or not isinstance(spec['sample_shape'], list)
                    or any(type(d) is not int or d < 1 for d in spec['sample_shape'])):
                raise ValueError('invalid batched input specification: ' + name)
        else:
            if spec['dtype'] != 'torch.int64':
                raise ValueError('ragged input requires int64 values and offsets')
            peer_key = 'offsets' if kind == 'ragged_values' else 'values'
            peer = specs.get(spec[peer_key]) if isinstance(spec[peer_key], str) else None
            opposite = 'ragged_offsets' if kind == 'ragged_values' else 'ragged_values'
            reverse = 'values' if kind == 'ragged_values' else 'offsets'
            if not isinstance(peer, dict) or peer.get('kind') != opposite or peer.get(reverse) != name:
                raise ValueError('unpaired ragged input specification: ' + name)
            if kind == 'ragged_values' and (type(spec['cardinality']) is not int or spec['cardinality'] < 1):
                raise ValueError('invalid ragged input cardinality')
    return specs


def tensor_metadata(inputs, layout=None):
    specs = validate_layout(layout)
    return {name: {'shape': list(t.shape), 'dtype': str(t.dtype),
                   'sample_axis': 0 if specs is None or specs[name]['kind'] == 'batch' else None}
            for name, t in inputs.items()}


def validate_tensors(inputs, samples, layout=None):
    import torch
    specs = validate_layout(layout)
    if specs is not None and set(inputs) != set(specs):
        raise ValueError('fixed input tensors differ from sealed layout')
    for name, tensor in inputs.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.ndim < 1:
            raise ValueError('fixed inputs must be CPU tensors')
    for name, tensor in inputs.items():
        spec = specs[name] if specs is not None else None
        if spec is None:
            if tensor.shape[0] != samples:
                raise ValueError('invalid fixed input tensor sample axis')
            continue
        if str(tensor.dtype) != spec['dtype']:
            raise ValueError('fixed input tensor dtype differs: ' + name)
        if spec['kind'] == 'batch':
            if list(tensor.shape) != [samples, *spec['sample_shape']]:
                raise ValueError('fixed input tensor shape differs: ' + name)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError('nonfinite fixed input: ' + name)
        elif spec['kind'] == 'ragged_values':
            if tensor.ndim != 1 or bool(((tensor < 0) | (tensor >= spec['cardinality'])).any()):
                raise ValueError('illegal ragged categorical IDs: ' + name)
        else:
            values = inputs[spec['values']]
            if (list(tensor.shape) != [samples + 1] or tensor[0].item() != 0
                    or tensor[-1].item() != values.numel() or bool((tensor[1:] < tensor[:-1]).any())):
                raise ValueError('invalid ragged offsets: ' + name)
