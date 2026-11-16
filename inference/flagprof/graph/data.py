# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Versioned graph bundle metadata, integrity and tensor-free graph projections."""
import hashlib
import ast
import json
from pathlib import Path

SCHEMA = 'flagprof-graph/1'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                    allow_nan=False) + '\n')


def file_info(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def read_manifest(path):
    value = json.loads((Path(path) / 'manifest.json').read_text())
    if value.get('schema') != SCHEMA:
        raise ValueError('unsupported graph bundle schema')
    return value


def verify_graph(path):
    root = Path(path).resolve()
    value = read_manifest(root)
    if value['status'] != 'completed':
        raise ValueError('graph export is not completed: ' + value['status'])
    if not value.get('formats') or set(value['formats']) != set(value['requested_formats']):
        raise ValueError('graph requested formats are incomplete')
    files = value['files']
    required = {'index.html'}
    for fmt, result in value['formats'].items():
        if result['status'] != 'completed':
            raise ValueError('graph format is not completed: ' + fmt)
        required.add(fmt + '.json')
        if fmt == 'fx':
            required.add('model.fx.txt')
        if value['bundle'] == 'full':
            required.add('model.pt2' if fmt == 'fx' else 'model.onnx')
            required.add('example.pt')
            required.update(result.get('external_data', []))
    if not required.issubset(files):
        raise ValueError('graph manifest missing required files: ' + str(sorted(required - files.keys())))
    for name, expected in files.items():
        target = (root / name).resolve()
        if root not in target.parents:
            raise ValueError('graph file outside bundle: ' + name)
        if file_info(target) != expected:
            raise ValueError('graph file checksum mismatch: ' + name)
    return dict(status='verified', files=len(files), bundle=value['bundle'])


def tensor_meta(value):
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return dict(shape=[int(d) for d in value.shape], dtype=str(value.dtype))
    if isinstance(value, (tuple, list)):
        return [tensor_meta(v) for v in value]
    if isinstance(value, dict):
        return {str(k): tensor_meta(v) for k, v in value.items()}
    return None


def fx_graph(program):
    import torch
    def argument(value):
        if isinstance(value, torch.fx.Node):
            return {'node': value.name}
        if isinstance(value, torch.Tensor):
            return tensor_meta(value)
        if isinstance(value, (tuple, list)):
            return [argument(v) for v in value]
        if isinstance(value, dict):
            return {str(k): argument(v) for k, v in value.items()}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and __import__('math').isfinite(value):
            return value
        return str(value)
    nodes = []
    for n in program.graph.nodes:
        modules = [dict(path=v[0], type=str(v[1])) for v in n.meta.get('nn_module_stack', {}).values()]
        nodes.append(dict(id=n.name, op=n.op, target=str(n.target),
                          inputs=[v.name for v in n.all_input_nodes], users=[v.name for v in n.users],
                          args=argument(n.args), kwargs=argument(n.kwargs),
                          value=tensor_meta(n.meta.get('val')), modules=modules,
                          source=n.meta.get('stack_trace')))
    return dict(format='fx', nodes=nodes, parameters={k:tensor_meta(v) for k,v in program.state_dict.items()},
                signature=str(program.graph_signature), range_constraints=str(program.range_constraints))


def onnx_graph(model):
    import onnx
    def value_info(v):
        t = v.type.tensor_type
        return dict(shape=[d.dim_value if d.HasField('dim_value') else d.dim_param or None for d in t.shape.dim],
                    dtype=onnx.TensorProto.DataType.Name(t.elem_type)) if t.elem_type else None
    def graph(g):
        values = {v.name:value_info(v) for v in (*g.input, *g.output, *g.value_info)}
        ids = ['onnx-' + str(i) for i in range(len(g.node))]
        producers = {name:ids[i] for i,n in enumerate(g.node) for name in n.output if name}
        nodes = []
        for i,n in enumerate(g.node):
            metadata = {v.key:v.value for v in n.metadata_props}
            modules = []
            try:
                scopes = ast.literal_eval(metadata.get('pkg.torch.onnx.name_scopes', '[]'))
                classes = ast.literal_eval(metadata.get('pkg.torch.onnx.class_hierarchy', '[]'))
                if isinstance(scopes,list) and isinstance(classes,list) and len(scopes)==len(classes):
                    modules = [dict(path=s,type=c) for s,c in zip(scopes[:-1],classes[:-1])
                               if isinstance(s,str) and isinstance(c,str) and s]
            except (ValueError, SyntaxError):
                pass
            subgraphs = []
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    subgraphs.append(dict(name=a.name, graph=graph(a.g)))
                elif a.type == onnx.AttributeProto.GRAPHS:
                    subgraphs.extend(dict(name=a.name, graph=graph(x)) for x in a.graphs)
            nodes.append(dict(id=ids[i], name=n.name, op=n.op_type, target=n.domain+'::'+n.op_type,
                              inputs=list(dict.fromkeys(producers[x] for x in n.input if x in producers)), users=[],
                              tensor_inputs=list(n.input), tensor_outputs=list(n.output),
                              value=[values.get(x) for x in n.output], modules=modules,
                              metadata=metadata, subgraphs=subgraphs))
        by_id = {n['id']:n for n in nodes}
        for n in nodes:
            for src in n['inputs']:
                by_id[src]['users'].append(n['id'])
        params = {t.name:dict(shape=list(t.dims), dtype=onnx.TensorProto.DataType.Name(t.data_type)) for t in g.initializer}
        return dict(nodes=nodes, inputs={v.name:values[v.name] for v in g.input},
                    outputs={v.name:values[v.name] for v in g.output}, parameters=params)
    return dict(format='onnx', **graph(model.graph), ir_version=model.ir_version,
                opsets={v.domain:v.version for v in model.opset_import},
                functions=[dict(domain=f.domain, name=f.name, inputs=list(f.input), outputs=list(f.output),
                                nodes=[dict(name=n.name, domain=n.domain, op=n.op_type,
                                            inputs=list(n.input), outputs=list(n.output)) for n in f.node])
                           for f in model.functions])
