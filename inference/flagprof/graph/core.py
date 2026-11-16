# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Fixed-shape CPU capture, per-format validation and portable bundle creation."""
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import time
import traceback

from .data import SCHEMA, file_info, fx_graph, onnx_graph, tensor_meta, write_json


def outputs(value):
    import torch
    values = (value,) if isinstance(value, torch.Tensor) else value
    if not isinstance(values, (tuple, list)) or not values or not all(isinstance(v, torch.Tensor) for v in values):
        raise ValueError('export outputs must be a Tensor or a nonempty flat tuple/list of Tensors')
    return tuple(values)


def compare_outputs(expected, actual, names, *, exact):
    import numpy as np
    import torch
    expected, actual = outputs(expected), outputs(actual)
    if len(expected) != len(actual) or len(expected) != len(names):
        raise ValueError('output count differs from output_names')
    checks = []
    for name, ref, got in zip(names, expected, actual):
        if ref.shape != got.shape or ref.dtype != got.dtype:
            raise ValueError('output shape/dtype mismatch: ' + name)
        if not bool(torch.isfinite(ref).all() and torch.isfinite(got).all()):
            raise ValueError('non-finite output: ' + name)
        a, b = ref.detach().double().numpy(), got.detach().double().numpy()
        atol = 1e-4 if name == 'pooled' else 1e-5
        samples = []
        count = a.shape[0] if a.ndim else 1
        for x,y in zip(a.reshape(count,-1), b.reshape(count,-1)):
            delta = np.linalg.norm(y-x); denom = np.linalg.norm(x)
            samples.append(dict(max_abs=float(np.max(np.abs(y-x))) if x.size else 0.,
                                relative_l2=float(delta/denom) if denom else (0. if not delta else None)))
        equal = bool(torch.equal(ref, got))
        passed = equal if exact else bool(torch.allclose(ref, got, rtol=1e-4, atol=atol))
        checks.append(dict(name=name, **tensor_meta(ref), exact_equal=equal, passed=passed,
                           rtol=0. if exact else 1e-4, atol=0. if exact else atol, samples=samples))
    return dict(status='passed' if all(v['passed'] for v in checks) else 'failed', outputs=checks)


def validate_onnx(path, args, kwargs, expected, input_names, output_names):
    import numpy as np
    import torch
    import onnxruntime as ort
    from torch.utils._pytree import tree_flatten
    leaves, _ = tree_flatten((args, kwargs))
    tensors = [v for v in leaves if isinstance(v, torch.Tensor)]
    if any(v.dtype == torch.bfloat16 for v in tensors) or any(v.dtype != torch.float32 for v in outputs(expected)):
        raise ValueError('ONNX runtime verification currently requires FP32 outputs and NumPy-compatible inputs')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    actual_inputs = [v.name for v in session.get_inputs()]
    actual_outputs = [v.name for v in session.get_outputs()]
    if actual_inputs != list(input_names) or actual_outputs != list(output_names):
        raise ValueError('ONNX runtime input/output names or order differ from export contract')
    values = session.run(list(output_names), {name:v.detach().cpu().numpy() for name,v in zip(input_names,tensors)})
    result = compare_outputs(expected, tuple(torch.from_numpy(np.asarray(v)) for v in values), output_names, exact=False)
    result.update(provider='CPUExecutionProvider', version=ort.__version__)
    return result


def _check(spec, formats, bundle, verify_onnx):
    import torch
    from torch.utils._pytree import tree_flatten
    from flagprof import current_attachment
    if current_attachment() is not None:
        raise ValueError('graph export requires an independent task without an active attachment')
    if not isinstance(spec.model, torch.nn.Module) or any(m.training for m in spec.model.modules()):
        raise ValueError('export model must be an eval torch.nn.Module')
    if not formats or len(set(formats)) != len(formats) or set(formats) - {'fx','onnx'}:
        raise ValueError('formats must select fx, onnx or both once')
    if bundle not in ('structure','full'):
        raise ValueError('bundle must be structure or full')
    if verify_onnx and 'onnx' not in formats:
        raise ValueError('ONNX verification requires the onnx format')
    leaves, _ = tree_flatten((spec.args, spec.kwargs))
    tensors = [v for v in leaves if isinstance(v, torch.Tensor)]
    if any(v.device.type != 'cpu' for v in (*tensors, *spec.model.parameters(), *spec.model.buffers())):
        raise ValueError('graph export requires CPU model and inputs')
    if verify_onnx and any(v.is_floating_point() and v.dtype != torch.float32
                           for v in (*tensors, *spec.model.parameters(), *spec.model.buffers())):
        raise ValueError('ONNX runtime verification currently requires FP32 model and inputs')
    if len(tensors) != len(spec.input_names):
        raise ValueError('input_names must name all tensor leaves in args/kwargs order')
    for names in (spec.input_names, spec.output_names):
        if not names or len(set(names)) != len(names) or not all(isinstance(n,str) and n for n in names):
            raise ValueError('input/output names must be unique nonempty strings')
    return tensors


def finish(root, result):
    """Publish the readable outcome after files and validation have settled."""
    from .view import write_view
    graphs = {fmt:json.loads((root/(fmt+'.json')).read_text())
              for fmt in result.get('formats', {}) if (root/(fmt+'.json')).is_file()}
    write_view(root/'index.html', result, graphs)
    result['files'] = {p.relative_to(root).as_posix():file_info(p) for p in sorted(root.rglob('*'))
                       if p.is_file() and p.name not in ('manifest.json','export.log')}
    write_json(root/'manifest.json', result)
    return result


def export_graph(spec, out, *, formats=('fx','onnx'), bundle='structure', verify_onnx=False):
    _check(spec, formats, bundle, verify_onnx)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    return export_into(spec, root, formats=formats, bundle=bundle, verify_onnx=verify_onnx)


def export_into(spec, root, *, formats, bundle, verify_onnx):
    import torch
    start = time.monotonic()
    result = dict(schema=SCHEMA, status='failed', bundle=bundle, identity=spec.identity,
                  device='cpu', shape_policy='static', requested_formats=list(formats), formats={}, input_names=list(spec.input_names),
                  output_names=list(spec.output_names), settings=dict(strict=False, dynamo=True, opset=18,
                    attention=spec.identity.get('attention'), components=spec.identity.get('components','unavailable'), verify_onnx=verify_onnx),
                  software={}, timings_s={})
    for package in ('torch','transformers','onnx','onnxscript','onnxruntime','numpy'):
        try:result['software'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:result['software'][package] = None
    work = root/'.work'
    try:
        tensors = _check(spec, formats, bundle, verify_onnx)
        result['inputs'] = {n:tensor_meta(v) for n,v in zip(spec.input_names,tensors)}
        result['input_sha256'] = {n:__import__('hashlib').sha256(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                                  for n,v in zip(spec.input_names,tensors)}
        work.mkdir()
        with torch.inference_mode():
            before = time.monotonic()
            expected_value = spec.model(*spec.args, **spec.kwargs)
            expected = outputs(expected_value)
            if len(expected) != len(spec.output_names):
                raise ValueError('output_names must name all outputs')
            result['outputs'] = {n:tensor_meta(v) for n,v in zip(spec.output_names,expected)}
            torch.save(dict(args=spec.args, kwargs=spec.kwargs, outputs=expected), work/'example.pt')
            result['timings_s']['native'] = time.monotonic()-before
            before = time.monotonic()
            program = torch.export.export(spec.model, spec.args, spec.kwargs, strict=False)
            result['timings_s']['capture'] = time.monotonic()-before
            for fmt in formats:
                before = time.monotonic()
                row = dict(status='failed')
                result['formats'][fmt] = row
                try:
                    if fmt == 'fx':
                        torch.export.save(program, work/'model.pt2')
                        loaded = torch.export.load(work/'model.pt2')
                        actual_value = loaded.module()(*spec.args, **spec.kwargs)
                        from torch.utils._pytree import tree_structure
                        if tree_structure(expected_value) != tree_structure(actual_value):
                            raise ValueError('FX output tree structure differs')
                        row['roundtrip'] = compare_outputs(expected, actual_value, spec.output_names, exact=True)
                        if row['roundtrip']['status'] != 'passed':
                            raise ValueError('FX roundtrip output mismatch')
                        write_json(root/'fx.json', fx_graph(program))
                        (root/'model.fx.txt').write_text(str(program.graph_module.graph)+'\n')
                    else:
                        import onnx
                        torch.onnx.export(program, args=(), f=str(work/'model.onnx'), dynamo=True,
                                          external_data=True, input_names=list(spec.input_names),
                                          output_names=list(spec.output_names), opset_version=18)
                        onnx.checker.check_model(str(work/'model.onnx'), full_check=True)
                        model = onnx.load(str(work/'model.onnx'), load_external_data=False)
                        if [v.name for v in model.graph.input] != list(spec.input_names) or [v.name for v in model.graph.output] != list(spec.output_names):
                            raise ValueError('ONNX input/output names or order differ')
                        row['structural_check'] = 'passed'
                        row['external_data'] = sorted({e.value for t in model.graph.initializer for e in t.external_data if e.key=='location'})
                        for name in row['external_data']:
                            target = (work/name).resolve()
                            if work.resolve() not in target.parents or not target.is_file():
                                raise ValueError('ONNX external data is outside bundle or missing')
                        write_json(root/'onnx.json', onnx_graph(model))
                        row['runtime'] = validate_onnx(work/'model.onnx', spec.args, spec.kwargs, expected,
                                                        spec.input_names, spec.output_names) if verify_onnx else dict(status='not_run')
                        if verify_onnx and row['runtime']['status'] != 'passed':
                            raise ValueError('ONNX runtime output mismatch')
                    row['status'] = 'completed'
                    row['graph_id'] = file_info(root/(fmt+'.json'))['sha256']
                except Exception as error:
                    row.update(error=str(error), traceback=traceback.format_exc())
                result['timings_s'][fmt] = time.monotonic()-before
            success = [row['status']=='completed' for row in result['formats'].values()]
            result['status'] = 'completed' if all(success) else 'partial' if any(success) else 'failed'
            if bundle == 'full':
                for path in list(work.iterdir()):
                    shutil.move(str(path), root/path.name)
                work.rmdir()
            elif result['status'] == 'completed':
                shutil.rmtree(work)
    except Exception as error:
        result.update(error=str(error), traceback=traceback.format_exc())
    result['timings_s']['total'] = time.monotonic()-start
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    return finish(root, result)
