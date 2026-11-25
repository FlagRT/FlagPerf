# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Training per-layer precision diff (oracle): boundary activation capture and
cross-run join.

Capture emits the same `t='tensor'` comparison payload as the inference
accuracy probe (shared tensor_event builder), keyed
`path#iteration#boundary#sample` where the iteration is the training step
taken from the live phase state. Purely observational: forward hooks only
read tensors and never change execution order or numerics. Gradients are
compared from the accuracy snapshots' named tensors, so no backward hook is
installed.

CLI: python -m flagprof.training.oracle runA runB --out DIR joins two runs
(activations from tensor events, gradients from step-N.pt snapshots) into an
error table with first-divergence location, written as oracle.json/oracle.md.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


# These overloads have positional tensor operands, one tensor result, and
# no state or mutation. Only add.Tensor takes an additional scalar (alpha).
REPLAY_ARITY = {'aten::mm': 2, 'aten::bmm': 2, 'aten::t': 1,
                'aten::tanh': 1, 'aten::relu': 1, 'aten::mul.Tensor': 2,
                'aten::add.Tensor': 2}


def _replay_kwargs(name, args, kwargs):
    import torch
    if len(args) != REPLAY_ARITY.get(name) or not all(isinstance(arg, torch.Tensor) for arg in args):
        return None
    if not kwargs:
        return {}
    if (name == 'aten::add.Tensor' and set(kwargs) == {'alpha'} and
            type(kwargs['alpha']) in (int, float) and math.isfinite(kwargs['alpha'])):
        return {'alpha': kwargs['alpha']}
    return None


def normalize_boundaries(configured):
    """Normalize selector strings to <root>.-prefixed module paths."""
    return sorted(p if p.startswith('<root>') else '<root>.'+p for p in configured)


def auto_boundaries(model):
    """Same discovery heuristic as the accuracy probe: input embeddings plus
    the largest uniform ModuleList (the transformer block stack)."""
    import torch
    paths = {id(m): '<root>'+('.'+name if name else '') for name, m in model.named_modules()}
    boundaries = set()
    embed = model.get_input_embeddings() if hasattr(model, 'get_input_embeddings') else getattr(model, 'embedding', None)
    if embed is not None:
        boundaries.add(paths[id(embed)])
    groups = [m for m in model.modules() if isinstance(m, torch.nn.ModuleList) and len(m) and len({type(v) for v in m}) == 1]
    if groups:
        boundaries.update(paths[id(v)] for v in max(groups, key=len))
    return boundaries, paths


class OracleCapture:
    """Forward-hook activation capture for one training run, with optional
    op-level drilling inside one selected module (replay inputs)."""

    def __init__(self, handle, boundaries=None, budget_bytes=1 << 26, ops_module=None):
        self.handle = handle
        self.writer = handle.writer
        self.boundaries = boundaries
        self.budget_bytes = budget_bytes
        self.ops_module = ops_module
        self.blob_bytes = 0
        self.boundary_calls = {}
        self.drill_calls = {}
        self.call_step = None
        self.drill_call_key = None
        self.stream = None
        self.handles = []
        self.model = None
        self.paths = {}
        self.drilling = False
        self.drill_path = None
        self.op_calls = {}
        self.mode = None
        self.busy = False
        self.metrics = dict(events=0, events_rejected=0, blob_bytes=0, blob_skipped=0, boundaries=0, ops=0)
        handle.state['oracle'] = self.metrics
        self.identity = dict(torch=__import__('torch').__version__,
                             cuda=__import__('torch').version.cuda if hasattr(__import__('torch').version, 'cuda') else None)

    def register_model(self, model):
        if self.model is model:
            return
        self.model = model
        self.paths = {id(m): '<root>'+('.'+name if name else '') for name, m in model.named_modules()}
        if self.boundaries is None:
            self.boundaries, _ = auto_boundaries(model)
        else:
            self.boundaries = normalize_boundaries(self.boundaries)
        missing = [b for b in self.boundaries if b not in set(self.paths.values())]
        if missing:
            raise ValueError('oracle scope is not a registered module path: '+', '.join(sorted(missing)))
        for name, module in model.named_modules():
            path = self.paths[id(module)]
            if path in self.boundaries:
                self.handles.append(module.register_forward_hook(
                    lambda m, a, o, path=path: self.safe(self.capture, path, o)))
        self.handle.state['oracle_boundaries'] = sorted(self.boundaries)
        self.metrics['boundaries'] = len(self.boundaries)
        if self.ops_module:
            drill = normalize_boundaries([self.ops_module])[0]
            if drill not in set(self.paths.values()):
                raise ValueError('oracle ops scope is not a registered module path: '+drill)
            self.drill_path = drill
            self._install_drill()
            self.handle.state['oracle_ops_module'] = drill
            self.handle.state['oracle_identity'] = self.identity

    def _install_drill(self):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        owner = self

        class Drill(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if owner.busy or not owner.drilling:
                    return func(*args, **(kwargs or {}))
                return owner.invoke(func, args, kwargs or {})

        self.mode = Drill()
        self.mode.__enter__()
        for name, module in self.model.named_modules():
            if self.paths[id(module)] == self.drill_path:
                self.handles.append(module.register_forward_pre_hook(
                    lambda m, a, k: self.safe(self.drill_enter, a, k or {}), with_kwargs=True))
                self.handles.append(module.register_forward_hook(
                    lambda m, a, k, o: self.safe(self.drill_leave, o), with_kwargs=True, always_call=True))
                break

    def drill_enter(self, args, kwargs):
        self.drilling = True
        self.drill_call_key = self._invocation_key(self.drill_path, self.drill_calls)
        self.op_calls.clear()

    def drill_leave(self, output):
        self.drilling = False

    def invoke(self, func, args, kwargs=None):
        try:
            schema = str(func._schema)
            name = schema.split('(')[0]
        except Exception:
            name = str(func)
        index = self.op_calls.get(name, 0)
        self.op_calls[name] = index + 1
        base = self.drill_call_key+'#op:'+name+'#'+str(index)
        self.busy = True
        try:
            self.safe(self._record_op_io, base, name, args, kwargs, None)
        finally:
            self.busy = False
        output = func(*args, **(kwargs or {}))
        self.busy = True
        try:
            if output is not None:
                self.safe(self._record_op_io, base, name, args, kwargs or {}, output)
            self.metrics['ops'] += 1
        finally:
            self.busy = False
        return output

    def _emit_op_event(self, key, name, tensor, role, **replay_metadata):
        from flagprof.probes.tensor import tensor_event
        # Input blobs are the replay material: keep them up to 32 MiB per
        # tensor (the run-level budget still bounds the total); outputs are
        # kept up to 8 MiB so the digest comparison has a reference.
        size = tensor.numel() * tensor.element_size()
        small = size <= (32 << 20) if role == 'in' else size <= (8 << 20)
        event, raw, blob_fmt = tensor_event(key, self.drill_path, self._step(),
                                            'op:'+name, role, tensor,
                                            role=role, ext={'training_schema': 'training/1',
                                                            'op_schema': name,
                                                            'capture_scope': 'oracle-replay',
                                                            **replay_metadata})
        self._emit(event, raw, blob_fmt, keep_blob=small)

    def _emit(self, event, raw, blob_fmt, *, keep_blob=True):
        if keep_blob and self.blob_bytes + len(raw) <= self.budget_bytes:
            if self.stream is None:
                folder = self.writer.run_dir/'blobs'
                folder.mkdir(exist_ok=True)
                self.stream = (folder/('train-oracle-'+str(os.getpid())+'-'+str(abs(id(self)))+'.bin')).open('xb')
            offset = self.stream.tell()
            self.stream.write(raw)
            event['blob'] = dict(path=str(Path(self.stream.name).relative_to(self.writer.run_dir)),
                                 offset=offset, nbytes=len(raw), hash=event['digest'], fmt=blob_fmt)
            self.blob_bytes += len(raw)
            self.metrics['blob_bytes'] = self.blob_bytes
        else:
            self.metrics['blob_skipped'] += 1
        if self.writer.emit(event):
            self.metrics['events'] += 1
        else:
            self.metrics['events_rejected'] += 1

    def _record_op_io(self, base, name, args, kwargs, output):
        import torch
        if output is not None:
            replay_kwargs = _replay_kwargs(name, args, kwargs) if isinstance(output, torch.Tensor) else None
            if isinstance(output, torch.Tensor):
                leaves = [('0', output)]
            else:
                leaves = list(_leaves(output))[:64]
            for leafpath, tensor in leaves:
                if isinstance(tensor, torch.Tensor):
                    self._emit_op_event(base+'#out:'+leafpath, name, tensor, 'out', replay_kwargs=replay_kwargs)
            return
        items = list(_leaves(tuple(args))) + list(_leaves(tuple(sorted(kwargs.items()))))
        for leafpath, tensor in items[:64]:
            if isinstance(tensor, torch.Tensor):
                layout = 'unsupported'
                if name in REPLAY_ARITY and tensor.layout == torch.strided:
                    if tensor.is_contiguous():
                        layout = 'contiguous'
                    elif tensor.ndim >= 2 and tensor.transpose(-1, -2).is_contiguous():
                        layout = 'transposed-last-two'
                self._emit_op_event(base+'#in:'+leafpath, name, tensor, 'in', replay_layout=layout)

    def safe(self, fn, *args):
        try:
            fn(*args)
        except Exception as error:  # capture must never break training
            self.metrics.setdefault('errors', []).append(type(error).__name__+': '+str(error)[:120])

    def _step(self):
        state = self.handle.context.probe_state().get('training') or {}
        step = state.get('step')
        return step if type(step) is int and step >= 0 else -1

    def _invocation_key(self, path, calls):
        step = self._step()
        if step != self.call_step:
            self.boundary_calls.clear()
            self.drill_calls.clear()
            self.call_step = step
        state = self.handle.context.probe_state().get('training') or {}
        micro = state.get('microstep', 0)
        index = calls.get((path, micro), 0)
        calls[(path, micro)] = index + 1
        key = f'{path}#{step}'
        # Keep the existing key for the first forward of a simple step.
        if micro or index:
            key += f'#micro:{micro}#call:{index}'
        return key

    def capture(self, path, output):
        if output is None:
            return
        import torch
        first = None
        for _, v in _leaves(output):
            first = v
            break
        if not isinstance(first, torch.Tensor):
            return
        # Our own CPU-snapshot ops must not re-enter the drill mode.
        self.busy = True
        try:
            self._capture_inner(path, first)
        finally:
            self.busy = False

    def _capture_inner(self, path, first):
        from flagprof.probes.tensor import tensor_event
        step = self._step()
        key = self._invocation_key(path, self.boundary_calls)+'#out#0'
        event, raw, blob_fmt = tensor_event(key, path, step, 'out', '0', first,
                                           ext={'training_schema': 'training/1', 'capture_scope': 'oracle'})
        self._emit(event, raw, blob_fmt)

    def close(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if self.mode is not None:
            self.mode.__exit__(None, None, None)
            self.mode = None
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.stream = None


def _leaves(value, path='0'):
    import torch
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from _leaves(v, path+'['+str(i)+']')
    elif isinstance(value, dict):
        for k, v in sorted(value.items()):
            yield from _leaves(v, path+'{'+str(k)+'}')


def start_from_env(handle):
    """Instantiate a capture on this attachment when FLAGPROF_TRAIN_ORACLE_AT
    is set ('auto' discovers boundaries); returns the capture or None."""
    configured = os.environ.get('FLAGPROF_TRAIN_ORACLE_AT')
    if not configured or handle.state.get('status') != 'running':
        return None
    if configured.strip() == 'auto':
        selectors = None
    else:
        selectors = [p.strip() for p in configured.split(',') if p.strip()]
        if not selectors:
            selectors = None
    try:
        budget = int(os.environ.get('FLAGPROF_TRAIN_ORACLE_BYTES', str(1 << 26)))
    except ValueError:
        budget = 1 << 26
    ops_module = (os.environ.get('FLAGPROF_TRAIN_ORACLE_OPS') or '').strip() or None
    capture = OracleCapture(handle, boundaries=selectors, budget_bytes=budget, ops_module=ops_module)
    handle.state['oracle_at'] = sorted(selectors) if selectors else 'auto'
    return capture


# --- cross-run join -------------------------------------------------------


def _resolve_run(path):
    """Accept either a run directory (with manifest.json) or its rank parent
    holding exactly one run."""
    p = Path(path)
    if (p/'manifest.json').exists():
        return p
    if p.is_dir():
        children = [c for c in sorted(p.iterdir()) if (c/'manifest.json').exists()]
        if len(children) == 1:
            return children[0]
    raise ValueError('not a training store run (no manifest.json under '+str(p)+')')


def _read_tensor_events(run, *, ops_only=False):
    from flagprof.store import StoreView
    rows = {}
    with StoreView(run) as view:
        for stored in view.iter_events(types=('tensor',)):
            event = stored.record
            if ops_only and event.get('ext', {}).get('capture_scope') != 'oracle-replay':
                continue
            key = event['call_key']
            if key in rows:
                raise ValueError(f'ambiguous repeated tensor key {key!r} in {run}; recapture with invocation keys')
            rows[key] = event
    return rows


def _read_gradient_snapshots(run):
    """(step, param-name) -> raw grad bytes, from accuracy step snapshots."""
    out = {}
    for path in sorted(Path(run).glob('step-*.pt')):
        import torch
        step = int(path.stem.removeprefix('step-'))
        snap = torch.load(path, map_location='cpu', weights_only=True)
        grads = snap.get('gradients', snap.get('grads')) if isinstance(snap, dict) else None
        if not isinstance(grads, dict):
            raise ValueError(f'{path} does not contain a gradient mapping')
        for name, tensor in grads.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError(f'invalid gradient entry in {path}: {name!r}')
            out[(step, name)] = tensor
    return out


def _capture_coverage(run):
    from flagprof.store.manifest import load_json, writer_fragments
    rejected = errors = 0
    for path in (run/'attachments').glob('*/state.json'):
        oracle = load_json(path).get('oracle', {})
        rejected += oracle.get('events_rejected', 0)
        errors += len(oracle.get('errors', []))
    # Includes all tensor producers on this rank and may overlap rejections.
    drops = sum(fragment.get('dropped', {}).get('tensor', 0) for fragment in writer_fragments(run))
    return dict(events_rejected=rejected, capture_errors=errors, writer_tensor_drops=drops)


def _probe_comparison(left, right, atol, rtol):
    from flagprof.tensor_data import values
    lp, rp = left.get('probe'), right.get('probe')
    dtype = left['dtype'].removeprefix('torch.')
    # p4v1 serializes float64 samples even for complex and integer inputs.
    # For equal shape/dtype integer arrays different digests prove an exact
    # mismatch, even if the sample lost the differing bit or missed the index.
    integer = dtype.startswith(('int', 'uint')) or dtype == 'bool'
    different_integer_bytes = integer and left.get('digest') and right.get('digest') and left['digest'] != right['digest']
    if different_integer_bytes and dtype in ('int64', 'uint64'):
        return dict(verdict='differs', basis='digest', max_abs=None, max_rel=None)
    if (not lp or not rp or len(lp) != len(rp) or
            left.get('probe_alg') != 'p4v1' or right.get('probe_alg') != 'p4v1' or
            dtype.startswith('complex') or dtype in ('int64', 'uint64')):
        return dict(verdict='differs' if different_integer_bytes else 'incomparable',
                    basis='digest' if different_integer_bytes else 'probe-sampled', max_abs=None, max_rel=None)
    differs = False
    absolute, relative = [], []
    for a, b in zip(values(None, left, 'sampled'), values(None, right, 'sampled')):
        if not (math.isfinite(a) and math.isfinite(b)):
            differs |= not (a == b or (math.isnan(a) and math.isnan(b)))
            continue
        delta = abs(a-b)
        absolute.append(delta)
        relative.append(delta/max(abs(a), 1e-12))
        differs |= a != b if integer else delta > atol + rtol*abs(a)
    # A different digest and equal samples cannot certify the unobserved elements.
    verdict = 'within_tolerance' if len(lp) == math.prod(left['shape']) else 'incomparable'
    if differs:
        verdict = 'differs'
    elif different_integer_bytes:
        return dict(verdict='differs', basis='digest', max_abs=None, max_rel=None)
    return dict(verdict=verdict, basis='probe-sampled',
                max_abs=max(absolute, default=None), max_rel=max(relative, default=None))


def _tensor_comparison(left, right, atol, rtol):
    import torch
    if not left.numel():
        return dict(verdict='within_tolerance', max_abs=0.0, max_rel=0.0)
    integer = not (left.is_floating_point() or left.is_complex())
    if integer and left.element_size() == 8:
        # Python integer subtraction retains all 65 bits of a signed int64
        # difference. Chunk conversion bounds the temporary Python objects.
        max_abs, max_rel = 0, 0.0
        a, b = left.reshape(-1), right.reshape(-1)
        for offset in range(0, a.numel(), 8192):
            for x, y in zip(a[offset:offset+8192].tolist(), b[offset:offset+8192].tolist()):
                delta = abs(x - y)
                max_abs = max(max_abs, delta)
                max_rel = max(max_rel, delta / max(abs(x), 1e-12))
        return dict(verdict='differs' if max_abs else 'within_tolerance', max_abs=max_abs, max_rel=max_rel)
    dtype = torch.complex128 if left.is_complex() else torch.float64
    a, b = left.to(dtype), right.to(dtype)
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a-b).abs()
    same_special = (a == b) | (torch.isnan(a) & torch.isnan(b))
    if integer:
        differs = not torch.equal(left, right)
    else:
        differs = bool(torch.where(finite, delta > atol + rtol*a.abs(), ~same_special).any())
    return dict(verdict='differs' if differs else 'within_tolerance',
                max_abs=float(delta[finite].max()) if bool(finite.any()) else None,
                max_rel=float((delta[finite]/a[finite].abs().clamp(min=1e-12)).max()) if bool(finite.any()) else None)


def _blob_tensor(run, event):
    import torch
    from flagprof.tensor_data import blob_path
    blob = event['blob']
    dtype = getattr(torch, event['dtype'].removeprefix('torch.'))
    expected = math.prod(event['shape']) * torch.empty((), dtype=dtype).element_size()
    if blob['nbytes'] != expected or blob['offset'] < 0:
        raise ValueError('invalid tensor blob range for '+event['call_key'])
    with blob_path(run, blob).open('rb') as stream:
        stream.seek(blob['offset'])
        raw = stream.read(expected)
    actual = 'sha256:'+hashlib.sha256(raw).hexdigest()
    if len(raw) != expected or actual != blob['hash'] or actual != event['digest']:
        raise ValueError('tensor blob differs from its recorded digest: '+event['call_key'])
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype) if raw else torch.empty(0, dtype=dtype)
    return tensor.view(event['shape'])


def _capture_order(left, right):
    # Keep the reference stream order. Place a right-only event before its
    # next common boundary so missing layers are located in execution order.
    before = {}
    pending = []
    for key in right:
        if key not in left:
            pending.append(key)
        elif pending:
            before[key] = pending
            pending = []
    for key in left:
        yield from before.get(key, ())
        yield key
    yield from pending


def join_runs(run_left, run_right, *, atol=0.0, rtol=0.0, out=None, snaps_left=None, snaps_right=None):
    """Join two training runs into the per-layer error table.

    Activations join on tensor-event call_keys (step-level boundary outputs);
    gradients join on (step, parameter name) from accuracy snapshots taken
    from the snapshots dirs (the training outputs directories) when given.
    Exact when both sides carry blobs, summarized over the 256-probe samples
    otherwise (rows carry the verdict basis explicitly).
    """
    if any(not math.isfinite(t) or t < 0 for t in (atol, rtol)):
        raise ValueError('atol and rtol must be finite and nonnegative')
    for root in (snaps_left, snaps_right):
        if root is not None and not any(Path(root).glob('step-*.pt')):
            raise ValueError(f'no gradient snapshots in {root}')
    run_left, run_right = _resolve_run(run_left), _resolve_run(run_right)
    left_events, right_events = _read_tensor_events(run_left), _read_tensor_events(run_right)
    left_grads, right_grads = _read_gradient_snapshots(snaps_left or run_left), _read_gradient_snapshots(snaps_right or run_right)
    rows = []
    for key in _capture_order(left_events, right_events):
        l, r = left_events.get(key), right_events.get(key)
        parts = key.split('#')
        step = int(parts[1]) if len(parts) > 1 and parts[1].lstrip('-').isdigit() else -1
        row = dict(step=step, key=key, kind='activation')
        if l is None or r is None:
            row.update(verdict='missing_side', present=['left' if l else 'right'])
        elif l['shape'] != r['shape'] or l['dtype'] != r['dtype']:
            row.update(verdict='differs', basis='metadata', shape=l['shape'], shape_right=r['shape'],
                       dtype=l['dtype'], dtype_right=r['dtype'])
        elif l.get('digest') and l['digest'] == r.get('digest'):
            row.update(verdict='exact', basis='digest', max_abs=0.0, max_rel=0.0)
        else:
            if l.get('blob') and r.get('blob'):
                row.update(_tensor_comparison(_blob_tensor(run_left, l), _blob_tensor(run_right, r), atol, rtol),
                           basis='blob-elementwise')
            else:
                row.update(_probe_comparison(l, r, atol, rtol))
            row.update(shape=l['shape'], dtype=l['dtype'],
                       digest_left=l.get('digest', '')[:19], digest_right=r.get('digest', '')[:19])
        rows.append(row)
    for (step, name) in sorted(set(left_grads) | set(right_grads)):
        l, r = left_grads.get((step, name)), right_grads.get((step, name))
        row = dict(step=step, key=name, kind='gradient')
        if l is None or r is None:
            row.update(verdict='missing_side')
        elif l.shape != r.shape or l.dtype != r.dtype:
            row.update(verdict='differs', basis='metadata', shape=list(l.shape), shape_right=list(r.shape),
                       dtype=str(l.dtype), dtype_right=str(r.dtype))
        else:
            import torch
            if torch.equal(l.contiguous().reshape(-1).view(torch.uint8), r.contiguous().reshape(-1).view(torch.uint8)):
                row.update(verdict='exact', basis='snapshot', max_abs=0.0, max_rel=0.0)
            else:
                row.update(_tensor_comparison(l, r, atol, rtol), basis='snapshot')
        rows.append(row)
    # Within an optimizer step the StoreView stream preserves reference capture order.
    # Gradient snapshots are recorded after its forwards.
    rows.sort(key=lambda row: row['step'] if row['step'] >= 0 else float('inf'))
    first = next((r for r in rows if r['verdict'] in ('differs', 'missing_side')), None)
    left_coverage, right_coverage = _capture_coverage(run_left), _capture_coverage(run_right)
    result = dict(schema='training-oracle/1', atol=atol, rtol=rtol,
                  runs=dict(left=str(run_left), right=str(run_right)),
                  coverage=dict(partial=any(left_coverage.values()) or any(right_coverage.values()),
                                left=left_coverage, right=right_coverage),
                  counts=dict(rows=len(rows),
                              incomparable=sum(1 for r in rows if r['verdict'] == 'incomparable'),
                              exact=sum(1 for r in rows if r['verdict'] == 'exact'),
                              differs=sum(1 for r in rows if r['verdict'] == 'differs'),
                              missing=sum(1 for r in rows if r['verdict'] == 'missing_side')),
                  first_divergence=first, rows=rows)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out/'oracle.json').write_text(json.dumps(result, ensure_ascii=False, indent=1)+'\n')
        (out/'oracle.md').write_text(_markdown(result))
    return result


def replay(run, op_key, device='cpu'):
    """Re-execute one drilled op from its captured inputs (blobs required) and
    compare the replayed output against the captured output digest. Cross-vendor
    replay = run this on the other vendor's stack against the same store copy;
    identity fields record both capture-time and replay-time versions."""
    import torch
    run = _resolve_run(run)
    events = _read_tensor_events(run, ops_only=True)
    if op_key not in events:
        prefix = op_key.split('#out')[0]
        outs = sorted(k for k in events if k.startswith(prefix + '#out'))
        raise ValueError('unknown op key; sibling out keys: ' + ', '.join(outs[:4]))
    out_event = events[op_key]
    name = out_event['ext']['op_schema']
    if name not in REPLAY_ARITY or out_event.get('role') != 'out' or not op_key.endswith('#out:0'):
        raise ValueError('unsupported replay overload or output: '+op_key)
    prefix = op_key[:-len('#out:0')]
    expected = [prefix + f'#in:0[{i}]' for i in range(REPLAY_ARITY[name])]
    inputs = {k: e for k, e in events.items() if k.startswith(prefix + '#in:')}
    if set(inputs) != set(expected):
        raise ValueError('replay requires every positional tensor operand: '+op_key)
    # Historical tensor-only overloads have no omitted scalar settings.
    # Legacy add.Tensor did not record alpha and cannot be replayed faithfully.
    kwargs = out_event['ext'].get('replay_kwargs', None if name == 'aten::add.Tensor' else {})
    if not isinstance(kwargs, dict):
        raise ValueError('unsupported or unrecorded replay arguments: '+op_key)
    capture_identity = None
    for att in sorted((run/'attachments').glob('*/state.json')) if (run/'attachments').is_dir() else []:
        try:
            state = json.loads(att.read_text())
            capture_identity = state.get('oracle_identity')
            break
        except ValueError:
            pass
    missing = [k for k, e in inputs.items() if not e.get('blob')]
    if missing:
        raise ValueError('inputs without blobs (replay impossible): '+', '.join(missing[:3]))
    args = []
    layouts = []
    for key in expected:
        event = inputs[key]
        if event.get('role') != 'in' or event['ext'].get('op_schema') != name:
            raise ValueError('replay input does not match output overload: '+key)
        layout = event['ext'].get('replay_layout', 'legacy-contiguous')
        tensor = _blob_tensor(run, event).to(device)
        if layout == 'transposed-last-two' and tensor.ndim >= 2:
            tensor = tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
        elif layout not in ('contiguous', 'legacy-contiguous'):
            raise ValueError('unsupported replay input layout: '+key)
        args.append(tensor)
        layouts.append(layout)
    if _replay_kwargs(name, args, kwargs) is None:
        raise ValueError('unsupported replay keyword arguments: '+op_key)
    op_name, _, overload = name.removeprefix('aten::').partition('.')
    opctor = getattr(torch.ops.aten, op_name)
    callable_op = getattr(opctor, overload) if overload else opctor.default
    replayed = callable_op(*args, **kwargs)
    replay_digest = 'sha256:'+hashlib.sha256(
        replayed.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    match = (replay_digest == out_event['digest'] and list(replayed.shape) == out_event['shape']
             and str(replayed.dtype) == out_event['dtype'])
    result = dict(schema='training-oracle-replay/1', op=name, key=op_key, device=device,
                  captured_digest=out_event['digest'], replayed_digest=replay_digest,
                  match=match, capture_identity=capture_identity,
                  replay_identity=dict(torch=torch.__version__,
                                       cuda=torch.version.cuda if hasattr(torch.version, 'cuda') else None),
                  inputs=len(args), input_layouts=layouts, kwargs=kwargs)
    return result


def _markdown(result):
    lines = ['# 训练逐层精度差分（oracle）', '',
             f"- 左：`{result['runs']['left']}`", f"- 右：`{result['runs']['right']}`",
             f"- 容差：atol={result['atol']} rtol={result['rtol']}",
             f"- 行数：{result['counts']['rows']}（exact {result['counts']['exact']} / "
             f"differs {result['counts']['differs']} / missing {result['counts']['missing']}）", '']
    first = result.get('first_divergence')
    if result['coverage']['partial']:
        lines += ['**采集记录不完整**：存在写入丢失或采集错误，以下结论仅覆盖已保留的记录。', '']
    if first:
        lines += [f"**首个发散**：step {first['step']} `{first['key']}`（{first['kind']}，verdict={first['verdict']}）", '']
    elif result['counts']['incomparable'] or result['coverage']['partial'] or not result['rows']:
        lines += ['**缺少可比较证据**：当前结果不能确认全部行在容差内。', '']
    else:
        lines += ['**未发现发散**：全部比较行在容差内。', '']
    lines += ['| step | kind | key | verdict | max_abs | max_rel |', '|---|---|---|---|---|---|']
    for r in result['rows'][:400]:
        lines.append('| {step} | {kind} | `{key}` | {verdict} | {a} | {b} |'.format(
            step=r['step'], kind=r['kind'], key=r['key'], verdict=r['verdict'],
            a=('%.3g' % r['max_abs']) if r.get('max_abs') is not None else '-', b=('%.3g' % r['max_rel']) if r.get('max_rel') is not None else '-'))
    if len(result['rows']) > 400:
        lines.append(f'（其余 {len(result["rows"])-400} 行见 oracle.json）')
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    verbs = parser.add_subparsers(dest='verb', required=True)
    join = verbs.add_parser('join', help='join two training runs into the per-layer error table')
    join.add_argument('left')
    join.add_argument('right')
    join.add_argument('--out', required=True)
    join.add_argument('--atol', type=float, default=0.0)
    join.add_argument('--rtol', type=float, default=0.0)
    join.add_argument('--snapshots-left', help='accuracy snapshot outputs dir of the left run')
    join.add_argument('--snapshots-right', help='accuracy snapshot outputs dir of the right run')
    replay_p = verbs.add_parser('replay', help='re-execute one drilled op from captured inputs')
    replay_p.add_argument('run')
    replay_p.add_argument('--op-key', required=True, help='the out-event call key of the op call')
    replay_p.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.verb == 'join':
        result = join_runs(args.left, args.right, atol=args.atol, rtol=args.rtol, out=args.out,
                           snaps_left=args.snapshots_left, snaps_right=args.snapshots_right)
        print(json.dumps(result['counts'], ensure_ascii=False))
        return 0
    result = replay(args.run, args.op_key, device=args.device)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
