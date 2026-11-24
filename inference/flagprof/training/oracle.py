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
import os
from pathlib import Path


def normalize_boundaries(configured):
    """Normalize selector strings to <root>.-prefixed module paths."""
    return {p if p.startswith('<root>') else '<root>.'+p for p in configured}


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
    """Forward-hook activation capture for one training run."""

    def __init__(self, handle, boundaries=None, budget_bytes=1 << 26):
        self.handle = handle
        self.writer = handle.writer
        self.boundaries = boundaries
        self.budget_bytes = budget_bytes
        self.blob_bytes = 0
        self.blob_exhausted = False
        self.stream = None
        self.handles = []
        self.model = None
        self.paths = {}
        self.metrics = dict(events=0, blob_bytes=0, blob_skipped=0, boundaries=0)
        handle.state['oracle'] = self.metrics

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

    def safe(self, fn, *args):
        try:
            fn(*args)
        except Exception as error:  # capture must never break training
            self.metrics.setdefault('errors', []).append(type(error).__name__+': '+str(error)[:120])

    def _step(self):
        state = self.handle.context.probe_state().get('training') or {}
        step = state.get('step')
        return step if type(step) is int and step >= 0 else -1

    def capture(self, path, output):
        from flagprof.probes.tensor import tensor_event
        if output is None:
            return
        import torch
        first = None
        for _, v in _leaves(output):
            first = v
            break
        if not isinstance(first, torch.Tensor):
            return
        step = self._step()
        key = path+'#'+str(step)+'#out#0'
        if self.blob_exhausted:
            return
        event, raw, blob_fmt = tensor_event(key, path, step, 'out', '0', first,
                                            ext={'training_schema': 'training/1', 'capture_scope': 'oracle'})
        if self.blob_bytes+len(raw) <= self.budget_bytes:
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
            self.blob_exhausted = True
            self.metrics['blob_skipped'] += 1
        self.writer.emit(event)
        self.metrics['events'] += 1

    def close(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
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
    capture = OracleCapture(handle, boundaries=selectors, budget_bytes=budget)
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


def _read_tensor_events(run):
    from flagprof.store import StoreView
    rows = {}
    with StoreView(run) as view:
        for stored in view.iter_events(types=('tensor',)):
            event = stored.record
            rows[event['call_key']] = event
    return rows


def _read_gradient_snapshots(run):
    """(step, param-name) -> raw grad bytes, from accuracy step snapshots."""
    out = {}
    import torch
    for path in sorted(Path(run).glob('step-*.pt')):
        try:
            snap = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            continue
        try:
            step = int(path.stem.split('-')[1])
        except (ValueError, IndexError):
            continue
        grads = snap.get('gradients') or snap.get('grads') or {}
        for name, tensor in grads.items():
            try:
                raw = tensor.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            except Exception:
                continue
            out[(step, name)] = dict(raw=raw, shape=list(tensor.shape), dtype=str(tensor.dtype))
    return out


def _digest(raw):
    return 'sha256:'+hashlib.sha256(raw).hexdigest()


def _max_abs_from_probe(left, right):
    """Summarized comparison when blobs are absent: compare the 256 probe
    samples and per-chunk means; returns (max_abs, samples) or None."""
    lp, rp = left.get('probe'), right.get('probe')
    if not lp or not rp or len(lp) != len(rp):
        return None
    def val(v):
        return 0.0 if v == 'nan' else float('inf') if v == '+inf' else float('-inf') if v == '-inf' else v
    diffs = [abs(val(a)-val(b)) for a, b in zip(lp, rp)]
    if not diffs:
        return None
    return max(diffs)


def _blob_max_abs(run_left, run_right, l, r):
    """Element-wise max abs/rel from both sides' blobs; None when either
    side lacks a blob or the formats disagree."""
    import torch
    lb, rb = l.get('blob'), r.get('blob')
    if not lb or not rb or lb.get('fmt') != rb.get('fmt') or lb['nbytes'] != rb['nbytes']:
        return None
    dtype_name = lb['fmt'].split(':')[1]
    try:
        torch_dtype = getattr(torch, dtype_name)
    except AttributeError:
        return None
    out = []
    for run, b in ((run_left, lb), (run_right, rb)):
        path = Path(run)/b['path']
        with path.open('rb') as stream:
            stream.seek(b['offset'])
            raw = stream.read(b['nbytes'])
        out.append(torch.frombuffer(bytearray(raw), dtype=torch_dtype).double())
    a, b = out
    if a.numel() != b.numel() or not a.numel():
        return None
    diff = (a-b).abs()
    denom = a.abs().clamp(min=1e-12)
    return float(diff.max()), float((diff/denom).max())


def join_runs(run_left, run_right, *, atol=0.0, rtol=0.0, out=None):
    """Join two training runs into the per-layer error table.

    Activations join on tensor-event call_keys (step-level boundary outputs);
    gradients join on (step, parameter name) from accuracy snapshots. Exact
    when both sides carry blobs, summarized over the 256-probe samples
    otherwise (rows carry the verdict basis explicitly).
    """
    run_left, run_right = _resolve_run(run_left), _resolve_run(run_right)
    left_events, right_events = _read_tensor_events(run_left), _read_tensor_events(run_right)
    left_grads, right_grads = _read_gradient_snapshots(run_left), _read_gradient_snapshots(run_right)
    rows = []
    for key in sorted(set(left_events) | set(right_events)):
        l, r = left_events.get(key), right_events.get(key)
        parts = key.split('#')
        step = int(parts[1]) if len(parts) > 1 and parts[1].lstrip('-').isdigit() else -1
        if l is None or r is None:
            rows.append(dict(step=step, key=key, kind='activation', verdict='missing_side',
                             present=['left' if l else None, 'right' if r else None]))
            continue
        same_meta = l['shape'] == r['shape'] and l['dtype'] == r['dtype'] and l['digest'] == r['digest']
        if same_meta:
            rows.append(dict(step=step, key=key, kind='activation', verdict='exact', max_abs=0.0, max_rel=0.0))
            continue
        blob_diff = _blob_max_abs(run_left, run_right, l, r)
        if blob_diff is not None:
            max_abs, max_rel = blob_diff
            basis = 'blob-elementwise'
        else:
            max_abs = _max_abs_from_probe(l, r)
            max_rel = None
            basis = 'probe-sampled'
            if max_abs is not None and l.get('stats', {}).get('absmean'):
                denom = max(l['stats']['absmean'], 1e-12)
                max_rel = max_abs/denom
        verdict = 'differs' if (max_abs is not None and (max_abs > atol or (max_rel or 0) > rtol)) else 'within_tolerance'
        rows.append(dict(step=step, key=key, kind='activation', verdict=verdict, basis=basis,
                         max_abs=max_abs, max_rel=max_rel,
                         shape=l['shape'], dtype=l['dtype'],
                         digest_left=l['digest'][:19], digest_right=r['digest'][:19]))
    for (step, name) in sorted(set(left_grads) | set(right_grads)):
        l, r = left_grads.get((step, name)), right_grads.get((step, name))
        if l is None or r is None:
            rows.append(dict(step=step, key=name, kind='gradient', verdict='missing_side'))
            continue
        if l['raw'] == r['raw'] and l['shape'] == r['shape']:
            rows.append(dict(step=step, key=name, kind='gradient', verdict='exact', max_abs=0.0, max_rel=0.0))
            continue
        max_abs = _raw_max_abs(l, r)
        rows.append(dict(step=step, key=name, kind='gradient', verdict='differs', max_abs=max_abs,
                         shape=l['shape'], dtype=l['dtype']))
    first = None
    for row in sorted(rows, key=lambda r: (r['step'] if r['step'] >= 0 else 1 << 30, r['key'])):
        if row['verdict'] in ('differs', 'missing_side'):
            first = row
            break
    result = dict(schema='training-oracle/1', atol=atol, rtol=rtol,
                  runs=dict(left=str(run_left), right=str(run_right)),
                  counts=dict(rows=len(rows),
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


def _raw_max_abs(l, r):
    import torch
    dtype = l['dtype']
    try:
        torch_dtype = getattr(torch, dtype.removeprefix('torch.'))
    except AttributeError:
        return None
    a = torch.frombuffer(bytearray(l['raw']), dtype=torch_dtype).double()
    b = torch.frombuffer(bytearray(r['raw']), dtype=torch_dtype).double()
    if a.numel() != b.numel():
        return None
    return float((a-b).abs().max()) if a.numel() else 0.0


def _markdown(result):
    lines = ['# 训练逐层精度差分（oracle）', '',
             f"- 左：`{result['runs']['left']}`", f"- 右：`{result['runs']['right']}`",
             f"- 容差：atol={result['atol']} rtol={result['rtol']}",
             f"- 行数：{result['counts']['rows']}（exact {result['counts']['exact']} / "
             f"differs {result['counts']['differs']} / missing {result['counts']['missing']}）", '']
    first = result.get('first_divergence')
    if first:
        lines += [f"**首个发散**：step {first['step']} `{first['key']}`（{first['kind']}，verdict={first['verdict']}）", '']
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
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('--out', required=True)
    parser.add_argument('--atol', type=float, default=0.0)
    parser.add_argument('--rtol', type=float, default=0.0)
    args = parser.parse_args(argv)
    result = join_runs(args.left, args.right, atol=args.atol, rtol=args.rtol, out=args.out)
    print(json.dumps(result['counts'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
