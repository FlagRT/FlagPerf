# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Supplemental evidence, never an alternative correctness gate."""
import copy
import time

from runtime.evidence import read, write

PROTOCOL = 'failure-diagnostics-v1'


def errors(actual, expected, atol, rtol):
    import torch
    a, e = actual.detach().cpu().double(), expected.detach().cpu().double()
    if a.shape != e.shape:
        return {'status': 'shape-mismatch'}
    match = torch.isclose(a, e, atol=atol, rtol=rtol, equal_nan=True)
    finite = torch.isfinite(a) & torch.isfinite(e)
    result = {'elements': a.numel(), 'mismatch_count': int((~match).sum()),
              'mismatch_fraction': float((~match).double().mean()) if a.numel() else 0,
              'nonfinite_elements': int((~finite).sum())}
    if finite.any():
        absolute = (a[finite] - e[finite]).abs()
        normalized = absolute / (atol + rtol * e[finite].abs()).clamp_min(torch.finfo(torch.float64).tiny)
        ids = finite.flatten().nonzero().flatten()
        worst = int(ids[normalized.argmax()])
        result.update(absolute_error_quantiles=dict(zip(('p50', 'p90', 'p99', 'max'),
                      torch.quantile(absolute, torch.tensor([.5, .9, .99, 1.], dtype=torch.float64)).tolist())),
                      max_tolerance_ratio=float(normalized.max()), worst_flat_index=worst,
                      actual_at_worst=float(a.flatten()[worst]), reference_at_worst=float(e.flatten()[worst]))
    return result


def numeric(root, task):
    if task.get('case') in ('dropout', 'native_dropout'):
        write(root / 'numeric-diagnostic.json', {'status': 'unsupported', 'protocol': PROTOCOL,
              'gate_changed': False, 'reason': 'stochastic case uses structural/mask-gradient checks in correctness.json, not a cross-device FP64 output oracle'})
        return
    import torch
    from runtime.worker import build, map_tensors, invocation, tensors, cpu
    started = time.monotonic()
    fn, _, bp, _ = build(task)
    bundle = torch.load(root / 'inputs.pt', map_location='cpu', weights_only=True)
    if isinstance(fn, torch.nn.Module):
        fn.load_state_dict(bundle['state']);fn = copy.deepcopy(fn).float()
    inputs = map_tensors(bundle['inputs'], lambda t: t.detach().to(dtype=torch.float32 if t.is_floating_point() else t.dtype).requires_grad_(t.requires_grad))
    cpu32 = cpu(invocation(fn, inputs, bp, nonzero=True))
    expected = torch.load(root / 'reference.pt', map_location='cpu', weights_only=True)
    check = read(root / 'correctness.json')
    atol, rtol = check.get('atol', 1e-5), check.get('rtol', 1e-4)
    result = {'status': 'completed', 'protocol': PROTOCOL, 'gate_changed': False,
              'atol': atol, 'rtol': rtol, 'comparisons': {}}
    for label, path in [('probe', 'output.pt'), ('measure', 'measurement-output.pt')]:
        actual = torch.load(root / path, map_location='cpu', weights_only=True)
        aa, ee, cc = tensors(actual), tensors(expected), tensors(cpu32)
        if not (len(aa) == len(ee) == len(cc)):
            raise ValueError('diagnostic output structure mismatch')
        forward_count = len(tensors(actual[0])) if bp else len(aa)
        result['comparisons'][label] = [
            {'tensor': i, 'role': 'output' if i < forward_count else 'input-gradient',
             'device_vs_fp64': errors(a, e, atol, rtol),
             'cpu_fp32_vs_fp64': errors(c, e, atol, rtol),
             'device_vs_cpu_fp32': errors(a, c, atol, rtol)}
            for i, (a, e, c) in enumerate(zip(aa, ee, cc))]
    result['elapsed_s'] = time.monotonic() - started
    write(root / 'numeric-diagnostic.json', result)


def collect(pool, root, task, item):
    """Isolate diagnostic errors while preserving cleanup failures and interrupts."""
    if task.get('diagnostics_mode', 'off') != 'failures':
        return
    modes = []
    if item.get('correctness', {}).get('status') == 'failed' or item.get('measurement', {}).get('correctness', {}).get('status') == 'failed':
        modes.append(('diagnose', 'numeric-diagnostic.json'))
    if item.get('routing', {}).get('status') == 'partial':
        modes.append(('trace', 'route-diagnostic.json'))
    if not modes:
        return
    from runtime.cli import CleanupError
    started = time.monotonic()
    item['diagnostics'] = {'protocol': PROTOCOL, 'results': {}}
    for mode, filename in modes:
        try:
            pool.phase(root, task, mode)
            item['diagnostics']['results'][mode] = {'status': read(root / filename)['status'], 'evidence': filename}
        except CleanupError:
            raise
        except Exception as exc:
            item['diagnostics']['results'][mode] = {'status': 'failed', 'error': str(exc)}
            # A timeout may leave work running. Do not reuse an uncertain pool.
            pool.close()
            if mode == 'trace' and pool.args.execution == 'docker':
                try:
                    pool.adapter.preflight(pool.root, [task['device_id']], label=f'diagnostic-health-{root.name}')
                except Exception as health_error:
                    raise CleanupError(f'diagnostic device health check failed: {health_error}') from health_error
    item['diagnostics']['elapsed_s'] = time.monotonic() - started
