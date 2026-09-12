# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Isolated CPU oracle, routing probe and uninstrumented measurement phases."""
import argparse
import copy
import importlib.util
from importlib import metadata
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.evidence import read, write, sha

_GEMS_ENABLED = False


def load_case(task):
    sys.path.insert(0, str(ROOT / 'benchmarks'))
    path = ROOT / 'benchmarks' / task['case'] / 'main.py'
    spec = importlib.util.spec_from_file_location('operation_case', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for x in value for t in tensors(x)]
    if isinstance(value, dict):
        return [t for x in value.values() for t in tensors(x)]
    return []


def map_tensors(value, fn):
    import torch
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(map_tensors(x, fn) for x in value)
    if isinstance(value, list):
        return [map_tensors(x, fn) for x in value]
    if isinstance(value, dict):
        return {k:map_tensors(v, fn) for k,v in value.items()}
    return value


def cpu(value):
    return map_tensors(value, lambda t: t.detach().cpu())


def compare(actual, expected, dtype):
    import torch
    aa, ee = tensors(actual), tensors(expected)
    if len(aa) != len(ee) or not aa:
        return {'status': 'failed', 'reason': 'output structure mismatch'}
    atol, rtol = {'FP32': (1e-5, 1e-4), 'FP16': (2e-3, 2e-2), 'BF16': (2e-2, 8e-2)}.get(dtype, (0, 0))
    checks = []
    for a, e in zip(aa, ee):
        a, e = a.detach().cpu(), e.detach().cpu()
        if a.shape != e.shape:
            return {'status': 'failed', 'reason': 'shape mismatch'}
        if not a.is_floating_point() or not e.is_floating_point():
            ok = a.dtype == e.dtype and torch.equal(a, e)
            checks.append({'passed': ok, 'exact': True});continue
        a, e = a.double(), e.double()
        match = torch.isclose(a, e, atol=atol, rtol=rtol, equal_nan=True)
        finite = torch.isfinite(a) & torch.isfinite(e)
        err = (a[finite] - e[finite]).abs()
        checks.append({'passed': bool(match.all()), 'mismatch_count': int((~match).sum()),
                       'max_absolute_error': float(err.max()) if err.numel() else 0})
    return {'status': 'passed' if all(x['passed'] for x in checks) else 'failed',
            'atol': atol, 'rtol': rtol, 'checks': checks, 'reference': 'CPU FP64 on actual quantized inputs; exact discrete outputs'}


def gradient_weights(output):
    """Binary-exact, nonuniform CPU weights shared by all reference/device paths."""
    import torch
    values = (torch.arange(output.numel(), dtype=torch.float64, device='cpu') % 17) / 16 + 0.5
    return values.reshape(output.shape).to(device=output.device, dtype=output.dtype)


def invocation(fn, inputs, backward=False, nonzero=False, grad_cache=None):
    import torch
    out = fn(*inputs)
    if not backward:
        return out
    required = [t for t in tensors(inputs) if t.requires_grad]
    if nonzero:
        gradients = torch.autograd.grad(out, required, grad_outputs=gradient_weights(out))
    else:
        total = out.sum()
        if grad_cache is not None and grad_cache:
            grad = grad_cache[0]
        else:
            grad = torch.zeros_like(total)
            if grad_cache is not None: grad_cache.append(grad)
        gradients = torch.autograd.grad(total, required, grad_outputs=grad)
    return out, gradients


def build(task):
    import torch
    torch.manual_seed(task['seed'])
    module = load_case(task)
    conf = SimpleNamespace(vendor=task['vendor'], device=torch.device('cpu'),
                           dataformat=task['dtype'], case_name=task['case'], oplib=task['oplib'])
    cfg = SimpleNamespace(**task['case_config'])
    return module.build_case(conf, cfg)


def reference(root, task):
    import torch
    fn, inputs, bp, count = build(task)
    bundle = {'inputs': inputs, 'state': fn.state_dict() if isinstance(fn, torch.nn.Module) else None}
    torch.save(bundle, root / 'inputs.pt')
    # Store quantized inputs before converting to a higher precision independent CPU path.
    ref_inputs = map_tensors(inputs, lambda t: t.detach().to(dtype=torch.float64 if t.is_floating_point() else t.dtype).requires_grad_(t.requires_grad))
    ref_fn = copy.deepcopy(fn).double() if isinstance(fn, torch.nn.Module) else fn
    if task['case'] not in ('dropout', 'native_dropout'):
        expected = invocation(ref_fn, ref_inputs, bp, nonzero=True)
        torch.save(cpu(expected), root / 'reference.pt')
    write(root / 'reference.json', {'status': 'passed', 'input_sha256': sha(root / 'inputs.pt'),
          'inputs': [{'shape': list(t.shape), 'dtype': str(t.dtype), 'requires_grad': t.requires_grad} for t in tensors(inputs)],
          'backward': bp, 'gradient_rule': 'binary-exact nonuniform upstream weights for backward', 'torch': torch.__version__, 'work_per_call': count(1) * (3 if bp else 1),
          'oracle': 'CPU FP64' if task['case'] not in ('dropout', 'native_dropout') else 'dropout structural/statistical check'})


def dropout_check(output, inputs, bp, dtype):
    import torch
    out, grads = output if bp else (output, None)
    out = out.detach().cpu().double(); x = inputs[0].detach().cpu().double()
    keep = out != 0
    nz = x != 0
    sample = int(nz.sum())
    fraction = float(keep[nz].double().mean())
    tolerance = max(0.02, 6 * (0.16 / max(sample, 1))**0.5)
    expected = x * keep / 0.8
    result = compare(out, expected, dtype)
    result['keep_fraction'] = fraction
    result['keep_fraction_tolerance'] = tolerance
    if abs(fraction - 0.8) > tolerance:
        result['status'] = 'failed'
    if bp:
        weights = gradient_weights(out).cpu()
        actual_grad = grads[0].detach().cpu().double()
        grad_check = compare(actual_grad[nz], (keep.to(torch.float64) * weights / 0.8)[nz], dtype)
        # A zero input hides whether dropout kept the element; either legal gradient is valid.
        zeros = ~nz
        zero_ok = (actual_grad[zeros] == 0) | torch.isclose(actual_grad[zeros], (weights / 0.8)[zeros], atol=grad_check.get('atol',0), rtol=grad_check.get('rtol',0))
        if not bool(zero_ok.all()): grad_check['status'] = 'failed'
        grad_check['zero_input_rule'] = 'gradient is zero or scaled upstream weight; mask cannot be inferred from zero output'
        result['gradient'] = grad_check
        if grad_check['status'] != 'passed': result['status'] = 'failed'
    result['reference'] = 'dropout p=0.2: output scaling, mask frequency, and mask-consistent nonzero gradient'
    return result


def device_phase(root, task, phase):
    from vendors import get_vendor
    adapter = get_vendor(task['vendor'])
    device = adapter.bootstrap(task['worker_device'])
    import torch
    fn, _, bp, count = build(task)
    bundle = torch.load(root / 'inputs.pt', map_location='cpu', weights_only=True)
    inputs = map_tensors(bundle['inputs'], lambda t: t.detach().to(device).requires_grad_(t.requires_grad))
    if isinstance(fn, torch.nn.Module):
        fn.load_state_dict(bundle['state']);fn = fn.to(device)
    # CPU construction and parameter initialization precede explicit FlagGems registration.
    global _GEMS_ENABLED
    if task['oplib'] == 'flaggems' and not _GEMS_ENABLED:
        import flag_gems
        flag_gems.enable()
        _GEMS_ENABLED = True
    packages = {}
    for name in ('torch', 'torch-fl', 'flag-gems', 'triton', 'triton-ascend'):
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: pass
    identity = {'correctness_protocol': 'nonuniform-gradient-v1', 'measurement_protocol': 'cached-zero-gradient-v1', 'vendor_runtime': adapter.identity(), 'packages': packages, 'device': str(device), 'case_sha256': sha(ROOT / 'benchmarks' / task['case'] / 'main.py'),
                'input_sha256': sha(root / 'inputs.pt'), 'input_devices': [str(t.device) for t in tensors(inputs)]}
    if phase == 'trace':
        adapter.synchronize()
        print('OPERATION_TARGET_BEGIN', file=sys.stderr, flush=True)
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                with torch.profiler.record_function('operation_target'):
                    out = invocation(fn, inputs, bp, nonzero=True)
                    adapter.synchronize()
        finally:
            print('OPERATION_TARGET_END', file=sys.stderr, flush=True)
        prof.export_chrome_trace(str(root / 'route-trace.json'))
        write(root / 'route-diagnostic.json', {**identity, 'status': 'completed',
              'route_status': 'partial', 'boundary': 'CPU ATen events do not prove device kernel execution',
              'events': [{'name': e.name, 'parent': e.cpu_parent.name if e.cpu_parent else None,
                          'shapes': e.input_shapes} for e in prof.events()],
              'output_devices': [str(t.device) for t in tensors(out)]})
        return
    if phase == 'probe':
        calls = {}
        launcher_states = {}
        profile_call = getattr(adapter, 'profile_call', lambda frame: None)
        def profile(frame, event, arg):
            path = frame.f_code.co_filename
            if event == 'call' and ('flag_gems' in path or 'triton' in path):
                key = (path, frame.f_code.co_name)
                calls[key] = calls.get(key, 0) + 1
                state = profile_call(frame)
                if state is not None:
                    states = launcher_states.setdefault(key, [])
                    if state not in states: states.append(state)
        adapter.synchronize()
        print('OPERATION_TARGET_BEGIN', file=sys.stderr, flush=True)
        sys.setprofile(profile)
        try:
            out = invocation(fn, inputs, bp, nonzero=True)
            adapter.synchronize()
        finally:
            sys.setprofile(None)
            print('OPERATION_TARGET_END', file=sys.stderr, flush=True)
        write(root / 'probe.json', {**identity, 'output_devices': [str(t.device) for t in tensors(out)],
             'calls': [{'path': p, 'function': f, 'count': n, 'sha256': sha(p) if Path(p).is_file() else None,
                        **({'launcher_states': launcher_states[(p,f)]} if (p,f) in launcher_states else {})}
                       for (p,f),n in sorted(calls.items())]})
        actual = cpu(out)
        if task['case'] in ('dropout', 'native_dropout'):
            check = dropout_check(actual, bundle['inputs'], bp, task['dtype'])
        else:
            check = compare(actual, torch.load(root / 'reference.pt', weights_only=True), task['dtype'])
        torch.save(actual, root / 'output.pt')
        write(root / 'correctness.json', check)
        return
    cfg = task['case_config']
    grad_cache = []
    def timed_call(backward=bp):
        return invocation(fn, inputs, backward, grad_cache=grad_cache)
    def timed_once(backward):
        adapter.synchronize(); start = time.perf_counter_ns()
        timed_call(backward)
        adapter.synchronize()
        return (time.perf_counter_ns() - start) / 1000
    cold = timed_once(False)
    for _ in range(cfg['WARMUP']): timed_call()
    warm = timed_once(False)
    per_call = []
    for _ in range(cfg['rounds']):
        adapter.synchronize(); start = time.perf_counter_ns()
        for _ in range(cfg['ITERS']): timed_call()
        adapter.synchronize()
        per_call.append((time.perf_counter_ns() - start) / cfg['ITERS'] / 1000)
    kernel_seconds = adapter.kernel_time(timed_call, cfg)
    # Check an actual measurement-process result too, outside the timer.
    checked = cpu(invocation(fn, inputs, bp, nonzero=True));adapter.synchronize()
    check = dropout_check(checked, bundle['inputs'], bp, task['dtype']) if task['case'] in ('dropout','native_dropout') else compare(checked, torch.load(root / 'reference.pt', weights_only=True), task['dtype'])
    if task.get('diagnostics_mode') == 'failures' and (check['status'] == 'failed' or read(root / 'correctness.json')['status'] == 'failed'):
        torch.save(checked, root / 'measurement-output.pt')
    median = statistics.median(per_call)
    rate = count(1) * (3 if bp else 1) / (median / 1e6) / 1e12
    write(root / 'measurement.json', {**identity, 'correctness': check,
          'cold_us': cold, 'warm_us': warm, 'per_call_us': per_call, 'median_us': median,
          'throughput_op_s': 1e6 / median, 'equivalent_tflops': rate if task['dtype'].startswith(('FP','BF')) else None,
          'work_units_per_second': rate * 1e12, 'formula': 'original op2flops, backward multiplier 3 is an estimate',
          'fu_percent': 100 * rate / task['spectflops'] if task.get('spectflops') and task['dtype'].startswith(('FP','BF')) else None,
          'kernel_us': kernel_seconds * 1e6 if kernel_seconds is not None else None,
          'kernel_status': 'measured' if kernel_seconds is not None else 'not-supported',
          'timing_boundary': 'synchronized host batches; cold/warm forward; original zero-gradient backward batches'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--serve', help='persistent worker control directory relative to root')
    parser.add_argument('--phase', choices=['reference', 'probe', 'measure', 'diagnose', 'trace'], required=True)
    args = parser.parse_args()
    if args.serve:
        control = args.root / args.serve
        previous = None
        count = 0
        while True:
            request_path = control / 'request.json'
            if not request_path.exists():
                time.sleep(0.05);continue
            request = read(request_path)
            if request['id'] == previous:
                time.sleep(0.05);continue
            root = (args.root / request['directory']).resolve()
            if not root.is_relative_to(args.root.resolve()):
                raise ValueError('request escapes evidence root')
            count += 1
            saved_out, saved_err = os.dup(1), os.dup(2)
            record = {'id':request['id'], 'returncode':0, 'request_count':count}
            try:
                with (root / f'{args.phase}.log').open('w') as log:
                    sys.stdout.flush();sys.stderr.flush()
                    os.dup2(log.fileno(), 1);os.dup2(log.fileno(), 2)
                    code = run_phase(root, args.phase)
                    record['returncode'] = code
                    if code:
                        record['error'] = read(root / f'{args.phase}-error.json')['error']
                    sys.stdout.flush();sys.stderr.flush()
            finally:
                os.dup2(saved_out,1);os.dup2(saved_err,2);os.close(saved_out);os.close(saved_err)
            write(control / 'done.json', record)
            previous = request['id']
    return run_phase(args.root, args.phase)


def run_phase(root, phase):
    task = read(root / 'task.json')
    try:
        if phase == 'reference': reference(root, task)
        elif phase == 'diagnose':
            from runtime.diagnostics import numeric
            numeric(root, task)
        else: device_phase(root, task, phase)
        return 0
    except Exception as exc:
        write(root / f'{phase}-error.json', {'type':type(exc).__name__, 'error':str(exc)})
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
