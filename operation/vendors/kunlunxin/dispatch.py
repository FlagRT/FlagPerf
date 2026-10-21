# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Observe ATen calls in an isolated probe without replacing operator implementations."""
from contextlib import contextmanager

from runtime.evidence import write, sha
from .profiling import analyze_events, number

# These operators only construct metadata/aliases; copy/allocation never qualifies
# as target computation. Any unrecognized device operation requires a real kernel.
METADATA = {'aten::view', 'aten::_unsafe_view', 'aten::reshape', 'aten::detach',
            'aten::alias', 'aten::as_strided', 'aten::transpose', 'aten::t',
            'aten::expand', 'aten::squeeze', 'aten::unsqueeze', 'aten::slice',
            'aten::select', 'aten::permute', 'aten::lift_fresh'}
PREPARATION = {'aten::empty', 'aten::empty_like', 'aten::empty_strided', 'aten::new_empty',
               'aten::copy_', 'aten::_to_copy', 'aten::clone', 'aten::to'}


def analyze_dispatch(events, rows, device=0, name='first', *, case=None, dtype=None):
    """Join each non-metadata device computation to its own host marker and kernel.

    A copy kernel, mere output placement or another thread's nearby launch cannot
    substitute for the selected device computation. CPU-only host bookkeeping is
    recorded separately, not interpreted as device work.
    """
    marker = 'p800_target_' + name + '_'
    result = {'protocol': 'p800-dispatch-correlation-v1', 'status': 'partial',
              'target_cpu_fallback': False, 'operations': rows, 'computations': [],
              'reason': 'no complete device computation evidence'}
    try:
        if not isinstance(rows, list) or not rows:
            return result
        if any(not isinstance(row, dict) or type(row.get('index')) is not int or row.get('index') != i
               or not isinstance(row.get('op'), str)
               or any(type(row.get(k)) is not bool for k in ('device_input', 'device_output', 'cpu_transfer'))
               for i, row in enumerate(rows)):
            return result
        result['target_cpu_fallback'] = any(row.get('cpu_transfer') for row in rows)
        if result['target_cpu_fallback']:
            result.update(status='failed', reason='device-input dispatch transferred data or returned a tensor on CPU')
            return result
        timeline = analyze_events(events, device, count=1, marker=marker)
        result['timeline'] = timeline
        if timeline['status'] != 'measured':
            result['reason'] = timeline.get('reason');return result
        kernel_corr = {k['correlation'] for k in timeline['kernels']}
        computations = []
        missing = []
        for row in rows:
            if (not row.get('device_input') or row['op'] in METADATA | PREPARATION
                    or case == 'linear' and row['op'] == 'aten::_has_same_storage_numel'):
                continue
            if not row.get('device_output'):
                missing.append(row['index']);continue
            marks = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
                     and e.get('name') == 'p800_dispatch_' + str(row['index'])]
            if len(marks) != 1:
                missing.append(row['index']);continue
            mark = marks[0];lo=number(mark['ts']);hi=lo+number(mark['dur'])
            launches = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cuda_runtime'
                        and e.get('name') == 'cudaLaunchKernel'
                        and (e.get('pid'), e.get('tid')) == (mark.get('pid'), mark.get('tid'))
                        and lo <= number(e['ts']) <= number(e['ts'])+number(e['dur']) <= hi
                        and e.get('args',{}).get('correlation') in kernel_corr]
            if not launches:
                missing.append(row['index']);continue
            computations.append({'index':row['index'], 'op':row['op'],
                'correlations':[e['args']['correlation'] for e in launches]})
        preparation = set()
        for row in rows:
            if row['op'] not in PREPARATION:
                continue
            marks = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
                     and e.get('name') == 'p800_dispatch_' + str(row['index'])]
            if len(marks) != 1:
                continue
            mark = marks[0]; lo = number(mark['ts']); hi = lo + number(mark['dur'])
            preparation.update(e.get('args', {}).get('correlation') for e in events
                               if e.get('cat') == 'cuda_runtime' and e.get('ph') == 'X'
                               and e.get('name') == 'cudaLaunchKernel'
                               and (e.get('pid'), e.get('tid')) == (mark.get('pid'), mark.get('tid'))
                               and lo <= number(e['ts']) <= number(e['ts']) + number(e['dur']) <= hi)
        # The locked XPyTorch LinearFunction launches directly from C++ and is
        # invisible to TorchDispatchMode apart from allocation/views. Require its
        # own concrete CPU record, matching target-thread launch and reviewed
        # XBLAS FC kernel; no generic 'Function' or nearby-kernel shortcut.
        if not computations and not missing and case == 'linear':
            opaque = [e for e in events if e.get('cat') == 'cpu_op' and e.get('ph') == 'X'
                      and e.get('name') == 'LinearFunction']
            targets = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X'
                       and e.get('name') == marker + '0']
            if len(opaque) == len(targets) == 1:
                op, target = opaque[0], targets[0]
                lo = number(op['ts']); hi = lo + number(op['dur'])
                a = number(target['ts']); b = a + number(target['dur'])
                launches = [e for e in events if e.get('cat') == 'cuda_runtime' and e.get('ph') == 'X'
                            and e.get('name') == 'cudaLaunchKernel'
                            and (e.get('pid'), e.get('tid')) == (op.get('pid'), op.get('tid'))
                            and lo <= number(e['ts']) <= number(e['ts']) + number(e['dur']) <= hi]
                launches = [e for e in launches if e.get('args', {}).get('correlation') not in preparation]
                attributed = {e.get('args', {}).get('correlation') for e in launches}
                selected = [k for k in timeline['kernels'] if k['correlation'] not in preparation]
                names = [k['name'] for k in selected]
                fc = lambda name: name.startswith('void xblas_xpu3::fc_one_loop<')
                bf16 = (len(names) == 5
                        and names[0].startswith('void xpu3::findmax1d_cluster<bfloat16>')
                        and names[1].startswith('hydra_internal_kernel::cast_te(bfloat16 const*')
                        and names[2].startswith('void xpu3::findmax1d_cluster<bfloat16>')
                        and names[3].startswith('hydra_internal_kernel::cast_te(bfloat16 const*')
                        and names[4].startswith('void xblas_xpu3::fc_bf16_rn_one_loop<'))
                reviewed = (dtype in ('FP32', 'FP16') and len(names) == 1 and fc(names[0])
                            or dtype == 'BF16' and bf16)
                if ((op.get('pid'), op.get('tid')) == (target.get('pid'), target.get('tid'))
                        and a <= lo < hi <= b and attributed == kernel_corr - preparation and reviewed
                        and any(row['device_input'] and row['device_output'] for row in rows)):
                    # Nested helper/preparation launch cannot qualify the opaque target.
                    helpers = [e for e in events if e.get('cat') == 'cpu_op' and e.get('ph') == 'X'
                               and e.get('name') in METADATA | PREPARATION
                               and (e.get('pid'), e.get('tid')) == (op.get('pid'), op.get('tid'))]
                    if not any(number(h['ts']) <= number(e['ts']) < number(h['ts']) + number(h['dur'])
                               for h in helpers for e in launches):
                        computations.append({'op':'LinearFunction', 'case':'linear',
                                             'evidence':'reviewed-cpp-record-and-xblas-fc',
                                             'correlations':sorted(attributed)})
        result['computations'] = computations
        result['missing_device_computations'] = missing
        covered = {corr for row in computations for corr in row['correlations']}
        unattributed = kernel_corr - covered - preparation
        result['unattributed_computation_correlations'] = sorted(unattributed)
        if computations and not missing and not unattributed:
            result.update(status='passed',reason=None)
        elif unattributed:
            result['reason'] = 'device kernels outside observed computation or preparation windows'

        elif missing:
            result['reason']='one or more device computations lacks an attributed kernel or device output'
        return result
    except (KeyError, TypeError, ValueError, ArithmeticError, AttributeError) as exc:
        result.update(status='partial', reason='invalid dispatch evidence: '+str(exc))
        return result


@contextmanager
def capture(root, adapter, observer, name):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from runtime.worker import tensors
    rows = []
    class Audit(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            values = tensors((args, kwargs))
            index = len(rows)
            row = {'index':index, 'op':func._schema.name, 'overload':func._schema.overload_name,
                   'inputs':[str(t.device) for t in values], 'device_input':any(t.device == adapter.device for t in values)}
            rows.append(row)
            with torch.profiler.record_function('p800_dispatch_'+str(index)):
                output = func(*args, **kwargs)
            outputs = tensors(output)
            row.update(outputs=[str(t.device) for t in outputs],
                       device_output=bool(outputs) and all(t.device == adapter.device for t in outputs),
                       cpu_transfer=row['device_input'] and any(t.device.type == 'cpu' for t in outputs))
            return output
    target_error = None
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as profiler:
            with torch.profiler.record_function('p800_target_'+name+'_0'), Audit():
                yield
    except BaseException as exc:
        target_error = exc
        raise
    finally:
        # Keep original target failures, and retain the attempted trace when the
        # exporter is usable. An evidence error must never replace that failure.
        try:
            path = root / ('route-'+name+'-trace.json')
            profiler.export_chrome_trace(str(path))
            import json
            result = analyze_dispatch(json.loads(path.read_text()).get('traceEvents',[]), rows, adapter.device.index, name, case=observer.case, dtype=observer.dtype)
            if target_error is not None:
                result.update(status='partial', reason='target raised '+type(target_error).__name__)
            result.update(trace=path.name, trace_sha256=sha(path))
            write(root / ('route-'+name+'-dispatch.json'), result)
            observer.dispatch[name] = result
        except Exception as exc:
            write(root / ('route-'+name+'-capture-error.json'),
                  {'type':type(exc).__name__, 'error':str(exc), 'target_error_preserved':target_error is not None})
            if target_error is None:
                raise
