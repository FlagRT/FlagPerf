# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Supplemental evidence, never an alternative correctness gate."""
import copy
import json
import time

from runtime.evidence import read, write, sha

PROTOCOL = 'failure-diagnostics-v2'
CHECK_PROTOCOL = 'operation-checks-v1'


def errors(actual, expected, atol, rtol):
    import torch
    a, e = actual.detach().cpu(), expected.detach().cpu()
    if a.shape != e.shape:
        return {'status': 'shape-mismatch'}
    if not a.is_floating_point() or not e.is_floating_point():
        mismatch = int((a != e).sum()) if a.dtype == e.dtype else a.numel()
        return {'elements': a.numel(), 'exact': True, 'dtype_match': a.dtype == e.dtype,
                'mismatch_count': mismatch, 'mismatch_fraction': mismatch / a.numel() if a.numel() else 0}
    a, e = a.double(), e.double()
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
        write(root / 'numeric-diagnostic.json', {'status': 'not-applicable' if task.get('common_diagnosis') else 'unsupported', 'protocol': PROTOCOL,
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
        if not (root / path).exists():
            result.setdefault('missing_outputs', []).append(label)
            continue
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


def reference_check(root, task):
    """Recompute a CPU reference from saved inputs/state without regenerating them."""
    import torch
    from runtime.worker import build, map_tensors, invocation, tensors, cpu
    fn, prototype, backward, _ = build(task)
    bundle = torch.load(root / 'inputs.pt', map_location='cpu', weights_only=True)
    inputs = bundle['inputs']
    def structure(value):
        if isinstance(value, torch.Tensor):
            return ('tensor', tuple(value.shape), str(value.dtype), value.requires_grad)
        if isinstance(value, (tuple, list)):
            return (type(value).__name__, tuple(structure(x) for x in value))
        if isinstance(value, dict):
            return ('dict', tuple((k, structure(v)) for k, v in value.items()))
        return (type(value).__name__, value)
    if structure(inputs) != structure(prototype):
        raise ValueError('saved inputs differ from the current Case shape/dtype/gradient contract')
    if isinstance(fn, torch.nn.Module):
        if structure(bundle.get('state')) != structure(fn.state_dict()):
            raise ValueError('saved module state differs from the current Case contract')
        fn.load_state_dict(bundle['state'], strict=True)
        fn = copy.deepcopy(fn).double()
    elif bundle.get('state') is not None:
        raise ValueError('unexpected module state in saved input')
    details = {'input_sha256': sha(root / 'inputs.pt'),
               'inputs': [{'shape': list(t.shape), 'dtype': str(t.dtype), 'stride': list(t.stride()),
                           'requires_grad': t.requires_grad, 'nonfinite_count': int((~torch.isfinite(t)).sum())} for t in tensors(inputs)],
               'backward': backward, 'torch': torch.__version__, 'gate_changed': False}
    if task['case'] in ('dropout', 'native_dropout'):
        write(root / 'reference-validation.json', {**details, 'status': 'not-applicable',
              'reason': 'input contract checked; stochastic output uses the original structural/mask-gradient checks'})
        return
    inputs64 = map_tensors(inputs, lambda t: t.detach().to(dtype=torch.float64 if t.is_floating_point() else t.dtype).requires_grad_(t.requires_grad))
    actual = cpu(invocation(fn, inputs64, backward, nonzero=True))
    expected = torch.load(root / 'reference.pt', map_location='cpu', weights_only=True)
    aa, ee = tensors(actual), tensors(expected)
    same = structure(actual) == structure(expected) and len(aa) == len(ee) and bool(aa)
    for a, e in zip(aa, ee):
        if a.shape != e.shape or a.dtype != e.dtype:
            same = False;break
        match = (a == e) | (torch.isnan(a) & torch.isnan(e)) if a.is_floating_point() else a == e
        same = same and bool(match.all())
    write(root / 'reference-validation.json', {**details, 'status': 'passed' if same else 'failed',
          'reference_reproduced': same, 'boundary': 'exact value comparison, matching NaNs; CPU FP64 is not infinite precision'})
    if not same:
        raise ValueError('saved reference differs from recomputation; stop before device replay')

def numeric_observation(value):
    comparisons = value.get('comparisons', {})
    phase = 'measure' if comparisons.get('measure') else 'probe'
    rows = comparisons.get(phase, [])
    if not rows:
        return value.get('reason', '没有可用的逐张量数值对照。')
    parts = []
    for row in rows:
        device = row.get('device_vs_fp64', {})
        cpu32 = row.get('cpu_fp32_vs_fp64', {})
        parts.append(f"{phase} tensor {row.get('tensor')} ({row.get('role', 'output')})："
                     f"设备超阈值 {device.get('mismatch_count', '未记录')}/{device.get('elements', '未记录')}，"
                     f"最大绝对误差 {device.get('absolute_error_quantiles', {}).get('max', '不适用/未记录')}，"
                     f"CPU FP32 超阈值 {cpu32.get('mismatch_count', '未记录')}。")
    return ' '.join(parts) + ' 误差大小不直接证明原因。'


def inspect_evidence(source, task, result, identity, digests, *, scope='saved-source', prefix='source-task/', sealed=True):
    checks = []
    def add(name, status, observation, evidence, next_step=''):
        checks.append({'check': name, 'status': status, 'observation': observation,
                       'evidence': evidence, 'next_step': next_step, 'scope': scope, 'execution_status': 'recorded', 'elapsed_s': 0})
    add('integrity', 'passed' if sealed else 'recorded',
        '源任务及所有消费文件的 SHA-256 校验通过；哈希不独立证明原测量有效。' if sealed else '当前任务尚未封存；运行结束时统一生成哈希索引。',
        ['source-artifacts.json'] if sealed else ['task.json'])
    result_path = 'source-result.json' if sealed else 'result.json'
    identity_path = 'source-identity.json' if sealed else 'diagnostic-context.json'
    has_input = 'inputs.pt' in digests
    add('inputs', 'recorded' if has_input else 'missing', '已保存实际输入。' if has_input else '未保存实际输入，不能同输入复放。', [prefix + 'inputs.pt'] if has_input else [], '缺失时补采原任务输入。' if not has_input else '')
    stochastic = task.get('case') in ('dropout', 'native_dropout')
    reference = 'reference.pt' in digests
    add('reference', 'not-applicable' if stochastic else 'recorded' if reference else 'missing',
        '随机算子使用原结构/mask/梯度契约。' if stochastic else '保存了参考；离线检查未重算其数值。' if reference else '参考缺失，数值对照不可执行。',
        [prefix + 'reference.pt'] if reference else [], '需要重算校验时使用 --replay。' if reference else '')
    for name, filename, value in [('correctness', 'correctness.json', result.get('correctness', {})), ('routing', 'probe.json', result.get('routing', {}))]:
        if filename == 'correctness.json' and filename in digests:
            value = read(source / filename)
        status = value.get('status', 'missing')
        evidence = [prefix + filename] if filename in digests else [result_path]
        if name == 'correctness':
            observation = '原正确性状态：' + status + '；保留原阈值，不确认根因。'
            next_step = '核对误差统计、参考及相同输入复放结果。' if status == 'failed' else ''
        else:
            observation = '原路由状态：' + status + '；CPU 调用链不等于设备 kernel 证据。'
            next_step = '补充目标计算的执行路径证据。' if status != 'passed' else ''
        add(name, status, observation, evidence, next_step)
    measured_gate = result.get('measurement', {}).get('correctness')
    if measured_gate:
        add('measurement-correctness', measured_gate.get('status', 'missing'),
            '测量进程实际结果的原门禁；与 probe 门禁分别保留。',
            [prefix + 'measurement.json'] if 'measurement.json' in digests else [result_path],
            '任一进程门禁失败均应检查参考和数值证据。' if measured_gate.get('status') == 'failed' else '')
    error_files = sorted(n for n in digests if n.endswith('-error.json'))
    error = result.get('error')
    if error or result.get('status') in ('blocked', 'failed') or error_files:
        hint = result.get('diagnosis')
        add('execution', result.get('status', 'failed'),
            f"原阶段：{result.get('failure_stage', '未记录')}；错误：{error or '见原门禁/错误文件'}；已有分类提示：{hint or '未分类'}。分类提示不等于已验证根因。",
            [result_path, *(prefix + n for n in error_files)], '按失败阶段检查原日志、注册与运行栈版本；数值对照需要先有输出。')
    else:
        add('execution', result.get('status', 'missing'), '原任务执行状态；不代表本次执行了设备任务。', [result_path])
    saved_checks = []
    if 'diagnostic-checks.json' in digests:
        saved = read(source / 'diagnostic-checks.json')
        if saved.get('check_protocol') == CHECK_PROTOCOL:
            for row in saved.get('checks', []):
                if row.get('execution_status') != 'recorded':
                    saved_checks.append({**row, 'scope': scope,
                        'observation': '原补采检查：' + row['observation'],
                        'evidence': [prefix + p for p in row['evidence']]})
    recorded_names = {row['check'] for row in saved_checks}
    file_checks = {'numeric-diagnostic.json': 'diagnose', 'route-diagnostic.json': 'trace',
                   'reference-validation.json': 'reference-check'}
    for name in ('numeric-diagnostic.json', 'route-diagnostic.json', 'reference-validation.json', 'reference.json', 'input-roundtrip.json'):
        if name in digests and file_checks.get(name, name) not in recorded_names:
            value = read(source / name)
            observation = numeric_observation(value) if name == 'numeric-diagnostic.json' else '已有附加证据，沿用原观察边界。'
            if name == 'reference.json':
                observation = '已记录输入元数据：' + json.dumps(value.get('inputs', []), ensure_ascii=False) + '；参考：' + value.get('oracle', '未记录')
            add(file_checks.get(name, name), value.get('status', 'recorded'), observation, [prefix + name])
    checks.extend(saved_checks)
    health = identity.get('postflight_error') or identity.get('cleanup_error')
    add('runtime', 'failed' if health else 'recorded' if identity.get('actual_image_id') else 'missing',
        '原运行身份与健康记录；不表示当前服务器空闲。' + (f' 原记录异常：{health}' if health else ''),
        [identity_path], '实机复放前重新检查身份与设备。')
    return checks



CHECK_FILES = {'reference-check': 'reference-validation.json', 'probe': 'correctness.json',
               'diagnose': 'numeric-diagnostic.json', 'trace': 'route-diagnostic.json'}


def check_record(check, status, scope, observation, evidence=(), *, execution_status='skipped', next_step='', elapsed_s=0):
    return {'check': check, 'status': status, 'execution_status': execution_status, 'scope': scope,
            'observation': observation, 'evidence': list(evidence), 'next_step': next_step, 'elapsed_s': elapsed_s}


def run_check(pool, root, task, mode, *, scope, records, close_after=False):
    """One shared, bounded check. Preserve records even when cleanup must abort."""
    from runtime.cli import CleanupError
    started = time.monotonic()
    filename = CHECK_FILES[mode]
    row = check_record(mode, 'running', scope, '', execution_status='running')
    records.append(row)
    stochastic = task.get('case') in ('dropout', 'native_dropout')
    if mode == 'diagnose' and stochastic:
        row.update(status='not-applicable', execution_status='skipped',
                   observation='随机算子沿用 correctness.json 的结构/mask/梯度契约，不执行 FP64 输出对照。',
                   evidence=['correctness.json'] if (root / 'correctness.json').exists() else [])
        return row
    required = ['inputs.pt']
    if mode != 'trace' and not stochastic: required.append('reference.pt')
    if mode == 'diagnose': required.append('correctness.json')
    missing = [n for n in required if not (root / n).is_file()]
    if mode == 'diagnose' and not any((root / n).is_file() for n in ('output.pt', 'measurement-output.pt')):
        missing.append('output.pt or measurement-output.pt')
    if missing:
        row.update(status='missing', execution_status='skipped', observation='缺少检查输入：' + ', '.join(missing),
                   next_step='保留失败阶段和日志；取得必要输入/输出后再补查。')
        return row
    try:
        pool.phase(root, task, mode)
        value = read(root / filename)
        row.update(status=value['status'], execution_status='completed', evidence=[filename],
                   observation=value.get('reason', value.get('boundary', '公共检查已完成；不改变原任务门禁。')))
        if mode == 'diagnose': row['observation'] = numeric_observation(value)
        if mode == 'trace':
            row.update(observation='CPU profiler 调用链已采集；不证明设备 kernel 执行，不提升原路由状态。',
                       evidence=[filename, 'route-trace.json'])
    except (KeyboardInterrupt, SystemExit):
        row.update(status='failed', execution_status='failed', observation='公共检查被中断。')
        pool.close()
        raise
    except CleanupError:
        row.update(status='failed', execution_status='failed', observation='设备清理未确认，停止后续工作。')
        raise
    except Exception as exc:
        row.update(status='failed', execution_status='failed', observation=str(exc), next_step='检查本阶段原始日志。')
        row['evidence'] = [n for n in (filename, mode + '.log', mode + '-error.json') if (root / n).is_file()]
        # A timeout can leave the worker alive. Never reuse uncertain state.
        pool.close()
        if mode == 'trace' and pool.args.execution == 'docker':
            try:
                pool.adapter.preflight(pool.root, [task['device_id']], label=f'diagnostic-health-{root.name}')
            except Exception as health_error:
                raise CleanupError(f'diagnostic device health check failed: {health_error}') from health_error
    finally:
        row['elapsed_s'] = time.monotonic() - started
        if close_after:
            try: pool.close()
            except CleanupError:
                row.update(status='failed', execution_status='failed', observation='设备清理未确认，停止后续工作。')
                raise
    return row


def collect(pool, root, task, item, *, identity=None, trigger='failure', reference_row=None):
    """Automatic and explicit replay share selection, execution and result schema.

    Execution failures only receive a host-side evidence inventory. Numeric
    mismatches add a CPU reference check; partial routes add at most one trace.
    """
    if trigger == 'failure' and task.get('diagnostics_mode', 'off') != 'failures': return
    numeric_failure = any(c.get('status') == 'failed' for c in
                          (item.get('correctness', {}), item.get('measurement', {}).get('correctness', {})))
    partial_route = item.get('routing', {}).get('status') == 'partial'
    execution_error = bool(item.get('error') or item.get('diagnostic_execution_blocked'))
    if trigger == 'failure' and not (numeric_failure or partial_route or execution_error or item.get('status') in ('failed', 'blocked', 'partial')):
        return
    started = time.monotonic()
    scope = 'automatic' if trigger == 'failure' else 'new-replay'
    result = {'protocol': PROTOCOL, 'check_protocol': CHECK_PROTOCOL, 'trigger': trigger,
              'status': 'running', 'results': {}, 'checks': [], 'elapsed_s': 0}
    item['diagnostics'] = result
    try:
        # Metadata checks never load tensors or launch a worker, including OOM/
        # initialization/timeout cases where another device call is inappropriate.
        context = {k: v for k, v in (identity or {}).items() if k in
                   ('actual_image_id', 'runtime', 'preflight', 'postflight', 'cleanup_error', 'postflight_error')}
        write(root / 'diagnostic-context.json', context)
        files = {p.name for p in root.iterdir() if p.is_file()}
        result['checks'] = inspect_evidence(root, task, item, context, files, scope=scope, prefix='', sealed=False)
        active = not execution_error
        needs_numeric = numeric_failure or trigger == 'replay'
        if active and needs_numeric:
            if reference_row is None:
                reference_row = run_check(pool, root, task, 'reference-check', scope=scope,
                                          records=result['checks'], close_after=trigger == 'replay')
            valid_reference = reference_row['execution_status'] == 'completed' and reference_row['status'] in ('passed', 'not-applicable')
            if valid_reference:
                run_check(pool, root, task, 'diagnose', scope=scope, records=result['checks'], close_after=trigger == 'replay')
            else:
                result['checks'].append(check_record('diagnose', 'missing', scope,
                    '输入/参考校验未通过，跳过数值补充。', next_step='先检查 reference-check 证据。'))
                active = False
        if not active and partial_route and not execution_error:
            result['checks'].append(check_record('trace', 'missing', scope,
                '输入/参考校验未通过，跳过额外设备调用。', next_step='先检查 reference-check 证据。'))
        if active and partial_route:
            run_check(pool, root, task, 'trace', scope=scope, records=result['checks'], close_after=trigger == 'replay')
    except Exception as exc:
        from runtime.cli import CleanupError
        result['checks'].append(check_record('collection', 'failed', scope, str(exc), execution_status='failed'))
        if isinstance(exc, CleanupError): raise
    finally:
        for row in result['checks']:
            if row['check'] in CHECK_FILES and row.get('execution_status') != 'recorded':
                result['results'][row['check']] = {'status': row['status'], 'execution_status': row['execution_status'],
                    'evidence': row['evidence'][0] if row['evidence'] else None, 'error': row['observation'],
                    'elapsed_s': row['elapsed_s']}
        failed = any(c.get('execution_status') == 'failed' for c in result['checks'])
        gaps = any(c['status'] in ('missing', 'partial', 'not-run', 'unsupported') for c in result['checks'])
        result['status'] = 'failed' if failed else 'partial' if gaps else 'completed'
        result['elapsed_s'] = time.monotonic() - started
        write(root / 'diagnostic-checks.json', result)
    return result


def check_table(checks, prefix=''):
    """The same check columns and evidence links in automatic/offline reports."""
    lines = ['| 范围 | 检查 | 观察状态 | 检查执行 | 耗时 s | 观察与下一步 | 证据 |',
             '|---|---|---|---|---:|---|---|']
    escape = lambda value: str(value).replace('|', '\\|').replace('\n', ' ')
    for row in checks:
        text = row['observation'] + (' 下一步：' + row['next_step'] if row.get('next_step') else '')
        links = ' '.join(f'[{p}]({prefix}{p})' for p in row['evidence'])
        values = [row['scope'], row['check'], row['status'], row.get('execution_status', 'recorded'),
                  f"{row.get('elapsed_s', 0):.3f}", text]
        lines.append('| ' + ' | '.join(escape(v) for v in values) + ' | ' + links + ' |')
    return lines
