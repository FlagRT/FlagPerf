# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Evidence-derived explanations; no execution gates or precision thresholds."""
from collections import Counter, defaultdict
from runtime.common import read_json
from runtime.performance import summarize


REASONS = {
    'not_scheduled': '尚未调度，没有执行证据',
    'estimate_does_not_fit': '剩余探测额度放不下估计成本，尚未执行',
    'global_budget_exhausted': '本次总探测预算耗尽',
    'search_allowance_exhausted': '探测额度耗尽，仍可能保留最终复验额度',
    'worker_timeout': '单 worker 执行上限触发，不能据此认定总预算不足',
    'confirmation_incomplete': '已尝试，但重复或对照未完成',
    'resource_failure': '设备或内存等资源故障，不能认定算子不支持',
    'evidence_incomplete': '执行检查证据不完整或不一致',
    'unresolved_failure': '失败原因未确定或对照不成立',
    'budget_exhausted': '预算耗尽，尚未充分探测',
    'not_observed': '未观测到函数命中',
    'failure_not_confirmed_within_budget': '失败待确认，预算不足',
    'unresolved_or_resource_failure': '原因未确定或资源异常',
    'not_in_joint_verified_selection': '单侧接受，未进入共同集合',
}


def coverage(policy):
    """Keep environment counts separate; joint selection is not unsupportedness."""
    common = policy.get('include', [])
    profiles = policy.get('profiles') or {'single': policy}
    rows = {}
    for key, entry in profiles.items():
        unknown = [r for r in entry.get('unknown', [])
                   if r.get('reason') != 'not_in_joint_verified_selection']
        omitted = [r['function'] for r in entry.get('unknown', [])
                   if r.get('reason') == 'not_in_joint_verified_selection']
        accepted = entry.get('accepted_include', sorted(set(common) | set(omitted)))
        rows[key] = {
            'candidate_count': entry.get('candidate_count'),
            'accepted_include': accepted, 'common_include': common,
            'not_in_joint_selection': omitted, 'excluded': entry.get('excluded', []),
            'unknown': unknown, 'unknown_reasons': dict(Counter(r.get('reason', 'unrecorded') for r in unknown)),
            'candidate_policy': entry.get('candidate_policy'),
            'verification': entry.get('verification'), 'stack': entry.get('stack'),
        }
    return {'verification_scope': 'execution_hits_and_finite_boundaries_not_accuracy_acceptance',
            'selection_sha256': policy.get('verified_selection_sha256'), 'environments': rows}


def participation(result):
    axis = result.get('comparison_component')
    if result.get('command') == 'preview':
        return {'status': 'not_evaluated', 'reason': 'preview 验证函数可执行性，不评价组件收益'}
    if not axis:
        return {'status': 'single_path', 'reason': '单路径执行，没有组件对照'}
    if axis == 'flagcx' and result.get('parallelism', 'single') == 'single':
        return {'status': 'not_applicable', 'reason': '单卡没有模型集合通信，不能评价 FlagCX 收益'}
    sources = result.get('component_audits', {}) if result.get('command') == 'performance' else result.get('paths', {})
    sides = ['off', 'on']
    missing, unobserved = [], []
    expected_ranks = [str(i) for i in range(len(result.get('devices', [])))]
    for side in sides:
        state = sources.get(side, {})
        if state.get('status') != 'completed':
            missing.append(side + ': 执行未完成或取证缺失')
            continue
        ranks = state.get('ranks') if result.get('parallelism') == 'tp' else {'single': state}
        if not ranks or (result.get('parallelism') == 'tp' and
                         (not expected_ranks or set(ranks) != set(expected_ranks))):
            missing.append(side + ': rank 证据不完整')
            continue
        for rank, row in ranks.items():
            name = side + '/' + rank
            if row.get('status') != 'completed' or row.get('numerical_anomalies'):
                missing.append(name + ': 执行失败或数值异常')
                continue
            if axis == 'flaggems' and side == 'off':
                continue
            if axis == 'flaggems':
                route = row.get('route', {})
                allowed = route.get('allowed')
                calls = route.get('actual_function_calls')
                if not allowed or calls is None:
                    missing.append(name + ': 缺少选择集合或函数计数')
                elif not all(calls.get(n, 0) > 0 for n in allowed):
                    unobserved.append(name)
            elif axis == 'flagtree':
                ev = row.get('components', {}).get('compiler', {})
                expected = 'vendor' if side == 'off' else 'flagtree'
                if ev.get('provider') != expected or ev.get('jit_run_calls') is None:
                    missing.append(name + ': 编译器身份或 JIT 计数缺失/不匹配')
                elif ev['jit_run_calls'] <= 0:
                    unobserved.append(name)
            else:
                ev = row.get('components', {}).get('communication', {})
                expected = 'hccl' if side == 'off' else 'flagcx'
                if ev.get('status') == 'mismatch' or ev.get('model_collective_calls') is None:
                    missing.append(name + ': 通信后端证据缺失/不匹配')
                elif ev['model_collective_calls'] <= 0:
                    unobserved.append(name)
                elif set(ev.get('observed_backends', [])) != {expected}:
                    missing.append(name + ': 模型通信后端不匹配')
    if missing:
        return {'status': 'incomplete', 'reason': '；'.join(missing)}
    if unobserved:
        return {'status': 'not_observed', 'reason': '以下路径未观测到比较所需调用：' + ', '.join(unobserved)}
    return {'status': 'observed', 'reason': '逐侧/逐 rank 已观测到所需组件参与；不代表稳定收益或 kernel 归因'}


def grouped_performance(result):
    """Pair exact repeat/cycle/batch workloads before calculating any ratios."""
    groups = {'shape': defaultdict(lambda: defaultdict(list)),
              'repeat': defaultdict(lambda: defaultdict(list))}
    records = {}
    for side, runs in result.get('paths', {}).items():
        records[side] = {}
        for repeat, state in runs.items():
            if state.get('status') != 'completed':
                continue
            for row in state.get('batches', []):
                if any(k not in row for k in ('cycle', 'batch_index', 'samples', 'tokens', 'latency_ns')):
                    return {'status': 'unavailable', 'reason': '历史批次字段不足，无法可靠配对'}
                key = (repeat, row['cycle'], row['batch_index'])
                if key in records[side] or row['latency_ns'] <= 0:
                    return {'status': 'unavailable', 'reason': '批次重复或时间无效'}
                records[side][key] = row
                groups['repeat'][repeat][side].append(row)
                if row.get('input_shape'):
                    groups['shape'][str(row['input_shape'])][side].append(row)
    paired = bool(records.get('off')) and bool(records.get('on'))
    reason = '单路径或没有可配对的完整批次'
    if paired:
        paired = all(state.get('status') == 'completed' for runs in result['paths'].values() for state in runs.values())
    if paired:
        paired = records['off'].keys() == records['on'].keys()
        if paired:
            paired = all(all(row.get(k) == records['on'][key].get(k)
                             for k in ('samples', 'tokens', 'input_shape', 'sample_ids'))
                         for key, row in records['off'].items())
        reason = '按 repeat/cycle/batch 配对成功' if paired else '两侧批次、形状或工作量不一致，不生成分组比值'
    output = {'status': 'available', 'paired': paired, 'reason': reason, 'shape': [], 'repeat': []}
    for kind, items in groups.items():
        for label, sides in sorted(items.items()):
            summaries = {side: summarize(rows) for side, rows in sides.items()}
            value = {'label': label, 'paths': summaries}
            if paired and {'off', 'on'} <= summaries.keys():
                off, on = summaries['off'], summaries['on']
                value['on_over_off_latency'] = on['latency_mean_ms'] / off['latency_mean_ms']
                value['on_over_off_throughput'] = on['samples_per_second'] / off['samples_per_second']
            output[kind].append(value)
    if any(not row.get('input_shape') for rows in records.values() for row in rows.values()):
        output['shape'] = []
        output['shape_note'] = '历史数据未记录实际输入形状；不从文本长度猜测形状'
    return output


def assess(root, result):
    """Projection used both before sealing a run and during read-only report replay."""
    output = {'comparison': participation(result)}
    path = root / ('preview/policy.yaml' if result.get('command') == 'preview' else 'policy.yaml')
    if path.is_file():
        import yaml
        output['policy'] = coverage(yaml.safe_load(path.read_text()))
    if result.get('command') == 'performance':
        if result.get('level') == 'layer':
            output['layer_status'] = result.get('layer', {}).get('timing', {}).get('status', 'incomplete')
            output['layer_pairing'] = result.get('layer', {}).get('timing', {}).get('paired', False)
        output['performance_groups'] = grouped_performance(result)
        output['measurement_status'] = 'completed' if result.get('summary') else 'incomplete'
        profiles = result.get('profiles', {})
        complete_profiles = (bool(profiles) and set(profiles) == set(result.get('paths', {})) and
            all(bool(runs) and set(runs) == set(result['paths'][side]) and
                all(v.get('status') == 'completed' for v in runs.values()) for side,runs in profiles.items()))
        output['communication_status'] = ('not_applicable' if result.get('parallelism') != 'tp' else
            'completed' if complete_profiles else 'incomplete')
    identity = root / 'prepared/identity.json'
    if identity.is_file():
        env = read_json(identity).get('environment', {})
        output['environment'] = {k: env[k] for k in ('TASK_QUEUE_ENABLE', 'TRITON_ENABLE_TASKQUEUE') if k in env}
    lifetimes = set()
    def visit(value):
        if isinstance(value, dict):
            if value.get('native_backend_lifetime'):
                lifetimes.add(value['native_backend_lifetime'])
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(result.get('paths', {})); visit(result.get('component_audits', {}))
    output['native_backend_lifetimes'] = sorted(lifetimes)
    return output
