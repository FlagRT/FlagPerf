# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Evidence-derived layer reading view. Capture and execution verdicts stay source-owned."""
from collections import defaultdict
from copy import deepcopy
import json
import math
import shlex
from statistics import mean, stdev

from runtime.common import digest, read_json, write_json
from analysis.layer import ranks, overhead, statistics
from analysis.assessment import grouped_performance


def shape_key(value):
    if isinstance(value, str):
        value = json.loads(value if value.strip().startswith('[') else '['+value+']')
    if not isinstance(value, (list, tuple)) or not value or any(type(v) is not int or v < 1 for v in value):
        raise ValueError('shape 使用正整数维度，例如 4,20')
    return str(list(value))


def selection(result, layers=None, selected_ranks=None, shapes=None):
    groups = result.get('layer', {}).get('timing', {}).get('groups', [])
    options = {'layers': sorted({g['module'] for g in groups}),
               'ranks': sorted({str(g['rank']) for g in groups}),
               'shapes': sorted({shape_key(g['shape']) for g in groups})}
    chosen = {'layers': list(layers) if layers and layers != ['all'] else [],
              'ranks': [str(r) for r in selected_ranks] if selected_ranks is not None else [],
              'shapes': [shape_key(s) for s in shapes] if shapes else []}
    for name, values in chosen.items():
        missing = set(values)-set(options[name])
        if missing:
            raise ValueError(f'{name} 请选择已采集值 {options[name]}；本次请求 {sorted(missing)}')
    if any(chosen.values()) and not any(matches(g, chosen) for g in groups):
        raise ValueError('请选择具有共同记录的 layer/rank/shape 组合')
    return chosen


def matches(row, selected, rank=None):
    return (not selected.get('layers') or row.get('module') in selected['layers']) and (
        not selected.get('ranks') or str(row.get('rank', rank)) in selected['ranks']) and (
        not selected.get('shapes') or shape_key(row.get('shape', row.get('input_shape'))) in selected['shapes'])


def filtered(result, selected):
    """Filter a presentation copy; raw files and global coverage retain original scope."""
    view = deepcopy(result)
    layer = view.get('layer', {})
    layer.get('timing', {})['groups'] = [g for g in layer.get('timing', {}).get('groups', []) if matches(g, selected)]
    for repeats in layer.get('runs', {}).get('layer_memory', {}).values():
        for state in repeats.values():
            for rank, value in ranks(state).items():
                value['rows'] = [r for r in value.get('rows', []) if matches(r, selected, rank)]
    for repeats in layer.get('profiles', {}).values():
        for states in repeats.values():
            for rank, value in states.items():
                value['rows'] = [r for r in value.get('rows', []) if matches(r, selected, rank)]
                events = []
                for event in value.get('communication', {}).get('events', []):
                    event['layer_calls'] = [r for r in event.get('layer_calls', []) if matches(r, selected, rank)]
                    if event['layer_calls']: events.append(event)
                if any(selected.values()): value.get('communication', {})['events'] = events
    for states in layer.get('routes', {}).values():
        for rank, modules in list(states.items()):
            states[rank] = {name: r for name, r in modules.items()
                            if (not selected.get('layers') or name in selected['layers'])
                            and (not selected.get('ranks') or rank in selected['ranks'])}
    return view


def build(result, selected):
    layer = result.get('layer', {})
    samples, memory, profiles = defaultdict(list), defaultdict(list), defaultdict(list)
    for phase, target in [('layer_timing', samples), ('layer_memory', memory)]:
        for side, repeats in layer.get('runs', {}).get(phase, {}).items():
            for label, state in repeats.items():
                if state.get('status') != 'completed': continue
                for rank, value in ranks(state).items():
                    if value.get('status') != 'completed': continue
                    for row in value.get('rows', []):
                        target[(row['module'], str(rank), str(row['input_shape']), side)].append((label, row))
    for side, repeats in layer.get('profiles', {}).items():
        for label, states in repeats.items():
            for rank, state in states.items():
                for row in state.get('rows', []):
                    profiles[(row['module'], str(rank), str(row['input_shape']), side)].append((state, row))
    groups = []
    for group in layer.get('timing', {}).get('groups', []):
        if not matches(group, selected): continue
        item = {'module': group['module'], 'rank': str(group['rank']), 'shape': shape_key(group['shape']),
                'anchor': 'layer-'+digest([group['module'], str(group['rank']), group['shape']])[:12],
                'ratio': group.get('on_over_off_device_time'), 'paths': {}}
        for side, old in group['paths'].items():
            key = (item['module'], item['rank'], item['shape'], side)
            raw = samples[key]
            stats = statistics([r for _, r in raw], 'device_ns') or old.get('device') or {}
            repeats = defaultdict(list)
            for label, r in raw:
                if isinstance(r.get('device_ns'), (int, float)) and math.isfinite(r['device_ns']) and r['device_ns'] > 0:
                    repeats[label].append(r['device_ns']/1e6)
            means = [mean(v) for v in repeats.values()]
            mem = [r for _, r in memory[key] if r.get('peak_increment')]
            prof = profiles[key]
            events = {}
            for kind in ('compute', 'communication', 'copy'):
                values = [r.get('events', {}).get(kind, {}).get('duration_sum_ns') for _, r in prof]
                events[kind] = mean(values)/1e6 if values and all(v is not None for v in values) else None
            route = layer.get('routes', {}).get(side, {}).get(item['rank'], {}).get(item['module'], {})
            item['paths'][side] = {'device': stats, 'host': old.get('host'), 'repeats': len(means),
                'repeat_means_ms': dict(zip(repeats, means)),
                'repeat_mean_sd_ms': stdev(means) if len(means) > 1 else None,
                'peak_increment_mib': max(r['peak_increment']['allocated_bytes'] for r in mem)/2**20 if mem else None,
                'net_change_mib': mean(r['net_change']['allocated_bytes'] for r in mem)/2**20 if mem else None,
                'memory_calls': len(mem), 'profile_calls': len(prof), 'event_mean_ms': events,
                'communication_applicable': result.get('parallelism') == 'tp',
                'profile_states': sorted({s['status'] for s, _ in prof}),
                'profile_evidence': sorted({s['evidence'] for s, _ in prof}),
                'route_native_ratio': route.get('classification', {}).get('policy_native_call_ratio'),
                'route_function_calls': sum(route.get('actual_function_calls', {}).values()) if route else None}
        groups.append(item)
    perturbation = [r for r in overhead(result, by_shape=True)
                    if (not selected.get('ranks') or str(r['rank']) in selected['ranks'])
                    and (not selected.get('shapes') or r.get('shape') in selected['shapes'])] if 'layer' in result else []
    model = grouped_performance(result)
    return {'schema_version': 1, 'selection': selected, 'model': model, 'groups': groups,
            'overhead_by_shape': perturbation,
            'scope': {'model': 'full sealed workload', 'layers': 'selected module/rank/shape',
                      'events': 'per-call means from independent profiler replay',
                      'routes': 'all audited shapes for each module/rank',
                      'quantiles': 'linear interpolation of observed calls; expected upper-tail count n*(1-p)',
                      'replicates': 'independent worker repeats; calls within a repeat share process/environment'}}


def number(v, digits=3):
    return '待采集' if v is None else f'{v:.{digits}f}'


def cell(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


def unified_rows(groups, prefix=''):
    lines = ['| 层 / rank / 输入shape | 路径 | 计时调用 / 独立重复 | 设备均值 [范围] ms | 窗口显存增量最大 MiB | 计算 / 通信事件均值 ms | on/off层窗口 |',
             '|---|---|---:|---|---:|---|---:|']
    for g in groups:
        for side in ('off', 'on'):
            if side not in g['paths']: continue
            p = g['paths'][side]; d = p['device']; ev = p['event_mean_ms']
            name = f"{g['module']} / {g['rank']} / {g['shape']}"
            lines.append(f"| [{cell(name)}]({prefix}#{g['anchor']}) | {side} | {d.get('calls', 0)} / {p['repeats']} | "
                         f"{number(d.get('mean_ms'))} [{number(d.get('min_ms'))}, {number(d.get('max_ms'))}] | "
                         f"{number(p['peak_increment_mib'])} | {number(ev['compute'])} / {number(ev['communication']) if p['communication_applicable'] else '不适用'} | "
                         f"{number(g['ratio']) if side == 'on' else '—'} |")
    return lines


def render_summary(root, result, evidence):
    """A short landing page plus a navigable, full per-layer index."""
    from reporting.components import NAMES
    from analysis.assessment import participation
    groups = evidence['groups']; model = evidence['model']
    axis = result.get('comparison_component')
    lines = ['# Inference 性能测试：Layer level', '',
             f"比较对象：{NAMES.get(axis, '固定组件组合')}；off 为参照、on 为待比较路径。" if axis else '本次观察固定组件组合，路径名沿用 FlagGems 状态。', '',
             '## 本次发现', '',
             '完整模型耗时来自设备驻留输入上的 forward、pooling、归一化至最终同步；各层窗口来自独立 Event 采样。', '']
    if (root/'prepared/identity.json').is_file():
        ident = read_json(root/'prepared/identity.json')
        model_info, inputs = ident.get('model', {}), ident.get('inputs', {})
        devices = ident.get('parallelism', {}).get('devices', ident.get('physical_device', '见身份文件'))
        lines += [f"模型：{model_info.get('name')}；dtype：{model_info.get('dtype')}；物理设备：{devices}；"
                  f"batch size：{inputs.get('batch_size')}；max length：{inputs.get('max_length')}。", '']
    if result.get('error'): lines += ['执行记录：'+cell(result['error']), '']
    selected = evidence['selection']
    if any(selected.values()):
        lines += ['逐层视图筛选：'+ '；'.join(k+'='+', '.join(v) for k, v in selected.items() if v)+'。', '']
    model_rows = model.get('shape', [])
    if model.get('paired'):
        off = sum(s['paths']['off']['measured_seconds'] for s in model_rows)
        on = sum(s['paths']['on']['measured_seconds'] for s in model_rows)
        if off > 0:
            ratio = on/off
            lines += [f"**本轮整模型 on/off 耗时为 {ratio:.2f} 倍（{'增加' if ratio >= 1 else '减少'} {abs(ratio-1)*100:.2f}%）。**", '']
        worst = max(model_rows, key=lambda r: abs(math.log(r['on_over_off_latency'])), default=None)
        if worst:
            lines += [f"优先查看输入 {worst['label']}：on/off 批次耗时 {worst['on_over_off_latency']:.2f} 倍。"
                      '下一步可沿该输入的重点层查看主机窗口、设备事件和路由，再增加独立重复验证变化方向。', '']
    if model_rows:
        lines += ['| 整模型输入shape | 路径 | 批次数 | 平均批次 ms | 样本/s | on/off耗时 |', '|---|---|---:|---:|---:|---:|']
        for g in model_rows:
            for side in ('off', 'on'):
                if side not in g['paths']: continue
                v = g['paths'][side]
                lines.append(f"| {g['label']} | {side} | {v['batches']} | {number(v['latency_mean_ms'])} | {number(v['samples_per_second'])} | {number(g.get('on_over_off_latency')) if side == 'on' else '—'} |")
        lines += ['', '整模型表保留完整运行工作量；层筛选作用于下面的逐层视图。', '']
    n = [p['device'].get('calls', 0) for g in groups for p in g['paths'].values()]
    repeat_counts = [p['repeats'] for g in groups for p in g['paths'].values()]
    if n:
        lines += [f"**采样信息：每个层/rank/shape 有 {min(n)}～{max(n)} 次有效调用，独立 worker 重复 {min(repeat_counts)}～{max(repeat_counts)} 次。** "
                  '正文用均值与观察范围描述结果；分位数和逐重复均值见层详情。', '']
        if min(n) < 100:
            lines += ['部分分组样本较少，p99 标为描述值。增加独立重复和每组调用数可进一步观察尾部波动。', '']
    over = [r['instrumented_over_baseline'] for r in evidence['overhead_by_shape'] if r.get('instrumented_over_baseline') is not None]
    if over:
        lines += [f"**各输入插桩/无插桩整模型时间范围：{min(over):.3f}～{max(over):.3f} 倍。** "
                  '该观察包含采样影响与环境波动，可与层间变化幅度一起阅读。', '']
    lines += ['## 重点层对照', '',
              '按层窗口绝对变化排序；单路径按窗口均值排序。每行包含子模块，各 rank 独立展示。'
              '计算/通信为 profiler 每次调用的已关联事件累计均值，显存为同步重放的进程窗口增量。'
              '每组采样数、覆盖状态与原始证据可点击层名查看。', '']
    def priority(g):
        means = {s: p['device'].get('mean_ms', 0) for s, p in g['paths'].items()}
        return abs(means['on']-means['off']) if g['ratio'] is not None and {'off','on'} <= means.keys() else max(means.values(), default=0)
    focused = sorted(groups, key=priority, reverse=True)[:10]
    lines += unified_rows(focused, 'layer-index.md')
    lines += ['', f'显示 {len(focused)}/{len(groups)} 组。查看[全部层及证据](layer-index.md)、[完整指标明细](layer-details.md)、[结构化视图](layer-view.json)。', '',
              '## 如何使用这些指标', '',
              '| 要判断的问题 | 当前可用信息 | 下一步 |', '|---|---|---|',
              '| 哪些层最值得检查 | 同输入、rank 的层窗口变化及主机均值 | 查看同层设备事件和调用路由 |',
              '| 哪段执行伴随显存增长 | 进程 allocator 的窗口峰值增量、净变化和层内权重 storage | 沿张量生命周期检查激活、临时张量及缓存 |',
              '| FlagGems 参与了哪些层 | 实际函数调用数及策略原生调用占比，覆盖审计全部输入 | 结合原始 trace 检查设备任务来源 |',
              '| 通信是否值得深入分析 | 已关联 collective 和独立 profiler 通信事件 | 对照各 rank 与链路证据 |', '',
              '## 采集状态与后续操作', '',
              f"采集状态：**{result.get('status', 'failed')}**；层计时：**{result.get('layer', {}).get('timing', {}).get('status', 'partial')}**；组件证据：**{participation(result)['status']}**。", '']
    if result.get('report_view', {}).get('reanalysis'):
        info = result['report_view']['reanalysis']
        lines += [f"本次副本分析：**{info['status']}**；来源运行：**{result['report_view']['source_status']}**。过程见 [重新分析记录](reanalysis.json)。", '']
    if result.get('status') != 'completed' and result.get('report_view', {}).get('reanalysis', {}).get('status') != 'completed':
        lines += ['已有计时、显存与路由可分别查看；profiler 后续解析可写入新的分析目录。', '', '```bash',
                  shlex.join(['python', 'run.py', 'report', '--source', str(root), '--level', 'layer', '--reanalyze', '--output', str(root)+'-reanalyzed']), '```', '',
                  '该入口在副本中解析设备 trace；缺失 trace 时使用原镜像和驱动管理接口导出原始采集。新运行状态和逐阶段结果分别保留。', '']
        lines += ['计时或显存采样待完成时，可使用本次 effective.yaml 在新目录执行完整采集：', '', '```bash',
                  shlex.join(['python', 'run.py', 'performance', '--config', str(root/'effective.yaml'),
                              '--level', 'layer', '--output', str(root)+'-rerun']), '```', '',
                  '正式采集入口会核验当前执行身份与策略，身份更新时可先运行对应配置的 preview。', '']
    lines += ['[组件配置、覆盖率及完整明细](layer-details.md) · [封存结果](result.json) · [视图来源](report-source.json)', '']
    if result.get('component_profiles'):
        lines += ['| 路径 | FlagGems | FlagTree | FlagCX |', '|---|---|---|---|']
        for side, p in result['component_profiles'].items():
            lines.append(f"| {side} | {p['flaggems']} | {p['flagtree']} | {p['flagcx']} |")
        lines += ['']
    (root/'report.md').write_text('\n'.join(lines))
    index = ['# 逐层诊断索引', '', '[返回摘要](report.md) · [完整指标明细](layer-details.md)', '',
             '本页使用相同模块/rank/shape 关联独立采集阶段，路由计数覆盖该模块审计的全部输入。', '']
    for g in groups:
        index += [f'<a id="{g["anchor"]}"></a>', '', f"## {g['module']} / rank {g['rank']} / {g['shape']}", '']
        index += unified_rows([g])
        for side, p in g['paths'].items():
            d = p['device']; calls = d.get('calls', 0)
            tag = '描述值' if calls < 100 else '观测插值'
            index += ['', f"{side}：主机均值 {number((p['host'] or {}).get('mean_ms'))} ms；p50/p90/p99 {tag}："
                      f"{number(d.get('p50_ms'))} / {number(d.get('p90_ms'))} / {number(d.get('p99_ms'))} ms；"
                      f"p99 上尾期望样本数 n×1%={calls*.01:.2f}。",
                      f"独立重复均值 ms：{cell(p['repeat_means_ms'])}；均值的样本标准差 {number(p['repeat_mean_sd_ms'])} ms。",
                      f"显存调用 {p['memory_calls']} 次，净变化均值 {number(p['net_change_mib'])} MiB；profiler 调用 {p['profile_calls']} 次，状态 {p['profile_states']}。",
                      f"审计全部输入：FlagGems 函数调用 {p['route_function_calls'] if p['route_function_calls'] is not None else '待采集'} 次；策略原生调用占比 {number(p['route_native_ratio'])}。",
                      ' '.join(f'[设备关联 {i+1}]({path})' for i, path in enumerate(p['profile_evidence'])), '']
    index += ['## 按输入观察采样扰动', '',
              '同一逻辑输入的独立重放按 repeat/cycle/batch 配对；比例包含环境波动与采样影响。', '',
              '| 路径 | repeat | rank | 输入shape | 批次数 | 插桩/无插桩 | 配对 |', '|---|---|---|---|---:|---:|---|']
    for r in evidence['overhead_by_shape']:
        index.append(f"| {r['side']} | {r['repeat']} | {r['rank']} | {r.get('shape', '待记录')} | {r.get('batches', 0)} | {number(r.get('instrumented_over_baseline'))} | {r['status']} |")
    (root/'layer-index.md').write_text('\n'.join(index)+'\n')
    write_json(root/'layer-view.json', evidence)
