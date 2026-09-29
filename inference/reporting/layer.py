# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Layer-focused human view of sealed measurements, with explicit timer boundaries."""
from collections import defaultdict
from runtime.common import read_json
from analysis.layer import ranks
from reporting.components import description


def fmt(value, scale=1):
    return '未采集/未定义' if value is None else f'{value/scale:.4f}'


def render(root, result):
    from reporting.layer_view import filtered
    result = filtered(result, result.get('report_view', {}).get('selection', {}))
    layer = result.get('layer', {})
    timing = layer.get('timing', {})
    from analysis.assessment import participation
    observation = participation(result)
    if observation['status'] == 'observed':
        observation = dict(observation, reason='逐侧、逐 rank 已观测到所需组件参与')
    lines = ['# Inference 性能测试：Layer level', '', description(result), '',
             f"执行状态：**{result.get('status', 'failed')}**；层级计时：**{timing.get('status', 'incomplete')}**；off/on 配对：**{timing.get('paired', False)}**。", '',
             '在完整模型前向中观察模块；每行包含其子模块。按同一层级、同一 rank 阅读指标。', '',
             '设备窗口来自当前流 Event，覆盖该流上的执行、等待及主机供给间隙；主机窗口覆盖模块调用。profiler、显存与路由分别独立采集。', '']
    lines += [f"比较组件生效证据：**{observation['status']}**。{observation['reason']}。", '']
    if result.get('error'): lines += ['失败/不完整原因：'+result['error'], '']
    lines += [f"层级采集 worker/组失败数：{layer.get('failure_count', '未记录')}；计数单位为采集 worker/组。", '']
    for issue in timing.get('issues', []): lines.append('- '+issue)
    if result.get('component_profiles'):
        lines += ['| 路径 | FlagGems | FlagTree | FlagCX |', '|---|---|---|---|']
        for side, profile in result['component_profiles'].items():
            lines.append(f"| {side} | {profile['flaggems']} | {profile['flagtree']} | {profile['flagcx']} |")
        lines += ['']
    identity = root/'prepared/identity.json'
    if identity.is_file():
        ident = read_json(identity)
        lines += [f"模型：{ident.get('model', {}).get('name')}；dtype：{ident.get('model', {}).get('dtype')}；输入与设备身份见 [identity](prepared/identity.json)。", '']
    lines += ['## 逐层时延与层执行速率', '',
              '先按同一 shape、rank 比较 off/on；下表均值及分位数从全部原始调用重算。层执行样本/s的分母是该层设备窗口总和。有效 token 由输入 attention_mask 统计。', '',
              '| 层 | rank | 输入shape | 路径 | 调用数 | 设备均值 ms | p50 ms | p90 ms | p99 ms | 主机均值 ms | 调用/s | 层样本/s | 有效token/s | on/off设备时间 |',
              '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in timing.get('groups', []):
        for side, values in group['paths'].items():
            device, host = values.get('device') or {}, values.get('host') or {}
            ratio = group.get('on_over_off_device_time') if side == 'on' else None
            lines.append(f"| {group['module']} | {group['rank']} | {group['shape']} | {side} | {values['observed_calls']} | " +
                         ' | '.join(fmt(device.get(k)) + ('（描述值）' if k != 'mean_ms' and device.get('calls', 0) < 100 else '') for k in ('mean_ms','p50_ms','p90_ms','p99_ms')) +
                         f" | {fmt(host.get('mean_ms'))} | {fmt(device.get('calls_per_second'))} | {fmt(device.get('samples_per_second'))} | {fmt(device.get('tokens_per_second'))} | {fmt(ratio) if ratio is not None else '—'} |")
    hot = defaultdict(list)
    for group in timing.get('groups', []):
        for side, values in group['paths'].items():
            if values.get('device'): hot[(group['rank'], group['shape'], side)].append((values['device']['mean_ms'], group['module']))
    lines += ['', '### 各输入形状的热点', '', '排名依据包含子模块的设备窗口均值；父子模块分别呈现各自包含范围。', '']
    for (rank, shape, side), values in hot.items():
        lines.append(f'- rank {rank} / {shape} / {side}：'+ '；'.join(f'{name} {ms:.4f} ms' for ms, name in sorted(values, reverse=True)[:5]))
    lines += ['', '## 层执行窗口内的进程显存', '',
              '独立同步重放。窗口峰值包含进入该层前仍存活的张量，峰值增量描述该进程窗口的新增 allocator 高水位。父子模块分组采集；权重 storage 按层内去重。当前 use_cache=False；MiB=2^20 bytes。', '',
              '| 层 | rank | shape | 路径 | 本地权重 MiB | 进入allocated均值 MiB | 窗口allocated峰值最大 MiB | allocated增量最大 MiB | reserved峰值最大 MiB | allocated净变化均值 MiB |',
              '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    groups = defaultdict(list); weights = {}
    for side, repeats in layer.get('runs', {}).get('layer_memory', {}).items():
        for state in repeats.values():
            for rank, value in ranks(state).items():
                for m in value.get('modules', []): weights[(side, rank, m['module'])] = m['weight_storage_bytes']
                for row in value.get('rows', []):
                    if row.get('peak'): groups[(row['module'], str(rank), str(row['input_shape']), side)].append(row)
    for (module, rank, shape, side), rows in sorted(groups.items()):
        numbers = [weights.get((side, rank, module)),
            sum(r['before']['allocated_bytes'] for r in rows)/len(rows),
            max(r['peak']['allocated_bytes'] for r in rows), max(r['peak_increment']['allocated_bytes'] for r in rows),
            max(r['peak']['reserved_bytes'] for r in rows), sum(r['net_change']['allocated_bytes'] for r in rows)/len(rows)]
        lines.append(f'| {module} | {rank} | {shape} | {side} | '+' | '.join(fmt(n, 2**20) for n in numbers)+' |')
    lines += ['', '完整进入/退出 allocated/reserved、峰值、增量和净变化见 [逐层证据](layer/summary.json)。负净变化表示退出窗口时分配量减少。', '',
              '## 独立 profiler：kernel、通信与拷贝', '',
              '每个 profiler 窗口保留事件时长之和与区间并集，刻画独立重放中的设备工作。通信内部 Memcpy 保留在通信 trace；通用拷贝单独统计。未知关联在原始证据中保留。', '',
              '主归因覆盖分母为计算kernel、collective整体窗口和可识别的普通Memcpy。通知/等待队列、底层SDMA及通信实现AICPU另列为辅助任务，原始记录保留；主事件和辅助任务分别展示覆盖情况。覆盖表使用本次完整采集范围。', '',
              '| 路径 | repeat | rank | 主已关联/事件 | 主覆盖率 | 主状态 | 辅助任务/未知 | 证据 |', '|---|---|---|---:|---:|---|---:|---|']
    event_groups = defaultdict(lambda: defaultdict(list)); comm_groups = defaultdict(list)
    for side, repeats in layer.get('profiles', {}).items():
        for label, values in repeats.items():
            for rank, state in values.items():
                c = state.get('coverage', {})
                aux = state.get('auxiliary_coverage', {})
                lines.append(f"| {side} | {label} | {rank} | {c.get('associated_events','?')}/{c.get('device_events','?')} | {fmt(c.get('fraction'))} | {state['status']} | {aux.get('tasks','?')}/{aux.get('unknown_tasks','?')} | [原始关联]({state['evidence']}) |")
                for row in state.get('rows', []):
                    for kind, value in row['events'].items():
                        event_groups[(row['module'], rank, str(row['input_shape']), kind)][side].append(value)
                for row in state.get('communication', {}).get('events', []):
                    owners = row.get('layer_calls', [])
                    for owner in owners:
                        comm_groups[(owner['module'], rank, str(owner['input_shape']), side)].append(row)
                    if not owners:
                        comm_groups[(row.get('layer_attribution_status','未归因'), rank, '未选择/未归因', side)].append(row)
    for side, repeats in layer.get('profiles', {}).items():
        for label, values in repeats.items():
            for rank, state in values.items():
                recovery=state.get('recovery')
                if recovery:
                    lines += ['', f"{side}/{label}/rank {rank}：首次trace缺失，独立解析恢复状态 **{recovery['status']}**；处理对象为原始数据副本。详情与导出日志入口见该行原始关联文件。"]
                if state.get('reason'):
                    lines += ['', f"{side}/{label}/rank {rank}：{state['reason']}"]
    lines += ['', '| 层 | rank | shape | 类型 | 路径 | 事件数 | 累计时长和 ms | 逐窗口并集累计 ms |', '|---|---|---|---|---|---:|---:|---:|']
    for (module, rank, shape, kind), sides in event_groups.items():
        for side in ('off', 'on'):
            if side not in sides: continue
            values = sides[side]
            def total_ns(field):
                numbers = [v[field] for v in values]
                return sum(numbers) if all(n is not None for n in numbers) else None
            lines.append(f'| {module} | {rank} | {shape} | {kind} | {side} | {sum(v["count"] for v in values)} | {fmt(total_ns("duration_sum_ns"),1e6)} | {fmt(total_ns("duration_union_ns"),1e6)} |')
    lines += ['', '表内零表示已关联事件计数为零，关联完整程度见覆盖表。拷贝方向和字节来自 trace 字段，缺失字段标为待采集。', '',
              '| 选定层 | rank | shape | 路径 | collective数 | 逻辑payload bytes | CANN elapsed累计 ms | wait累计 ms |', '|---|---|---|---|---:|---:|---:|---:|']
    for (module, rank, shape, side), rows in sorted(comm_groups.items()):
        def total(field, nested=False):
            values = [r.get('time', {}).get(field) if nested else r.get(field) for r in rows]
            return sum(values) if all(v is not None for v in values) else None
        lines.append(f'| {module} | {rank} | {shape} | {side} | {len(rows)} | {fmt(total("logical_bytes"))} | {fmt(total("Elapse Time(ms)",True))} | {fmt(total("Wait Time(ms)",True))} |')
    if result.get('parallelism') != 'tp': lines += ['', '单卡模型集合通信：不适用。']
    lines += ['', '通信链路矩阵及底层传输量见关联文件内 communication.links；它们与框架逻辑 payload 不同，链路数据按已有 rank 粒度展示。模型加载、输入H2D与输出D2H在模型阶段证据中展示。', '',
              '## 逐层 FlagGems 路由', '',
              '以下来自独立调用审计，覆盖审计全部输入；父子作用域使用包含子模块的计数。实际函数调用说明 FlagGems 参与位置；策略原生占比描述调用路由。硬件 kernel 来源作为后续取证项。', '',
              '| 层 | rank | 路径 | 顶层ATen数 | 实际FlagGems函数调用数 | 策略原生调用占比 | 硬件kernel fallback |', '|---|---|---|---:|---:|---:|---|']
    route_rows = []
    for side, values in layer.get('routes', {}).items():
        for rank, modules in values.items():
            for name, row in modules.items(): route_rows.append((name, rank, side, row))
    for name, rank, side, row in sorted(route_rows):
        lines.append(f"| {name} | {rank} | {side} | {sum(v['calls'] for v in row['inventory'].values())} | {sum(row['actual_function_calls'].values())} | {fmt(row.get('classification',{}).get('policy_native_call_ratio'))} | 未采集 |")
    lines += ['', '## 采样扰动与 TP rank 差异', '',
              '下表比较相同逻辑输入的两次独立运行；比值共同反映环境波动与采样影响。', '',
              '| 路径 | repeat | rank | 插桩/无插桩整模型时间 | 配对状态 |', '|---|---|---|---:|---|']
    for row in layer.get('overhead', []):
        lines.append(f"| {row['side']} | {row['repeat']} | {row['rank']} | {fmt(row.get('instrumented_over_baseline'))} | {row['status']} |")
    imbalance = timing.get('rank_imbalance', [])
    if imbalance:
        worst = max(imbalance, key=lambda r: r['spread_ns'])
        lines += ['', f"本次最大 rank 局部窗口差：{worst['module']}，{fmt(worst['spread_ns'],1e6)} ms（{worst['side']}，repeat {worst['repeat']}，cycle {worst['cycle']}，batch {worst['batch_index']}）。各 rank 使用本地 Event 时钟，最大窗口表示该次逻辑调用中最慢的本地窗口。"]
    lines += ['', '## 核验与离线视图', '',
              '[完整结果](result.json) · [层级原始记录索引及汇总](layer/summary.json) · [最终配置](effective.yaml) · [源码快照](source-snapshot.json)', '',
              'layer/layer_timing、layer/layer_memory、layer/layer_profile 下按路径、repeat、rank 保存 layers.jsonl、batches.jsonl和模块清单；profiles中的 evidence 指向原始事件关联。整模型基准保存在原 off/on repeat 目录。', '',
              '相同证据生成模型视图：`python run.py report --source <本次目录> --level total --output <新目录>`。原证据只读。', '']
    (root/'report.md').write_text('\n'.join(lines))
