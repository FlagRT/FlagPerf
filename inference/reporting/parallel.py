# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Human-readable TP report projected only from sealed evidence."""
from collections import defaultdict
from reporting.components import description
from pathlib import Path
import re
from statistics import mean, stdev

import yaml

from runtime.common import ROOT, file_hash, read_json, write_json


MIB = 2 ** 20
CATEGORIES = {
    'attention_output': 'Attention 输出聚合',
    'mlp_output': 'MLP 输出聚合',
    'unattributed': '未归因',
}
ROUTES = {
    'excluded': ('策略排除', '当前策略已确认排除'),
    'unverified': ('未复验', '候选函数未在本策略复验'),
    'uncovered': ('无覆盖', '没有对应候选函数'),
    'allowed_function_observed': ('候选函数已观测命中', '函数命中过，不能逐次证明该类调用都命中'),
    'ambiguous': ('归属不明', '不能计作已证实回退'),
}


def number(value, digits=3):
    return '未知/未采集' if value is None else f'{value:.{digits}f}'


def stat(values, digits=3, scale=1):
    seen = [value * scale for value in values if value is not None]
    if not seen:
        return '未采集'
    result = f'{mean(seen):.{digits}f}'
    result += f' ± {stdev(seen):.{digits}f}' if len(seen) > 1 else '（n=1，SD 不适用）'
    if len(seen) < len(values):
        result += f'（已采集 {len(seen)}/{len(values)}）'
    return result


def span(values):
    seen = [value for value in values if value is not None]
    return f'{min(seen):.3f}–{max(seen):.3f}' if seen else '未采集'


def field(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def mib(value):
    return None if value is None else value / MIB


def evidence(root, relative, label):
    return f'[{label}]({relative})' if (root / relative).exists() else f'{label}（证据缺失：{relative}）'


def coverage(rows):
    seen = [row['attributed_collectives'] for row in rows if row.get('attributed_collectives') is not None]
    expected = [row['expected_collectives'] for row in rows if row.get('expected_collectives') is not None]
    return f'{sum(seen) if seen else "未采集"}/{sum(expected) if expected else "未采集"}'


def sides(result):
    found = set(result.get('summary', {})) | set(result.get('paths', {})) | set(result.get('profiles', {}))
    return [side for side in ('off', 'on') if side in found] + sorted(found - {'off', 'on'})


def paired_order(keys, *group_indices):
    """Keep off/on adjacent for each identical set of comparison dimensions."""
    side_order = {'off': 0, 'on': 1}

    def dimension(value):
        return (0, int(value)) if isinstance(value, int) or str(value).isdecimal() else (1, str(value))

    return sorted(keys, key=lambda key: tuple(dimension(key[index]) for index in group_indices)
                  + ((side_order.get(key[0], 2), str(key[0])),))


def rank_rows(runs):
    for repeat, run in sorted(runs.items()):
        if run.get('status') != 'completed':
            continue
        for rank, state in sorted(run.get('ranks', {}).items(), key=lambda item: int(item[0])):
            if state.get('status') == 'completed':
                yield repeat, rank, state


def profile_rows(runs, world_size=None):
    for repeat, run in sorted(runs.items()):
        comm = run.get('communication') or {}
        if not profile_complete(run, world_size):
            continue
        for rank, state in sorted(comm.get('ranks', {}).items(), key=lambda item: int(item[0])):
            yield repeat, rank, state


def profile_complete(run, world_size=None):
    comm = run.get('communication') or {}
    ranks = comm.get('ranks') or {}
    if run.get('status') != 'completed' or comm.get('status') != 'completed' or not ranks:
        return False
    if world_size is not None and len(ranks) != world_size:
        return False
    return all(row.get('status') == 'completed' and
               row.get('expected_collectives') is not None and
               row.get('attributed_collectives') == row['expected_collectives']
               for row in ranks.values())


def compute_path_hint(result):
    if result.get('status') != 'completed':
        return False
    profiles = result.get('profiles') or {}
    if not {'off', 'on'} <= set(profiles):
        return False
    world_size = len(result.get('devices') or []) or None
    values = {}
    for side in ('off', 'on'):
        rows = list(profile_rows(profiles[side], world_size))
        if not rows:
            return False
        compute = [field(row, 'intervals', 'compute_union_ms') for _, _, row in rows]
        comm = [field(row, 'intervals', 'communication_union_ms') for _, _, row in rows]
        if any(value is None for value in compute + comm):
            return False
        values[side] = (compute, comm)
    return (min(values['on'][0]) > max(values['off'][0]) and
            max(values['on'][1]) <= max(values['off'][1]))


def batch_shapes(samples, inputs):
    size, limit = inputs.get('batch_size'), inputs.get('max_length')
    if not samples or not isinstance(size, int) or size < 1 or not isinstance(limit, int):
        return {}
    shapes = {}
    for offset in range(0, len(samples), size):
        lengths = [row.get('original_tokens') for row in samples[offset:offset + size]]
        if all(isinstance(length, int) for length in lengths):
            shapes[offset // size] = f'[{len(lengths)}, {max(min(length, limit) for length in lengths)}]'
    return shapes


def snapshot_processes(path):
    """Map npu-smi's NPU/chip process rows to physical device IDs when readable."""
    if not path.is_file():
        return None
    rows = path.read_text().splitlines()
    mapping = {}
    for index, row in enumerate(rows[:-1]):
        card = re.match(r'^\|\s*(\d+)\s+Ascend\w*\s*\|', row)
        chip = re.match(r'^\|\s*(\d+)\s+(\d+)\s+\|', rows[index + 1])
        if card and chip:
            mapping[(int(card[1]), int(chip[1]))] = int(chip[2])
    if not mapping or not any('Process id' in row for row in rows):
        return None
    processes = set()
    for row in rows:
        match = re.match(r'^\|\s*(\d+)\s+(\d+)\s+\|\s*(\d+)\s+\|\s*[^|]+\|\s*\d+', row)
        if match and (int(match[1]), int(match[2])) in mapping:
            processes.add(mapping[(int(match[1]), int(match[2]))])
    return processes


def conditions(lines, result, identity, config, samples, shapes, root):
    model, inputs = identity.get('model', {}), identity.get('inputs', {})
    perf = config.get('performance', {})
    devices = identity.get('devices') or []
    mapping = ', '.join(f'rank {row["rank"]} → 物理设备 {row["physical_device"]}'
                        for row in devices if 'rank' in row and 'physical_device' in row)
    if not mapping:
        mapping = str(result.get('devices') or '未记录')
    lines += ['## 模型、输入与测量条件', '',
              '| 项目 | 本次条件 |', '|---|---|',
              f'| 模型 | {model.get("name", "未记录")}；{model.get("dtype", "未记录")}；attention={model.get("attention", "未记录")} |',
              f'| TP 设备 | {mapping}；{identity.get("device_name", "型号未记录")} |',
              f'| 输入 | {len(samples) if samples else "未记录"} 条固定文本；batch={inputs.get("batch_size", "未记录")}；max_length={inputs.get("max_length", "未记录")}；{inputs.get("padding_side", "未记录")} padding |',
              f'| 批次负载 | {"、".join(shapes.values()) if shapes else "形状未记录"}；每组预热 {perf.get("warmup_rounds", "未记录")} 轮、测量 {perf.get("measure_rounds", "未记录")} 轮，共 {perf.get("repeats", "未记录")} 组 |',
              f'| 镜像 | {identity.get("image_id", "未记录")} |', '']
    if samples and all(isinstance(row.get('original_tokens'), int) for row in samples):
        lengths = [row['original_tokens'] for row in samples]
        lines += [f'原始输入长度 {min(lengths)}–{max(lengths)} token；截断 {sum(bool(row.get("truncated")) for row in samples)}/{len(samples)} 条。多轮样本处理次数不代表不同文本数。', '']
    lines += ['一次批次的执行路径：所有 rank 读取同一批输入 → 各自用 TP 权重分片计算、通过所选模型通信后端执行 AllReduce → 以最早开始至最晚结束形成一个全局批次窗口。样本和有效输入 token 只计一次。',
              description(result) if 'component_profiles' in result else 'FlagGems off 使用设备原生算子；on 按已验证策略选择性注册 FlagGems 函数。两侧使用独立进程组，正式计时不启用 profiler。', '']
    before = snapshot_processes(root / 'npu-before.txt')
    after = snapshot_processes(root / 'npu-after.txt')
    selected = set(result.get('devices') or [])
    if before is not None and after is not None and selected:
        occupied = before | after
        observed = '、'.join(str(device) for device in sorted(occupied)) if occupied else '无'
        selected_status = ('所选设备在这两个快照中无其他进程' if not occupied & selected
                           else '所选设备在快照中出现进程，需检查占用身份')
        lines += [f'运行前后设备快照记录的进程所在物理设备：{observed}；{selected_status}。快照不覆盖整个运行过程，不能证明整机独占。', '']
    else:
        lines += ['运行时资源隔离情况未完整记录；结果不能直接作为独占整机性能基线。', '']


def performance(lines, result, ordered, shapes):
    summary = result.get('summary') or {}
    lines += ['## 正式总级性能', '']
    if not summary:
        lines += ['没有通过完整测量门禁的总级汇总；已完成的组与失败阶段见附录。', '']
        return
    lines += ['| 路径 | 批次 | 样本处理次数 | 有效输入 token | 平均批次 ms | p50 / p90 / p99 ms | 样本/s | token/s | 测量累计 s |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for side in ordered:
        if side not in summary:
            continue
        row = summary[side]
        quantiles = ' / '.join(number(row.get(key)) for key in ('latency_p50_ms', 'latency_p90_ms', 'latency_p99_ms'))
        lines.append(f'| {side} | {row["batches"]} | {row["samples"]} | {row["tokens"]} | {number(row.get("latency_mean_ms"))} | {quantiles} | {number(row.get("samples_per_second"))} | {number(row.get("tokens_per_second"))} | {number(row.get("measured_seconds"))} |')
    lines += ['', '上表合并所有成功 repeat 的全局批次；p50/p90/p99 对全部批次时延排序并线性插值，不是各组分位数的平均。', '']
    comparison = result.get('comparison') or {}
    ratio = comparison.get('off_over_on_measured_time')
    if ratio is not None and ratio > 0:
        lines += [f'on/off 样本吞吐比为 **{number(comparison.get("on_over_off_samples_per_second"), 4)}**，与开头的主窗口用时比来自同一组配对结果；独立通信采样不进入此比较。', '']
    lines += ['### 跨 repeat 稳定性', '',
              '| 路径 | 成功/已记录组数 | 组内平均批次 ms 的跨组均值 ± SD | 组内样本/s 的跨组均值 ± SD | 组内 token/s 的跨组均值 ± SD |',
              '|---|---:|---:|---:|---:|']
    for side in ordered:
        if side not in summary:
            continue
        runs = result.get('paths', {}).get(side, {})
        groups = [run['summary'] for _, run in sorted(runs.items())
                  if run.get('status') == 'completed' and run.get('summary')]
        lines.append(f'| {side} | {len(groups)}/{len(runs)} | {stat([group.get("latency_mean_ms") for group in groups])} | {stat([group.get("samples_per_second") for group in groups])} | {stat([group.get("tokens_per_second") for group in groups])} |')
    lines += ['', 'SD 是成功 repeat 汇总值之间的样本标准差，只有一组时不适用。', '']
    grouped = defaultdict(lambda: defaultdict(list))
    token_counts = defaultdict(set)
    for side in ordered:
        if side not in summary:
            continue
        for run in result.get('paths', {}).get(side, {}).values():
            if run.get('status') != 'completed':
                continue
            for batch in run.get('batches', []):
                index, latency = batch.get('batch_index'), batch.get('latency_ns')
                if index is None or latency is None:
                    continue
                grouped[index][side].append(latency / 1e6)
                if batch.get('tokens') is not None:
                    token_counts[index].add(batch['tokens'])
    if grouped:
        lines += ['### 输入形状与时延', '',
                  '| 批次形状 | 每批有效输入 token | 路径 | 完成批次 | 平均批次 ms |',
                  '|---|---:|---|---:|---:|']
        for index in sorted(grouped):
            tokens = str(next(iter(token_counts[index]))) if len(token_counts[index]) == 1 else '未记录/不一致'
            for side in ordered:
                values = grouped[index].get(side, [])
                if values:
                    lines.append(f'| {shapes.get(index, "批次 " + str(index))} | {tokens} | {side} | {len(values)} | {mean(values):.3f} |')
        lines += ['', '按封存批次索引分组，形状由同次封存的样本长度与配置还原。混合形状的总级分位数不能解释为某一形状或在线单请求时延。', '']
    lines += ['### 主指标说明', '',
              '| 指标 | 计算与适用边界 |', '|---|---|',
              '| 全局批次时延 | 各 rank 用同机 perf_counter_ns()；取 max(end) − min(start)，包含前向、last-token pooling、FP32 归一化和末尾 NPU 同步，也含主机调用开销，不是纯 kernel 时间。 |',
              '| 测量累计 s | 全部全局批次窗口之和；不含模型加载、数据拷贝、预热、Gloo 协调、独立 profiler、落盘和批间等待。不是端到端服务耗时。 |',
              '| 样本/s 与 token/s | 全局样本处理次数或 attention_mask 非零数除以测量累计秒数；同一批输入不按 rank 数翻倍。token 为有效输入 token，不是生成 token。 |',
              '| p50/p90/p99 | 全部批次时延排序后在 (N−1)×p/100 位置线性插值；不能当作单请求时延。 |', '']


def resources(lines, result, ordered):
    by_rank = defaultdict(list)
    for side in ordered:
        for _, rank, state in rank_rows(result.get('paths', {}).get(side, {})):
            by_rank[(side, rank)].append(state)
    if not by_rank:
        return
    lines += ['## 阶段与显存：按 rank 汇总', '',
              '同一路径、同一 rank 的成功 repeat 先聚合为均值 ± 样本 SD；表中按 rank 将 off/on 相邻排列。不同 rank 的加载、预热与峰值不能相加成全局耗时或同一时刻的整机峰值。', '',
              '| 路径/rank | 成功组数 | 模型加载 s | 预热 s | 权重 storage MiB |',
              '|---|---:|---:|---:|---:|']
    for side, rank in paired_order(by_rank, 1):
        states = by_rank[(side, rank)]
        lines.append(f'| {side}/{rank} | {len(states)} | {stat([s.get("model_load_seconds") for s in states])} | {stat([s.get("warmup_seconds") for s in states])} | {stat([mib(field(s, "memory", "weight_storage_bytes")) for s in states])} |')
    lines += ['', '| 路径/rank | 加载后 allocated 增量 MiB | 预热后基线 allocated / reserved MiB | 测量峰值 allocated / reserved MiB | 峰值减基线 allocated / reserved MiB |',
              '|---|---:|---:|---:|---:|']
    for side, rank in paired_order(by_rank, 1):
        states = by_rank[(side, rank)]
        loaded = []
        for state in states:
            before, after = field(state, 'memory', 'before_model_load', 'allocated_bytes'), field(state, 'memory', 'after_model_load', 'allocated_bytes')
            loaded.append(mib(after - before) if before is not None and after is not None else None)
        def pair(stage):
            return ' / '.join(stat([mib(field(s, 'memory', stage, kind + '_bytes')) for s in states], 1)
                              for kind in ('allocated', 'reserved'))
        lines.append(f'| {side}/{rank} | {stat(loaded, 1)} | {pair("measurement_baseline")} | {pair("measurement_peak")} | {pair("peak_increment_bytes")} |')
    lines += ['', '模型加载含权重读取、TP 切分、设备放置和同步；预热不是纯编译时间。同一 rank 的去重权重 storage 已包含在该 rank 的 allocated 中，两列不能相加；跨 rank storage 还可能包含复制的权重。',
              'allocated/reserved 来自 torch.npu 分配器；峰值减基线包括临时张量和工作区，不能称为纯 activation。use_cache=False，KV cache 不适用；分配器外设备显存未采集。', '']
    lines += ['## 数据传输：按 rank 汇总', '',
              '按传输方向与对象、rank 配对展示 off/on；缺少的一侧不补零。', '',
              '| 路径/rank | 方向与对象 | 每组逻辑数据量 | 同步窗口耗时 ms，均值 ± SD |',
              '|---|---|---:|---:|']
    for key, label in [('input', 'CPU→设备：全部预制 input_ids、attention_mask'),
                       ('output_embedding', '设备→CPU：首批 embedding')]:
        for side, rank in paired_order(by_rank, 1):
            states = by_rank[(side, rank)]
            items = [field(state, 'transfer', key) for state in states]
            sizes = [item.get('logical_bytes') if item else None for item in items]
            known = [value for value in sizes if value is not None]
            size_text = (f'{known[0] / 1024:.2f} KiB' if len(known) == len(sizes) and len(set(known)) == 1
                         else stat(sizes, 2, 1 / 1024) + ' KiB' if known else '未采集')
            lines.append(f'| {side}/{rank} | {label} | {size_text} | {stat([item.get("seconds") if item else None for item in items], 3, 1000)} |')
    lines += ['', '输入在预热前由每 rank 各拷贝一次；输出只测额外首批前向结束后的一次 embedding 拷回，均不进入正式计时。',
              'TP 加载器融合权重读取、切分和设备放置，权重 CPU→设备的独立数据量及耗时未采集；模型加载总时间不能冒充纯 H2D。表内逻辑字节不是物理总线流量，同步窗口也不是纯链路时间。', '']


def communication(lines, result, ordered, config):
    profiles = result.get('profiles') or {}
    lines += ['## 独立通信采样', '']
    if not profiles:
        lines += ['未封存 TP 通信采样，通信量和时间为未采集。', '']
        return
    lines += ['通信数据来自同配置的独立 profiler 进程组。以下时间不能除以无 profiler 的正式时延作为通信占比，也不能从正式时延中扣除。', '']
    repeats = field(config, 'performance', 'repeats')
    world_size = len(result.get('devices') or []) or None
    complete = {}
    for side in ordered:
        runs = profiles.get(side, {})
        if not runs:
            continue
        complete[side] = list(profile_rows(runs, world_size))
        finished = sum(profile_complete(run, world_size) for run in runs.values())
        expected_groups = repeats if isinstance(repeats, int) else len(runs)
        all_ranks = [row for run in runs.values() for row in (run.get('communication') or {}).get('ranks', {}).values()]
        full = len(runs) == expected_groups and finished == expected_groups
        qualifier = '完整' if full else '不完整，仅列已记录值'
        lines.append(f'- {side}：完整采样组 {finished}/{expected_groups}；已记录归因 {coverage(all_ranks)} 次；{qualifier}。')
    lines += ['', '归因使用模块范围标记 → Enqueue/Dequeue correlation ID → CANN launch connection ID → HCCL 事件；同步路径可使用同线程标记内唯一 launch 的明确 connection ID。缺失或多义关联不能按顺序猜测。初始化、预热与 Gloo 测试协调不计入模型 HCCL active 窗口。', '']
    if not any(complete.values()):
        lines += ['没有可汇总的完整 rank 采样；问题和原始文件见附录。', '']
        return
    categories = defaultdict(list)
    links = defaultdict(list)
    intervals = defaultdict(list)
    skews = defaultdict(list)
    for side, runs in profiles.items():
        for run in runs.values():
            if profile_complete(run, world_size):
                for row in field(run, 'communication', 'rank_entry_skew') or []:
                    if row.get('host_entry_skew_ns') is not None:
                        skews[side].append(row['host_entry_skew_ns'] / 1e6)
    for side, rows in complete.items():
        for _, rank, state in rows:
            for category, value in state.get('categories', {}).items():
                categories[(side, rank, category)].append(value)
            for key in ('communication_union_ms', 'compute_union_ms', 'communication_compute_overlap_ms'):
                intervals[(side, key)].append(field(state, 'intervals', key))
            per_link = defaultdict(lambda: {'size': None, 'time': None})
            for edge in state.get('links', []):
                key = (side, rank, edge.get('src_rank'), edge.get('dst_rank'), edge.get('transport'))
                if edge.get('transit_size_mb') is not None:
                    per_link[key]['size'] = (per_link[key]['size'] or 0) + edge['transit_size_mb']
                if edge.get('transit_time_ms') is not None:
                    per_link[key]['time'] = (per_link[key]['time'] or 0) + edge['transit_time_ms']
            for key, value in per_link.items():
                links[key].append(value)
    lines += ['### HCCL 类别', '',
              '本表先在**一个独立 profiler 采样组**内，将同一 rank、同一类别的 HCCL 算子字段分别求和；再对完整采样组计算均值 ± 样本标准差（SD）。因此“每组 ms”是**算子耗时累计**，不是采样窗口墙钟时长，也不是无 profiler 的正式批次时延。', '',
              '| 指标 | CANN `communication.json` 原字段 | 读法 |',
              '|---|---|---|',
              '| elapsed | `Communication Time Info / Elapse Time(ms)` | 单个通信算子所有事件的总耗时；表中累计同类算子。官方口径将其分为 Wait、Transit 和 Idle，其中 Idle 是算子下发耗时。 |',
              '| Wait | `Wait Time(ms)` | 通信前等待对端完成同步的耗时；属于 elapsed 内的等待口径，不代表数据在链路上传输的时间。 |',
              '| Synchronization | `Synchronization Time(ms)` | 首次传输数据前的卡间同步等待；它与 Wait 的时间可能重叠，是进一步观察等待来源的字段，不是可额外加到 elapsed 上的一段时间。 |', '',
              '官方定义 `Idle = Elapse − Transit − Wait`，因此 `elapsed − Wait` 还包含 Transit 与 Idle，不能直接当作纯传输时间；**不要再把 Synchronization 加入这个分解**。Wait 与 Synchronization 即便同值，也不能解释成两段独立等待。字段定义见 [MindStudio Insight 26.1 通信算子说明](https://www.hiascend.com/document/detail/en/mindstudio/2610/visualization_tool/MindStudioInsight/docs/en/user_guide/system_tuning.md)。', '',
              '下表按类别、rank 排列，同一条件的 off/on 相邻；缺少的一侧不补零。', '',
              '| 路径/rank | 类别 | 完整采样组 | 每组次数，均值 ± SD | 每组逻辑 payload MiB，均值 ± SD | 每组 elapsed ms，均值 ± SD | 每组 Wait ms，均值 ± SD | 每组 Synchronization ms，均值 ± SD |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for side, rank, category in paired_order(categories, 2, 1):
        items = categories[(side, rank, category)]
        times = ' | '.join(stat([item.get(key) for item in items])
                           for key in ('device_elapsed_ms_sum', 'wait_ms_sum', 'synchronization_ms_sum'))
        lines.append(f'| {side}/{rank} | {CATEGORIES.get(category, category)} | {len(items)} | {stat([item.get("collectives") for item in items], 1)} | {stat([item.get("logical_payload_bytes") for item in items], 3, 1 / MIB)} | {times} |')
    example = next(((key, values) for key, values in sorted(categories.items())
                    if all(value.get('device_elapsed_ms_sum') is not None and
                           value.get('wait_ms_sum') is not None and
                           value.get('synchronization_ms_sum') is not None for value in values)
                    and values and
                    all(abs(value['wait_ms_sum'] - value['synchronization_ms_sum']) < 1e-9
                        for value in values)), None)
    if example:
        (side, rank, category), values = example
        elapsed = mean(value['device_elapsed_ms_sum'] for value in values)
        wait = mean(value['wait_ms_sum'] for value in values)
        lines += ['', f'例如 {side}/{rank} 的 {CATEGORIES.get(category, category)}，每组 elapsed 为 {elapsed:.3f} ms、Wait 与 Synchronization 均为 {wait:.3f} ms（均值）；这不表示有两段各 {wait:.3f} ms 的等待。']
    lines += ['', '逻辑 payload 是每 rank 真实 collective 输入 tensor 的 numel × element_size，不是物理线路流量；各 rank 的 elapsed 也不能相加成全局时延。下方 CANN 链路任务来自另一统计口径，不能与这里的算子字段混算。', '']
    if links:
        lines += ['### CANN 链路任务', '',
                  '按相同 rank、源/目标和链路类型配对展示 off/on。', '',
                  '| 路径/rank | 源 → 目标 | 类型 | 每组 CANN 任务量 MB，均值 ± SD | 每组任务时间和 ms，均值 ± SD |',
                  '|---|---|---|---:|---:|']
        for side, rank, src, dst, transport in paired_order(links, 1, 2, 3, 4):
            items = links[(side, rank, src, dst, transport)]
            lines.append(f'| {side}/{rank} | {src} → {dst} | {transport} | {stat([item["size"] for item in items], 6)} | {stat([item["time"] for item in items], 6)} |')
        lines += ['', '仅汇总通信矩阵中源 rank 自己的 total@group 有向边；LOCAL 是设备内任务，和 HCCS 分列。MB 为十进制，时间是 CANN 任务字段之和，不是链路墙钟时延。HCCS/SDMA 以及 top/middle/bottom/total 视图不得重复累计。', '']
    lines += ['### 同一采样窗口内的区间与到达差异', '',
              '| 路径 | 通信区间并集 ms，范围 | 已识别计算区间并集 ms，范围 | 两者交集 ms，范围 | rank 主机调用到达差最大 ms |',
              '|---|---:|---:|---:|---:|']
    for side in ordered:
        if not complete.get(side):
            continue
        values = lambda key: span(intervals[(side, key)])
        skew = max(skews[side]) if skews.get(side) else None
        lines.append(f'| {side} | {values("communication_union_ms")} | {values("compute_union_ms")} | {values("communication_compute_overlap_ms")} | {number(skew)} |')
    lines += ['', '每个范围来自同一 rank、同一次独立采样的设备区间。计算仅含已识别 AI Core/Vector 任务；到达差是同一 collective 的 rank CPU 标记起点差。交集为零只代表本次采样，不证明正式计时全程无重叠或链路无瓶颈。', '']
    if compute_path_hint(result):
        lines += ['**推断**：本次独立采样中 on 的已识别计算区间显著增加，通信区间未同步增长；应优先核验计算路径。尚无算子级反事实证据，不能据此确认具体慢算子。', '']


def route_section(lines, result, root):
    route = result.get('route_summary')
    if not route:
        return
    total, counts = route.get('observed_top_level_aten_calls'), route.get('counts') or {}
    lines += ['## FlagGems 策略调用分类', '',
              f'独立 on 取证在所有 rank 上累计观察到 {total if total is not None else "未采集"} 次顶层设备 ATen 调用；该计数不进入正式性能计时，也不用于模型吞吐分子。', '',
              '| 分类 | 调用次数 | 占比 | 解释 |', '|---|---:|---:|---|']
    for key, (name, meaning) in ROUTES.items():
        count = counts.get(key)
        ratio = f'{count / total:.3%}' if count is not None and total else '未定义'
        lines.append(f'| {name} | {count if count is not None else "未采集"} | {ratio} | {meaning} |')
    native = route.get('policy_native_call_ratio')
    native_text = f'{native:.3%}' if native is not None else '未采集'
    lines += ['', f'策略明确原生调用占比：**{native_text}**＝（策略排除＋未复验＋无覆盖）÷顶层 ATen 调用数。',
              '这只是调用级策略分类，不是硬件 kernel fallback 比例或耗时占比；函数命中过也不证明该类 ATen key 的每次调用均命中 FlagGems。硬件 kernel fallback 未采集。',
              evidence(root, 'route-summary.json', '完整路由分类') + ' · ' + evidence(root, 'audit/on/result.json', 'on 取证结果'), '']


def appendix(lines, result, ordered, root):
    lines += ['## 附录：逐组与原始证据', '',
              '正文只聚合完整组；未完成组保留状态、首个错误和原始证据，未执行批次不算失败批次。', '',
              '<details><summary>正式计时：逐组与逐 rank 明细</summary>', '',
              '| 路径/组 | 状态 | 完成/预期批次 | 平均批次 ms | rank 加载/预热 s | 失败 rank 数 |',
              '|---|---|---:|---:|---|---:|']
    for side in ordered:
        for repeat, run in sorted(result.get('paths', {}).get(side, {}).items()):
            values = ', '.join(f'{rank}: {number(row.get("model_load_seconds"))}/{number(row.get("warmup_seconds"))}'
                               for rank, row in sorted(run.get('ranks', {}).items(), key=lambda item: int(item[0]))
                               if row.get('status') == 'completed') or '未采集'
            label = evidence(root, f'{side}/{repeat}/result.json', f'{side}/{repeat}')
            lines.append(f'| {label} | {run.get("status", "未知")} | {run.get("completed_batches", "未知")}/{run.get("expected_batches", "未知")} | {number(field(run, "summary", "latency_mean_ms"))} | {values} | {run.get("failure_count", "未知")} |')
    lines += ['', '| 路径/组/rank | 权重 storage MiB | 基线 allocated/reserved MiB | 峰值 allocated/reserved MiB | 输入 KiB/ms | 首批输出 KiB/ms |',
              '|---|---:|---:|---:|---:|---:|']
    for side in ordered:
        for repeat, rank, row in rank_rows(result.get('paths', {}).get(side, {})):
            pair = lambda stage: '/'.join(number(mib(field(row, 'memory', stage, kind + '_bytes')))
                                            for kind in ('allocated', 'reserved'))
            def transfer(key):
                size, seconds = field(row, 'transfer', key, 'logical_bytes'), field(row, 'transfer', key, 'seconds')
                return f'{number(size / 1024 if size is not None else None)}/{number(seconds * 1000 if seconds is not None else None)}'
            lines.append(f'| {side}/{repeat}/{rank} | {number(mib(field(row, "memory", "weight_storage_bytes")))} | {pair("measurement_baseline")} | {pair("measurement_peak")} | {transfer("input")} | {transfer("output_embedding")} |')
    lines += ['', '</details>', '', '<details><summary>通信采样：逐组、逐 rank、类别与链路明细</summary>', '',
              '| 路径/采样组/rank | 状态 | 归因/预期 | 通信/计算/交集 ms | 原始证据 |',
              '|---|---|---:|---:|---|']
    for side in ordered:
        for repeat, run in sorted(result.get('profiles', {}).get(side, {}).items()):
            comm = run.get('communication') or {}
            for rank, row in sorted(comm.get('ranks', {}).items(), key=lambda item: int(item[0])):
                base = f'profiles/{side}/{repeat}'
                periods = '/'.join(number(field(row, 'intervals', key))
                                   for key in ('communication_union_ms', 'compute_union_ms', 'communication_compute_overlap_ms'))
                source = evidence(root, f'{base}/communication.json', '结构化归因') + ' · ' + evidence(root, f'{base}/rank-{rank}/profiler', '原始 profiler')
                issues = '；'.join(str(item) for item in row.get('issues', []))
                status = row.get('status', '未知') + ('：' + issues if issues else '')
                lines.append(f'| {side}/{repeat}/{rank} | {status} | {row.get("attributed_collectives", "未知")}/{row.get("expected_collectives", "未知")} | {periods} | {source} |')
    lines += ['', '| 路径/采样组/rank | 类别 | 次数 | 逻辑 payload MiB | elapsed ms | Wait ms | Synchronization ms |',
              '|---|---|---:|---:|---:|---:|---:|']
    world_size = len(result.get('devices') or []) or None
    for side in ordered:
        for repeat, rank, row in profile_rows(result.get('profiles', {}).get(side, {}), world_size):
            for category, item in sorted(row.get('categories', {}).items()):
                periods = ' | '.join(number(item.get(key)) for key in ('device_elapsed_ms_sum', 'wait_ms_sum', 'synchronization_ms_sum'))
                lines.append(f'| {side}/{repeat}/{rank} | {CATEGORIES.get(category, category)} | {item.get("collectives", "未知")} | {number(mib(item.get("logical_payload_bytes")))} | {periods} |')
    lines += ['', '| 路径/采样组/rank | 源 → 目标 | 类型 | CANN 任务量 MB | CANN 任务时间 ms |',
              '|---|---|---|---:|---:|']
    for side in ordered:
        for repeat, rank, row in profile_rows(result.get('profiles', {}).get(side, {}), world_size):
            for edge in sorted(row.get('links', []), key=lambda item: (str(item.get('step')), str(item.get('src_rank')), str(item.get('dst_rank')), str(item.get('transport')))):
                lines.append(f'| {side}/{repeat}/{rank} | {edge.get("src_rank", "?")} → {edge.get("dst_rank", "?")} | {edge.get("transport", "未知")} | {number(edge.get("transit_size_mb"), 6)} | {number(edge.get("transit_time_ms"), 6)} |')
    lines += ['', '</details>', '',
              ' · '.join(evidence(root, path, label) for path, label in (
                  ('result.json', '机器可读总结果'), ('effective.yaml', '有效配置'),
                  ('prepared/identity.json', '模型、环境与执行源码身份'),
                  ('prepared/samples.json', '输入与截断记录'),
                  ('npu-before.txt', '运行前设备快照'), ('npu-after.txt', '运行后设备快照'))), '']


def render(root, result):
    """Rebuild only presentation files; measurement and profiler evidence stay sealed."""
    root = Path(root)
    identity_path, config_path = root / 'prepared/identity.json', root / 'effective.yaml'
    samples_path = root / 'prepared/samples.json'
    identity = read_json(identity_path) if identity_path.is_file() else {}
    config = (yaml.safe_load(config_path.read_text()) or {}) if config_path.is_file() else {}
    samples = read_json(samples_path) if samples_path.is_file() else []
    shapes = batch_shapes(samples, identity.get('inputs', {}))
    ordered = sides(result)
    profiles = result.get('profiles') or {}
    lines = ['# Inference 性能测试：单机 TP 模型级', '',
             f'执行状态：**{result.get("status", "未知")}**。正式无 profiler 性能与独立通信采样分别解释；采样不完整时仍可查看已封存的正式计时。', '']
    if result.get('error'):
        lines += [f'首个未完成项：{result["error"]}', '']
    comparison = result.get('comparison') or {}
    ratio = comparison.get('off_over_on_measured_time')
    if ratio is not None and ratio > 0:
        lines += [f'**本次观察**：相同逻辑负载下，on 的主计时累计用时为 off 的 **{1 / ratio:.3f} 倍**。这只适用于本次镜像、策略、设备和输入，不代表所比较组件的普遍性能。', '']
    if profiles:
        known = [row for runs in profiles.values() for run in runs.values()
                 for row in (run.get('communication') or {}).get('ranks', {}).values()]
        repeats = field(config, 'performance', 'repeats')
        expected_sides = set(result.get('summary') or profiles)
        world_size = len(result.get('devices') or []) or None
        full = bool(known) and expected_sides <= set(profiles) and all(
            (not isinstance(repeats, int) or len(profiles[side]) == repeats) and
            all(profile_complete(run, world_size) for run in profiles[side].values())
            for side in expected_sides)
        lines += [f'独立通信采样：{"完整归因" if full else "已记录归因，覆盖不完整"} **{coverage(known)}** 次 rank 事件；采样时间不进入上述比值。', '']
    if compute_path_hint(result):
        lines += ['**诊断线索**：独立采样提示优先核验计算路径；具体慢算子和性能根因尚未确认，依据与边界见后文通信采样。', '']
    conditions(lines, result, identity, config, samples, shapes, root)
    performance(lines, result, ordered, shapes)
    resources(lines, result, ordered)
    communication(lines, result, ordered, config)
    route_section(lines, result, root)
    appendix(lines, result, ordered, root)
    temporary = root / 'report.md.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(root / 'report.md')
    write_json(root / 'presentation-source-tp.json',
               {'reporting/parallel.py': file_hash(ROOT / 'reporting/parallel.py')})
