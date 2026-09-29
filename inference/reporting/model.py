# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Human report is a projection of inspectable artifacts, never a new verdict."""
from runtime.common import read_json
from reporting.components import description


def render(root, result):
    _render(root,result)
    from reporting.components import add_evidence
    add_evidence(root,result)
    from reporting.components import add_overview
    add_overview(root,result)
    add_resume_evidence(root,result)
    from runtime.common import ROOT, file_hash, write_json
    write_json(root/'report-source.json', {
        'generator_sha256': {name:file_hash(ROOT/name) for name in
                            ['reporting/model.py','reporting/parallel.py',
                             'reporting/components.py','analysis/assessment.py']},
        'result_sha256':file_hash(root/'result.json') if (root/'result.json').is_file() else None})


def _render(root, result):
    if result.get('command') == 'performance':
        return render_performance(root,result)
    if result.get('command') == 'accuracy' and result.get('parallelism') == 'tp':
        return render_tp_accuracy(root,result)
    lines = ['# Inference 精度测试报告','']
    if result.get('error'):
        lines += [f"首个失败：{result['error']}",'']
    if result.get('comparison'):
        diff = read_json(root/result['comparison'])
        lines += [description(result) if 'component_profiles' in result else '**比较对象：** 同一模型和输入在同一设备上分别执行：`off` 关闭 FlagGems，使用设备原生 PyTorch 路径；`on` 按已验证策略选择性启用 FlagGems。下表以 `off` 输出为参照，计算 `on` 与 `off` 的差异。',
                  '',
                  '**参照边界：** `off` 只是本次同设备对照，不是数学真值，也不是 NVIDIA/A100 Oracle。报告说明两条路径的输出有多接近，不能据此证明绝对正确或精度达标。',
                  '',
                  '所有误差在 CPU FP64 上计算。相对误差分母为 max(abs(off), 1e-12)；非有限元素独立报告。Layer 统计排除 padding。','',
                  '| 输出边界 | MSE | MAE | 最大相对误差 | 最低样本余弦 | 有效/总元素 |',
                  '|---|---:|---:|---:|---:|---:|']
        def fmt(value): return '未定义' if value is None else f'{value:.7g}'
        for name, m in diff['aggregate'].items():
            lines.append(f"| {name} | {fmt(m['mse'])} | {fmt(m['mae'])} | {fmt(m['relative_max'])} | {fmt(m['sample_cosine_min'])} | {m['valid_elements']}/{m['total_elements']} |")
        lines += ['',f"异常记录数：{len(diff['anomalies'])}。异常包括非有限输出和余弦未定义等情况，不按精度阈值筛选。",'',
                  '## 最差样本','', '| 边界 | 指标 | 样本 ID | 值 |','|---|---|---|---:|']
        for name, rankings in diff['worst'].items():
            for metric, rows in rankings.items():
                if rows:
                    lines.append(f"| {name} | {metric} | {rows[0]['sample_id']} | {fmt(rows[0]['value'])} |")
        lines += ['', '[逐样本、各指标最差样本及异常完整结果](comparison.json)','']
    for side in ['off','on']:
        if side in result.get('paths',{}):
            state = result['paths'][side]
            route = state.get('route',{})
            lines += [f"## {side} 路径",'',f"执行：{state.get('status')}；请求 FlagGems：{route.get('requested','unknown')}；实际调用已观测：{route.get('observed','unknown')}。",'',
                      '路由证据是 FlagGems Python 函数调用，不等同于硬件 kernel trace 或 kernel 覆盖率。','',
                      f"[路径结果]({side}/result.json) · [运行日志]({side}/run.log)",'']
    if result.get('policy'):
        lines += ['## Preview 策略','',f"最终组合执行复验：{result['verified']}；逐环境覆盖与共同选择见首页。",'',
                  '有限误差不触发回退。排除结论只适用于已记录的模型、输入、dtype、设备及依赖组合。',
                  '未知项使用原生实现，但不能解释为该算子已确认不可用。','']
        evidence = '[策略与失败签名](preview/policy.yaml)'
        if (root/'preview/attempts.json').is_file():
            evidence += ' · [探测决定](preview/attempts.json)'
        elif (root/'preview/reuse-decision.json').is_file():
            evidence += ' · [历史复验记录](preview/reuse-decision.json)'
        lines += [evidence,'']
    if result.get('export'):
        lines += ['## 图导出','', '图为 CPU 原生路径、指定 dtype、首个固定输入 batch 的结构；不是 FlagGems 实际路由图。','',
                  '[各格式结果与验证边界](export/result.json)','']
    lines += ['## 执行与证据','',
              ('输入只 tokenize 一次 → 各环境累积探测 → 共同集合完整复验 → 执行验证策略。有限误差不设达标阈值。' if result.get('command') == 'preview' else '输入只 tokenize 一次 → 独立路径进程 → 原始输出 → CPU 指标 → 报告。模型和依赖身份不匹配时拒绝比较。'),'',
              '[最终配置](effective.yaml) · [环境、模型与代码身份](prepared/identity.json) · [输入样本与截断记录](prepared/samples.json) · [机器可读结果](result.json)','',
              '未实现：任务评分、A100 参考导入和算子级精度归因。诊断日志中的 wall time 不构成性能结果；性能指标见独立 performance 模式。','']
    (root/'report.md').write_text('\n'.join(lines))


def render_tp_accuracy(root, result):
    identity_path = root/'prepared/identity.json'
    identity = read_json(identity_path) if identity_path.is_file() else {}
    devices = result.get('devices') or identity.get('parallelism',{}).get('devices',[])
    paths = result.get('paths',{})
    comparison_scope = ('逐 rank 比较同一模型、同一输入在原生 PyTorch（off）和选择性 FlagGems（on）路径的输出。'
                        if result.get('comparison') else
                        f"已执行路径：{', '.join(paths) if paths else '尚无完成路径'}；本次没有生成 off/on 数值差分。")
    if 'component_profiles' in result: comparison_scope = description(result)
    lines = ['# Inference TP 精度测试报告','',
             f"执行状态：**{result['status']}**。{comparison_scope}off 是本设备参照，不是绝对真值。没有精度阈值或达标结论。",'']
    if result.get('error'):
        lines += [f"首个失败：{result['error']}",'']
    lines += ['## 执行范围','',
              f"模型：{identity.get('model',{}).get('name','未记录')}；dtype：{identity.get('model',{}).get('dtype','未记录')}；并行：单机 TP，{len(devices)} rank。每个 rank 处理同一批逻辑样本，统计表不跨 rank 相加。",'',
              '| rank | 物理设备 |','|---:|---:|']
    for rank, device in enumerate(devices):
        lines.append(f'| {rank} | {device} |')
    lines += ['', '模型级 pooled、embedding 是各 rank 的本地观测；所选模块的三维输出可能包含 TP 分片。比较只在同一 rank 的 off/on 之间进行，不跨 rank 拼接张量。Layer 输出排除 padding。', '']
    if result.get('comparison'):
        diff = read_json(root/result['comparison'])
        lines += ['## 逐 rank 数值差异','',
                  '指标在 CPU FP64 上计算；相对误差分母为 max(abs(off), 1e-12)。非有限元素、形状不一致及未定义余弦单独记录。','',
                  '| rank | 输出边界 | MSE | MAE | 最大相对误差 | 最低样本余弦 | 有效/总元素 |',
                  '|---:|---|---:|---:|---:|---:|---:|']
        def fmt(value): return '未定义' if value is None else f'{value:.7g}'
        for rank, row in diff['ranks'].items():
            for name, metric in row['aggregate'].items():
                lines.append(f"| {rank} | {name} | {fmt(metric['mse'])} | {fmt(metric['mae'])} | {fmt(metric['relative_max'])} | {fmt(metric['sample_cosine_min'])} | {metric['valid_elements']}/{metric['total_elements']} |")
        lines += ['',f"异常记录数：{result.get('numerical_anomalies',0)}，按 rank 保存在原始比较结果中。",'',
                  '### 各 rank 最差样本','',
                  '| rank | 边界 | 指标 | 样本 ID | 值 |','|---:|---|---|---|---:|']
        for rank, row in diff['ranks'].items():
            for name, rankings in row['worst'].items():
                for metric, samples in rankings.items():
                    if samples:
                        lines.append(f"| {rank} | {name} | {metric} | {samples[0]['sample_id']} | {fmt(samples[0]['value'])} |")
        lines += ['', '[逐 rank、逐样本完整比较与异常](comparison.json)','']
    for side in ['off','on']:
        if side not in paths:
            continue
        state = paths[side]
        lines += [f'## {side} 路径','',f"进程组：{state.get('status')}；{state.get('error','各 rank 完成') }。",'',
                  '| rank | 执行 | FlagGems 函数调用 | 原始输出与证据 |',
                  '|---:|---|---:|---|']
        for rank, device in enumerate(devices):
            key = str(rank)
            rank_state = state.get('ranks',{}).get(key,{})
            calls = rank_state.get('route',{}).get('actual_function_calls',{})
            count = sum(calls.values()) if calls else 0
            evidence = (f'[{side}/rank-{rank}/result.json]({side}/rank-{rank}/result.json)'
                        if (root/side/f'rank-{rank}/result.json').is_file() else '缺失')
            lines.append(f"| {rank} | {rank_state.get('status','缺失')} | {count if rank_state.get('route') else '未记录'} | {evidence} |")
        lines += ['',f'[{side} 进程组结果]({side}/result.json) · [{side} 运行日志]({side}/run.log)','',
                  '调用数仅证明 FlagGems Python 函数入口被触发，不代表硬件 kernel 覆盖；进程组失败时缺失的 rank 不计为数值通过。','']
    evidence = ['[最终配置](effective.yaml)','[机器可读总结果](result.json)']
    if identity_path.is_file():
        evidence.append('[环境、模型、源码与设备身份](prepared/identity.json)')
    if (root/'prepared/samples.json').is_file():
        evidence.append('[输入样本](prepared/samples.json)')
    if (root/'prepared/run.log').is_file():
        evidence.append('[输入准备日志](prepared/run.log)')
    lines += ['## 身份与复查','', ' · '.join(evidence),'',
              ('输入只 tokenize 一次；off/on 分别启动完整 TP 进程组。各 rank 保存原始 tensor，之后在 CPU 按相同 rank、样本 ID 和边界配对。'
               if identity_path.is_file() else '输入准备尚未完成；请先查看失败阶段和 worker 日志。') +
              '诊断 wall time 不是性能结果。','']
    (root/'report.md').write_text('\n'.join(lines))


def render_performance(root, result):
    if result.get('parallelism') == 'tp':
        from reporting.parallel import render
        return render(root,result)
    import re
    from statistics import mean, stdev

    def stat(values, digits=3, scale=1):
        values = [value * scale for value in values if value is not None]
        if not values:
            return '未采集'
        average = f'{mean(values):.{digits}f}'
        if len(values) == 1:
            return f'{average}（n=1）'
        return f'{average} ± {stdev(values):.{digits}f}'

    def mib(value):
        return value / 1048576

    def size(value):
        return f'{mib(value):.2f} MiB' if value >= 1048576 else f'{value / 1024:.2f} KiB'

    def memory_value(state, stage, kind):
        return mib(state['memory'][stage][kind + '_bytes'])

    def memory_pair(state, stage):
        return f"{memory_value(state, stage, 'allocated'):.1f} / {memory_value(state, stage, 'reserved'):.1f}"

    paths = result.get('paths', {})
    sides = list(dict.fromkeys([*result.get('summary', {}), *paths]))
    complete = {side: [(label, state) for label, state in paths.get(side, {}).items()
                       if state.get('status') == 'completed'] for side in sides}
    all_complete = [(side, label, state) for side in sides for label, state in complete[side]]
    identity_path = root / 'prepared/identity.json'
    identity = read_json(identity_path) if identity_path.is_file() else {}
    model = identity.get('model', {})
    inputs = identity.get('inputs', {})
    samples_path = root / 'prepared/samples.json'
    samples = read_json(samples_path) if samples_path.is_file() else []
    device = f"{identity['vendor'].capitalize()} 物理设备 {identity['physical_device']}" if identity.get('vendor') and identity.get('physical_device') is not None else '当前设备'
    batch_size = inputs.get('batch_size')
    max_length = inputs.get('max_length')
    shapes = []
    if samples and batch_size and max_length:
        for index in range(0, len(samples), batch_size):
            group = samples[index:index + batch_size]
            if all(row.get('original_tokens') is not None for row in group):
                shapes.append(f"[{len(group)}, {max(min(row['original_tokens'], max_length) for row in group)}]")
    first_rows = next((state.get('batches', []) for _, _, state in all_complete if state.get('batches')), [])
    batches_per_round = len({row['batch_index'] for row in first_rows}) if first_rows else None
    measure_rounds = len({row['cycle'] for row in first_rows}) if first_rows else None
    weights = [mib(state['memory']['weight_storage_bytes']) for _, _, state in all_complete if state.get('memory')]
    weight_text = stat(weights, 2) + ' MiB' if weights else '未采集'
    model_detail = '；28 层、hidden size 1024、1024 维 FP32 embedding' if model.get('name') == 'qwen3_embedding_0.6b' else ''
    config_path = root / 'effective.yaml'
    config_text = config_path.read_text() if config_path.is_file() else ''
    warmup_match = re.search(r'(?m)^  warmup_rounds: (\d+)$', config_text)
    warmup_rounds = warmup_match.group(1) if warmup_match else '未记录'

    path_scope = ('两侧分别在独立进程执行' if {'off', 'on'} <= set(sides)
                  else f"本次包含 {'、'.join(sides)} 路径" if sides else '本次暂无正式计时组')
    component_description = description(result) if 'component_profiles' in result else '对比路径：FlagGems off 使用同设备原生算子；FlagGems on 按已验证策略选择性注册 FlagGems 函数。'
    lines = ['# Inference 性能测试：模型级', '',
             f"执行状态：**{result['status']}**。模型级对应 `--level total` 的整体前向窗口。{component_description}{path_scope}。", '']
    if result.get('error'):
        lines += [f"首个失败：{result['error']}", '']
    lines += ['## 模型、输入与测量条件', '',
              '| 项目 | 本次条件 |', '|---|---|',
              f"| 模型 | {model.get('name', '未记录')}；{model.get('dtype', '未记录')}；attention={model.get('attention', '未记录')}{model_detail}；设备权重及 buffer 去重 storage {weight_text} |",
              f"| 输入 | {len(samples) if samples else '未记录'} 条固定文本；batch size {batch_size or '未记录'}；max length {max_length or '未记录'} token；{inputs.get('padding_side', '未记录')} padding |",
              f"| 批次负载 | {'、'.join(shapes) if shapes else '形状未记录'}；每轮 {batches_per_round or '未记录'} 批；预热 {warmup_rounds} 轮/组；测量 {measure_rounds or '未记录'} 轮/组 |",
              f"| 设备 | {device}；{identity.get('device_name', '型号未记录')}；单进程、串行同步批次 |", '',
              ('模型名中的 0.6B 是型号标称规模；' if model.get('name') == 'qwen3_embedding_0.6b' else '') +
              '表中 storage 是实际设备上参数与 buffer 去重后的占用，不是模型文件大小，也不是精确参数个数。输入在各路径共用同一批已预制数据。', '']
    if samples:
        truncated = sum(bool(row.get('truncated')) for row in samples)
        lines += [f"输入原始 token 长度 {min(row['original_tokens'] for row in samples)}–{max(row['original_tokens'] for row in samples)}；截断 {truncated}/{len(samples)} 条。样本数在多轮中重复计算，不代表不同文本的数量。", '']
    lines += ['结构化结果未记录同板卡其他任务的负载，不能据此排除共享资源干扰；数值只适用于本次封存的设备、模型、输入与策略。', '']

    summary = result.get('summary', {})
    if summary:
        lines += ['## 主指标：全部 repeat 的合并统计', '',
                  '| 路径 | 批次 | 样本处理次数 | 有效输入 token | 平均批次时延 ms | p50 / p90 / p99 ms | 样本/s | token/s | 测量累计 s |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for side, state in summary.items():
            lines.append(f"| {side} | {state['batches']} | {state['samples']} | {state['tokens']} | {state['latency_mean_ms']:.3f} | {state['latency_p50_ms']:.3f} / {state['latency_p90_ms']:.3f} / {state['latency_p99_ms']:.3f} | {state['samples_per_second']:.4f} | {state['tokens_per_second']:.3f} | {state['measured_seconds']:.3f} |")
        lines += ['', '上表从所有成功 repeat 的批次原始记录合并计算；分位数不是各组分位数的平均。下表单独展示各 repeat 汇总值的均值 ± 样本标准差（SD）。', '',
                  '| 路径 | 成功 repeat 数 | 组内平均批次时延 ms 的跨组均值 ± SD | 组内样本/s 的跨组均值 ± SD | 组内 token/s 的跨组均值 ± SD |',
                  '|---|---:|---:|---:|---:|']
        for side in summary:
            groups = [state['summary'] for _, state in complete.get(side, []) if state.get('summary')]
            lines.append(f"| {side} | {len(groups)} | {stat([item['latency_mean_ms'] for item in groups])} | {stat([item['samples_per_second'] for item in groups], 4)} | {stat([item['tokens_per_second'] for item in groups])} |")
        lines += ['', '### 主指标说明', '',
                  '| 指标 | 测量或计算方式 | 解读边界 |', '|---|---|---|',
                  '| 批次、样本处理次数 | 批次是完成的同步前向次数；样本处理次数为各批实际样本数之和。 | 多轮会重复计算同一文本，不是不同文本数。 |',
                  '| 有效输入 token | 各批 `attention_mask` 非零元素之和。 | 不含 padding，也不是生成 token 数。 |',
                  '| 测量累计 s | 各批 `perf_counter_ns()` 计时窗口之和。批次输入已驻留设备，前向、最后有效 token pooling、FP32 归一化和末尾设备同步计入窗口。 | 不含 tokenization、模型加载、传输、预热、取证、落盘及批间等待；不是端到端服务耗时或纯 kernel 时间。 |',
                  '| 平均批次时延 ms | 测量累计时间除以完成批次数。 | 同步批次完成时间，不能当作单请求时延。 |',
                  '| p50 / p90 / p99 ms | 将所有完成批次时延排序，在 `(N-1)×p/100` 位置线性插值。 | 描述批次分布；不同输入形状可能形成多个峰。 |',
                  '| 样本/s | 样本处理次数除以测量累计秒数。 | 固定输入的计算窗口吞吐，不含服务端排队或在线并发。 |',
                  '| token/s | 有效输入 token 数除以同一测量累计秒数。 | 输入处理吞吐，不是生成吞吐。 |',
                  '| 跨 repeat 均值 ± SD | 先分别计算每组的平均批次时延、样本/s、token/s，再对成功组求算术均值和样本 SD。 | 与上表的合并批次统计口径不同；只有一组时 SD 不定义。 |', '']
    if result.get('comparison'):
        ratio = result['comparison']['off_over_on_measured_time']
        if ratio:
            lines += [f"相同批次负载下，on 的主测量累计耗时为 off 的 **{1 / ratio:.2f} 倍**；on/off 样本吞吐比为 **{result['comparison']['on_over_off_samples_per_second']:.4f}**。这只比较主计时窗口。", '']

    if all_complete:
        has_cpu = any(state.get('model_cpu_load_seconds') is not None for _, _, state in all_complete)
        has_weights_transfer = any(state.get('transfer', {}).get('model_weights') for _, _, state in all_complete)
        lines += ['## 阶段耗时：按 repeat 汇总', '',
                  '耗时单元格为成功 repeat 的均值 ± 样本 SD，单位秒；每组都独立加载模型、预热，再正式计时。只有一组时显示 n=1，不计算 SD。', '',
                  '| 路径 | 成功/总组数 | 模型 storage MiB | 模型加载总 s | 预热轮数/组 | 预热 s | 失败批次数 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for side in sides:
            runs = complete[side]
            failures = sum(s.get('failure_count', 0) for s in paths.get(side, {}).values())
            failure_text = str(failures) if len(runs) == len(paths.get(side, {})) else f'{failures}；另有未完成组'
            lines.append(f"| {side} | {len(runs)}/{len(paths.get(side, {}))} | {stat([mib(s['memory']['weight_storage_bytes']) for _, s in runs], 2)} | {stat([s['model_load_seconds'] for _, s in runs])} | {warmup_rounds} | {stat([s['warmup_seconds'] for _, s in runs])} | {failure_text} |")
        lines += ['']
        if has_cpu or has_weights_transfer:
            lines += ['本次可独立计量的加载子阶段：', '', '| 路径 | CPU 模型准备 s | 参数迁移到设备 s |', '|---|---:|---:|']
            for side in sides:
                runs = complete[side]
                lines.append(f"| {side} | {stat([s.get('model_cpu_load_seconds') for _, s in runs]) if has_cpu else '—'} | {stat([s.get('transfer', {}).get('model_weights', {}).get('seconds') for _, s in runs]) if has_weights_transfer else '—'} |")
            lines += ['']
        else:
            lines += ['本次封存结果只有加载总时间，不能据此拆分 CPU 准备或权重迁移耗时。', '']

        lines += ['### 阶段指标说明', '',
                  '| 指标 | 测量或计算方式 | 解读边界 |', '|---|---|---|',
                  '| 成功/总组数 | 完整结束的 repeat 数 / 已启动的 repeat 数。 | 未完成组的缺失批次不计作已测结果。 |',
                  '| 模型 storage MiB | 设备上参数与 buffer 去重后的 storage 字节换算 MiB。 | 提供加载耗时的规模背景；不是模型文件大小或精确参数数。 |',
                  '| 模型加载总 s | 从在 CPU 读取并构建模型开始，到模型迁入设备且设备同步结束；每组计一次。 | 含读取、构建、迁移及同步，不是纯磁盘或纯传输时间；不进入主计时。 |',
                  '| 预热轮数/组、预热 s | 一轮遍历所有预制批次；计时记录配置轮数及完成这些轮次的总时间。 | 含前向与同步，不能直接解释为编译时间；不进入主计时。 |',
                  '| 失败批次数 | 累加 worker 明确记录的失败批次；未完成组另行标注。 | 未执行或状态未知的批次不能填为失败；阶段和退出原因见各组 `stage.json`、`result.json`。 |']
        if has_cpu:
            lines.append('| CPU 模型准备 s | 从开始加载 CPU 模型到 CPU storage 统计结束。 | 仅在原始结果实际采集时显示，是模型加载总时间的一部分。 |')
        if has_weights_transfer:
            lines.append('| 参数迁移到设备 s | CPU 模型构建后的一次 `model.to(device)`，直到设备同步。 | 含分配与同步；是模型加载总时间的一部分，不能再与总时间相加。 |')
        lines += ['']

        lines += ['## 显存：按 repeat 汇总', '',
                  '单位 MiB；每项为成功 repeat 的均值 ± 样本 SD。', '',
                  '| 路径 | 权重 storage | 加载后 allocated 增量 | 预热后基线 allocated / reserved | 测量峰值 allocated / reserved | 峰值减基线 allocated / reserved |',
                  '|---|---:|---:|---:|---:|---:|']
        for side in sides:
            runs = [s for _, s in complete[side]]
            def ms(stage, kind):
                return stat([memory_value(s, stage, kind) for s in runs], 1)
            loaded = [mib(s['memory']['after_model_load']['allocated_bytes'] - s['memory']['before_model_load']['allocated_bytes']) for s in runs]
            lines.append(f"| {side} | {stat([mib(s['memory']['weight_storage_bytes']) for s in runs], 1)} | {stat(loaded, 1)} | {ms('measurement_baseline', 'allocated')} / {ms('measurement_baseline', 'reserved')} | {ms('measurement_peak', 'allocated')} / {ms('measurement_peak', 'reserved')} | {ms('peak_increment_bytes', 'allocated')} / {ms('peak_increment_bytes', 'reserved')} |")
        lines += ['', '### 显存指标说明', '',
                  '| 指标 | 测量或计算方式 | 解读边界 |', '|---|---|---|',
                  '| 权重 storage | 模型参数及 buffer 按设备 storage 去重后的字节数。 | 已包含在 allocated 中，不能与 allocated 相加。 |',
                  '| 加载后 allocated 增量 | 模型加载后 allocated 减加载前 allocated。 | 还可能含模型初始化分配，不等于纯权重大小。 |',
                  '| 预热后基线 allocated / reserved | 完成预热、正式计时前读取的分配器状态。 | 包含已驻留模型、输入和运行时分配。 |',
                  '| 测量峰值 allocated / reserved | 预热后重置峰值统计，再记录正式测量窗口的最大分配器状态。 | allocated 为已分配内存，reserved 为分配器保留的缓存；两者不能相加，也非整卡物理显存。 |',
                  '| 峰值减基线 allocated / reserved | 各自的测量峰值减预热后基线。 | 仅是分配器窗口增量，不能拆成纯 activation 或 workspace。 |',
                  '| KV 与分配器外内存 | 模型前向使用 `use_cache=False`。 | KV cache 显存不适用；分配器外的物理设备内存未采集。 |', '']

        kinds = [('input', f'CPU→{device}：input_ids、attention_mask', '全部预制 batch 各同步拷贝一次'),
                 ('output_embedding', f'{device}→CPU：embedding', '首批额外前向完成后，只计一次输出拷回')]
        if has_weights_transfer:
            kinds.insert(0, ('model_weights', f'CPU→{device}：模型参数、buffer', 'CPU 构建后一次迁移并同步'))
        lines += ['## 数据传输：按 repeat 汇总', '',
                  '窗口耗时为成功 repeat 的均值 ± 样本 SD，单位毫秒；仅展示本次确实单独采集的对象。', '',
                  '| 路径 | 方向与对象 | 每组逻辑数据量 | 窗口耗时 ms |',
                  '|---|---|---:|---:|']
        for side in sides:
            runs = [s for _, s in complete[side]]
            for key, direction, _ in kinds:
                transfers = [s.get('transfer', {}).get(key) for s in runs]
                if not any(transfers):
                    continue
                bytes_values = [item['logical_bytes'] for item in transfers if item]
                lines.append(f"| {side} | {direction} | {size(bytes_values[0]) if len(set(bytes_values)) == 1 else stat(bytes_values, 0) + ' B'} | {stat([item['seconds'] for item in transfers if item], 3, 1000)} |")
        lines += ['', '### 传输指标说明', '',
                  '| 指标 | 测量或计算方式 | 解读边界 |', '|---|---|---|',
                  '| 输入 CPU→设备 | 预热前将全部预制 batch 的 `input_ids`、`attention_mask` 各同步拷贝一次，耗时逐批累加。 | 正式主计时使用已驻留输入，不包含这项传输。 |',
                  '| 输出 设备→CPU | 额外运行首批前向并同步，随后单独计一次 `embedding.cpu()` 及设备同步。 | 只覆盖首批输出，不含额外前向，也不能外推为全部批次输出传输。 |']
        if has_weights_transfer:
            lines.append('| 模型参数、buffer CPU→设备 | CPU 模型构建后一次 `model.to(device)` 并同步。 | 是模型加载总时间的子阶段，不与加载总时间相加。 |')
        lines += ['| 逻辑数据量 | 输入和输出为 tensor 元素数 × 元素字节数；模型迁移为 CPU 去重 storage 字节。 | 不是物理总线流量；实际互连未采集。 |',
                  '| 窗口耗时 | 主机单调时钟覆盖拷贝调用至设备同步。 | 含主机调用、设备分配和同步，不能解释为纯链路延迟。 |',
                  '| 通信开销 | 本次单卡串行，无分布式集合通信。 | 集合通信不适用；其他运行时内部通信时间未采集，不能填零。 |', '']

    route = result.get('route_summary')
    if route:
        counts = route['counts']; total = route['observed_top_level_aten_calls']
        labels = [('excluded', '策略排除', '当前策略已确认排除，走原生实现'),
                  ('unverified', '未复验', '有候选函数，但本次策略未复验，保持原生'),
                  ('uncovered', '无覆盖', '当前候选映射没有对应 FlagGems 函数'),
                  ('allowed_function_observed', '使用FlagGems', '对应候选函数已观测到命中；本行是 ATen key 调用数，不能逐次确认每次都走 FlagGems'),
                  ('ambiguous', '归属不明', '候选归属无法确定，不能计作已证实回退')]
        lines += ['## FlagGems fallback 比例与原因', '',
                  f'独立 on 取证观察到 {total} 次顶层设备 ATen 调用。以下比例的分母均为这个调用数，取证进程不参与正式性能计时。', '',
                  '| 分类 | 调用次数 | 占比 | 原因或解释 |', '|---|---:|---:|---|']
        for key, label, reason in labels:
            count = counts.get(key, 0)
            lines.append(f'| {label} | {count} | {count / total:.3%} | {reason} |' if total else f'| {label} | {count} | 未定义 | {reason} |')
        lines += ['', '### fallback 指标说明', '',
                  '| 指标 | 数值与计算 | 解读边界 |', '|---|---|---|',
                  f"| 策略明确原生调用占比 | **{route['policy_native_call_ratio']:.3%}**＝（策略排除＋未复验＋无覆盖）÷{total} 次顶层设备 ATen 调用。 | 调用级策略分类，不是硬件 kernel fallback 比例或耗时占比。 |",
                  '| “使用FlagGems”分类 | 对应候选函数至少在独立取证中实际命中一次。 | 本行计数仍是该类 ATen key 的调用数，不能证明其中每次都由 FlagGems 执行。 |',
                  '| 硬件 kernel fallback 比例 | 未采集。 | 无 kernel trace，不能填 0，也不能由调用分类推算。 |', '',
                  '[完整路由分类](route-summary.json) · [off 原始取证](audit/off/result.json) · [on 实际命中](audit/on/route.json)', '']

    if paths:
        lines += ['## 附录：每次 repeat 的原始汇总', '',
                  '逐批纳秒记录、失败阶段和未舍入的数值见对应组的 `batches.jsonl`、`stage.json` 和 `result.json`。', '',
                  '### 批次与阶段时间', '',
                  '| 路径 / 组 | 状态 | 批次 | 平均批次 ms | p50 / p90 / p99 ms | 样本/s | token/s | 加载总 s | 预热 s | 失败批次 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for side in sides:
            for label, state in paths.get(side, {}).items():
                link = f'[{side}/{label}]({side}/{label}/result.json)'
                if state.get('status') != 'completed':
                    lines.append(f"| {link} | {state.get('status', 'unknown')} | {state.get('completed_batches', 0)}/{state.get('expected_batches', '?')} | | | | | | | |")
                    continue
                s = state.get('summary', {})
                if s:
                    lines.append(f"| {link} | completed | {s['batches']} | {s['latency_mean_ms']:.3f} | {s['latency_p50_ms']:.3f} / {s['latency_p90_ms']:.3f} / {s['latency_p99_ms']:.3f} | {s['samples_per_second']:.3f} | {s['tokens_per_second']:.3f} | {state['model_load_seconds']:.3f} | {state['warmup_seconds']:.3f} | {state.get('failure_count', 0)} |")
                else:
                    lines.append(f"| {link} | completed | 未记录 | | | | | {state['model_load_seconds']:.3f} | {state['warmup_seconds']:.3f} | {state.get('failure_count', 0)} |")
        if all_complete:
            if has_cpu or has_weights_transfer:
                lines += ['', '### 已采集的加载子阶段', '', '| 路径 / 组 | CPU 模型准备 s | 参数迁移到设备 s |', '|---|---:|---:|']
                for side, label, state in all_complete:
                    cpu = state.get('model_cpu_load_seconds')
                    weights_transfer = state.get('transfer', {}).get('model_weights')
                    cpu_text = f'{cpu:.3f}' if cpu is not None else '—'
                    transfer_text = f"{weights_transfer['seconds']:.3f}" if weights_transfer else '—'
                    lines.append(f'| {side}/{label} | {cpu_text} | {transfer_text} |')
            lines += ['', '### 显存明细', '',
                      '| 路径 / 组 | 权重 storage MiB | 加载后 allocated 增量 MiB | 基线 allocated / reserved MiB | 峰值 allocated / reserved MiB | 峰值减基线 allocated / reserved MiB |',
                      '|---|---:|---:|---:|---:|---:|']
            for side, label, state in all_complete:
                m = state['memory']
                loaded = mib(m['after_model_load']['allocated_bytes'] - m['before_model_load']['allocated_bytes'])
                lines.append(f"| {side}/{label} | {mib(m['weight_storage_bytes']):.1f} | {loaded:.1f} | {memory_pair(state, 'measurement_baseline')} | {memory_pair(state, 'measurement_peak')} | {memory_pair(state, 'peak_increment_bytes')} |")
            lines += ['', '### 数据传输明细', '',
                      '| 路径 / 组 | 方向与对象 | 逻辑数据量 | 窗口耗时 ms |', '|---|---|---:|---:|']
            for side, label, state in all_complete:
                for key, direction, _ in kinds:
                    item = state.get('transfer', {}).get(key)
                    if item:
                        lines.append(f"| {side}/{label} | {direction} | {size(item['logical_bytes'])} | {item['seconds'] * 1000:.3f} |")
    evidence = ['[机器可读总结果](result.json)', '[配置](effective.yaml)']
    if identity_path.is_file():
        evidence.append('[模型、输入、环境与执行源码身份](prepared/identity.json)')
    if samples_path.is_file():
        evidence.append('[输入样本与截断记录](prepared/samples.json)')
    if (root / 'policy.yaml').is_file():
        evidence.append('[当前策略](policy.yaml)')
    lines += ['', '## 原始证据', '', ' · '.join(evidence), '']
    (root / 'report.md').write_text('\n'.join(lines))


def add_resume_evidence(root, result):
    """Explain provenance and measured discovery costs without changing execution verdicts."""
    snapshot = root/'source-snapshot.json'
    if not snapshot.is_file() and not result.get('budget'):
        return
    lines = ['', '## 版本与进度来源', '']
    if snapshot.is_file():
        value = read_json(snapshot)
        lines += [f"执行源码摘要：`{value['execution_source_key']}`。",
                  f"本次分析摘要：`{value['analysis_key']}`。", '',
                  '策略匹配执行身份；纯分析源码可以独立更新，依赖与镜像变化仍要求重新探测。', '']
    if result.get('resume'):
        source = result['resume'].get('source')
        lines += [f"来源：{source or '全新 preview'}；续探代数：{result['resume']['generation']}。", '',
                  '| 环境 | 继承接受 | 本次新增接受 | 本次共同集合 |', '|---|---:|---:|---:|']
        for key,p in result.get('policy_profiles',{}).items():
            lines.append(f"| {key[:12]} | {len(p.get('inherited_include',[]))} | {len(p.get('new_include',[]))} | {result.get('accepted',0)} |")
        lines += ['', '累积探测有顺序依赖；续探不保证与一次长跑选择相同集合。历史失败只属于记录的累积组合。',
                  '轻量恢复证据已复制；历史 tensor/trace 仍引用来源目录，不是独立备份。', '',
                  '[原子检查点](preview/checkpoint.json) · [预算记录](preview/budget.json)', '']
    budget = result.get('budget')
    if budget:
        from collections import defaultdict
        groups=defaultdict(lambda:[0,0.0,0])
        for row in budget['trials']:
            group=groups[row['phase']];group[0]+=1;group[1]+=row['seconds'];group[2]+=int(row['timed_out'])
        lines += ['## 探测时间花在哪里', '',
                  f"预算策略：{budget['mode']}；本次预算 {budget['seconds']} 秒，已计费 {budget['charged_seconds']:.2f} 秒。", '',
                  '| 阶段 | worker 次数 | 实测秒数 | 超时次数 |', '|---|---:|---:|---:|']
        for phase,(count,seconds,timeouts) in groups.items():
            lines.append(f'| {phase} | {count} | {seconds:.2f} | {timeouts} |')
        lines += ['', f"因预算估计暂缓的候选：{len(budget['skipped'])}；建议下次预算至少约 {budget['estimated_minimum_next_budget_seconds']} 秒（估计，不保证完成）。",
                  '本预算覆盖基线、旧集合复验、候选及最终复验；准备、复制、清理另列。上述耗时是实验周转成本，不是模型推理性能。', '']
        for name,seconds in result.get('preparation_seconds',{}).items():
            lines.append(f'- {name}：{seconds:.2f} 秒。')
        lines.append(f"- 超时清理：{budget['cleanup_seconds']:.2f} 秒。")
    path=root/'report.md'
    path.write_text(path.read_text()+'\n'+'\n'.join(lines))
