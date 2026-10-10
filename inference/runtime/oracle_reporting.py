"""Additive reference report: keep same-device comparisons and label directions."""
import math
from runtime.common import read_json, write_json


LABELS = {'mse': 'MSE', 'mae': 'MAE', 'max_abs': '最大绝对误差',
          'normalized_rmse': '归一化 RMSE', 'sample_cosine_min': '最低样本余弦'}


def fmt(value):
    return '未定义/未记录' if value is None else f'{value:.8g}'


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def change(off, on, higher=False):
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (off, on)):
        return {'delta': None, 'relative_change': None, 'direction': 'undefined'}
    delta = on-off
    relative = delta/abs(off) if off else None
    return {'delta': delta if math.isfinite(delta) else None,
            'relative_change': relative if relative is not None and math.isfinite(relative) else None,
            'direction': 'unchanged' if delta == 0 else 'improved' if (delta > 0) == higher else 'worsened'}


def comparison_summary(diff, same_device=None):
    """Three comparisons share sample/rank labels, never subtract tensors or infer causality."""
    candidates = diff.get('candidates', {})
    off = candidates.get('off', {}).get('ranks', {})
    on = candidates.get('on', {}).get('ranks', {})
    rows, samples, issues = [], [], []
    for rank in sorted(set(off) | set(on)):
        a, b = off.get(rank, {}), on.get(rank, {})
        ah={r['sample_id']:r['compared_steps'] for r in a.get('generation',{}).get('trajectories',[])}
        bh={r['sample_id']:r['compared_steps'] for r in b.get('generation',{}).get('trajectories',[])}
        comparable=ah==bh
        if not comparable:
            issues.append(f'rank {rank}: A100→off 与 A100→on 的生成配对范围不同，保留各自误差，不计算改善或变差')
        native = ((same_device or {}).get('ranks', {}).get(rank, {})
                  if 'ranks' in (same_device or {}) else same_device or {})
        for boundary in sorted(set(a.get('aggregate', {})) | set(b.get('aggregate', {}))):
            am, bm = a.get('aggregate', {}).get(boundary, {}), b.get('aggregate', {}).get(boundary, {})
            pair = {'rank': rank, 'boundary': boundary, 'metrics': {}}
            for metric in LABELS:
                av, bv = am.get(metric), bm.get(metric)
                pair['metrics'][metric] = {'a100_to_off': av, 'a100_to_on': bv,
                    'off_to_on': native.get('aggregate', {}).get(boundary, {}).get(metric),
                    **change(av if comparable else None, bv if comparable else None, metric == 'sample_cosine_min')}
            rows.append(pair)
        ar = {(row['sample_id'], row['boundary']): row for row in a.get('samples', [])}
        br = {(row['sample_id'], row['boundary']): row for row in b.get('samples', [])}
        if a and b and ar.keys() != br.keys():
            issues.append(f'rank {rank}: off/on 的样本或边界覆盖不同，缺失项不参与变化判断')
        for key in sorted(ar.keys() | br.keys()):
            values = {}
            for metric in ('mse', 'max_abs', 'cosine_similarity'):
                av, bv = ar.get(key, {}).get(metric), br.get(key, {}).get(metric)
                values[metric] = {'a100_to_off': av, 'a100_to_on': bv,
                                 **change(av if ah.get(key[0])==bh.get(key[0]) else None,
                                          bv if ah.get(key[0])==bh.get(key[0]) else None, metric == 'cosine_similarity')}
            samples.append({'rank': rank, 'sample_id': key[0], 'boundary': key[1], 'metrics': values})
    return {'schema_version': 1, 'oracle_id': diff.get('oracle_id'), 'aggregate': rows,
            'samples': samples, 'issues': issues,
            'scope': 'off/on errors use the same A100 reference; off_to_on is a separate device comparison',
            'direction_definition': 'exact descriptive metric ordering, no tolerance, significance or task pass/fail',
            'relative_change_definition': '(on_error-off_error)/abs(off_error); zero denominator is undefined'}


def summary_lines(summary, diff):
    directions = {'improved': '该指标改善', 'worsened': '该指标变差',
                  'unchanged': '数值相同', 'undefined': '缺少可比值'}
    lines = ['## 开启组件后，哪些指标改变', '',
             '前两列误差都以 A100 为参照；第三列是同设备 off→on 的独立差分。变化只描述指标方向，'
             '业务可接受程度由任务标准决定。逐 rank 保留结果，多个 rank 的同一逻辑样本不累加。', '',
             '| rank / 边界 | 指标 | A100→off | A100→on | 同设备 off→on | 相对 A100 的变化 |',
             '|---|---|---:|---:|---:|---|']
    for row in summary['aggregate']:
        for metric, values in row['metrics'].items():
            direction = directions[values['direction']]
            if values['relative_change'] is not None and metric != 'sample_cosine_min':
                direction += f"（{values['relative_change']:+.2%}）"
            lines.append(f"| {row['rank']} / {row['boundary']} | {LABELS[metric]} | {fmt(values['a100_to_off'])} | "
                         f"{fmt(values['a100_to_on'])} | {fmt(values['off_to_on'])} | {direction} |")
    lines += ['', '### 相对 A100 的逐样本变化', '',
              '按每个样本、每个边界独立比较，避免平均改善掩盖局部退化。改善/变差采用原数值大小排序，未使用业务阈值。', '',
              '| rank / 边界 | 指标 | 改善 / 变差 / 相同 / 未定义 | 最需复查的退化样本（off→on 指标） |',
              '|---|---|---|---|']
    for rank, boundary in sorted({(r['rank'], r['boundary']) for r in summary['samples']}):
        rows = [r for r in summary['samples'] if r['rank'] == rank and r['boundary'] == boundary]
        for metric, label in [('mse', 'MSE'), ('max_abs', '最大绝对误差'), ('cosine_similarity', '余弦')]:
            counts = {key: sum(r['metrics'][metric]['direction'] == key for r in rows) for key in directions}
            worse = [r for r in rows if r['metrics'][metric]['direction'] == 'worsened']
            worst = max(worse, key=lambda r: abs(r['metrics'][metric]['delta']), default=None)
            text = '无观测退化' if not worse and counts['undefined'] == 0 else '未形成完整判断'
            if worst:
                values = worst['metrics'][metric]
                text = f"{cell(worst['sample_id'])}（{fmt(values['a100_to_off'])} → {fmt(values['a100_to_on'])}）"
            lines.append(f"| {rank} / {boundary} | {label} | " + ' / '.join(str(counts[k]) for k in ('improved', 'worsened', 'unchanged', 'undefined')) + f' | {text} |')
    lines += ['', '### A100 差分中的最差样本', '',
              '| 被测路径 / rank | 边界 | 指标 | 样本 | 值 |', '|---|---|---|---|---:|']
    for side, target in diff.get('candidates', {}).items():
        for rank, row in target.get('ranks', {}).items():
            for boundary in row.get('aggregate',{}):
                for metric, label in [('mse', 'MSE'), ('max_abs', '最大绝对误差'), ('cosine_similarity', '最低余弦')]:
                    worst = row.get('worst', {}).get(boundary, {}).get(metric, [])
                    if worst:
                        lines.append(f"| {side} / {rank} | {boundary} | {label} | {cell(worst[0]['sample_id'])} | {fmt(worst[0]['value'])} |")
    lines += ['', '[所有边界与逐样本改善/退化明细](oracle-summary.json) · [原始 A100 差分](oracle-comparison.json)', '']
    if summary['issues']:
        lines += [cell(issue) for issue in summary['issues']] + ['']
    return lines


def interpretation_lines(diff):
    lines = ['## 误差尺度、分布与样本权重', '',
             '归一化 RMSE＝误差 RMS / 参考 RMS，用参考张量的幅度解释误差；参考全零时未定义。'
             '绝对误差分位数按有限配对元素线性插值。近零统计使用 abs(reference)≤1e-12，'
             '与原相对误差分母下限一致，是数值解释口径，不是精度容差。', '',
             '| 路径 / rank / 模型输出 | 参考 RMS | 归一化 RMSE | 绝对误差 p50 / p90 / p99 | 近零元素 / 有限配对 | 近零处最大绝对误差 |',
             '|---|---:|---:|---|---:|---:|']
    weights = []
    for side, target in diff.get('candidates', {}).items():
        for rank, row in target.get('ranks', {}).items():
            for boundary, metric in row.get('aggregate', {}).items():
                if metric:
                    quantiles = metric.get('absolute_error_quantiles', {})
                    near = metric.get('near_zero_reference', {})
                    lines.append(f"| {side} / {rank} / {boundary} | {fmt(metric.get('reference_rms'))} | {fmt(metric.get('normalized_rmse'))} | "
                                 + ' / '.join(fmt(quantiles.get(f'p{q}')) for q in (50, 90, 99)) +
                                 f" | {near.get('elements', '未记录')} / {near.get('finite_pairs', '未记录')} | {fmt(near.get('max_abs_error'))} |")
                sample_weights = metric.get('sample_element_weights', [])
                if sample_weights:
                    heaviest = max(sample_weights, key=lambda r: r.get('weight') or 0)
                    macro = metric.get('sample_equal_weight', {}).get('mse', {})
                    weights.append((side, rank, boundary, metric.get('mse'), macro, heaviest))
    lines += ['', '### 按元素汇总与按样本等权汇总', '',
              '原 MSE 按有限配对元素加权：有效元素更多的样本权重更大。等权 MSE 先计算每条样本误差，再对有定义的样本取平均。'
              '两种统计同时保留；发生非有限值时，有限元素与有定义样本数一并展示。', '',
              '| 路径 / rank / 边界 | 原元素加权 MSE | 样本等权 MSE | 有定义样本 / 总样本 | 元素权重最大的样本 | 权重 |',
              '|---|---:|---:|---:|---|---:|']
    for side, rank, boundary, mse, macro, heaviest in weights:
        lines.append(f"| {side} / {rank} / {boundary} | {fmt(mse)} | {fmt(macro.get('value'))} | "
                     f"{macro.get('defined_samples')} / {macro.get('total_samples')} | {cell(heaviest['sample_id'])} | "
                     + (f"{heaviest['weight']:.2%}" if heaviest.get('weight') is not None else '未定义') + ' |')
    if not weights:
        lines += ['', '历史指标未保存新增尺度、分位数或元素权重时显示未记录；需使用保留的原始张量在新目录重算这些新增指标。']
    return lines + ['']


def append_report(root,result):
    oracle=result.get('oracle',{})
    text=(root/'report.md').read_text()
    text=text.replace('未实现：任务评分、A100 参考导入和算子级精度归因。','未实现：任务评分和算子级精度归因。')
    if oracle.get('role')=='consumer':
        text=text.replace('输入只 tokenize 一次','直接复用参考包封存的固定输入')
        text=text.replace('## 最差样本', '## 同设备 off→on 最差样本').replace('### 各 rank 最差样本', '### 同设备 off→on 各 rank 最差样本')
    lines=['# A100 可复用精度参考','']
    if oracle.get('role')=='producer':
        lines += ['基准已封存；两次独立进程前向的全部所选张量逐位相同。此结论仅适用于本次环境与输入。',
                  '',f"基准标识：`{oracle['oracle_id']}`。",'',
                  '[完整基准清单](package/manifest.json) · [重复性验证](repeatability.json)','',
                  '导入时传入 `--oracle <本目录>/package`。包可独立迁移；无需连接本机。权重和安装介质不在包内。','']
    elif oracle.get('role')=='consumer':
        lines += [f"参考标识：`{oracle['oracle_id']}`；差分状态：**{oracle['status']}**。",'',
                  '比较方向：A100 为 reference，被测路径为 candidate。同设备 off/on 对照仍单独保留。',
                  'A100 是工程参考，不是数学真值；无精度阈值或达标结论。依赖、设备及数值实现差异均可能贡献误差。','',
                  '指标沿用 CPU FP64 公式；相对误差分母为 max(abs(A100), 1e-12)，层输出剔除 padding。',
                  'TP 每个 rank 分别对照同一参考，逻辑样本不乘 rank 数；不拼接分片。','']
        diff=read_json(root/'oracle-comparison.json')
        from runtime.metric_quality import description as metric_description
        lines += [side+'/'+str(rank)+'：'+metric_description(row)
                  for side,target in diff.get('candidates', {}).items() for rank,row in target.get('ranks', {}).items()] + ['']
        same = read_json(root/result['comparison']) if isinstance(result.get('comparison'), str) and (root/result['comparison']).is_file() else None
        summary = comparison_summary(diff, same)
        write_json(root/'oracle-summary.json', summary)
        lines += summary_lines(summary, diff)
        lines += interpretation_lines(diff)
        lines += ['## A100 差分完整指标', '',
                  '最大相对误差保留原始公式；参考近零位置的解释请结合上方近零统计和绝对误差。', '',
                  '| 被测路径 | rank | 边界 | MSE | MAE | 最大相对误差 | 最低样本余弦 | 有效/总元素 |',
                  '|---|---|---|---:|---:|---:|---:|---:|']
        trends=[]
        for side,target in diff.get('candidates',{}).items():
            for rank,row in target['ranks'].items():
                for name,m in row['aggregate'].items():
                    lines.append(f"| {side} | {rank} | {name} | {fmt(m['mse'])} | {fmt(m['mae'])} | {fmt(m['relative_max'])} | {fmt(m['sample_cosine_min'])} | {m['valid_elements']}/{m['total_elements']} |")
                for point in row.get('layer_trend', []):
                    metric = row['aggregate'][point['boundary']]
                    trends.append(f"| {side} | {rank} | {point['boundary']} | {fmt(point['mse'])} | {fmt(point['adjacent_mse_change'])} | "
                                  f"{fmt(metric.get('reference_rms'))} | {fmt(metric.get('normalized_rmse'))} |")
        if trends:
            lines += ['', '## 逐层误差变化','',
                      '相邻完整 block 的 MSE 变化是描述值；同时观察参考 RMS 与归一化 RMSE，区分幅度变化与相对误差变化。'
                      '定位误差来源还需受控输入或算子实验。非相邻选择不计算差值。','',
                      '| 路径 | rank | 层 | MSE | 相邻层 MSE 变化 | 参考 RMS | 归一化 RMSE |','|---|---|---|---:|---:|---:|---:|',*trends]
        lines += ['', '## 证据与保存','',
                  f"被测张量保存状态：`{oracle.get('retention',{}).get('status','unknown')}`。",'',
                  '成功后默认只保留完整指标、配置、日志、输入与来源摘要；失败保留现场。`--keep-tensors` 可保留原始输出。',
                  '仅留指标时可复核已保存结果与公式来源，但新增张量指标需重新运行被测路径。','',
                  '[逐样本、最差样本、异常及环境差异](oracle-comparison.json) · [参考清单副本](oracle-manifest.json) · [参考环境](oracle-producer.json)','']
        if (root/'tensor-retention.json').exists(): lines += ['[张量清理清单](tensor-retention.json)','']
        if diff.get('errors') or diff.get('error'): lines += [str(diff.get('errors') or diff['error']),'']
    else:
        lines += ['参考生成未完成，不能作为可复用基准。','']
    (root/'report.md').write_text('\n'.join(lines)+'\n---\n\n'+text)
