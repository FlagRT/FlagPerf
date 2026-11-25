# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Reader questions answered from saved results, with stable source-row targets.

No thresholds or causal diagnoses are introduced here. These observations keep
the analyzer's population, rank, window and comparison status.
"""
from .fields import display, LABELS
from .model import read_document


def scalar(value):
    return value.get('value') if isinstance(value, dict) else value


def identity_text(identity):
    parts = []
    for key, label in (('model', '模型'), ('device', '设备'), ('dtype', '精度'),
                       ('comm_backend', '通信后端'), ('samples', '样本')):
        if identity.get(key) is not None:
            value = identity[key]
            shown = str(len(value))+' 个 · '+display(value[:3])+'…' if key=='samples' and isinstance(value,list) and len(value)>4 else display(value)
            parts.append(label+' '+shown)
    gems = identity.get('flaggems')
    if isinstance(gems, dict):
        parts.append('FlagGems '+display(gems.get('enabled')))
    return '；'.join(parts) or '来源未记录这些条件'


def comparison_sources(doc):
    result = {}
    for i, row in enumerate(doc.get('sources', ())):
        identity = row.get('context', {}).get('workload_identity', {})
        if row.get('report'):
            result[row['report']] = '来源 '+str(i+1)+' · '+' / '.join(display(identity[k]) for k in ('comm_backend','device') if identity.get(k))
    return result


def comparison_facts(model, doc, source):
    """Return whether a comparison has its own visible identities."""
    if doc.get('schema') == 'device-comparison/1':
        for i, row in enumerate(doc.get('sources', ())):
            model.meta['facts'].append(dict(label='来源 '+str(i+1), value=identity_text(
                row.get('context', {}).get('workload_identity', {}))))
        return bool(doc.get('sources'))
    if doc.get('schema') != 'oracle-three-way/1':
        return False
    roles = [('参考侧', 'reference-baseline', 'ref'),
             ('基线侧', 'reference-baseline', 'cmp'),
             ('候选侧', 'reference-candidate', 'cmp')]
    for label, pair, side in roles:
        identity = {}
        path = source/pair/'result.json' if source else None
        if path and path.is_file():
            child = read_document(path, model)
            if label=='参考侧':
                layers=child.get('results',{}).get('oracle.layer',{}).get('data',())
                model.meta['boundary_order']=[dict(scope=r.get('scope'),boundary=r.get('boundary'),order=r['order'])
                                              for r in layers if isinstance(r.get('order'),int)]
            for row in child.get('results', {}).get('oracle.identity_diff', {}).get('data', ()):
                if row.get(side) is not None:
                    identity[row['field']] = row[side]
        model.meta['facts'].append(dict(label=label, value=identity_text(identity)))
    return True


def performance_comparison(model, doc):
    sources = list(doc.get('sources', ()))
    names = comparison_sources(doc)
    for index, row in enumerate(model.rows('comparison.metrics')):
        if row.get('metric') != 'mean_ns' or row.get('scope') not in ('request', 'forward', 'group_request') or row.get('delta') is None:
            continue
        rank = '组请求' if row['scope'] == 'group_request' else 'rank '+str(row.get('rank'))
        candidate = next((s for s in sources if s.get('report') == row.get('source')), {})
        counts = []
        repeats = []
        for src in [sources[0] if sources else {}, candidate]:
            group = next((g for g in src.get('runner', {}).get('groups', ())
                if all(g.get(k) == row.get(k) for k in ('rank','scope','phase','generation_step','shape'))), {})
            counts.append(str(group.get('observations', '未记录')))
            repeats.append(group.get('independent_repeats', 0))
        baseline = identity_text(sources[0].get('context', {}).get('workload_identity', {})) if sources else '基准来源'
        detail = '均值 '+display(row['baseline'], 'duration_ns')+' → '+display(row['candidate'], 'duration_ns')
        detail += '，变化 '+display(row.get('relative_change'), 'relative_change')+'（'+display(row['delta'], 'duration_ns')+'）。'
        detail += '有效观测 '+counts[0]+' → '+counts[1]+' 次；'
        detail += ('按各 rank 的 '+row['scope']+' 计时，组墙钟另列。' if row['scope'] != 'group_request' else '范围为组请求墙钟。')
        if min(repeats) < 3:
            detail += '独立进程重复依据不足，以上为本批次观测变化。'
        p95 = next((p for p in model.rows('comparison.metrics') if p.get('metric') == 'p95_ns'
                    and all(p.get(k) == row.get(k) for k in ('source','rank','scope','phase','generation_step','shape'))), {})
        if p95.get('delta') is not None:
            detail += ' p95 '+display(p95.get('baseline'),'duration_ns')+' → '+display(p95.get('candidate'),'duration_ns')+'。'
        detail += ' 比较：'+baseline+' → '+names.get(row.get('source'), '候选来源')+'。'
        model.finding(rank+' '+row['scope']+' 平均时延变化', detail,
                      'comparison.metrics', level='result', row_index=index)
    differences = []
    for row in doc.get('comparisons', ()):
        for change in row.get('differences', ()):
            field=str(change.get('field')).removeprefix('workload_identity.')
            differences.append(LABELS.get(field,field)+'：'+display(change.get('left'))+' → '+display(change.get('right'))+
                               ('（已声明）' if change.get('declared') else '（未声明）'))
    if differences:
        model.meta['facts'].append(dict(label='变化条件', value='；'.join(dict.fromkeys(differences))))


def accuracy(model):
    for index, row in enumerate(model.rows('oracle.layer')):
        if row.get('scope') != '<root>' or row.get('boundary') not in ('output', 'pooled'):
            continue
        max_abs = row.get('max_abs') or row.get('worst', {}).get('max_abs')
        l2 = row.get('relative_l2') or row.get('worst', {}).get('relative_l2')
        model.finding('模型 '+str(row['boundary'])+' 边界的误差',
            '覆盖 '+str(row.get('samples', '未记录'))+' 个配对样本，证据 '+display(row.get('levels'))+
            '；最大绝对误差 '+display(max_abs)+'；相对 L2 '+display(l2)+
            '。最差指标分别保留对应样本；数值差异是否满足业务要求需结合任务精度标准。',
            'oracle.layer', level='result', row_index=index)
    for index, row in enumerate(model.rows('oracle.three_way')):
        if row.get('scope') != '<root>' or row.get('boundary') != 'output' or row.get('metric') not in ('max_abs', 'relative_l2'):
            continue
        metric = {'max_abs':'最大绝对误差', 'relative_l2':'相对 L2'}[row['metric']]
        detail = '样本 '+str(row.get('sample'))+'；'+('rank '+str(row['rank'])+'；' if row.get('rank') is not None else '')+'证据 '+display(row.get('level'))+'。'
        detail += '相对参考侧：基线 '+display(row.get('reference_to_baseline'))+' → 候选 '+display(row.get('reference_to_candidate'))+'。'
        if row.get('delta') is not None:
            change = row['delta']
            relative = row.get('relative_change')
            detail += ('误差增大 ' if change > 0 else '误差减小 ' if change < 0 else '误差不变 ')+display(abs(relative) if relative is not None else None, 'relative_change')+'。'
        else:
            detail += '变化量未计算：'+str(row.get('reason'))+'。'
        detail += '基线与候选直接差异 '+display(row.get('baseline_to_candidate'))+'；不同指标分别判断。'
        model.finding('最终 output：'+metric, detail, 'oracle.three_way', level='result', row_index=index)
    for index, row in enumerate(model.rows('oracle.propagation')):
        if row.get('ratio') is not None:
            model.finding('误差传播中最大的相对 L2 倍数线索',
                '样本 '+str(row.get('sample'))+'；'+str(row.get('before'))+' → '+str(row.get('after'))+
                '，'+display(row['ratio'])+' 倍。这里只定位变化边界；输入差异与同输入引入的差异需继续核对。',
                'oracle.propagation', row_index=index)
            break  # Producer orders largest_jumps before zero-to-nonzero transitions.
    replay = list(model.rows('oracle.replay'))
    for index, row in enumerate(replay):
        if not row.get('valid'):
            continue
        metric = row.get('metrics', {})
        self_results = row.get('self_reproduction', {})
        model.finding('回放实际结果：'+str(row.get('call_key')),
            '原输出复现：'+ '；'.join(side+' 最大绝对误差 '+display(v.get('max_abs')) for side,v in self_results.items())+
            '。同输入跨侧最大绝对误差 '+display(metric.get('max_abs'))+'，不同元素 '+str(metric.get('differing_elements'))+
            ' / 有效元素 '+str(metric.get('valid_elements'))+'。这是隔离调用的观测。',
            'oracle.replay', row_index=index)
        if index >= 2:
            break


def windows_and_resources(model):
    for index, row in enumerate(model.rows('perf.gap')):
        if row.get('kind') != 'official_capture':
            continue
        model.finding('采集窗口 '+display(row.get('window_ns'), 'window_ns'),
            '已记录设备活动并集 '+display(row.get('all_activity_union_ns'), 'duration_ns')+
            '；未被这些活动覆盖的时间 '+display(row.get('blank_ns'), 'duration_ns')+
            '（'+display(row.get('blank_ratio'), 'blank_ratio')+'）。这是完整采集窗口，不能作为正式请求时延；空白也不直接等于主机开销或可优化时间。',
            'perf.gap', level='result', row_index=index)
    seen = set()
    for index, row in enumerate(model.rows('resource')):
        if row.get('mean') is None or row.get('name') != 'run' or row.get('metric') not in ('power.draw', 'memory.used'):
            continue
        key = (row.get('entity'), row.get('metric'))
        if key in seen:
            continue
        seen.add(key)
        maximum = '区间均值的最大值' if row.get('maximum_kind') == 'window_mean_maximum' else '记录最大值'
        unit=row.get('unit')
        def measured(value):
            return display(value,'bytes') if unit=='B' else display(value)+' '+str(unit)
        model.finding(str(row.get('entity'))+' · '+{'power.draw':'功率','memory.used':'设备显存占用'}[row['metric']],
            '任务窗口均值 '+measured(row['mean'])+'；'+maximum+' '+measured(row.get('maximum'))+
            '；时间覆盖 '+display(row.get('coverage'), 'coverage_ratio')+'。'+
            ('窗口为近似对应，时间相关性不自动说明任务因果。' if row.get('approximate_window') else '指标对象与功率/内存域随来源记录。'),
            'resource', row_index=index)
