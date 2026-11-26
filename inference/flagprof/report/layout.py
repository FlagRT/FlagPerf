# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Reader workspaces and bounded Markdown excerpts shared by both renderers."""
import json
from heapq import nlargest
from itertools import islice

from .fields import display, cell
from .curves import preferred_curve

# Longer prefixes win over their parent workspace.
GROUPS = [
    ('diagnosis', '任务诊断', ('diagnosis',)),
    ('compare', '比较条件', ('comparisons', 'window.comparisons')),
    ('metrics', '指标对照', ('comparison',)),
    ('identity', '比较条件与生成轨迹', ('oracle.identity_diff', 'oracle.generation')),
    ('layers', '逐层数值差异与首差异', ('oracle.layer', 'oracle.first_divergence', 'oracle.propagation')),
    ('operators', '算子差异归因', ('oracle.op_sensitivity',)),
    ('verify', '回放与独立性能', ('oracle.replay', 'oracle.perf_links')),
    ('samples', '样本与配对', ('oracle.samples', 'oracle.alignment')),
    ('threeway', '三路误差对照', ('oracle.three_way', 'oracle.pair_statuses')),
    ('window', '请求窗口与活动构成', ('perf.gap', 'forwards')),
    ('device', '设备热点与归属', ('perf.device',)),
    ('ops', '逐层与算子耗时', ('perf.layer', 'perf.op')),
    ('route', '组件路由与编译', ('route.coverage', 'compile.health', 'compile_issues')),
    ('generation', '生成请求与 KV 缓存', ('generation',)),
    ('group', '各 rank 与通信对照', ('group', 'communication.aligned')),
    ('communication', '通信与设备证据', ('communication',)),
    ('resource', '显存与硬件资源', ('resource', 'memory')),
    ('baseline', '性能基准与稳定性', ('performance',)),
    ('monitor', '节点资源监控', ('monitor',)),
    ('training', '训练步与阶段', ('training',)),
]
MODES = {'diagnosis':'故障诊断','oracle-three-way':'三路精度','oracle':'精度差分',
         'baseline':'性能基准','compare':'性能对照','training':'训练阶段',
         'monitor':'节点监控','profile':'性能分析','group':'多 rank 组'}
MODE_SECTIONS = [('group.ranks','group'), ('monitor.series','monitor'), ('comparisons','compare'),
                 ('oracle.three_way','oracle-three-way'), ('oracle.layer','oracle'),
                 ('oracle.identity_diff','oracle'), ('training','training'), ('diagnosis','diagnosis'),
                 ('performance.baseline','baseline')]
SORT = {'perf.device':'device_work_ns', 'perf.op':'host_ns', 'perf.layer':'inclusive_ns',
        'oracle.propagation':'ratio', 'route.coverage.missing_ops':'forward_non_hit_calls',
        'communication.aligned':'group_seq'}
EXCERPTS = {
    'profile': ('perf.gap','perf.device','perf.layer','route.coverage.missing_ops','compile.health'),
    'training': ('training','training.phases','perf.device'),
    'baseline': ('performance.baseline','performance.perturbations'),
    'compare': ('comparison.metrics','window.comparisons'),
    'oracle': ('oracle.layer','oracle.first_divergence','oracle.propagation','oracle.op_sensitivity.introduced','oracle.replay'),
    'oracle-three-way': ('oracle.three_way',), 'diagnosis': ('diagnosis','diagnosis.coverage'),
    'monitor': ('monitor.series','monitor.findings'), 'group': ('group.ranks','group.counts','group.communication_coverage','communication.aligned','group.control_calls','performance.baseline')}
SUMMARY_COLUMNS = {
    'group.counts':'rank forwards warmup_forwards measure_forwards unknown_forwards steps allreduce allgather allreduce_average',
    'performance.baseline':'rank shape phase observations median_ns p95_ns samples_per_second independent_repeats repeat_cv',
    'comparison.metrics':'rank phase metric baseline candidate delta relative_change',
    'oracle.three_way':'sample scope boundary metric reference_to_baseline reference_to_candidate relative_change',
    'perf.device':'category name events device_work_ns category_share',
    'perf.gap':'kind request_id window_ns all_activity_union_ns blank_ns',
    'diagnosis':'task_label event_label node_event_ns diagnostic_time_ns trigger location',
    'monitor.series':'entity metric unit mean maximum p95 coverage valid missing',
}


CHARTS = {
    'perf.device':'pareto_bars', 'perf.op':'pareto_bars', 'perf.layer':'layer_tree',
    'perf.layer.forwards':'forward_layer_ranking',
    'perf.gap':'window_union', 'route.coverage':'route_share', 'oracle.layer':'oracle_layer_curve',
    'oracle.three_way':'oracle_three_way', 'comparison.metrics':'compare_diverging',
    'performance.baseline':'latency_range', 'generation.steps':'step_latency',
    'communication.aligned':'rank_sequence', 'group.ranks':'rank_cards', 'diagnosis':'fault_timeline',
    'monitor.series':'monitor_grid', 'memory.series':'memory_curve', 'resource.series':'resource_curve',
    'resource':'resource_statistics', 'resource.stages':'resource_statistics'}
FILTER_KEYS = set('dev rank sample level metric category route_decision entity group comparison_status rule state introduced equal instrumentation op_verdict'.split())
SHORTCUTS = [('category','compute','仅看计算'),('category','memcpy','仅看搬运'),('category','comm','仅看通信'),
             ('route_decision','fallback','仅看回退'),('introduced','true','仅看同输入引入'),
             ('op_verdict','进入算子前输入已不同','仅看输入已不同的调用'),
             ('equal','false','仅看不同项'),('comparison_status','not_comparable','仅看不可比项')]


def filters(ident,keys):
    scope = keys & FILTER_KEYS
    if ident.startswith('diagnosis') and 'task_label' in keys:scope.add('task_label')
    for dim,fields in [('step',('generation_step','step')),('shape',('input_shape','shape')),
                       ('sid',('sid','sid_cmp','sid_ref')),('level',('levels',)),('phase',('phase',))]:
        if keys.intersection(fields):scope.add(dim)
    if ident.startswith(('performance.','comparison.metrics')) and 'scope' in keys:scope.add('scope')
    if 'scopes' in keys or (ident.startswith('perf.layer') and 'name' in keys) or (
            ident.startswith(('oracle.','perf.op','perf.device')) and 'scope' in keys and ident!='oracle.identity_diff'):
        scope.add('layer')
    if ident.startswith('oracle.') and keys.intersection(('introduced','digest_equal')):scope.add('diff_status')
    if ident in ('perf.gap','resource.series'):scope.add('phase')
    return sorted(scope)


def compact(value, key=''):
    if isinstance(value,list) and len(value)>4:
        return str(len(value))+' 项 · '+display(value[:3],key)+'…'
    return display(value,key)


def decorate(model):
    meta = model.meta
    ids = {s['id'] for s in model.sections}
    if 'diagnosis.coverage' in ids:
        from .diagnosis import timeline_axis
        meta['diagnosis_axis'] = timeline_axis(model.rows('diagnosis.coverage'), model.rows('diagnosis'),
                                               meta.get('diagnosis_session_ns'))
    mode = meta.get('mode') or next((m for k,m in MODE_SECTIONS if k in ids), 'profile')
    meta['mode'] = mode
    if mode=='training':
        meta['title']='训练步与阶段观测'
    if mode=='diagnosis' and meta['title']=='推理运行结果':
        meta['title']='任务执行与告警'
    context = []
    for fact in meta['facts']:
        label, value = fact['label'], fact['value']
        field = fact.get('field', '')
        if isinstance(value,dict) and 'ref' in value and 'cmp' in value:
            def fact_text(v):
                if field == 'samples' and isinstance(v, list):
                    return str(len(v))+' 个（'+display(v[:3], 'samples')+('…' if len(v)>3 else '')+'）'
                if field in ('inputs_sha256','oracle_id') and isinstance(v,str) and len(v) == 64:
                    other = value['cmp'] if v == value['ref'] else value['ref']
                    if v == other or not isinstance(other,str) or (v[:12],v[-6:]) != (other[:12],other[-6:]):
                        return v[:12]+'…'+v[-6:]
                if label == 'FlagGems' and isinstance(v, dict) and v.get('enabled') is not None:
                    return display(v['enabled'])
                return compact(v, field)
            context.append(dict(label=label,field=field,left=fact_text(value['ref']),right=fact_text(value['cmp']),
                                different=bool(fact.get('status_text')) or value['ref']!=value['cmp'],
                                status_text=fact.get('status_text')))
        else:
            context.append(dict(label=label,field=field,value=compact(value, field)))
    if not any(f['label'] in ('设备','参考侧','来源 1') for f in context) and mode=='profile':
        context.append(dict(label='设备与模型身份',value='历史记录未提供'))
    for row in model.rows('comparisons'):
        for diff in row.get('differences') or ():
            if isinstance(diff,dict):
                field = diff.get('field') or ''
                key = field.rsplit('.',1)[-1]
                context.append(dict(label=display(field,'field'),left=compact(diff.get('left'),key),
                                    right=compact(diff.get('right'),key),different=True))
    meta['context_summary'] = sorted(context,key=lambda f:not f.get('different'))
    meta['banners'] = [f for f in meta['findings'] if f['level']=='fault']
    meta['credibility'] = [dict(label=f['title'],detail=f['detail'],section=f.get('section'))
                           for f in meta['findings'] if f['level']=='coverage' or f.get('section') in
                           ('perf.gap','diagnosis.coverage','oracle.pair_statuses','performance.perturbations')][:4]
    if not meta['credibility']:
        meta['credibility'] = [dict(label='观测范围',detail=n,section='availability')
                               for n in meta['notices'] if n][:2]
    kpis = []
    for finding in meta['findings']:
        if meta['banners'] and finding == meta['banners'][0] and finding['section']=='diagnosis':
            continue
        sid, ordinal = finding.get('section'), finding.get('row_index')
        row = {}
        if ordinal is not None and sid:
            record = model.db.execute('select payload from report_rows where section=? and ordinal=?',(sid,ordinal)).fetchone()
            if record:
                row = json.loads(record[0])
        preferred = {'performance.baseline':'median_ns','comparison.metrics':'relative_change',
                     'route.coverage.missing_ops':'forward_non_hit_calls','compile.health':'count',
                     'perf.device':'device_work_ns','perf.gap':'window_ns','oracle.layer':'max_abs',
                     'oracle.three_way':'relative_change','communication.aligned':'cross_rank_difference_ns'}
        key = preferred.get(sid)
        value = row.get(key)
        if isinstance(value,dict):
            value=value.get('value')
        kpis.append(dict(finding,metric_value=display(value,key) if value is not None else '',
                         eyebrow=' · '.join(compact(row[k]) for k in ('shape','phase') if row.get(k) is not None)))
    meta['kpis'] = kpis
    descriptions={s['id']:s.get('description') for s in model.sections}
    curve_order = {}
    for s in model.sections:
        ident = s['id']
        if ident == 'diagnosis' and 'diagnosis_axis' in meta:
            origin = meta['diagnosis_axis']['origin']
            note = {'task': '横轴从本任务的诊断启动（0 s）算起。',
                    'node': '各任务共用节点监控会话的时间刻度。',
                    'first_task': '各任务共用主机时钟，横轴从最早一次诊断启动算起。表中时刻相对各自任务启动。',
                    'unmapped': '各任务缺少共同的时钟基准。可在事件详情中核对各自的时间记录。'}[origin]
            s['description'] = note + '选择事件可查看对应记录和证据。'
            time_field = 'node_event_ns' if origin == 'node' else 'diagnostic_time_ns'
            s['columns'] = list(dict.fromkeys(time_field if k in ('node_event_ns','diagnostic_time_ns') else k
                                              for k in s['columns']))
        group,title,_ = next((g for g in GROUPS if any(ident==p or ident.startswith(p+'.') for p in g[2])),
                             (ident,s['title'],()))
        if group == 'identity' and 'oracle.generation' not in ids:
            title = '比较条件'
        parent = ident.removesuffix('.quality')
        if group in ('layers','operators') and parent != ident and parent in ids:
            s['quality_for'] = parent
        if ident.rpartition('.')[0] in descriptions and descriptions[ident.rpartition('.')[0]]==s.get('description'):
            s['description']=''
        if 'filter_scope' not in s:
            keys=set(s['columns'])
            for row in model.rows(ident):keys.update(row)
            s['filter_scope']=filters(ident,keys)
        s['quick_filters']=[dict(field=k,value=v,label=label) for k,v,label in SHORTCUTS if k in s['filter_scope']]
        if ident in CHARTS:s['chart']=dict(type=CHARTS[ident])
        if ident in ('resource.series', 'memory.series'):
            rank, series = preferred_curve(model.rows(ident))
            s['chart']['default_series'] = series
            curve_order[ident] = rank
        if ident in ('resource', 'resource.stages'):
            s['filter_scope'] = sorted(set(s['filter_scope']) | {'phase'})
        if ident=='diagnosis.coverage' and 'diagnosis' not in ids:s['chart']=dict(type='fault_timeline')
        s.update(group=group,group_title=title,tab_label=s['title'],default_sort=SORT.get(ident),
                 default_desc=ident not in ('diagnosis','communication.aligned'),preserve_order=ident not in SORT)
    # Stable ordering also retains the source ordinal of every record.
    first_groups = {'profile':['window','device'], 'oracle':['identity','layers','operators'],
                    'oracle-three-way':['threeway'], 'baseline':['baseline'],'group':['group'],
                    'compare':['compare','metrics'],'monitor':['monitor'],'training':['training']}.get(mode,[])
    priority = (['diagnosis'] if meta['banners'] else []) + first_groups
    order = list(dict.fromkeys(priority+[s['group'] for s in model.sections]))
    layer_tabs = ['oracle.layer','oracle.first_divergence','oracle.propagation']
    def tab_order(s):
        if s['group'] == 'resource':
            return curve_order.get(s['id'], float('inf'))
        if s['group'] == 'layers':
            return layer_tabs.index(s['id']) if s['id'] in layer_tabs else len(layer_tabs)
        return int(s['id'] in ('diagnosis.coverage','diagnosis.evidence'))
    model.sections.sort(key=lambda s:(order.index(s['group']),tab_order(s)))


def excerpts(model):
    sections = {s['id']:s for s in model.sections}
    for ident in EXCERPTS[model.meta['mode']]:
        if ident not in sections:
            continue
        section = sections[ident]
        rows = iter(model.rows(ident))
        if ident=='oracle.three_way':
            rows = (r for r in rows if r.get('scope')=='<root>' and r.get('boundary')=='output')
        elif ident=='oracle.layer':
            rows = (r for r in rows if r.get('scope')=='<root>')
        key = 'cross_rank_difference_ns' if ident=='communication.aligned' else SORT.get(ident)
        if (key and section['default_desc']) or ident=='communication.aligned':
            result = nlargest(10,rows,key=lambda r:r[key] if isinstance(r.get(key),(float,int)) else -float('inf'))
        else:
            result = list(islice(rows,10))
        if ident=='perf.device':
            totals={}
            for r in model.rows(ident):
                category=r.get('category')
                totals[category]=totals.get(category,0)+(r.get('device_work_ns') or 0)
            for r in result:
                total=totals[r.get('category')]
                if total and r.get('device_work_ns') is not None:r['category_share']=r['device_work_ns']/total
        if result:
            columns = SUMMARY_COLUMNS.get(ident,'').split() or section['columns'][:7]
            if ident == 'diagnosis':
                columns = [k for k in columns if k not in ('diagnostic_time_ns','node_event_ns') or k in section['columns']]
            yield section, [k for k in columns if any(cell(r,k) is not None or k in r.get('_field_notes',{}) for r in result)], result
