# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Evidence-led result views shared by all public report entrypoints."""
from collections import Counter
import json
import hashlib
from pathlib import Path
import shutil
import re

from .fields import PANELS, LABELS, VALUES, display, prepare_fields
from .model import opened_model, read_document, digest


FIELD_DISPLAY_VERSION = 2


DESCRIPTIONS = {
    'perf.op':'全采集窗口内可用的调用记录。主机区间含探针开销；设备工作量为关联事件时长之和，含重叠。',
    'perf.layer':'全采集窗口。含子层耗时与独占耗时分列；父子层不能相加。设备工作量不等于层的墙钟时延。',
    'perf.device':'来源窗口内的设备记录，按名称、签名和关联状态分组。工作量含事件重叠；跨厂商名称不直接配对。',
    'perf.gap':'每个窗口独立统计：全部活动并集＋空白＝窗口长度。各类活动可能重叠，不能相加。',
    'route.coverage':'实际路由按签名保留。策略排除、组合算子与视图类分别阅读；非命中次数不等于性能损失。',
    'compile.health':'仅统计已经记录的编译观测。缓存不可观测时保留原因；编译未记录不等于没有编译。',
    'oracle.layer':'每个数值边界按逻辑样本配对。最差指标保留样本和位置；抽样结果仅描述抽样位置。',
    'oracle.op_sensitivity':'同配置、完整相同输入上的输出差异可以判为 introduced。上游输入已不同的调用另列，不能仅凭放大倍数判定根因。',
    'oracle.perf_links':'精度与性能来自独立运行，只按同侧身份与精确签名连接。性能数据不作为精度采集的逐事件计时。',
    'oracle.replay':'先检查各侧原输入能否复现封存输出，再阅读相同输入的跨侧差异；最终模型输出影响需独立证据。',
    'memory':'各来源、allocated / reserved / device-used 分开。峰值为观测峰值；分配事件量不替代总占用。',
    'resource':'指标按观测对象、窗口和单位统计。分辨率不足时不给出统计；时间对应关系不自动证明因果。',
    'training':'按训练步登记的主机区间。各阶段的主机区间与设备活动分别计时。',
    'generation':'step 0 为 prefill，后续为 decode。采集中的推进耗时与无插桩 runner 基准分别阅读。',
    'communication':'逻辑调用与设备表示分别计数；设备活动并集与时长和分别解释。主机返回不证明设备执行完成。',
}

EXTRAS = {
    'phases':('按阶段的设备工作量',('dev','phase','category','events','device_work_ns')),
    'headline':('按已选前向汇总', None),
    'groups':('关联与类别覆盖', ('dev','category','link_status','route_decision','events','missing_durations','device_work_ns')),
    'op_groups':('签名对应的设备工作量', ('sid','dev','category','route_decision','events','missing_durations','device_work_ns')),
    'layer_groups':('层对应的设备工作量', ('name','dev','category','route_decision','events','missing_durations','device_work_ns')),
    'introduced':('同输入引入的差异', PANELS['oracle.op_sensitivity'][1]),
    'introduced_routes':('引入差异的预期实现', ('impl','occurrences','scopes','ulp_histogram')),
    'amplification':('误差放大观测', PANELS['oracle.op_sensitivity'][1]),
    'steps':('逐生成步', ('request_id','generation_step','phase','host_duration_ns','kv_length_before','input_shape')),
    'kv':('KV 生命周期', ('event','request_id','generation_step','length','bytes')),
    'streams':('设备流与空白区间', ('window_id','kind','step','dev','stream','window_ns','all_activity_union_ns','blank_ns')),
    'device_calls':('通信调用的设备证据', ('group','group_seq','rank','op','device_link_status','device_union_ns','device_sum_ns')),
    'transfers':('数据搬运', ('dev','kind','events','bytes')),
    'allocations':('分配与释放观测', ('dev','component','native_owner','allocations','releases','allocated_bytes','released_bytes','release_not_observed')),
    'stages':('按阶段的资源统计', ('name','entity','metric','unit','mean','maximum','p95','coverage','status')),
    'energy':('能耗与功率域', ('name','entity','joules','coverage','joules_per_token','power_domain','token_reason')),
    'memory':('分配器外显存估算', ('entity','kind','ts','observed_bytes','preexisting_usage','bytes','status','reason')),
    'restrictions':('资源限制观测', ('name','entity','reason','window','restricted_ns','observed_window','conclusion')),
    'frequency':('观测频率区间', ('entity','window','samples','frequency_min_hz','frequency_max_hz','evidence')),
    'missing_ops':('回退主表：非命中调用与处理原因', ('name','kind','forward_non_hit_calls','non_hit_calls','non_hit_ratio','host_ns','reasons','impls')),
}


def rows(value):
    return value if value is not None and not isinstance(value, (str, dict)) else ()


def add_table(model, ident, title, source, columns=None, description='', status='ok', kind='table'):
    present = set()
    for original in rows(source):
        row = dict(original) if isinstance(original, dict) else dict(value=original)
        if ident == 'oracle.first_divergence':
            first = row.get('first_inequal')
            row = dict(first or {}, alignment_gaps=row.get('alignment_gaps'),
                       source_result=original)
        prepare_fields(ident, row, model.meta.get('forward_only',False))
        if 'scope_samples' in row and not row.get('scopes'):
            row['scope_evidence']=row['scope_samples']
        if 'observed_peak_bytes' in row:
            row['peak_bytes'] = row['observed_peak_bytes']
        if 'worst' in row:
            for key in ('max_abs','relative_l2','cosine_similarity'):
                if key in row['worst']:
                    row[key] = row['worst'][key]
        if isinstance(row.get('expected_route'), dict):
            row['expected_route_detail'] = row['expected_route']
            row['expected_route'] = row['expected_route'].get('impl',row['expected_route'].get('status'))
        present.update(row)
        model.add(ident, row)
    selected = [k for k in columns or () if k in present]
    if not selected:
        selected = [k for k in LABELS if k in present][:8]
    if not selected:
        selected = sorted(present)[:6]
    from .layout import filters
    section=model.section(ident, title, selected, description=description, status=status, kind=kind)
    section['filter_scope']=filters(ident,present)
    return model.rows(ident)


def _scalar(value):
    return value.get('value') if isinstance(value, dict) else value


def _facts(model, doc, source):
    from .insights import comparison_facts
    comparison_identity = comparison_facts(model, doc, Path(source) if source else None)
    context = doc.get('report_context', {})
    facts = dict(context.get('identity') or {})
    context_run = context.get('run', {})
    if doc.get('runner_summary'):
        identities=[r.get('identity',{}) for r in doc['runner_summary'].get('sources',[])]
        devices=sorted(set(str(r['device']) for r in identities if r.get('device')))
        if devices:facts['device']=devices
    if context_run:
        facts['run_id'] = context_run.get('run_id')
        facts['execution_status'] = context_run.get('state')
    diagnostic = doc.get('diagnosis', {}).get('coverage', {}).get('reference', {})
    if diagnostic.get('diagnostic_identity'):
        facts.setdefault('run_id', diagnostic['diagnostic_identity'].get('run_id'))
    if diagnostic.get('devices'):
        facts.setdefault('device', diagnostic['devices'])
    for panel in doc.get('results', {}).values():
        coverage = panel.get('coverage', {})
        identity = coverage.get('workload_identity') or {}
        for key in ('model','model_files','device','dtype','samples','attention','pooling','flaggems',
                    'software','parallelism','comm_backend'):
            if key not in facts and key in identity:
                facts[key] = identity[key]
        window = coverage.get('window_identity') or {}
        if window:
            facts.setdefault('capture_window', window)
    for row in rows(doc.get('results', {}).get('oracle.identity_diff', {}).get('data')):
        if row.get('field') in ('device','dtype','samples','flaggems','attention','pooling'):
            facts[row['field']] = dict(ref=row.get('ref'),cmp=row.get('cmp'))
    names = dict(model='模型',model_files='模型文件身份',device='设备',dtype='精度',samples='输入样本',
                 attention='Attention',pooling='输出语义',flaggems='FlagGems',software='软件版本',
                 parallelism='并行配置',comm_backend='通信后端',capture_window='采集窗口与配置',
                 run_id='运行标识',execution_status='运行状态')
    for key, title in names.items():
        if facts.get(key) is not None:
            value=facts[key]
            if key=='model_files' and isinstance(value,dict):
                fingerprint=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:16]
                add_table(model,'identity.files','模型文件身份',({'name':k,'value':v} for k,v in value.items()),
                          ('name','value'),'用于核对模型文件一致性；不是模型名称。',kind='quality')
                value=str(len(value))+' 个文件 · 身份 '+fingerprint
            elif key in ('capture_window','software') and isinstance(value,dict):
                value={k:v for k,v in value.items() if v is not None}
            model.meta['facts'].append(dict(label=title,value=value))
    if not facts and not comparison_identity and doc.get('schema')!='monitor-report/1':
        model.meta['notices'].append('此历史报告未保存完整任务身份；页面仅解释已保存的结果。')
    elif facts.get('model_files') and not facts.get('model'):
        model.meta['facts'].insert(0,dict(label='模型名称',value='历史记录未提供；按模型文件身份核验'))
    model.meta['source'] = 'result.json'
    model.meta['links'].extend([dict(label='完整机器结果',href='result.json')])


def _diagnosis(model, data, node=None):
    if not data:
        return
    observations = list(rows(data.get('progress_coverage')))
    findings = rows(data.get('findings'))
    valid = sum(row.get('registered_execution',0) for row in observations)
    gaps = sum(row.get('source_gap',0)+row.get('probe_gap',0) for row in observations)
    status = data.get('status','unavailable')
    source_findings = []
    evidence = {tuple(r['identity']): r['record'] for r in rows(data.get('evidence'))}
    starts = {(r.get('run_id'), r['start_clock']): r['started_ns'] for r in observations
              if r.get('start_clock') and r.get('started_ns') is not None}
    for record in evidence.values():
        detail = record.get('detail', {})
        if detail.get('started_ns') is not None:
            starts[(detail.get('identity', {}).get('run_id'), record.get('clk'))] = detail['started_ns']
    # Diagnostic findings are bounded by the existing producer/session contract.
    for row in findings:
        row = dict(row)
        origin = row.get('task_relative_ns')
        detected = row.get('first_detected',row.get('observed_at'))
        # Use the same mapped event's absolute and task-relative timestamps.
        if origin is not None and detected is not None and row.get('window') and row['window'][0] is not None:
            row['first_detected_relative_ns'] = origin+detected-row['window'][0]
        detail=row.get('detail') or {}
        events = [evidence[tuple(ref)] for ref in row.get('evidence', ()) if tuple(ref) in evidence]
        # The monitor wrapper contains the original business event. Unwrap only
        # this documented shape, retaining both records in the evidence panel.
        events = [event.get('detail', {}).get('record', event) for event in events]
        events = [event for event in events if event.get('ts') is not None]
        if events:
            event = min(events, key=lambda event: event['ts'])
            row['event_time_ns'] = event['ts']
            row['event_clock'] = event.get('clk')
            row['event_detail'] = event.get('detail', {})
            started = starts.get((row.get('run_id'), event.get('clk'))) if row.get('run_id') and event.get('clk') else None
            if started is not None:
                row['occurrence_task_relative_ns'] = event['ts']-started
            if not detail:
                detail = row['event_detail']
        row['trigger']=detail.get('message') or {k:detail[k] for k in ('elapsed_ns','timeout_ns','budget_ns','exit_code','returncode','signal','error','observation','name','group','group_seq','asynchronous') if k in detail}
        if row.get('rule')=='oom.device' and isinstance(row['trigger'],str):
            requested=re.search(r'Tried to allocate ([\d.]+\s+(?:[KMGT]iB|bytes))',row['trigger'])
            free=re.search(r'([\d.]+\s+(?:[KMGT]iB|bytes)) free',row['trigger'])
            if requested and free:
                row['allocation_summary']='申请 '+requested.group(1)+'；当时空闲 '+free.group(1)+'。完整异常原文见对应证据。'
        if row.get('window') and row['window'][0] is not None and row.get('observed_at') is not None:
            row['observed_since_onset_ns']=row['observed_at']-row['window'][0]
        settings=data.get('rule_settings',{}).get(row.get('config_version'),{})
        threshold={'hang.no_progress':settings.get('progress_timeout_s'),'comm.timeout':settings.get('comm_timeout_s')}.get(row.get('rule'))
        row['configured_timeout_ns']=threshold*1000000000 if threshold is not None else None
        if row.get('rule')=='hang.heartbeat':row['configured_heartbeat_misses']=settings.get('heartbeat_misses')
        if not row['trigger']:row['trigger']='触发过程见引用事件；此历史记录未保存完整触发参数'

        row['location']=detail.get('active') or detail.get('scope_path') or detail.get('name') or '事件未记录活动层或算子'
        row['next_step']={
            'oom.device':'先核对分配申请与当时可用显存，再检查任务输入规模及已有占用；有活动层记录时继续定位该层。',
            'hang.no_progress':'核对最后一次有效进度、超时阈值和该线程最后活动。',
            'hang.heartbeat':'核对目标进程状态、最后心跳和来源读取状态，再区分目标停顿与采集缺口。',
            'comm.timeout':'核对通信域、序号、各 rank 退出与设备通信证据。',
            'comm.exception':'核对异常消息和同序号的其他 rank。',
            'proc.exit':'核对退出码、信号和先前事件；退出记录与根因分别解释。',
        }.get(row.get('rule'),'从引用的原始事件核对触发值、观测范围和最后活动。')
        source_findings.append(row)
    from .diagnosis import by_task, conclusion, node_axis
    tasks, timing = by_task(observations, source_findings)
    model.meta['diagnosis_layout_version'] = 4
    duration = node_axis(node, evidence.values(), tasks, source_findings) if node is not None else None
    if duration is not None:
        model.meta['diagnosis_session_ns'] = duration
    add_table(model, 'diagnosis.coverage', '任务与观测范围', tasks,
              ('task_label','exit_summary','alert_count','registered_execution','source_gap','probe_gap','run_id'),
              '每个运行标识对应一个任务。有效执行观测按心跳计数。')
    add_table(model,'diagnosis','告警与状态变化',source_findings,
              ('task_label','event_label','node_event_ns','diagnostic_time_ns','rank'),
              status=status)
    from flagprof.monitor.diagnostic_report import anchor
    add_table(model,'diagnosis.evidence','故障的原始观测记录',
              (dict(r,anchor=anchor(r['identity'])) for r in rows(data.get('evidence'))),
              ('identity','record'), '证据引用随报告保存，展开记录可核对触发条件与时间。')
    if source_findings or valid or any('exit_summary' in t for t in tasks):
        conclusion(model, tasks, source_findings, timing)
    if status != 'unavailable' and valid == 0:
        model.finding('本次没有有效执行进度观测',
                      '诊断结果可读取，但有效执行观测为 0。'+
                      ('未记录告警不能据此判断执行过程正常。' if not source_findings else '故障事件与进度观测分别解释。'),
                      'diagnosis.coverage','coverage')
    elif gaps:
        model.finding('执行观测存在缺口', f'来源或探针记录了 {gaps} 次采集缺口。', 'diagnosis.coverage','coverage')
    elif status == 'unavailable':
        model.meta['notices'].append('本次未提供业务故障诊断：'+display(data.get('reason','未采集诊断来源'))+'。无法据此判断执行期间是否发生业务故障。')


def build(model, doc, source=None):
    schema_modes={'device-comparison/1':'compare','oracle-three-way/1':'oracle-three-way','monitor-report/1':'monitor',
                  'oracle-group/1':'group','flagprof-group-report/1':'group','runner-result/1':'baseline'}
    if doc.get('schema') in schema_modes:model.meta['mode']=schema_modes[doc['schema']]
    model.meta['field_display_version'] = FIELD_DISPLAY_VERSION
    identity = ((doc.get('report_context') or {}).get('identity') or
                (doc.get('runner_summary') or {}).get('condition') or {})
    task = identity.get('task')
    model.meta['forward_only'] = (task in ('embedding','rerank','ctr') or
                                  not task and bool(identity.get('pooling')) and not identity.get('generation'))
    _facts(model,doc,source)
    if doc.get('schema')!='monitor-report/1' or (doc.get('diagnosis') or {}).get('findings') or (doc.get('diagnosis') or {}).get('progress_coverage'):
        _diagnosis(model,doc.get('diagnosis'),doc if doc.get('schema')=='monitor-report/1' else None)
    panels = doc.get('results', {})
    oracle = any(key.startswith('oracle.') for key in panels)
    if oracle:
        model.meta['title'] = '精度对照结果'
        order = ('oracle.identity_diff','oracle.generation','oracle.first_divergence',
                 'oracle.op_sensitivity','oracle.layer','oracle.perf_links','oracle.replay')
    else:
        order = ('perf.device','perf.op','perf.layer','perf.gap','generation',
                 'route.coverage','compile.health','communication','memory','resource')
    for ident in list(order)+[key for key in panels if key not in order]:
        if ident not in panels or ident == 'perf.timeline':
            continue
        result = panels[ident]
        title, columns = PANELS.get(ident,(ident,()))
        coverage = result.get('coverage',{})
        data = result.get('data',())
        status = result.get('status','unavailable')
        resource_children = ((ident == 'resource' and coverage.get('stages')) or
                             (ident == 'memory' and coverage.get('series')))
        if not len(data) and not resource_children:
            if model.meta['forward_only'] and ident in ('generation','oracle.generation'):
                continue
            empty_reasons={'route.coverage':'本次未提供组件路由记录；未启用组件时此项不适用。',
                           'compile.health':'本次没有可汇总的编译观测；不据此判断没有编译。',
                           'memory':'本次未提供官方显存事件；设备资源观测另见资源结果。'}
            model.meta['notices'].append(title+'：'+empty_reasons.get(ident,'；'.join(display(v) for v in result.get('notes',[])) or display(status)))
            # Preserve identity and first-divergence sections even when unavailable.
            if ident not in ('oracle.identity_diff','oracle.first_divergence'):
                continue
        description = DESCRIPTIONS.get(ident,'')
        if ident in ('perf.op','perf.layer'):
            description += ' 前向摘要范围：'+str(coverage.get('headline_note','未记录'))+'。'
        if ident == 'resource':
            columns = ('name','entity','metric','unit','mean','maximum','p95','coverage','status')
        add_table(model,ident,title,data,columns,description,status)
        if ident=='oracle.first_divergence':
            for record in data:
                add_table(model,'oracle.propagation','逐层误差变化线索',record.get('largest_jumps',[])+record.get('zero_to_nonzero',[]),
                    ('sample','before','after','ratio','change'),'按已观测边界记录误差变化；变化线索不等于算子因果。')
        for key, (subtitle, subcolumns) in EXTRAS.items():
            if ident.startswith('oracle.') and key in ('introduced','introduced_routes','amplification') and 'oracle.op_sensitivity' in panels and ident!='oracle.op_sensitivity':
                continue
            values = coverage.get(key)
            if values is None or isinstance(values,(str,dict)) or not len(values):
                continue
            if ident=='training' and key=='phases':
                subtitle,subcolumns='逐阶段主机区间',('rank','step','phase','microstep','host_duration_ns')
            if key == 'headline' and ident == 'perf.device':
                subtitle = '设备类别与关联概览'
                subcolumns = ('dev','category','association_label','events','device_work_ns')
            note = (str(coverage.get('headline_note','来源已选范围')) if key == 'headline' else description)
            add_table(model,ident+'.'+key,subtitle,values,subcolumns or columns,note,status)
        if ident == 'memory':
            from .curves import memory_envelope
            count=memory_envelope(model,rows(coverage.get('series')))
            model.section('memory.series','显存全范围趋势',('dev','component','source_time_ns','values'),
                description='覆盖全部 '+str(count)+' 个指标观测；分桶保留端点、峰谷和缺失。精确统计来自原始记录；完整点见 exports/memory-series.jsonl。',kind='memory')
        quality_rows = []
        for key in ('phase_status','headline_note','selected_path','totals','window_identity','clock',
                    'collection','alignment','resource_limit','unattributed_compile','probe'):
            value = coverage.get(key)
            if isinstance(value,(dict,str,int,float,bool)) and value not in ({},'',None):
                quality_rows.append(dict(field=key,value=value))
        if quality_rows:
            add_table(model,ident+'.quality',title+'的范围与依据',quality_rows,('field','value'),
                      '这里说明本面板可回答的问题和观测范围；完整采集身份保留在机器结果中。',kind='quality')
    _summarize(model,panels)
    quality = doc.get('quality',{})
    selected = quality.get('selected_forwards')
    if selected:
        add_table(model,'forwards','前向与阶段登记',selected,
                  ('run_id','forward','phase','phase_source','compile_count','compile_ns','selected'),
                  '只使用已记录的阶段依据。未确定阶段不会被当成正式测量。')
    if 'training' in panels and len(model.rows('training')):
        steps=model.rows('training')
        measured=sum(r.get('host_duration_ns') is not None for r in steps)
        model.finding('训练步主机区间',str(len(steps))+' 条训练步记录，其中 '+str(measured)+' 条保存了主机区间长度。','training')
    if doc.get('schema') == 'monitor-report/1':
        _monitor(model,doc)
    if doc.get('schema') in ('flagprof-group-report/1','oracle-group/1'):
        _group(model,doc,source)
    if doc.get('schema') == 'device-comparison/1':
        _comparison(model,doc)
    if doc.get('schema') == 'oracle-three-way/1':
        _three_way(model,doc)
    if source:
        _export_tables(model,Path(source))
        _timeline(model,Path(source),doc.get('diagnosis') or {})
    if doc.get('schema')=='runner-result/1':model.meta['title']='性能基准结果'
    if doc.get('runner_summary'):
        add_runner(model,doc['runner_summary'])
    _insights(model)
    from .insights import accuracy, windows_and_resources
    accuracy(model)
    windows_and_resources(model)
    # Findings are ordered by direct impact, not by analyzer execution order.
    priority = dict(fault=0,result=1,observation=2,coverage=3)
    model.meta['findings'].sort(key=lambda v:priority.get(v['level'],2))
    for section in model.sections:
        if section['id'] in ('oracle.samples','perf.op.forwards','perf.layer.forwards') and len(model.rows(section['id']))>2000:
            section['kind']='evidence'
    model.sections = [s for s in model.sections if len(model.rows(s['id']))]
    known={s['id'] for s in model.sections}
    for finding in model.meta['findings']:
        if finding.get('section') not in known:finding['section']=None
    from .layout import decorate
    decorate(model)
    return model


def _summarize(model, panels):
    device = model.rows('perf.device')
    maxima = {};category_totals={}
    positions = {}
    for index, row in enumerate(device):
        category=row.get('category','unknown')
        category_totals[category]=category_totals.get(category,0)+(row.get('device_work_ns') or 0)
        maximum=maxima.get(category)
        if row.get('device_work_ns') is not None and (maximum is None or row['device_work_ns']>maximum['device_work_ns']):
            maxima[category]=row
            positions[category]=index
    for category in ('compute','memcpy'):
        maximum=maxima.get(category)
        if not maximum:continue
        name=maximum.get('native_owner') or maximum.get('name','未命名活动')
        short=name if len(name)<=90 else name[:87]+'…'
        share=maximum['device_work_ns']/category_totals[category] if category_totals[category] else None
        model.finding('全采集窗口'+display(category)+'热点',
                      short+'；设备工作量 '+display(maximum['device_work_ns'],'device_work_ns')+
                      '，占同类别已记录时长和 '+display(share,'duration_share')+
                      '；记录 '+str(maximum.get('events',0))+' 次。完整名称与来源见对应聚合项。','perf.device',row_index=positions[category])
    p = panels.get('perf.device',{})
    totals = p.get('coverage',{}).get('totals',{})
    if totals.get('events'):
        model.finding('设备归属覆盖',
                      '设备事件 '+str(totals['events'])+' 条；原生框架归属 '+
                      str(totals.get('native_owned','未记录'))+' 条；已关联项目调用 '+
                      str(totals.get('linked','未记录'))+' 条。项目调用关联和设备记录采用独立的计数口径。',
                      'perf.device.groups','coverage')
    p = panels.get('oracle.first_divergence')
    if p:
        first = next(iter(rows(p.get('data'))),{}).get('first_inequal')
        if first:
            model.finding('首个已观测差异：'+str(first.get('scope',first.get('boundary','未记录位置'))),
                          '样本 '+str(first.get('sample','未记录'))+'；最大绝对误差 '+
                          display(first.get('max_abs'))+'；相对 L2 '+display(first.get('relative_l2'))+
                          '；不同 / 有效元素 '+str(first.get('differing_elements','未记录'))+' / '+str(first.get('valid_elements','未记录'))+
                          '；证据 '+display(first.get('level'))+'。上一个相等边界：'+str(first.get('last_equal_boundary','未记录'))+
                          '。首差异描述已配对的观测顺序。','oracle.first_divergence',row_index=0)
        elif p.get('status') == 'ok':
            model.finding('已配对观测中未发现数值差异',
                          '比较范围、样本和完整 / 抽样证据见下方；该结果不替代任务精度标准。','oracle.layer')
        else:
            model.finding('首差异结论尚不完整',
                          '需要先查看比较身份、未配对项和数值证据范围。','oracle.identity_diff','coverage')
    introduced = model.rows('oracle.op_sensitivity.introduced')
    if len(introduced):
        model.finding('同输入引入差异的调用：'+str(len(introduced))+' 条',
                      '这些调用满足已有配置和输入判据；从调用详情继续查看实现、回放与独立性能证据。',
                      'oracle.op_sensitivity.introduced')


def _export_tables(model, source):
    extras = (
        ('oracle.perf_links','oracle.perf_links.jsonl'),
        ('oracle.samples','oracle.samples.jsonl'),('oracle.alignment','oracle.alignment.jsonl'),
        ('compile_issues','compile_issues.jsonl'),('forwards','forwards.jsonl'),
        ('perf.op.forwards','perf-op-forwards.jsonl'),('perf.layer.forwards','perf-layer-forwards.jsonl'),
    )
    titles = {'oracle.samples':'逐样本数值证据','oracle.alignment':'数值配对与缺口',
              'compile_issues':'编译问题记录','forwards':'前向与阶段登记',
              'perf.op.forwards':'按前向的调用记录','perf.layer.forwards':'按前向的层记录'}
    known = {s['id'] for s in model.sections}
    for ident,name in extras:
        path = source/'exports'/name
        if ident in known or not path.is_file() or not path.stat().st_size:
            continue
        def records():
            with path.open() as stream:
                while True:
                    line = stream.readline(model.limits.record_bytes+1)
                    if not line: break
                    if len(line.encode())>model.limits.record_bytes:raise ValueError('report export record exceeds budget')
                    yield json.loads(line)
        title, columns = PANELS.get(ident,(titles.get(ident,ident),()))
        add_table(model,ident,title,records(),columns,DESCRIPTIONS.get(ident,'完整来源记录；按表内条件筛选。'))
    path = source/'exports/timeline.json'
    if path.is_file():
        model.meta['links'].append(dict(label='完整时间轴（Chrome / Perfetto）',href='exports/timeline.json'))


def _insights(model):
    issues=model.rows('compile_issues')
    if len(issues):model.finding('编译问题观测 '+str(len(issues))+' 条','先查看错误消息、缓存状态和对应调用；缓存命中率只覆盖已记录的监听路径。','compile_issues')
    for index, row in sorted(enumerate(model.rows('compile.health')),key=lambda pair:pair[1].get('total_ns') or 0,reverse=True)[:1]:
        model.finding('编译观测：'+str(row.get('backend')),'共 '+str(row.get('count'))+' 次，记录耗时 '+display(row.get('total_ns'),'total_ns')+'；命中比例 '+display(row.get('cache_hit_ratio'),'cache_hit_ratio')+'。阶段登记用于区分预热和测量期。','compile.health',row_index=index)
    candidates=[(i,r) for i,r in enumerate(model.rows('route.coverage.missing_ops')) if r.get('section')=='main']
    if candidates:
        index,row=max(candidates,key=lambda pair:pair[1].get('forward_non_hit_calls') or 0)
        model.finding('前向内非命中调用最多：'+str(row.get('name')),
            str(row.get('forward_non_hit_calls',0))+' 次；非命中比例 '+display(row.get('non_hit_ratio'),'non_hit_ratio')+'。先按原因核对实现覆盖与策略排除，再评估性能影响。','route.coverage.missing_ops',row_index=index)
    peaks={}
    for index,row in enumerate(model.rows('memory')):
        key=(row.get('dev'),row.get('component'),row.get('metric'))
        if row.get('peak_bytes') is not None:peaks[key]=(index,row)
    if peaks:
        index,row=max(peaks.values(),key=lambda pair:pair[1]['peak_bytes'])
        model.finding('最高的单来源显存观测',str(row.get('dev'))+' / '+str(row.get('component'))+' / '+str(row.get('metric'))+'：'+display(row['peak_bytes'],'peak_bytes')+'。其他分配器与设备占用分别列出，不跨域相加。','memory',row_index=index)
    unavailable=valid=0
    for row in model.rows('resource'):
        if row.get('mean') is None:unavailable+=1
        else:valid+=1
    if unavailable or valid:
        model.finding('资源统计可用范围',str(valid)+' 个指标窗口有均值；'+str(unavailable)+' 个窗口缺少有效统计。先查看覆盖与采样分辨率，再阅读阶段功率、能耗或频率。','resource','coverage')
    replay=list(model.rows('oracle.replay'))
    if replay:
        valid=sum(bool(r.get('valid')) for r in replay)
        model.finding('同输入回放：'+str(valid)+' / '+str(len(replay))+' 组有效',
            '各侧原输出复现、输入签名与同输入误差分别保留。这里检验隔离调用，不直接推导最终模型输出变化。','oracle.replay')
    links=list(model.rows('oracle.perf_links'))
    if links:
        joined=sum(bool(r.get('matched_rows')) for r in links)
        model.finding('独立性能证据：'+str(joined)+' / '+str(len(links))+' 项匹配',
            '每项按同侧身份和精确签名核对；缺失、身份差异与路由不一致随记录给出。','oracle.perf_links')


def _timeline(model,source,diagnosis):
    path=source/'exports/timeline.json'
    if not path.is_file():return
    from .curves import memory_envelope,unit
    doc=read_document(path,model)
    if doc.get('metadata',{}).get('clock')!='same_boot_monotonic_raw':return
    origin=doc['metadata'].get('origin_raw_ns')
    reference=diagnosis.get('coverage',{}).get('reference',{})
    boot=reference.get('diagnostic_identity',{}).get('proc_key',{}).get('boot_id')
    if boot and origin is not None:
        times=(r['event_time_ns'] for r in model.rows('diagnosis')
               if r.get('event_clock')=='host_mono:'+boot and r.get('event_time_ns') is not None)
        first=min(times,default=None)
        if first is not None:model.meta['resource_fault_ns']=first-origin
    events=doc.get('traceEvents',())
    def points():
        for event in events:
            if event.get('ph') not in ('C','i'):continue
            name=event.get('name','');entity,_,metric=name.partition('/')
            if not metric:continue
            yield dict(dev=entity,component='同主机资源观测',source_time_ns=int(event['ts']*1000),
                values={metric:event.get('args',{}).get('value')},unit=unit(metric))
    count=memory_envelope(model,points(),'resource.series')
    if count:
        model.section('resource.series','资源与任务阶段',('dev','component','source_time_ns','values'),
            description='同一主机时钟下的资源观测；阶段背景来自已保存的任务区间。峰谷和缺失保留，统计不从曲线反推。',kind='memory')
        model.meta['phase_bands']=[dict(name=e['name'],start_ns=int(e['ts']*1000),end_ns=int((e['ts']+e['dur'])*1000))
            for e in events if e.get('ph')=='X' and e.get('cat')=='phase']
    if doc.get('metadata',{}).get('truncated'):model.meta['notices'].append('资源时间线达到来源导出上限；展示范围按已保存的时间线解释。')


def _monitor(model, doc):
    model.meta['title'] = '节点资源结果'
    model.meta['facts'].extend([dict(label='监控会话',value=doc.get('session_id')),
                               dict(label='观测窗口长度',value=display(doc['window'][1]-doc['window'][0],'duration_ns') if doc.get('window') else None)])
    add_table(model,'monitor.series','资源趋势',
              (dict(r,unit=('无量纲' if r['unit']=='' else r['unit']) if r.get('unit') is not None else '历史记录未保存单位') for r in rows(doc.get('series'))),
              ('entity','metric','unit','valid','missing','minimum','maximum'),
              '实心点标出保留的读数，悬停可看时间和值。细线连接相邻读数，状态码和位掩码只画点。采样缺失、监控中断或间隔超过通常采样周期的 2.5 倍时断线。',kind='monitor')
    add_table(model,'monitor.minutes','分钟观测均值',doc.get('minute_trends'),
              ('entity','metric','count','missing','total'),
              '分钟观测均值不替代原始覆盖率或故障规则证据。')
    business = {(r.get('id'), r.get('revision'), r.get('run_id'), r.get('rule'))
                for r in model.rows('diagnosis') if r.get('id')}
    node_findings = (r for r in rows(doc.get('findings'))
                     if (r.get('id'), r.get('revision'), r.get('run_id'), r.get('rule')) not in business)
    add_table(model,'monitor.findings','节点资源告警',node_findings,None)
    if len(model.rows('diagnosis.coverage')):
        model.meta['title'] = '节点任务诊断与资源监控'
    series=model.rows('monitor.series')
    if len(series):
        entities={r.get('entity') for r in series if r.get('entity') is not None}
        missing=sum(r.get('missing') or 0 for r in series)
        model.finding(str(len(entities))+' 个观测对象 · '+str(len(series))+' 条指标序列',
                      '各序列共登记 '+str(missing)+' 个缺失样本。按设备查看同一时间窗内的指标。','monitor.series')
    model.meta['monitor_window'] = doc.get('window')
    coverage = (doc.get('coverage') or {}).get('window_coverage') or {}
    model.meta['monitor_gaps'] = [dict(window=gap.get('window') or [gap['ts'], gap['ts']], label='监控中断')
                                  for gap in doc.get('gaps', []) if gap.get('window') or gap.get('ts') is not None]
    model.meta['monitor_gaps'].extend(dict(window=w, label='此段无留存采集记录')
                                      for w in coverage.get('missing_segment_intervals') or [])
    model.meta['notices'].extend(str(v) for v in doc.get('errors',[]))


def _rank_summary(receipt):
    status = receipt.get('status')
    outcome = display(status) if status is not None else '缺少该 rank 的状态回执，无法确认执行是否完成'
    if status in ('complete','completed','failed'):
        outcome = '执行'+outcome
    for field, label in (('process_returncode','退出码'),('exit_code','退出码'),('returncode','worker 自报退出码')):
        if receipt.get(field) is not None:
            return f'{outcome} · {label} {receipt[field]}'
    return outcome+' · '+display(None,'process_returncode')


def _group(model, doc, source):
    oracle = doc.get('schema') == 'oracle-group/1'
    model.meta['title'] = '多 rank 精度结果' if oracle else '多 rank 通信与推理结果'
    model.meta['facts'].extend(dict(label=LABELS.get(k,k),value=doc[k])
                              for k in ('backend','ref_backend','cmp_backend','samples','status') if k in doc)
    if oracle:model.finding('逻辑输入样本：'+str(doc.get('samples','未记录')),'对应 rank 分别比较内部算子；复制边界与逻辑样本只计一次。','group.ranks')
    reports = {r['rank']:r for r in rows(doc.get('ranks'))}
    receipts = {r['rank']:r for r in rows(doc.get('rank_execution'))}
    ranks = []
    for rank in sorted(reports.keys() | receipts.keys()):
        row = dict(reports.get(rank, {}), rank=rank)
        if rank in reports:
            href = row.get('report') or f'rank-{rank}/report.html'
            row['report'] = href
            model.meta['links'].append(dict(label=f'rank {rank} 结果',href=href))
        if not oracle:
            row['execution_summary'] = _rank_summary({**row, **receipts.get(rank,{})})
        ranks.append(row)
    add_table(model,'group.ranks','各 rank 报告',ranks,('rank','execution_summary','report'))
    add_table(model,'communication.aligned','同次通信的跨 rank 对照',doc.get('device_calls'),
              ('group','group_seq','rank_device_union_ns','cross_rank_min_observed_ns','cross_rank_max_observed_ns','cross_rank_difference_ns','reason'),
              '按通信域与序号配对；差值是观测占用差，不等于额外等待。缺少时钟关系时不画共同绝对时间轴。',
              kind='communication')
    add_table(model,'group.counts','模型通信计数与阶段',doc.get('rank_counts'),
              ('rank','forwards','steps','forward_phases','allreduce','allgather','model_count_matches'))
    if doc.get('control_calls'):
        add_table(model,'group.control_calls','控制组通信',doc['control_calls'],
                  ('backend','op','rank','group_seq','host_duration_ns','device_union_ns'),
                  '每行记录一个 rank 的控制调用。主机耗时从进入通信 API 到返回，设备占用仅统计已关联的设备区间。')
    add_table(model,'group.execution','各 rank 执行结果',receipts.values(),
              ('rank','status','process_returncode','returncode','exit_code','signal'))
    add_table(model,'group.request_windows','各 rank 请求窗口',doc.get('request_windows'),
              ('rank','request_id','coverage_status','window_ns','all_activity_union_ns','blank_ns'))
    aligned=list(rows(doc.get('device_calls')))
    usable=[(i,r) for i,r in enumerate(aligned) if r.get('cross_rank_difference_ns') is not None]
    if doc.get('model_call_count'):
        model.finding(f"模型通信已配对 {doc['paired_model_calls']} / {doc['model_call_count']} 次",
                      '每次按通信组和调用序号核对两个 rank 的设备区间。','communication.aligned')
    if usable:
        index,worst=max(usable,key=lambda pair:pair[1]['cross_rank_difference_ns'])
        model.finding('跨 rank 通信占用差最大的已配对调用',
            '通信序号 '+str(worst.get('group_seq'))+'；占用差 '+display(worst['cross_rank_difference_ns'],'cross_rank_difference_ns')+'。该差值不等于额外等待；从该序号进入各 rank 详情。','communication.aligned',row_index=index)
    unlinked = {str(rank): n for rank,n in (doc.get('unlinked_logical_calls') or {}).items() if n}
    host_barriers = Counter(str(row['rank']) for row in rows(doc.get('control_calls'))
                            if row.get('backend')=='gloo' and row.get('op')=='barrier'
                            and row.get('host_duration_ns') is not None and row.get('device_union_ns') is None)
    if unlinked and unlinked == host_barriers:
        model.finding('Gloo 控制组屏障仅记录主机耗时',
                      '控制组表列出各 rank 从进入屏障到返回的耗时。模型通信的设备区间配对情况单独统计。',
                      'group.control_calls','coverage')
    elif unlinked:
        counts = '、'.join(f'rank {rank} 有 {n} 次' for rank,n in sorted(unlinked.items()) if n)
        detail = counts+'逻辑通信调用缺少对应的设备活动区间，无法比较这些调用在各 rank 上的设备耗时。'
        model.finding('通信设备耗时的观测范围',detail,'group.counts','coverage')
    model.meta['notices'].extend(str(v) for v in doc.get('issues',[]))
    if source:
        for rank in ranks:
            path=Path(source)/('rank-'+str(rank['rank']))/'result.json'
            if path.is_file():
                child=read_document(path,model)
                first=child.get('results',{}).get('oracle.first_divergence',{})
                item=next(iter(rows(first.get('data'))),{}).get('first_inequal')
                if item:
                    model.finding('rank '+str(rank['rank'])+' 首差异：'+str(item.get('scope')),
                                  '样本 '+str(item.get('sample'))+'；最大绝对误差 '+display(item.get('max_abs')),
                                  'group.ranks')


def _comparison(model, doc):
    from .insights import comparison_sources, performance_comparison
    names = comparison_sources(doc)
    model.meta['title']='运行条件与性能对照'
    add_table(model,'comparisons','比较条件',
              (dict(r,left_label=names.get(r.get('left'),'来源未记录'),right_label=names.get(r.get('right'),'来源未记录')) for r in rows(doc.get('comparisons'))),
              ('left_label','right_label','status','differences','reasons'))
    add_table(model,'window.comparisons','窗口定义与完成情况',doc.get('window_comparisons'),
              ('left','right','kind','status','differences','reasons'))
    add_table(model,'comparison.metrics','同口径指标变化',
              (dict(r,source_label=names.get(r.get('source'),'来源未记录')) for r in rows(doc.get('metric_comparisons'))
               if not (r.get('scope')=='forward' and r.get('metric')=='generated_tokens_per_second')),
              ('source_label','rank','scope','phase','generation_step','metric','unit','baseline','candidate','delta','relative_change','comparison_status'))
    model.meta['notices'].append(doc.get('note',''))
    comparable=[r for r in rows(doc.get('metric_comparisons')) if r.get('delta') is not None]
    performance_comparison(model, doc)
    if comparable and not any(f['section']=='comparison.metrics' for f in model.meta['findings']):
        model.finding('已生成同口径指标变化','以第一份报告为基准；差值和相对变化随指标给出。多个条件同时变化时，不将差异归因于单一组件。','comparison.metrics')
    elif not comparable:model.finding('本组对照保留原值，尚无可解释的差值','先核对缺失身份、输出语义与窗口定义；原因逐项列在比较条件中。','comparisons','coverage')
    for n,source in enumerate(rows(doc.get('sources'))):
        if source.get('local_report'):model.meta['links'].append(dict(label='来源 '+str(n+1)+' 完整结果',href=source['local_report']))
        add_table(model,'comparison.source.'+str(n),'来源 '+str(n+1)+' 的观测原值',
                  source.get('data'),PANELS['perf.device'][1],
                  Path(str(source.get('report',''))).name+'；'+display(source.get('comparison_status')))


def _three_way(model, doc):
    model.meta['title']='三路精度对照结果'
    add_table(model,'oracle.three_way','相同参考下的误差变化',doc.get('comparisons'),
              ('scope','boundary','sample','rank','metric','reference_to_baseline','reference_to_candidate',
               'baseline_to_candidate','delta','relative_change','reason'))
    add_table(model,'oracle.pair_statuses','三路比较的身份核对',doc.get('pair_statuses'),('pair','rank','status'))
    model.finding('三组精度证据已分别计算','先核对三路身份状态，再沿同一样本和调用键比较误差变化；缺侧或证据范围不同的行不计算变化量。','oracle.three_way')
    for label,href in doc.get('reports',{}).items():
        model.meta['links'].append(dict(label=label,href=href))


def add_runner(model, summary):
    from .performance import GROUP_TIMING_NOTE, FORWARD_TIMING_NOTE
    add_table(model,'performance.baseline','无插桩与采集条件下的性能',summary.get('groups'),
              ('instrumentation','rank','scope','shape','phase','generation_step','observations',
               'mean_ns','median_ns','p95_ns','samples_per_second','generated_tokens_per_second',
               'independent_repeats','repeat_cv','spike_count','status'),
              '预热单列并排除。进程内重复与独立进程重复分开；吞吐的分子和计时边界保留在详情中。')
    add_table(model,'performance.spikes','需要复查的时延尖峰',summary.get('spikes'),
              ('source','rank','phase','generation_step','latency_ns','median_ns','repeat'))
    add_table(model,'performance.group_windows','组请求计时依据',summary.get('group_windows'),('group_execution','request_id','ranks','latency_ns','rule'))
    add_table(model,'performance.perturbations','配对实验中的采集扰动',summary.get('instrumentation_pairs'),
              ('pair_id','rank','scope','phase','generation_step','baseline_mean_ns','instrumented_mean_ns','overhead_ratio','status','reason'),
              '必须有共同配对标识与测量协议；比较同条件、同计时边界和相同观测次数。')
    add_table(model,'performance.sources','基准输入与计时条件',summary.get('sources'),('source','rank','instrumentation','load_s','semantics'))
    for index,group in enumerate(rows(summary.get('groups'))):
        if group.get('scope') not in ('forward','request','group_request'):continue
        model.finding(('无插桩' if group.get('instrumentation')=='none' else '采集条件下')+' '+str(group.get('scope'))+' 时延',
            'rank '+str(group.get('rank'))+'；有效观测 '+str(group.get('observations'))+' 次；中位数 '+display(group.get('median_ns'),'median_ns')+'，p95 '+display(group.get('p95_ns'),'p95_ns')+'。'+('需要复查重复波动。' if group.get('cv_screening')=='unstable' else ''),'performance.baseline',row_index=index)
    groups = rows(summary.get('groups'))
    forward_only = bool(groups) and all(row.get('scope')=='forward' for row in groups)
    model.meta['notices'].extend(FORWARD_TIMING_NOTE if forward_only and note == GROUP_TIMING_NOTE else note
                                 for note in summary.get('notes',[]))


def write_report(out, document=None, *, limits=None,  source=None, filename='report.html'):
    from .render import render
    out=Path(out)
    with opened_model(limits=limits) as model:
        document = document if document is not None else read_document(out/'result.json',model)
        build(model,document,source or out)
        model.save(out/'report-data.jsonl')
        render(model,out,filename=filename)
        return dict(output=str(out),html=filename,summary='report.md',sections=len(model.sections))


def redraw(source, out, *, limits=None, runner_results=None, refresh_view=False):
    from flagprof.offline.worker import output_operation
    return output_operation(out, [source], _redraw, source, limits=limits,
                            runner_results=runner_results, refresh_view=refresh_view)


def _redraw(source, out, *,  limits=None, runner_results=None, refresh_view=False):
    from .render import render
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('render output must be separate from its source')
    from flagprof.offline.worker import new_output
    new_output(out)
    # Preserve exports and linked rank reports; no Store or device access.
    for name in ('result.json','resources.json','exports'):
        path=source/name
        if path.is_dir():shutil.copytree(path,out/name)
        elif path.is_file():shutil.copyfile(path,out/name)
    for path in list(source.glob('rank-*'))+list(source.glob('source-*'))+[source/name for name in ('reference-baseline','reference-candidate','baseline-candidate')]:
        if path.is_dir() and (path/'result.json').is_file():
            redraw(path,out/path.name,limits=limits,refresh_view=refresh_view)
    filename='index.html' if (source/'index.html').exists() and not (source/'report.html').exists() else 'report.html'
    with opened_model(limits=limits) as model:
        snapshot=source/'report-data.jsonl'
        if snapshot.is_file() and not refresh_view and not runner_results and (source/'result.json').is_file():
            with snapshot.open() as stream:
                header = stream.readline(model.limits.record_bytes + 1)
            if len(header.encode()) <= model.limits.record_bytes:
                page = json.loads(header).get('value', {}) if header.strip() else {}
                refresh_view = (page.get('field_display_version') != FIELD_DISPLAY_VERSION or
                                page.get('mode') == 'monitor' and 'monitor_gaps' not in page or
                                page.get('mode') == 'monitor' and page.get('diagnosis_layout_version') in (2, 3) or
                                'diagnosis_layout_version' not in page and any(
                                    (f.get('section') or '').startswith('diagnosis') for f in page.get('findings', [])))
        if runner_results:
            from .performance import summarize
            from . import write_json
            document=read_document(source/'result.json',model)
            identity=document.get('report_context',{}).get('identity') or document.get('results',{}).get('perf.device',{}).get('coverage',{}).get('workload_identity')
            document['runner_summary']=summarize(runner_results,identity,limits=model.limits)
            with (out/'result.json').open('w') as stream:write_json(stream,document);stream.write('\n')
            build(model,document,out)
        elif snapshot.is_file() and not refresh_view:
            model.load(snapshot)
            from .layout import decorate
            decorate(model)
        else:
            build(model,read_document(source/'result.json',model),source)
        model.meta['rendered_from']=dict(path=str(source),sha256=digest(snapshot if snapshot.exists() and not refresh_view and not runner_results else source/'result.json'),refresh_view=refresh_view)
        model.save(out/'report-data.jsonl')
        render(model,out,filename=filename)
        return dict(output=str(out),html=filename,summary='report.md')
