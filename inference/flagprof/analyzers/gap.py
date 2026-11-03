# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Interval occupancy on one explicitly sourced clock and window at a time."""
import math
import json
import statistics
from collections import Counter
from . import Analyzer, Need, Requires, Result
from .workspace import encode

CATEGORIES=('compute','memcpy','comm','wait','unknown')


def interval_stats(intervals, start, end, *, gaps=False):
    """Input ordered by start. Sum includes overlap; union counts it once."""
    total=union=records=0;left=right=None;blank=[];cursor=start
    for a,b in intervals:
        a=max(start,a);b=min(end,b)
        if b<=a:continue
        total+=b-a;records+=1
        if right is None:
            left,right=a,b
            if gaps and a>cursor:blank.append(a-cursor)
        elif a<=right:right=max(right,b)
        else:
            union+=right-left
            if gaps:blank.append(a-right)
            left,right=a,b
    if right is not None:union+=right-left
    result=dict(union_ns=union,sum_ns=total,records=records)
    if gaps:
        cursor=right if right is not None else start
        if end>cursor:blank.append(end-cursor)
        result['blank_segments_ns']=distribution(blank)
    return result


def distribution(values):
    ordered=sorted(values)
    if not ordered:return dict(count=0,mean=None,median=None,p90=None,p95=None)
    return dict(count=len(ordered),mean=statistics.mean(ordered),median=statistics.median(ordered),
                p90=ordered[math.ceil(.9*len(ordered))-1],p95=ordered[math.ceil(.95*len(ordered))-1],
                quantile_rule='nearest_rank')


def comparable(left,right):
    """Missing identities are evidence, never permission to compute speedups."""
    issues=[];differences=[]
    for kind in ('workload_identity','window_identity'):
        a,b=left.get(kind),right.get(kind)
        if not a or not b:issues.append(kind+':missing')
        elif a!=b:
            issues.append(kind+':different')
            for field in sorted(set(a)|set(b)):
                if a.get(field)!=b.get(field):differences.append(dict(field=kind+'.'+field,left=a.get(field),right=b.get(field)))
    required={'window_identity':('warmup','forwards','repeats','profiler','dispatch','window_kind'),
              'workload_identity':('model_files','inputs_sha256','samples','dtype','attention','software','numerics')}
    for side,context in (('left',left),('right',right)):
        for kind,fields in required.items():
            identity=context.get(kind) or {}
            missing=[k for k in fields if identity.get(k) is None]
            if missing:issues.append(side+':incomplete_'+kind+':'+','.join(missing))
        window=context.get('window_identity') or {}
        if window.get('window_kind')=='project_request':
            missing=[k for k in ('boundary_rule','samples','generation_mode','generated_steps') if window.get(k) is None]
            if missing:issues.append(side+':incomplete_request_identity:'+','.join(missing))
            if window.get('coverage_status')!='complete':issues.append(side+':incomplete_request_coverage')
    return dict(status='comparable' if not issues else 'not_comparable',reasons=issues,differences=differences)


def gap(work,params):
    context=work.view.device_analysis_context()
    if context.get('status')!='available':
        return Result('perf.gap','unavailable',work.table('perf.gap'),context,['设备区间派生证据尚未记录。'])
    work.db.executescript('''create table activity(category text,dev integer,stream text,a integer,b integer,call_key text,payload text,source_clock text);
        create index activity_start on activity(a,b);
        create index activity_stream on activity(dev,stream,a,b);
        create index activity_call on activity(call_key,a);''')
    counts=Counter();clock=context.get('clock',{})
    for ordinal,row in enumerate(work.view.iter_device_events(projection='analysis')):
        counts['classification_conflicts']+=bool(row.get('classification_conflict'))
        counts['device_representation_records']+=row['category']=='comm'
        if not row['canonical']:continue
        counts['canonical_communication_records']+=row['category']=='comm'
        a,d=row['start_ns'],row['duration_ns']
        if a is None or d is None or d<0:
            counts['invalid_intervals']+=1;continue
        comm=row.get('communication',{});call=comm.get('call');key=encode((call['group'],call['group_seq'],call.get('rank'))) if call else None
        work.db.execute('insert into activity values(?,?,?,?,?,?,?,?)',(row['category'],row['dev'],encode(row.get('stream')),a,a+d,key,encode(row),row['source_clock']))
        if ordinal%256==0:work.db.commit()
    work.db.commit()
    for callkey, in work.db.execute('select distinct call_key from activity where call_key is not null'):
        if work.db.execute('select count(distinct source_clock) from activity where call_key=?',(callkey,)).fetchone()[0]!=1:
            counts['ambiguous_multi_window_calls']+=1
            continue
        source=work.db.execute('select payload from activity where call_key=? order by a limit 1',(callkey,)).fetchone()
        first=json.loads(source[0]);call=first['communication']['call']
        stats=interval_stats(work.db.execute('select a,b from activity where call_key=? order by a,b',(callkey,)),-(1<<63),(1<<63)-1)
        sources=[json.loads(r[0])['source'] for r in work.db.execute('select payload from activity where call_key=? limit ?',(callkey,work.limits.evidence_samples))]
        work.row('communication.device_calls',callkey,dict(call,device_link_status='linked',device_union_ns=stats['union_ns'],device_sum_ns=stats['sum_ns'],device_records=stats['records'],clock=clock,source_clock=context['source_clock'],evidence=sources,association=first['communication']['evidence']))
    counts['linked_logical_calls']=len(work.table('communication.device_calls'))
    grouped={};partial=bool(counts['classification_conflicts'])
    for index,window in enumerate(context.get('windows',[])):
        a,b=window['start_ns'],window['end_ns']
        if b<=a:continue
        source_clock=window.get('source_clock',context['source_clock'])
        filt='a<? and b>? and source_clock=?';args=[b,a,source_clock]
        if window.get('dev') is not None:filt+=' and dev=?';args.append(window['dev'])
        def stats(category=None):
            suffix=' and category=?' if category else ''
            return interval_stats(work.db.execute('select a,b from activity where '+filt+suffix+' order by a,b',args+([category] if category else [])),a,b,gaps=category is None)
        categories={kind:stats(kind) for kind in CATEGORIES};all_stats=stats();length=b-a
        for value in categories.values():value['union_ratio']=value['union_ns']/length;value['sum_ratio']=value['sum_ns']/length
        row=dict(window,window_id=str(index),window_ns=length,categories=categories,
                 all_activity_union_ns=all_stats['union_ns'],blank_ns=length-all_stats['union_ns'],
                 all_activity_ratio=all_stats['union_ns']/length,blank_ratio=(length-all_stats['union_ns'])/length,
                 blank_segments_ns=all_stats['blank_segments_ns'],
                 denominator='explicit_window_length_ns',source_clock=source_clock)
        duration_sum=sum(value['sum_ns'] for value in categories.values())
        row['unknown_duration_ratio']=categories['unknown']['sum_ns']/duration_sum if duration_sum else 0
        partial |= row['unknown_duration_ratio']>.01
        for category in ('compute','memcpy','comm','wait'):
            for metric in ('union_ns','sum_ns'):row[category+'_'+metric]=categories[category][metric]
        work.row('perf.gap',str(index).zfill(8),row)
        for dev,stream in work.db.execute('select distinct dev,stream from activity where '+filt,args):
            selected='dev is ? and stream=? and a<? and b>? and source_clock=?';values=[dev,stream,b,a,source_clock]
            stream_categories={category:interval_stats(work.db.execute('select a,b from activity where '+selected+' and category=? order by a,b',values+[category]),a,b) for category in CATEGORIES}
            occupied=interval_stats(work.db.execute('select a,b from activity where '+selected+' order by a,b',values),a,b,gaps=True)
            work.row('perf.gap.streams',encode((index,dev,stream)),dict(window_id=str(index),kind=window['kind'],step=window.get('step'),dev=dev,stream=json.loads(stream),window_ns=length,categories=stream_categories,all_activity_union_ns=occupied['union_ns'],blank_ns=length-occupied['union_ns'],blank_segments_ns=occupied['blank_segments_ns'],source_clock=source_clock))
        grouped.setdefault(window['kind'],[]).append(row)
    distributions={kind:{field:distribution([r[field] for r in rows]) for field in ('window_ns','all_activity_union_ns','blank_ns')} for kind,rows in grouped.items()}
    for kind,rows in grouped.items():
        distributions[kind]['categories']={category:{field:distribution([r['categories'][category][field] for r in rows]) for field in ('union_ns','sum_ns')} for category in CATEGORIES}
    coverage=dict(context,communication_counts=dict(counts),distributions=distributions,
                  device_calls=work.table('communication.device_calls'),streams=work.table('perf.gap.streams'),
                  unknown_duration_threshold=.01)
    work.analysis_context=context
    return Result('perf.gap',('partial' if partial else 'ok') if grouped else 'unavailable',work.table('perf.gap'),coverage,
        ['请求边界映射：'+('可用' if context.get('request_window_status')=='available' else context.get('request_window_status','unavailable'))+'；误差界见时钟证据，映射不会改变原设备事件时钟。',
         'project_request 从 prefill 前至最后设备同步后；模型加载、请求前准备和 KV 释放另计。coverage_status=partial 表示请求完成或采集覆盖不完整。',
         '区间占用不等于硬件利用率；各类区间可以重叠，不能相加。全部活动并集＋空白＝窗口长度。',
         '贡献使用逐窗口值或均值；分布使用中位数与 nearest-rank 分位数。decode 相邻选择窗口不覆盖完整请求。']+
        (['类别结构冲突或 unknown 时长和占比超过 1%，类别解释为 partial，原值保留。'] if partial else []))


def make_perf_gap():
    return Analyzer('perf.gap',gap,Requires((Need(views=('device_analysis',)),)))
