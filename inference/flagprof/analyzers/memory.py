# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Source-clock occupancy and transfer summaries over public Store events."""
import json
from . import Analyzer, Need, Requires, Result
from .workspace import encode


OCCUPANCY = ('allocated_bytes','reserved_bytes','active_bytes','device_used_bytes')


def memory(work, params):
    state = work.view.memory_status()
    points = 0
    for run,w,seq,event in work.events(('counter','memop')):
        evidence = dict(run_id=run,w=w,seq=seq)
        if event['t']=='memop' and event['kind'] in ('h2d','d2h','d2d'):
            group=dict(dev=event['dev'],kind=event['kind'])
            key=encode(group)
            previous=work.db.execute("select payload from rows where panel='memory.transfers' and key=?",(key,)).fetchone()
            row=json.loads(previous[0]) if previous else dict(group,events=0,bytes=0,evidence=[])
            row['events']+=1;row['bytes']+=event['bytes']
            if len(row['evidence'])<work.limits.evidence_samples:row['evidence'].append(evidence)
            work.row('memory.transfers',key,row)
            continue
        meta=event.get('ext',{}).get('memory')
        if not meta:continue
        points+=1
        evidence.update(source=meta['source'],loc=meta['loc'])
        point=dict(dev=event['dev'],component=meta['component'],values=event['values'],
            source_time_ns=meta['source_time_ns'],clock_status=meta['clock_status'],
            layer_status=meta['layer_status'],native_owner=meta.get('native_owner'),
            operation_status=meta.get('operation_status'),release_observed=meta.get('release_observed'),evidence=evidence)
        work.row('memory.series',encode((run,w,seq)),point)
        work.db.execute("update rows set metric=? where panel='memory.series' and key=?",
            (-(meta['source_time_ns'] if meta['source_time_ns'] is not None else 9e18),encode((run,w,seq))))
        values=event['values']
        if 'allocation_bytes' in values or 'delta_bytes' in values:
            group=dict(dev=event['dev'],component=meta['component'],native_owner=meta.get('native_owner'))
            key=encode(group)
            previous=work.db.execute("select payload from rows where panel='memory.allocations' and key=?",(key,)).fetchone()
            row=json.loads(previous[0]) if previous else dict(group,allocations=0,releases=0,
                allocated_bytes=0,released_bytes=0,release_not_observed=0 if 'allocation_bytes' in values else None,evidence=[],
                lifetime_pairing='official_operator_record' if 'allocation_bytes' in values else 'not_reconstructed',
                layer_status=meta['layer_status'],operation_status=meta.get('operation_status'))
            if 'allocation_bytes' in values:
                row['allocations']+=1;row['allocated_bytes']+=values['allocation_bytes']
                if meta.get('release_observed'):
                    row['releases']+=1;row['released_bytes']+=values['allocation_bytes']
                else:row['release_not_observed']+=1
            elif values['delta_bytes']>0:
                row['allocations']+=1;row['allocated_bytes']+=values['delta_bytes']
            elif values['delta_bytes']<0:
                row['releases']+=1;row['released_bytes']-=values['delta_bytes']
            if len(row['evidence'])<work.limits.evidence_samples:row['evidence'].append(evidence)
            work.row('memory.allocations',key,row)
        for metric in OCCUPANCY:
            if metric not in event['values']:continue
            group=dict(dev=event['dev'],component=meta['component'],metric=metric,unit='bytes')
            key=encode(group)
            previous=work.db.execute("select payload from rows where panel='memory' and key=?",(key,)).fetchone()
            row=json.loads(previous[0]) if previous else dict(group,observations=0,missing_time=0,
                observed_peak_bytes=None,peak_source_time_ns=None,clock_status='source_clock_only',
                layer_status='explicit_project_call_missing',evidence=None)
            row['observations']+=1;row['missing_time']+=meta['source_time_ns'] is None
            value=event['values'][metric]
            if row['observed_peak_bytes'] is None or value>row['observed_peak_bytes']:
                row.update(observed_peak_bytes=value,peak_source_time_ns=meta['source_time_ns'],evidence=evidence)
            work.row('memory',key,row)
        if points%256==0:work.check_budget()
    available=len(work.table('memory'))>0
    return Result('memory',('partial' if work.partial() or state['status']!='complete' else 'ok') if available else 'unavailable',
        work.table('memory'),dict(collection=state,points=points,series=work.table('memory.series'),
            allocations=work.table('memory.allocations'),transfers=work.table('memory.transfers'),
            transfer_status='available' if len(work.table('memory.transfers')) else 'no_representable_transfer_events',
            allocation_note='分配/释放字节为观测事件量；不用于推算总占用，未观测释放不等于泄漏。'),
        ['峰值为采集窗口内观测峰值；各组件及 allocated/reserved 分别报告。',
         '曲线使用官方来源自身时间；项目时间和层归因不可用。分配差额不替代总占用。'])


def make_memory():
    return Analyzer('memory',memory,Requires((Need(events=('counter',)),)))
