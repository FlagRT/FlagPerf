# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Device work and attribution, without treating overlap as wall time."""
import json
from . import Analyzer, Need, Requires, Result
from .workspace import encode


def category(event):
    if event['t'] == 'kernel':
        return event.get('ktype', 'compute')
    if event['t'] == 'memop':
        return 'allocation' if event['kind'] in ('alloc', 'free') else 'memcpy' if event['kind'] in ('h2d','d2h','d2d') else 'other'
    return 'comm'


def accumulate(work, panel, key, base, event, evidence):
    previous = work.db.execute('select payload from rows where panel=? and key=?', (panel, key)).fetchone()
    row = json.loads(previous[0]) if previous else dict(base, events=0, valid_durations=0,
        missing_durations=0, device_work_ns=None, evidence=[])
    row['events'] += 1
    duration = event.get('dur')
    row['missing_durations' if duration is None else 'valid_durations'] += 1
    if duration is not None:
        row['device_work_ns'] = (row['device_work_ns'] or 0) + duration
    if len(row['evidence']) < work.limits.evidence_samples:
        row['evidence'].append(evidence)
    work.row(panel, key, row)
    return row


def device(work, params):
    if not work.db.execute("select 1 from sqlite_master where name='spans'").fetchone():
        from .core import perf_layer
        perf_layer(work, params)
    totals = dict(events=0, linked=0, project_resolved=0, native_owned=0,
                  valid_durations=0, missing_durations=0, device_work_ns=0)
    for run, writer, seq, event in work.events(('kernel', 'memop', 'comm')):
        if event['t'] == 'comm' and event.get('phase') != 'device':
            continue
        kind = category(event)
        if kind == 'allocation':
            continue  # Allocator records belong to memory, not device execution coverage.
        totals['events'] += 1
        totals['missing_durations' if event.get('dur') is None else 'valid_durations'] += 1
        totals['device_work_ns'] += event.get('dur') or 0
        link = event.get('link', {})
        totals['linked'] += link.get('status') == 'linked'
        source_run = work.source.collection.run.name if work.source.collection else run
        op = work.event(source_run, link.get('corr')) if link.get('status') == 'linked' else None
        route = work.event(source_run, op['corr'], 'route') if op else None
        decision = route['decision'] if route else 'unknown'
        totals['project_resolved'] += op is not None
        source = event.get('_source')
        native = None
        if source:
            found = work.db.execute('select payload from native where trace=? and loc=?',
                                   (source['trace_uid'], str(source['loc']))).fetchone()
            if found:
                native = json.loads(found[0])
        owned = bool(native and native.get('state') == 'owned')
        totals['native_owned'] += owned
        evidence = dict(run_id=run, w=writer, seq=seq)
        if source:
            evidence['source'] = source
        base = dict(dev=event.get('dev'), category=kind, name=event.get('name', event.get('kind')),
                    sid=op.get('sid') if op else None, route_decision=decision,
                    link_status=link.get('status', 'unlinked'),link_reason=link.get('reason'),
                    association_label='已关联项目调用' if link.get('status')=='linked' else '未关联项目调用',
                    native_owner=native.get('owner_name') if owned else None,
                    aggregation='sum_of_device_durations_including_overlap')
        accumulate(work, 'perf.device', encode(base), base, event, evidence)
        group = {k:base[k] for k in ('dev', 'category', 'link_status', 'link_reason', 'route_decision')}
        accumulate(work, 'perf.device.coverage', encode(group), group, event, evidence)
        if op:
            key = encode((op['sid'], op.get('dev'), bool(op.get('parent_corr')),
                          route['decision'] if route else None))
            group = dict(sid=op['sid'], dev=base['dev'], category=kind, route_decision=decision,op_group_key=key)
            accumulate(work, 'perf.op.devices', encode((key, group)), group, event, evidence)
            scope = str(op.get('scope', 0))
            work.db.execute('delete from visited')
            while scope != '0':
                if work.db.execute('select 1 from visited where corr=?', (scope,)).fetchone():
                    break
                work.db.execute('insert into visited values(?)', (scope,))
                span = work.db.execute('select parent,payload from spans where run=? and corr=?',
                                       (source_run, scope)).fetchone()
                if not span:
                    break
                value = json.loads(span[1]); name = value.get('attrs', {}).get('fold_group', value['name'])
                group = dict(name=name, dev=base['dev'], category=kind, route_decision=decision)
                accumulate(work, 'perf.layer.devices', encode(group), group, event, evidence)
                scope = span[0]
        if totals['events'] % 256 == 0:
            work.check_budget()
    total = totals['events']
    totals.update(linked_ratio=totals['linked']/total if total else None,
                  project_resolved_ratio=totals['project_resolved']/total if total else None,
                  native_owned_ratio=totals['native_owned']/total if total else None)
    if not totals['valid_durations']:
        totals['device_work_ns'] = None
    for panel, groups, field in (('perf.op','perf.op.devices','op_group_key'),('perf.layer','perf.layer.devices','name')):
        work.db.execute("create index device_"+field+" on rows(panel,json_extract(payload,'$."+field+"'))")
        for key,payload in work.db.execute('select key,payload from rows where panel=?',(panel,)):
            row=json.loads(payload)
            if panel=='perf.op' and not row.get('sid'):continue
            counts=work.db.execute("select sum(json_extract(payload,'$.events')),sum(json_extract(payload,'$.valid_durations')),sum(json_extract(payload,'$.missing_durations')),sum(json_extract(payload,'$.device_work_ns')) from rows where panel=? and json_extract(payload,'$."+field+"')=?",(groups,key)).fetchone()
            row.update(device_events=counts[0] or 0,device_valid_durations=counts[1] or 0,device_missing_durations=counts[2] or 0,
                device_work_ns=counts[3],device_aggregation='sum_of_event_durations_including_overlap')
            work.row(panel,key,row)
    context=work.view.device_analysis_context()
    observed=set()
    for row in work.table('perf.device.coverage'):
        row['association_label']='已关联项目调用' if row['link_status']=='linked' else '未关联项目调用'
        observed.add((row['dev'],row['category'],row['link_status']))
        row.update(window_identity=context.get('window_identity'),workload_identity=context.get('workload_identity'))
        row['scope_note']='linked 子集不代表完整前向'
        work.row('perf.device.headline',encode({k:row[k] for k in ('dev','category','link_status','link_reason','route_decision')}),row)
    for dev,kind in sorted({(dev,kind) for dev,kind,status in observed},key=encode):
        if (dev,kind,'linked') not in observed:
            row=dict(dev=dev,category=kind,link_status='linked',association_label='已关联项目调用',route_decision='unknown',events=0,valid_durations=0,missing_durations=0,device_work_ns=0,evidence=[],scope_note='此类别未记录已关联项目调用；linked 子集不代表完整前向',window_identity=context.get('window_identity'),workload_identity=context.get('workload_identity'))
            work.row('perf.device.headline',encode((dev,kind,'linked')),row)
    coverage = dict(work.coverage(), totals=totals, groups=work.table('perf.device.coverage'),
                    op_groups=work.table('perf.op.devices'), layer_groups=work.table('perf.layer.devices'),headline=work.table('perf.device.headline'),
                    window_identity=context.get('window_identity'),workload_identity=context.get('workload_identity'))
    uncertain = any(q.get('format_status') != 'within' or q.get('import_status') != 'complete'
                    for q in work.qualities.values())
    return Result('perf.device', ('partial' if work.partial() or uncertain else 'ok') if total else 'unavailable',
                  work.table('perf.device'), coverage,
                  ['设备工作量为有效事件时长之和，包含重叠；原生 owner 与项目签名分别保留。'])


def make_perf_device():
    return Analyzer('perf.device', device, Requires((Need(events=('kernel',)), Need(events=('memop',)), Need(events=('comm',)))))
