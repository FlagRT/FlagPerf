# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Host, layer, route and compilation conclusions from the analysis index."""
import json
from . import Analyzer, Need, Requires, Result
from .workspace import encode


from flagprof.devices import accelerator



def key_name(sig):
    return sig['name'].removeprefix('aten::') + ('.'+sig['overload'] if sig['overload'] and sig['overload'] != 'default' else '')


def sample(values, value, limit):
    if value not in values and len(values) < limit:
        values.append(value)
    return values


def perf_op(work, params):
    limit = work.limits.evidence_samples
    counts = dict(each=0, summary=0, dev_unknown=0, out_of_scope=0, nested=0)
    for run, writer, seq, r in work.events(('op', 'opsum')):
        each = r['t'] == 'op'
        n, dur = (1, r['dur']) if each else (r['n'], r['dur_sum'])
        nested = bool(r.get('parent_corr')) if each else r.get('nested', False)
        route = work.event(run, r['corr'], 'route') if each else None
        decision = route['decision'] if route else r.get('route_decision')
        dev = r.get('dev')
        key = encode((r['sid'], dev, nested, decision))
        row = work.db.execute('select calls,host,each_n,summary_n,samples,corrs,scopes from opgroups where key=?', (key,)).fetchone()
        calls, host, en, sn, evidence, corrs, scopes = row if row else (0, 0, 0, 0, '[]', '[]', '[]')
        evidence = sample(json.loads(evidence), dict(run_id=run, w=writer, seq=seq), limit)
        corrs = sample(json.loads(corrs), r['corr'], limit) if each else json.loads(corrs)
        scopes = sample(json.loads(scopes), dict(run_id=run, corr=r['scope']), limit) if each and r.get('scope') else json.loads(scopes)
        work.db.execute('insert or replace into opgroups values(?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (key, r['sid'], encode(dev), int(nested), decision, calls+n, host+dur, 0,
             en+n if each else en, sn if each else sn+n, encode(evidence), encode(corrs), encode(scopes)))
        counts['each' if each else 'summary'] += n
        counts['nested'] += n if nested else 0
        counts['dev_unknown'] += n if dev is None else 0
        counts['out_of_scope'] += n if nested or dev is not None and not accelerator(dev) else 0
    linked, device_count = 0, 0
    # Device evidence remains a separate duration column, covering each only.
    for run, writer, seq, event in work.events(('kernel','memop','comm')):
        if event['t']=='comm' and event.get('phase') != 'device':
            continue
        device_count += 1
        link = event.get('link', {})
        if link.get('status') != 'linked':
            continue
        source_run = work.source.collection.run.name if work.source.collection else run
        op = work.event(source_run, link['corr'])
        if not op:
            continue
        linked += 1
        route = work.event(source_run, op['corr'], 'route')
        key = encode((op['sid'], op.get('dev'), bool(op.get('parent_corr')), route['decision'] if route else None))
        work.db.execute('update opgroups set device=device+? where key=?', (event.get('dur') or 0, key))
    host_total = work.db.execute('select coalesce(sum(host),0) from opgroups where nested=0').fetchone()[0]
    for key, sid, dev, nested, decision, calls, host, device, en, sn, evidence, corrs, scopes in work.db.execute('select * from opgroups order by key'):
        sig = work.signature(sid)
        if not sig:
            continue
        row = dict(name=sig['name'], overload=sig['overload'], sid=sid, signature=sig,
            dev=json.loads(dev), nested=bool(nested), route_decision=decision, calls=calls,
            host_ns=host, host_share=host/host_total if host_total and not nested else None,
            device_work_ns=device if linked else None, device_scope='each_window',
            each_calls=en, summary_calls=sn, evidence=json.loads(evidence), corr_samples=json.loads(corrs),
            scope_samples=json.loads(scopes))
        work.row('perf.op', key, row)
        if not nested and accelerator(row['dev']):
            inventory_key = key_name(sig)
            old = work.db.execute('select payload from rows where panel=? and key=?', ('inventory_v1_view', inventory_key)).fetchone()
            inventory = json.loads(old[0]) if old else dict(name=inventory_key, calls=0, each_calls=0, summary_calls=0)
            inventory.update(calls=inventory['calls']+calls, each_calls=inventory['each_calls']+en,
                             summary_calls=inventory['summary_calls']+sn)
            work.row('inventory_v1_view', inventory_key, inventory)
    native_rows = 0
    if not linked and work.native_available:
        for run,writer,seq,event in work.events(('kernel','memop','comm')):
            if event['t']=='comm' and event.get('phase')!='device':
                continue
            source=event.get('_source')
            if not source:
                continue
            native=work.db.execute('select payload from native where trace=? and loc=?',
                (source['trace_uid'],str(source['loc']))).fetchone()
            if not native:
                continue
            native=json.loads(native[0])
            if native['state']!='owned':
                continue
            owner=native['owner']; owner_name=native.get('owner_name') or 'native owner '+encode(owner)
            category=native['category']
            unknown_type=event.get('name') if category=='unknown' else None
            key='native:'+encode((owner['trace_uid'],owner_name,category,unknown_type))
            old=work.db.execute("select payload from rows where panel='perf.op' and key=?",(key,)).fetchone()
            row=json.loads(old[0]) if old else dict(name=owner_name,
                owner=owner,owner_samples=[],calls=0,device_events=0,host_ns=None,device_work_ns=0,source='native',
                category=category,unknown_type=unknown_type,
                scope='persisted_device_events',evidence=[],format_status=native.get('source_state',{}).get('format_status','unavailable'))
            inserted=work.db.execute('insert or ignore into native_seen values(?,?,?)',(key,owner['trace_uid'],str(owner['loc']))).rowcount
            row['calls']+=inserted;row['device_events']+=1;row['device_work_ns']+=event.get('dur') or 0
            row['owner_samples']=sample(row['owner_samples'],owner,limit)
            row['evidence']=sample(row['evidence'],source,limit)
            work.row('perf.op',key,row)
        native_rows=work.db.execute("select count(*) from rows where panel='perf.op' and key like 'native:%'").fetchone()[0]
    available = len(work.table('perf.op')) > 0
    uncertain=any(q.get('format_status')!='within' or q.get('import_status')!='complete' for q in work.qualities.values())
    status = 'partial' if work.partial() or (native_rows or linked) and uncertain else 'ok'
    coverage = dict(work.coverage(), op_granularity=counts, inventory_v1_view=work.table('inventory_v1_view'),
        selected_path='project' if not native_rows else 'project_host+native_device' if counts['each']+counts['summary'] else 'native', linked=linked, persisted_device_events=device_count,
        linked_ratio=linked/device_count if device_count else None)
    return Result('perf.op', status if available else 'unavailable', work.table('perf.op'), coverage,
        ['Device duration is work initiated by this call, not exclusive wall time.']+
        (['Native owner and source name are resolved through persisted evidence.'] if native_rows else [])+
        (['Device interpretation is partial: inspect format applicability and import state.'] if uncertain and (native_rows or linked) else [])+
        ([] if available else ['缺少可用的 op/opsum+sig 或已持久化设备的原生归属路径。']))


def perf_layer(work, params):
    work.db.executescript('''create table spans(run text,corr text,parent text,start integer,dur integer,clk text,payload text,
        primary key(run,corr)); create index span_parent on spans(run,parent,start);
        create table visited(corr text primary key);''')
    unmatched = 0
    for run, writer, seq, r in work.events(('span',)):
        if r['ph'] == 'b':
            continue
        if r['ph'] == 'e':
            begins = work.db.execute("select payload from events where run=? and corr=? and type='span' and json_extract(payload,'$.ph')='b'", (run, str(r['corr']))).fetchall()
            if len(begins) != 1:
                unmatched += 1
                continue
            begin = json.loads(begins[0][0])
            if begin.get('clk') != r.get('clk') or r['ts'] is None or begin['ts'] is None or r['ts'] < begin['ts']:
                unmatched += 1
                continue
            r = dict(begin, ph='x', dur=r['ts']-begin['ts'])
        if r['ph'] == 'x':
            work.db.execute('insert or replace into spans values(?,?,?,?,?,?,?)',
                (run, str(r['corr']), str(r.get('parent', 0)), r['ts'], r['dur'], r.get('clk'), encode(r)))
    dropped = any(w.get('dropped', 0) for report in work.read_reports for w in report.get('writers', {}).values())
    unmatched += work.db.execute("select count(*) from events e where type='span' and json_extract(payload,'$.ph')='b' and not exists(select 1 from spans s where s.run=e.run and s.corr=e.corr)").fetchone()[0]
    invalid = 0
    for run, corr, parent, start, dur, clk, payload in work.db.execute('select * from spans order by run,corr'):
        span = json.loads(payload)
        if start is None:
            continue
        exclusive, reason = dur, None
        end, union, right = start+dur, 0, start
        for child_start, child_dur, child_clk in work.db.execute('select start,dur,clk from spans where run=? and parent=? order by start', (run, corr)):
            if child_clk != clk or child_start is None or child_start < start or child_start+child_dur > end:
                reason = 'invalid_child_interval'
                continue
            a, b = max(right, child_start), child_start+child_dur
            union += max(0, b-a)
            right = max(right, b)
        if parent == corr:
            reason = 'cycle'
        work.db.execute('delete from visited')
        ancestor = parent
        while ancestor != '0':
            if ancestor == corr or work.db.execute('select 1 from visited where corr=?', (ancestor,)).fetchone():
                reason = 'cycle'
                break
            work.db.execute('insert into visited values(?)', (ancestor,))
            node = work.db.execute('select parent from spans where run=? and corr=?', (run, ancestor)).fetchone()
            if not node:
                reason = 'missing_parent'
                break
            ancestor = node[0]
        if dropped or unmatched:
            reason = 'incomplete_span_tree'
        if reason:
            exclusive = None
            invalid += 1
        else:
            exclusive -= union
        attrs = span.get('attrs', {})
        key = attrs.get('fold_group', span['name'])
        old = work.db.execute('select payload from rows where panel=? and key=?', ('perf.layer', key)).fetchone()
        row = json.loads(old[0]) if old else dict(name=key, folded=bool(attrs.get('folded')), members=0,
            inclusive_ns=0, exclusive_ns=0, min_ns=dur, max_ns=dur, corr_samples=[], op_calls=0,
            device_work_ns=None, device_status='unavailable', exclusive_reason=None)
        row.update(members=row['members']+1, inclusive_ns=row['inclusive_ns']+dur,
                   min_ns=min(row['min_ns'], dur), max_ns=max(row['max_ns'], dur),
                   exclusive_ns=None if exclusive is None or row['exclusive_ns'] is None else row['exclusive_ns']+exclusive,
                   exclusive_reason=reason or row['exclusive_reason'])
        row['mean_ns'] = row['inclusive_ns']/row['members']
        row['corr_samples'] = sample(row['corr_samples'], dict(run_id=run, corr=int(corr)), work.limits.evidence_samples)
        row['op_calls'] += work.db.execute("select count(*) from events where run=? and scope=? and type='op'", (run, corr)).fetchone()[0]
        work.db.execute('insert or ignore into members values(?,?,?,?)', ('perf.layer', key, run, corr))
        work.row('perf.layer', key, row)
    for run, writer, seq, event in work.events(('kernel','memop','comm')):
        if event['t']=='comm' and event.get('phase')!='device':
            continue
        link=event.get('link',{})
        if link.get('status')!='linked':
            continue
        source=work.source.collection.run.name if work.source.collection else run
        op=work.event(source,link['corr'])
        scope=str(op.get('scope',0)) if op else '0'
        work.db.execute('delete from visited')
        while scope!='0':
            if work.db.execute('select 1 from visited where corr=?',(scope,)).fetchone():
                break
            work.db.execute('insert into visited values(?)',(scope,))
            node=work.db.execute('select parent,payload from spans where run=? and corr=?',(source,scope)).fetchone()
            if not node:
                break
            span=json.loads(node[1]); key=span.get('attrs',{}).get('fold_group',span['name'])
            payload=work.db.execute("select payload from rows where panel='perf.layer' and key=?",(key,)).fetchone()
            if payload:
                row=json.loads(payload[0]); row.update(device_work_ns=(row['device_work_ns'] or 0)+(event.get('dur') or 0),
                    device_status='available',device_scope='each_window')
                work.row('perf.layer',key,row)
            scope=node[0]
    available = len(work.table('perf.layer')) > 0
    unscoped=work.db.execute("select count(*) from events where type='op' and scope='0'").fetchone()[0]
    unresolved=work.db.execute("select count(*) from events e where type='op' and scope!='0' and not exists(select 1 from spans s where s.run=e.run and s.corr=e.scope)").fetchone()[0]
    for tid,n in work.db.execute("select json_extract(payload,'$.tid'),count(*) from events where type='op' and scope='0' group by json_extract(payload,'$.tid')"):
        work.row('unscoped_by_tid',tid,dict(tid=tid,calls=n))
    return Result('perf.layer', 'partial' if available and (invalid or unresolved or work.partial()) else 'ok' if available else 'unavailable',
        work.table('perf.layer'), dict(work.coverage(), invalid_exclusive=invalid, unmatched=unmatched,
        unscoped_calls=unscoped,scope_unresolved=unresolved,unscoped_by_tid=work.table('unscoped_by_tid'),
        module_probe=work.probe('module')), ['Device duration, when available, is initiated work, not layer elapsed time.']+
        ([] if available else ['缺少可重建的完整 span 区间。']))


def route_coverage(work, params):
    status = work.probe('route')
    totals = dict(calls=0, hit=0, fallback=0, excluded=0, error=0, route_table_mismatch=0,
                  route_unobserved=sum(s.get('route_unobserved', 0) for s in work.statuses),
                  summary_only_sids=0, summary_only_calls=0)
    table = {}
    mismatch_samples=[]
    missing_summary_calls=0
    for state in work.statuses:
        table.update(state.get('route_table') or {})
    for run, writer, seq, event in work.events(('route',)):
        op=work.event(run,event['corr'])
        if not op or op.get('parent_corr') or not accelerator(op.get('dev')):
            continue
        sig=work.signature(op['sid']); rule=table.get(key_name(sig))
        expected_reason='not_registered' if rule is None else 'device_unsupported' if rule['device_unsupported'] else 'policy_excluded' if rule['policy_excluded'] else None
        if event['decision']=='error':
            expected_reason='runtime_error'
        expected_impl=rule.get('impl') if rule and event['decision']=='hit' else None
        if table and (event.get('reason')!=expected_reason or event.get('impl')!=expected_impl):
            totals['route_table_mismatch']+=1
            sample(mismatch_samples,dict(run_id=run,w=writer,seq=seq,corr=event['corr'],sid=op['sid']),work.limits.evidence_samples)
    for row in work.table('perf.op'):
        if row.get('source') == 'native' or row['nested'] or not accelerator(row['dev']):
            continue
        decision = row['route_decision']
        if decision is None:
            continue
        canonical = key_name(row['signature'])
        rule = table.get(canonical)
        reason = 'not_registered' if rule is None else 'device_unsupported' if rule['device_unsupported'] else 'policy_excluded' if rule['policy_excluded'] else None
        if decision == 'error':
            reason = 'runtime_error'
        impl = rule.get('impl') if rule and decision == 'hit' else None
        each_present = row['each_calls'] > 0
        reason_status='available'
        if row['summary_calls'] and not table:
            reason=impl=None
            reason_status='unavailable'
            missing_summary_calls+=row['summary_calls']
        elif not row['summary_calls'] and row['corr_samples']:
            observed=next((work.event(e['run_id'],c,'route') for e in row['evidence'] for c in row['corr_samples']
                           if work.event(e['run_id'],c,'route')),None)
            if observed:
                reason,impl=observed.get('reason'),observed.get('impl')
        item = dict(row, reason=reason, impl=impl, reason_source='route_table' if row['summary_calls'] else 'route',
            reason_status=reason_status,reason_unavailable='route_table_missing' if reason_status=='unavailable' else None,
            layers=row['scope_samples'] if each_present else None,
            layer_status='available' if each_present else 'unavailable', layer_reason=None if each_present else 'summary_only')
        totals['calls'] += row['calls']
        totals[decision] += row['calls']
        if not each_present:
            totals['summary_only_sids'] += 1
            totals['summary_only_calls'] += row['calls']
        work.row('route.coverage', encode((row['sid'], row['dev'], decision)), item)
    # Fractions use only this route scope, including its non-hit categories.
    host_total = sum(row['host_ns'] for row in work.table('route.coverage'))
    for row in work.table('route.coverage'):
        row.update(call_share=row['calls']/totals['calls'] if totals['calls'] else None,
                   host_share=row['host_ns']/host_total if host_total else None)
        work.row('route.coverage', encode((row['sid'], row['dev'], row['route_decision'])), row)
        if row['route_decision'] != 'hit':
            work.row('missing_ops', encode((row['sid'], row['dev'], row['route_decision'])), row)
    totals['hit_ratio'] = totals['hit']/totals['calls'] if totals['calls'] else None
    totals['summary_only_sids']=work.db.execute('select count(distinct sid) from opgroups g where summary_n>0 and not exists(select 1 from opgroups e where e.sid=g.sid and e.each_n>0)').fetchone()[0]
    available = status['status']=='available'
    notes=(['summary 缺少 route_table，reason/impl unavailable。'] if missing_summary_calls else [])+(['1.0 summary 缺少 dev/nested，无法恢复完整路由分母。'] if work.legacy_summary_calls else [])
    return Result('route.coverage', 'partial' if available and (work.partial() or missing_summary_calls or totals['route_table_mismatch']) else 'ok' if available else 'unavailable',
        work.table('route.coverage'), dict(work.coverage(), **totals, probe=status,route_table_mismatch_samples=mismatch_samples,
            summary_reason_unavailable_calls=missing_summary_calls,observation='observed' if totals['calls'] else 'no_observations'),
        notes+([] if available and totals['calls'] else ['判定范围内没有路由观测。'] if available else [status['reason']]))


def compile_health(work, params):
    status = work.probe('compile')
    count = attributed = 0
    for run, writer, seq, r in work.events(('compile',)):
        item = dict(r, evidence=dict(run_id=run, w=writer, seq=seq))
        op = work.event(run, r['corr']) if r['corr'] else None
        item['op'] = work.signature(op['sid'])['name'] if op else None
        attributed += bool(op)
        work.db.execute('insert into compilations values(?,?,?,?)', (r['backend'], r.get('ver', 'unknown'), r['dur'], encode(item)))
        if r.get('err') or r['cache']=='invalid':
            work.row('compile_issues', encode((run, writer, seq)), item)
        count += 1
    for backend, ver, n, lo, hi, total in work.db.execute('select backend,ver,count(*),min(dur),max(dur),sum(dur) from compilations group by backend,ver order by backend,ver'):
        med = [v for v, in work.db.execute('select dur from compilations where backend=? and ver=? order by dur limit ? offset ?',
            (backend, ver, 2 if n%2==0 else 1, (n-1)//2))]
        hits = work.db.execute("select count(*) from compilations where backend=? and ver=? and json_extract(payload,'$.cache')='hit'", (backend, ver)).fetchone()[0]
        work.row('compile.health', encode((backend, ver)), dict(backend=backend, ver=ver, count=n,
            min_ns=lo, max_ns=hi, total_ns=total, median_ns=sum(med)/len(med), cache_hit_ratio=hits/n))
    available = status['status']=='available'
    return Result('compile.health', 'partial' if available and (work.partial() or status.get('not_observable')) else 'ok' if available else 'unavailable',
        work.table('compile.health'), dict(work.coverage(), count=count, attributed=attributed,
            cache_hit_ratio=work.db.execute("select count(*) from compilations where json_extract(payload,'$.cache')='hit'").fetchone()[0]/count if count else None,
            attribution_status='available' if attributed else 'not_covered',
            observation='observed' if count else 'no_observations', probe=status),
        ['编译监听器覆盖成功编译与持久化缓存返回；失败、失效与进程内复用未覆盖。'] if available and status.get('not_observable') else [] if available else [status['reason']])


ANALYZERS = (
    Analyzer('perf.op', perf_op, Requires((Need(events=('op','sig')), Need(views=('native_relations',))))),
    Analyzer('perf.layer', perf_layer, Requires((Need(events=('span',)),))),
    Analyzer('route.coverage', route_coverage, Requires((Need(events=('op','sig','route')),))),
    Analyzer('compile.health', compile_health, Requires((Need(events=('compile',)),))),
)


def make_perf_op():
    from copy import deepcopy
    return deepcopy(ANALYZERS[0])


def make_perf_layer():
    from copy import deepcopy
    return deepcopy(ANALYZERS[1])


def make_route_coverage():
    from copy import deepcopy
    return deepcopy(ANALYZERS[2])


def make_compile_health():
    from copy import deepcopy
    return deepcopy(ANALYZERS[3])
