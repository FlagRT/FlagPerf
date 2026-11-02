"""Forward ownership and presentation derived from stored structural identities."""
from functools import lru_cache
import json
from flagprof.devices import accelerator, device_types
from .workspace import encode


def _add(work, panel, key, fields, calls, duration, *, exclusive=None):
    previous = work.db.execute('select payload from rows where panel=? and key=?', (panel, key)).fetchone()
    row = json.loads(previous[0]) if previous else dict(fields, calls=0, host_ns=0)
    row.update(calls=row['calls']+calls, host_ns=row['host_ns']+duration)
    if exclusive is not None:
        row['exclusive_ns']=row.get('exclusive_ns',0)+exclusive
        row['inclusive_ns']=row['host_ns']
    work.row(panel, key, row)


def prepare(work):
    work.db.executescript('''create table forward_spans(run text,corr text,parent text,start integer,end integer,
        pid integer,tid integer,clk text,payload text,primary key(run,corr));
        create index forward_children on forward_spans(run,parent,clk,start);
        create table forward_events(run text,w integer,seq integer,root text,primary key(run,w,seq));
        create index forward_root on forward_events(run,root);''')
    for run, w, seq, event in work.events(('span',)):
        if event['cat']!='module' or event['ph']!='x' or event['ts'] is None:
            continue
        work.db.execute('insert or replace into forward_spans values(?,?,?,?,?,?,?,?,?)',
            (run,str(event['corr']),str(event.get('parent',0)),event['ts'],event['ts']+event['dur'],
             event.get('pid'),event.get('tid'),event.get('clk'),encode(event)))

    @lru_cache(maxsize=4096)
    def root_of(run, corr):
        visited=set();root=None
        while corr!='0' and corr not in visited:
            visited.add(corr)
            row=work.db.execute('select parent from forward_spans where run=? and corr=?',(run,corr)).fetchone()
            if not row:break
            root=corr;corr=row[0]
        return root if corr not in visited else None

    roots={};indexes={}
    for run,corr,parent,start,end,pid,tid,clk,payload in work.db.execute('select * from forward_spans order by run,pid,tid,start'):
        if root_of(run,corr)!=corr:continue
        group=(run,pid,tid);index=indexes.get(group,0);indexes[group]=index+1
        event=json.loads(payload)
        roots[(run,corr)]=dict(run_id=run,root_corr=int(corr),forward=index,
            phase=event.get('attrs',{}).get('phase','unknown'),compile_in_forward=False,compile_count=0,compile_ns=0)
    # Old runner captures retain their exact argv. Only infer phases when its
    # explicit loop counts cover the entire root sequence in one thread.
    for view in work.source.views():
        manifest=json.loads((view.run/'manifest.json').read_text())
        argv=manifest.get('identity',{}).get('argv',[])
        if 'flagprof.runner' not in argv:continue
        try:
            warmup=int(argv[argv.index('--warmup')+1]);measure=int(argv[argv.index('--forwards')+1])
            repeats=int(argv[argv.index('--repeats')+1]) if '--repeats' in argv else 1
        except (ValueError,IndexError):continue
        groups=[key for key in indexes if key[0]==view.run.name]
        if warmup<0 or measure<1 or repeats<1 or len(groups)!=1:continue
        group=groups[0];cycle=warmup+measure
        if indexes[group]!=cycle*repeats:continue
        for (run,corr),row in roots.items():
            if run==view.run.name and row['phase']=='unknown':
                row['phase']='warmup' if row['forward']%cycle<warmup else 'measure'
                row['phase_source']='recorded_runner_argv'
    unattributed=dict(count=0,total_ns=0);excluded_meta=0
    for run,w,seq,event in work.events(('op','opsum','compile')):
        root=None
        if event['t']=='op':root=root_of(run,str(event.get('scope',0)))
        elif event['t']=='opsum':
            start,end=event['win']
            candidates=[corr for corr, in work.db.execute('''select corr from forward_spans s
                where run=? and start<=? and end>=? and pid is ? and tid is ? and clk is ?''',
                (run,start,end,event.get('pid'),event.get('tid'),event.get('clk'))) if (run,corr) in roots]
            root=candidates[0] if len(candidates)==1 else None
        else:
            direct=root_of(run,str(event.get('ext',{}).get('scope',0)))
            op=work.event(run,event['corr']) if event['corr'] else None
            via_op=root_of(run,str(op.get('scope',0))) if op else None
            root=None if direct and via_op and direct!=via_op else direct or via_op
            if root and (run,root) in roots:
                item=roots[(run,root)]
                item.update(compile_in_forward=True,compile_count=item['compile_count']+1,compile_ns=item['compile_ns']+event['dur'])
            else:
                unattributed['count']+=1;unattributed['total_ns']+=event['dur']
        work.db.execute('insert into forward_events values(?,?,?,?)',(run,w,seq,root))
        if event['t']!='compile' and 'meta' in device_types(event.get('dev')) and not accelerator(event.get('dev')):
            excluded_meta+=event.get('n',1)
    candidates=[(key,row) for key,row in roots.items() if row['phase']=='measure']
    if not candidates:candidates=[(key,row) for key,row in roots.items() if row['phase']=='unknown']
    clean=[(key,row) for key,row in candidates if not row['compile_in_forward']]
    selected={key for key,row in clean or candidates or list(roots.items())}
    for key,row in roots.items():
        work.row('forwards',encode(key),dict(row,selected=key in selected))
    work.forward_roots=roots;work.forward_selected=selected
    work.forward_root_of=root_of
    work.forward_quality=dict(unattributed_compile=unattributed,excluded_meta_calls=excluded_meta,
        selected_forwards=sorted((roots[key] for key in selected),key=lambda row:(row['run_id'],row['forward'],row['root_corr'])),phase_status='known' if candidates and all(r['phase']=='measure' for _,r in candidates) else 'unknown')
    work.forward_quality['headline_note']='测量前向含编译' if any(roots[k]['compile_in_forward'] for k in selected) else '相位未知' if any(roots[k]['phase']=='unknown' for k in selected) else '测量前向；未观测到已归属编译'


def finish(work, op_kinds=None):
    """Keep full exports and build separate forward rows and selected headlines."""
    roots=work.forward_roots;selected=work.forward_selected
    dispatch=any((s.get('effective_config') or {}).get('dispatch','off')!='off' for s in work.statuses)
    note='含 dispatch 探针开销' if dispatch else '主机调用区间；同步前向耗时见 runner'
    work.forward_quality['host_time_note']=note
    for run,w,seq,event in work.events(('op','opsum')):
        root=work.db.execute('select root from forward_events where run=? and w=? and seq=?',(run,w,seq)).fetchone()[0]
        nested=bool(event.get('parent_corr')) if event['t']=='op' else event.get('nested',False)
        if nested:continue
        sig=work.signature(event['sid'])
        if not sig:continue
        identity=roots.get((run,root),dict(run_id=run,forward='unknown',phase='unknown'))
        fields=dict(name=sig['name'],overload=sig['overload'],sid=event['sid'],dev=event.get('dev'),**identity)
        key=encode((run,root,event['sid'],event.get('dev')))
        n,dur=(1,event['dur']) if event['t']=='op' else (event['n'],event['dur_sum'])
        if accelerator(event.get('dev')):
            route=work.event(run,event['corr'],'route') if event['t']=='op' else None
            decision=route.get('decision') if route else event.get('route_decision')
            if decision:
                domain='forward' if root else 'outside_or_unknown'
                op_name=sig['name']+('.'+sig['overload'] if sig['overload'] else '')
                domain_key=encode((op_name,domain))
                saved=work.db.execute("select payload from rows where panel='route.domains' and key=?",(domain_key,)).fetchone()
                d=json.loads(saved[0]) if saved else dict(name=op_name,domain=domain,total_calls=0,non_hit_calls=0)
                d['total_calls']+=n;d['non_hit_calls']+=n if decision!='hit' else 0
                work.row('route.domains',domain_key,d)
        _add(work,'perf.op.forwards',key,fields,n,dur)
        if (run,root) in selected:
            _add(work,'perf.op.headline',encode((event['sid'],event.get('dev'))),dict(name=sig['name'],sid=event['sid'],dev=event.get('dev'),host_time_note=note),n,dur)
    for run,corr,parent,start,end,pid,tid,clk,payload in work.db.execute('select * from forward_spans'):
        event=json.loads(payload);root=work.forward_root_of(run,corr)
        right=start;union=0
        for child_start,child_end in work.db.execute('select start,end from forward_spans where run=? and parent=? and clk is ? order by start',(run,corr,clk)):
            union+=max(0,min(end,child_end)-max(right,child_start));right=max(right,child_end)
        exclusive=max(0,end-start-union)
        fields=dict(name=event['name'],**roots.get((run,root),dict(run_id=run,forward='unknown',phase='unknown')))
        _add(work,'perf.layer.forwards',encode((run,root,event['name'])),fields,1,end-start,exclusive=exclusive)
        if (run,root) in selected:
            _add(work,'perf.layer.headline',event['name'],dict(name=event['name'],host_time_note=note),1,end-start,exclusive=exclusive)
    for panel in ('perf.op.headline','perf.layer.headline'):
        for key,payload in list(work.db.execute('select key,payload from rows where panel=?',(panel,))):
            row=json.loads(payload);row['mean_ns']=row['host_ns']/row['calls']
            work.row(panel,key,row)
    tables=[s['op_kinds'] for s in work.statuses if s.get('op_kinds')]
    table=op_kinds or (tables[0] if tables else {})
    versions={s.get('torch_version') for s in work.statuses if s.get('torch_version')}
    if not versions:
        for _,_,_,event in work.events(('meta',)):
            env=event.get('data',{});v=env.get('torch_version') or env.get('torch')
            if isinstance(v,str):versions.add(v)
    external=table.get('external_run_environment',{})
    external_bound=not versions and table.get('run_id')==work.view.run.name and external.get('torch_version')==table.get('torch_version') and bool(external.get('sha256'))
    valid=bool(table) and (versions=={table.get('torch_version')} or external_bound)
    work.forward_quality['op_kinds_version_source']='registered_external_image' if external_bound else 'attachment_or_meta' if versions else 'unavailable'
    # Full route rows retain their sid-level evidence. Presentation is key-level.
    work.db.execute("delete from rows where panel='missing_ops'")
    route_host_total=sum(row['host_ns'] for row in work.table('route.coverage'))
    for row in work.table('route.coverage'):
        if row['route_decision']=='hit':continue
        name=row['name']+('.'+row['overload'] if row['overload'] else '')
        kind=table.get('entries',{}).get(name,{}).get('kind','unknown') if valid else 'unknown'
        reason=row.get('reason') or 'unknown';key=encode((name,kind))
        old=work.db.execute("select payload from rows where panel='missing_ops' and key=?",(key,)).fetchone()
        item=json.loads(old[0]) if old else dict(name=name,kind=kind,section='main' if kind in ('kernel','unknown') else kind,
            calls=0,non_hit_calls=0,reasons={},decisions={},impls=[],sids=[],host_ns=0,host_time_note=note,
            classification_note=('op_kind entry missing' if name not in table.get('entries',{}) else None) if valid else 'op_kinds missing or torch version mismatch')
        item['calls']+=row['calls'];item['non_hit_calls']+=row['calls'];item['host_ns']+=row['host_ns']
        item['reasons'][reason]=item['reasons'].get(reason,0)+row['calls']
        decision=row['route_decision'];item['decisions'][decision]=item['decisions'].get(decision,0)+row['calls']
        if row.get('impl') not in item['impls']:item['impls'].append(row.get('impl'))
        if row['sid'] not in item['sids']:item['sids'].append(row['sid'])
        item['sid_count']=len(item['sids'])
        item['host_share']=item['host_ns']/route_host_total if route_host_total else None
        item['host_share_denominator']='full route.coverage host_ns; includes dispatch overhead'
        domains=[json.loads(payload) for payload, in work.db.execute("select payload from rows where panel='route.domains' and json_extract(payload,'$.name')=?",(name,))]
        item['domains']=domains
        item['total_calls']=sum(d['total_calls'] for d in domains)
        item['non_hit_ratio']=item['non_hit_calls']/item['total_calls'] if item['total_calls'] else None
        item['forward_non_hit_calls']=sum(d['non_hit_calls'] for d in domains if d['domain']=='forward')
        item['priority_basis']='forward_non_hit_calls; outside_or_unknown listed separately'
        work.row('missing_ops',key,item)
        work.db.execute("update rows set metric=? where panel='missing_ops' and key=?",(item['forward_non_hit_calls']+(item['non_hit_ratio'] or 0)/2,key))
    return work.forward_quality
