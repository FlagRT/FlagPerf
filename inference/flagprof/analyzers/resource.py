"""Offline resource joins over explicit target identity and RAW-time support."""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from . import Analyzer, Need, Requires, Result
from .workspace import encode
from flagprof.store import StoreView
from flagprof.telemetry.quality import statistics, aggregate, energy, residual, violation, energy_per_token, low_frequency


def profiler_windows(work, fallback):
    windows = []
    for state in work.statuses:
        if state.get('preset') not in ('perf','modules'):
            continue
        found = False
        for view in work.source.views():
            directory = view.run/state.get('capture_relative', 'capture-staging/'+state['id'])
            for path in (directory/'session.json', directory/'producer.json'):
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                spans = [data.get('profiler_raw_window')]+[w.get('raw_window') for w in data.get('windows',[])]
                for span in spans:
                    if span and span[0] is not None:
                        windows.append((span[0], span[1] or fallback[1])); found = True
        if not found:
            windows.append(tuple(fallback))
    return windows


def selected_entities(ref, inventory):
    entities={e['entity']:e for e in inventory.get('entities',[])}
    selected={d['entity'] for d in ref.get('devices',[]) if d.get('status')=='available' and d['entity'] in entities}
    selected.update(entities[e]['power_domain'] for e in list(selected) if entities[e].get('power_domain'))
    key=ref.get('worker_proc_key')
    if key:
        process='proc:{pid_ns}:{pid}:{proc_start_ticks}'.format(**key)
        selected.add(process)
        selected.update(process+'/'+d['entity'] for d in ref.get('devices',[]) if d.get('status')=='available')
        selected.add('host:'+key['boot_id'])
    return selected


def root_module(work, run, writer, event):
    """Root modules may be nested in phase/user scopes; inspect ancestry."""
    parent=event.get('parent');seen=set()
    while parent and parent not in seen:
        seen.add(parent)
        row=work.db.execute('select payload from events where run=? and w=? and corr=? and type=? limit 1',
                            (run,writer,str(parent),'span')).fetchone()
        if row is None:return True
        ancestor=json.loads(row[0])
        if ancestor.get('cat')=='module':return False
        parent=ancestor.get('parent')
    return not parent


def resource(work, params):
    state = work.view.monitor_status()
    ref, monitor, inventory = (state[k] for k in ('reference','path','inventory'))
    coverage = dict(reference=ref)
    notes = []
    if monitor is None:
        return Result('resource','unavailable',work.table('resource'),coverage,[ref.get('reason','monitor unavailable')])
    target_source=ref.get('target_source_identity',{}).get('sha256')
    monitor_source=inventory.get('source_identity',{}).get('sha256')
    if target_source and monitor_source and target_source!=monitor_source:
        notes.append('monitor and target source identities differ')
    identities = [d for d in ref.get('devices',[]) if d.get('status') == 'available']
    entities = {e['entity']:e for e in inventory['entities']}
    selected = set()
    for identity in identities:
        device = entities.get(identity['entity'])
        if device is None:
            notes.append('device identity conflict: '+identity['entity']); continue
        selected.add(identity['entity'])
        if device.get('power_domain'):
            selected.add(device['power_domain'])
    if not selected:
        return Result('resource','unavailable',work.table('resource'),coverage,['target device identity unknown or conflicting'])
    work.db.executescript('''create table if not exists telemetry(entity text,metric text,ts integer,payload text);
        create index if not exists telemetry_lookup on telemetry(entity,metric,ts);''')
    descriptors = {d['name']:d for d in inventory['descriptors']}
    allowed = selected_entities(ref, inventory)
    count = 0
    with StoreView(work.view.run) as view:
        try:
            for event in view.iter_monitor_events(types=('counter',), work=work.directory, limits=asdict(work.limits)):
                if event.get('ts') is None:
                    continue
                entity = event['dev']
                if entity not in allowed:
                    continue
                ext = event['telemetry']
                for metric,value in event['values'].items():
                    point = dict(ts=event['ts'], value=value, dur=event.get('dur',0),
                                 interval_ns=ext.get('interval_ns',1_000_000_000), miss=event.get('miss',{}).get(metric))
                    if metric in ext.get('sample_times',{}):
                        point['sample_ts'] = ext['sample_times'][metric]
                    work.db.execute('insert into telemetry values(?,?,?,?)',
                                    (entity,metric,point.get('sample_ts',point['ts']),encode(point)))
                count += 1
                if count%256 == 0:work.check_budget()
        except (OSError, ValueError) as error:
            notes.append(str(error))
        coverage['monitor_read'] = view.monitor_coverage()
        notes.extend(error['reason'] for error in coverage['monitor_read']['read_report']['errors'])
    work.db.commit()
    coverage['counter_events'] = count
    begin = ref['window'][0]
    end = ref['window'][1] or inventory.get('ended_raw_ns')
    if end is None:
        from flagprof.telemetry.contract import raw_clock
        end = raw_clock()
    conflicts = profiler_windows(work, (begin,end))
    # SQLite holds the full source. Materialize only one metric at a time within
    # the caller's existing analysis memory budget; record any omitted suffix.
    max_points = max(2, work.limits.cache_bytes//512)
    groups = list(work.db.execute('select distinct entity,metric from telemetry'))
    totals = defaultdict(list)
    windows = [('run', 'run', begin, end, {})]
    requests=[]
    for run,w,seq,event in work.events(('span','compile')):
        if event.get('ts') is None or not event.get('dur'):
            continue
        if event['t']=='span' and (event.get('ph')!='x' or event['cat'] not in ('phase','step','module','user') or
                (event['cat']=='module' and not root_module(work,run,w,event))):
            continue
        item=(str((run,w,seq)),event.get('name','compile:'+event.get('backend','unknown')),
              event['ts'],event['ts']+event['dur'],event.get('attrs',{}))
        windows.append(item)
        if item[4].get('kind')=='generation_request':requests.append(item)
    preexisting={}
    def load_points(entity,metric):
        rows=work.db.execute('select payload from telemetry where entity=? and metric=? order by ts limit ?',
                            (entity,metric,max_points+1)).fetchall()
        if len(rows)>max_points:
            notes.append('resource sample limit: '+entity+'/'+metric);rows=rows[:max_points]
        return [json.loads(row[0]) for row in rows]
    for entity,metric in groups:
        if metric!='memory.used':continue
        baseline=inventory.get('idle_baselines',{}).get(entity,{})
        prior=work.db.execute('select payload from telemetry where entity=? and metric=? and ts<=? order by ts desc limit 1',
                              (entity,metric,begin)).fetchone()
        point=json.loads(prior[0]) if prior else {}
        value=point.get('value')
        extra=(value-baseline['bytes']>baseline['tolerance']) if value is not None and baseline and not point.get('miss') else None
        preexisting[entity]=extra
        work.row('resource.memory',encode((entity,'pre_sample')),dict(entity=entity,kind='pre_sample',
            status='available' if extra is not None else 'unavailable',preexisting_usage=extra,
            observed_bytes=value,baseline=baseline,reason='no_pre_sample_or_baseline' if extra is None else None))
        if extra:notes.append(entity+': 启动前已有额外显存占用')
    energy_rows=defaultdict(dict)
    for entity,metric in groups:
        points=load_points(entity,metric)
        descriptor = descriptors.get(metric,dict(name=metric,unit='unknown',kind='instant'))
        period = int(sorted(p['interval_ns'] for p in points)[len(points)//2]) if points else 1_000_000_000
        for key,name,a,b,attrs in windows:
            result = statistics(points,descriptor,(a,b),period,conflicts)
            totals[(entity,metric,name)].append(result)
            row = dict(entity=entity,metric=metric,unit=descriptor['unit'],name=name,level='instance',
                       window=[a,b], evidence='time_candidate', **{k:v for k,v in result.items() if not k.startswith('_')})
            # Start-of-interval baseline comes from the preceding observed point.
            prior = [p for p in points if p.get('sample_ts',p['ts']) <= a and p.get('value') is not None and not p.get('miss')]
            if prior and a-prior[-1].get('sample_ts',prior[-1]['ts']) <= period and result['maximum'] is not None:
                row.update(start_value=prior[-1]['value'],jump=result['maximum']-prior[-1]['value'])
            work.row('resource',encode((entity,metric,key)),row)
            restriction = next((r.rsplit('.',1)[-1] for r in descriptor.get('roles',[]) if r.startswith('restriction.cumulative.')),None)
            if restriction:
                value=violation(points,(a,b))
                work.row('resource.restrictions',encode((entity,metric,key)),dict(entity=entity,name=name,
                    reason=restriction,window=[a,b],**value))
            if metric in ('power.draw','energy.total'):
                effective=[dict(ts=p.get('sample_ts',p['ts']),value=p['value'],miss=p.get('miss')) for p in points]
                value=energy(effective,(a,b),cumulative=metric=='energy.total',max_gap_ns=2*period)
                members=entities.get(entity,{}).get('members',[entity])
                tokens=attrs.get('generated_tokens')
                if tokens is None:
                    contained=[r for r in requests if a<=r[2] and r[3]<=b]
                    tokens=sum(r[4].get('generated_tokens',0) for r in contained) or None
                energy_per_token(value,tokens,members,selected,any(preexisting.get(m) for m in members) or
                                 bool(set(members) & set(ref.get('shared_devices',[]))))
                value.update(entity=entity,name=name,window=[a,b],power_domain=entity,evidence='time_candidate',source_metric=metric)
                energy_rows[(entity,key)][metric]=value
        if metric == 'clock.compute':
            for index,row in enumerate(low_frequency(points,load_points(entity,'clock.compute.max'),period)):
                a,b=row['window']
                if a<end and b>begin:
                    row['window']=[max(a,begin),min(b,end)]
                    context={}
                    for related in (n for n,d in descriptors.items() if set(d.get('roles',[])) & {'activity','temperature','power'}):
                        domain=entities.get(entity,{}).get('power_domain',entity) if related=='power.draw' else entity
                        samples=load_points(domain,related)
                        if samples:
                            context[related]={k:v for k,v in statistics(samples,descriptors.get(related,{}),row['window'],period,conflicts).items() if not k.startswith('_')}
                    restrictions={role.rsplit('.',1)[-1]:violation(load_points(entity,name),row['window'])
                                  for name,d in descriptors.items() for role in d.get('roles',[])
                                  if role.startswith('restriction.cumulative.')}
                    work.row('resource.frequency',encode((entity,index)),dict(entity=entity,context=context,
                        restrictions=restrictions,restriction_scope='occurrence within interval; not stage performance causality',**row))
        if metric == 'memory.used':
            baseline = inventory.get('idle_baselines',{}).get(entity,{})
            extra = preexisting.get(entity)
            observed = 0
            for run,w,seq,event in work.events(('counter',)):
                if event.get('src')!='framework' or event.get('ext',{}).get('telemetry',{}).get('entity')!=entity or event.get('ts') is None:
                    continue
                ext=event.get('ext',{}).get('telemetry',{})
                if 'reserved_bytes' not in event['values']:continue
                observed += 1
                value=residual(dict(ts=event['ts'],value=event['values']['reserved_bytes'],observed=ext.get('observed',False)),
                    points,baseline=baseline.get('bytes'),tolerance=baseline.get('tolerance'),interval_ns=period,preexisting=extra)
                if extra is None and value['status']=='available':
                    value=dict(status='unavailable',reason='pre_sample_unavailable')
                if entity in ref.get('shared_devices',[]):
                    value=dict(status='unavailable',reason='shared_device_targets')
                work.row('resource.memory',encode((run,w,seq)),dict(entity=entity,ts=event['ts'],**value))
            if not observed:
                work.row('resource.memory',encode((entity,'framework')),dict(entity=entity,status='unavailable',reason='framework_unobserved'))
    for (entity,key), sources in energy_rows.items():
        counter=sources.get('energy.total',{})
        primary='energy.total' if counter.get('joules') is not None else 'power.draw'
        for metric,row in sources.items():
            row['primary']=metric==primary
            work.row('resource.energy',encode((entity,metric,key)),row)
    for (entity,metric,name), rows in totals.items():
        value=aggregate(rows)
        work.row('resource.stages',encode((entity,metric,name)),dict(entity=entity,metric=metric,name=name,
                 level='stage',evidence='time_candidate',**value))
    coverage.update(stages=work.table('resource.stages'),energy=work.table('resource.energy'),
                    memory=work.table('resource.memory'),restrictions=work.table('resource.restrictions'),frequency=work.table('resource.frequency'),profiler_conflicts=conflicts,
                    process_metrics='available' if any(e.startswith('proc:') for e,m in groups) else 'worker_not_observed')
    status='partial' if notes or ref['status']!='complete' else 'ok'
    if not count:status='unavailable'
    return Result('resource',status,work.table('resource'),coverage,notes+[
        '资源与阶段重叠表示时间候选；窗口均值最大值不代表瞬时峰值。',
        '显存估算保留配对时差和负残差；能耗覆盖率不补齐缺失区间。'])


def make_resource():
    return Analyzer('resource',resource,Requires((Need(events=('span','counter')),)),requires_min_schema='1.3')
