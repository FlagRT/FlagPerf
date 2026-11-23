"""Public session report: recent raw coverage, trends and local Findings."""
from html import escape
import json
from pathlib import Path
import re
from flagprof.store import StoreView
from flagprof.store.manifest import atomic_json
from flagprof.telemetry.contract import raw_clock
from .session import load


def duration(value):
    match=re.fullmatch(r'([1-9][0-9]*)([smhd])',value)
    if not match: raise ValueError('last requires positive duration, e.g. 1h')
    return int(match[1])*dict(s=1,m=60,h=3600,d=86400)[match[2]]


def lines(points,lo,hi):
    """Missing buckets and explicit null observations break the plotted path."""
    groups=[];group=[];previous=None
    for slot,value in sorted(points.items()):
        if value is None or previous is not None and slot>previous+1:
            if group:groups.append(group)
            group=[]
        if value is not None:
            group.append('%.1f,%.1f'%(slot*2,90-80*(value-lo)/max(1,hi-lo)))
        previous=slot
    if group:groups.append(group)
    return groups


def generate(session, out=None, *, last='1h'):
    root=Path(session).resolve();state=load(root)
    out=Path(out or str(root)+'-report').resolve()
    if root==out or root in out.parents:raise ValueError('report must be outside session')
    out.mkdir(parents=True,exist_ok=False)
    end=state.get('ended_raw_ns') or raw_clock()
    begin=max(state['started_raw_ns'],end-duration(last)*1_000_000_000)
    rows={};aggregates={};events=0;errors=[]
    from flagprof.report.curves import unit
    specifications={m['name']:m for m in state['inventory'].get('descriptors',[]) if isinstance(m,dict)}
    with StoreView(root) as view:
        try:
            for event in view.iter_monitor_events(('counter',)):
                ts=event.get('ts')
                if ts is None or not begin<=ts<=end:continue
                events+=1
                for metric,value in event['values'].items():
                    sample_ts=event.get('telemetry',{}).get('sample_times',{}).get(metric,ts)
                    if not begin<=sample_ts<=end:continue
                    key=(event['dev'],metric)
                    if key not in rows and len(rows)>=4096:raise ValueError('report series capacity')
                    row=rows.setdefault(key,dict(entity=key[0],metric=key[1],valid=0,missing=0,minimum=None,maximum=None,points={},envelope={},unit=specifications.get(metric,{}).get('unit',unit(metric))))
                    slot=min(255,int((sample_ts-begin)*256/max(1,end-begin)))
                    bucket=row['envelope'].setdefault(slot,dict(first=[sample_ts,value],last=[sample_ts,value],minimum=None,maximum=None,missing=False))
                    if sample_ts<bucket['first'][0]:bucket['first']=[sample_ts,value]
                    if sample_ts>=bucket['last'][0]:bucket['last']=[sample_ts,value]
                    if value is not None:
                        for kind,better in (('minimum',lambda a,b:a<b),('maximum',lambda a,b:a>b)):
                            if bucket[kind] is None or better(value,bucket[kind][1]):bucket[kind]=[sample_ts,value]
                    if value is None:
                        bucket['missing']=True
                        row['missing']+=1;row['points'][slot]=None;continue
                    row['valid']+=1
                    row['minimum']=value if row['minimum'] is None else min(value,row['minimum'])
                    row['maximum']=value if row['maximum'] is None else max(value,row['maximum'])
                    # A bucket containing any missing observation remains a gap.
                    if slot not in row['points'] or row['points'][slot] is not None:row['points'][slot]=value
        except (ValueError,OSError) as error:errors.append(str(error))
        coverage=view.monitor_coverage();errors.extend(coverage['read_report']['errors'])
        for minute in view.iter_monitor_minutes():
            ts=minute['minute']*60_000_000_000
            if ts+60_000_000_000<begin or ts>end:continue
            for value in minute['rows']:
                key=(value['entity'],value['metric'])
                if key not in aggregates and len(aggregates)>=4096:
                    raise ValueError('aggregate report series capacity')
                row=aggregates.setdefault(key,dict(entity=key[0],metric=key[1],count=0,missing=0,total=0.,points={}))
                row['count']+=value['count'];row['missing']+=value['missing'];row['total']+=value['total']
                slot=max(0,min(255,int((ts-begin)*256/max(1,end-begin))))
                bucket=row['points'].setdefault(slot,dict(total=0.,count=0,missing=0))
                for name in ('total','count','missing'):bucket[name]+=value[name]
    for row in rows.values():
        points=[]
        for slot in range(256):
            bucket=row['envelope'].get(slot)
            if not bucket:
                points.append([begin+slot*(end-begin)//256,None]);continue
            candidates=[bucket[k] for k in ('first','minimum','maximum','last') if bucket[k] is not None]
            if bucket['missing']:
                # Keep extrema as isolated points inside an explicitly missing bucket.
                for timestamp,value in sorted(candidates):points.extend([[timestamp,None],[timestamp,value],[timestamp,None]])
            else:points.extend(sorted({tuple(v) for v in candidates}))
        row['samples_points']=points;del row['envelope']
    findings=[]
    for folder in sorted((root/'findings').glob('*')):
        for path in sorted(folder.glob('*.json')):
            row=json.loads(path.read_text())
            if begin<=row['observed_at']<=end:
                if len(findings)>=4096:errors.append('finding display capacity');break
                findings.append(row)
    result=dict(schema='monitor-report/1',session_id=state['id'],status='partial' if errors else state['status'],
        window=[begin,end],raw_counter_events=events,series=list(rows.values()),findings=findings,errors=errors,
        minute_trends=list(aggregates.values()),minute_semantics='versioned derived sample means; not raw coverage or restriction evidence',
        coverage=coverage,
        gaps=state['gaps'],capability_missing=state['inventory'].get('unavailable',{}),
        interpretation='Node observations and fact rules; no inferred workload hang or recovery action.')
    from .diagnostic_report import panel, html as diagnostic_html
    result['diagnosis']=panel(root)
    atomic_json(out/'result.json',result)
    from flagprof.report.product import write_report
    write_report(out,result,filename='index.html')
    return dict(output=str(out),status=result['status'],events=events,findings=len(findings))
