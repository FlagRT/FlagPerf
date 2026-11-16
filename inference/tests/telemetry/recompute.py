"""Independent interval-cell oracle for delivery checks, never imported online."""
import argparse
import json
import math
from pathlib import Path
from flagprof.store import StoreView


def recompute(points,descriptor,window,period,conflicts):
    points=sorted(points,key=lambda p:p['time']);begin,end=window;segments=[]
    for index,p in enumerate(points):
        t=p['time'];step=p.get('period') or period;width=descriptor.get('window_ns') or step
        source=(t-width,t)
        if p['value'] is None or p.get('miss'):continue
        if descriptor.get('profiler_conflict') and any(source[0]<b and source[1]>a for a,b in conflicts):continue
        if descriptor.get('kind')=='window':
            if source[0]<begin or source[1]>end:continue
            a,b=source
            if index and (points[index-1]['value'] is None or points[index-1].get('miss')):a=max(a,points[index-1]['time'])
        else:a,b=t,min(t+step,points[index+1]['time'] if index+1<len(points) else t+step)
        a,b=max(a,begin),min(b,end)
        if b>a:segments.append((a,b,index,p['value']))
    boundaries=sorted({x for s in segments for x in s[:2]});weights={}
    for a,b in zip(boundaries,boundaries[1:]):
        owners=[s for s in segments if s[0]<=a and s[1]>=b]
        if owners:
            chosen=max(owners,key=lambda s:s[2]);key=(chosen[2],chosen[3]);weights[key]=weights.get(key,0)+b-a
    duration=sum(weights.values());minimum=(descriptor.get('window_ns') or (period if descriptor.get('kind')=='window' else 0))+period
    valid=len(weights)>=2 and end-begin>=minimum and duration>0
    values=sorted((k[1],w) for k,w in weights.items());p95=None;total=0
    if valid and len(weights)>=20:
        for value,weight in values:
            total+=weight
            if total>=.95*duration:p95=value;break
    return dict(status='available' if valid else 'insufficient_resolution',samples=len(weights),covered_ns=duration,
        coverage=duration/(end-begin),mean=sum(k[1]*w for k,w in weights.items())/duration if valid else None,
        maximum=max((k[1] for k in weights),default=None) if valid else None,p95=p95)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    ref=json.loads((a.run/'monitor-ref.json').read_text());monitor=(a.run/ref['monitor_path']).resolve()
    inv=json.loads((monitor/'inventory.json').read_text());descriptors={d['name']:d for d in inv['descriptors']};groups={}
    with StoreView(monitor) as view:
        for stored in view.iter_events(('counter',)):
            event=stored.record;ext=event.get('ext',{}).get('telemetry',{})
            for metric,value in event['values'].items():
                groups.setdefault((event['dev'],metric),[]).append(dict(time=ext.get('sample_times',{}).get(metric,event['ts']),
                    value=value,miss=event.get('miss',{}).get(metric),period=ext.get('interval_ns',10**9)))
    result=json.loads(a.report.read_text())['results']['resource'];checked=0
    # Report JSON previews can truncate. The resource table artifact contains every row.
    rows=result['data']
    table=a.report.parent/'exports/resource.jsonl'
    if table.exists():rows=[json.loads(line) for line in table.read_text().splitlines()]
    for row in rows:
        if row['name'] not in ('run','load','warmup','measure','inject.memory','inject.idle','control.decode'):continue
        points=groups[row['entity'],row['metric']];period=sorted(p['period'] for p in points)[len(points)//2]
        expected=recompute(points,descriptors[row['metric']],row['window'],period,result['coverage']['profiler_conflicts'])
        for key,value in expected.items():
            actual=row[key]
            if isinstance(value,(int,float)):
                assert actual is not None and math.isclose(actual,value,rel_tol=1e-10,abs_tol=1e-6),(row['name'],row['metric'],key,value,actual)
            else:assert actual==value,(row['name'],row['metric'],key,value,actual)
        checked+=1
    assert checked>0
    a.out.write_text(json.dumps(dict(checked_rows=checked,run=str(a.run),status='pass'),indent=2)+'\n')

if __name__=='__main__':main()
