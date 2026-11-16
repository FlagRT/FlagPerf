"""Bounded Chrome/Perfetto export of host spans and resource counters."""
import json
from . import Analyzer, Need, Requires, Result
from flagprof.store import StoreView
from .resource import selected_entities, root_module


def export(work, path, *, max_events=1_000_000):
    if type(max_events) is not int or max_events<1:
        raise ValueError('positive timeline event limit required')
    state=work.view.monitor_status()
    ref,monitor,inventory=(state[k] for k in ('reference','path','inventory'))
    fallback=work.db.execute('select min(ts) from events where ts is not null').fetchone()[0]
    origin=ref.get('window',[fallback or 0])[0]
    tracks={};count=0;truncated=False;read_errors=[]
    selected=selected_entities(ref,inventory or {})
    end=ref.get('window',[None,None])[1]
    def track(entity,metric):
        key=(entity,metric)
        if key not in tracks:tracks[key]=len(tracks)+1
        return tracks[key]
    def events():
        for run,w,seq,event in work.events(('span','compile')):
            if event.get('ts') is None or not event.get('dur'):continue
            if event['t']=='span' and (event.get('ph')!='x' or event['cat'] not in ('phase','step','module','user') or
                    (event['cat']=='module' and not root_module(work,run,w,event))):continue
            yield dict(ph='X',name=event.get('name',event.get('backend','compile')),
                cat=event.get('cat','compile'),ts=(event['ts']-origin)/1000,dur=event['dur']/1000,
                pid=1,tid=event.get('tid',w),args=dict(run_id=run,writer=w,seq=seq,**event.get('attrs',{})))
        if monitor is not None:
            with StoreView(work.view.run) as view:
                try:
                    for event in view.iter_monitor_events(('counter',),work=work.directory):
                        if event.get('ts') is None or event['dev'] not in selected:continue
                        times=event['telemetry'].get('sample_times',{})
                        for metric,value in event['values'].items():
                            # Explicit missing points are instant annotations,
                            # never numeric zeros or interpolation across a gap.
                            timestamp=times.get(metric,event['ts'])
                            if timestamp<origin or (end is not None and timestamp>end):continue
                            args=dict(value=value) if value is not None else dict(miss=event.get('miss',{}).get(metric,'unknown'))
                            row=dict(ph='C' if value is not None else 'i',name=event['dev']+'/'+metric,
                                ts=(timestamp-origin)/1000,pid=2,tid=track(event['dev'],metric),args=args)
                            if value is None:row['s']='t'
                            yield row
                except (ValueError,OSError) as error:
                    read_errors.append(str(error))
                read_errors.extend(e['reason'] for e in view.monitor_coverage()['read_report']['errors'])
    with open(path,'w') as stream:
        stream.write('{"traceEvents":[')
        for event in events():
            if count>=max_events:
                truncated=True;break
            if count:stream.write(',')
            json.dump(event,stream,ensure_ascii=False,allow_nan=False)
            count+=1
        metadata=dict(origin_raw_ns=origin,time_unit='us',events=count,max_events=max_events,
                      truncated=truncated,read_errors=read_errors,monitor_status=ref['status'],
                      clock='same_boot_monotonic_raw',counter_null='gap_annotation')
        stream.write('],"displayTimeUnit":"ms","metadata":')
        json.dump(metadata,stream,ensure_ascii=False);stream.write('}\n')
    return metadata


def timeline(work,params):
    if not params.get('out'):
        return Result('perf.timeline','unavailable',[],{},['export output path required'])
    meta=export(work,params['out'],max_events=params.get('max_events',1_000_000))
    return Result('perf.timeline','partial' if meta['truncated'] or meta['read_errors'] else 'ok',[],meta)


def make_perf_timeline():
    return Analyzer('perf.timeline',timeline,Requires((Need(events=('span','counter')),)))
