"""Kineto/CUPTI attribution through explicit correlation IDs and host-thread scopes."""
from collections import defaultdict
import gzip
import json
from runtime.common import read_json
from runtime.trace_utils import ns, union_ns, overlap_ns
from vendors.nvidia.scope_index import ThreadScopeIndex


def trace_files(folder):
    return sorted([*(folder/'profiler').rglob('*.pt.trace.json'),
                   *(folder/'profiler').rglob('*.pt.trace.json.gz')])


def load(folder, progress=None):
    files=trace_files(folder)
    if len(files)!=1: raise ValueError('expected exactly one CUDA trace')
    p=files[0]
    if progress: progress('loading', source=p.relative_to(folder).as_posix(), total_bytes=p.stat().st_size)
    from vendors.nvidia.trace_index import open_events
    indexed = open_events(p, folder/'.work', progress)
    value = dict(indexed.metadata, traceEvents=indexed)
    return value, p.relative_to(folder).as_posix()


def contains(scope,host):
    return (scope.get('pid')==host.get('pid') and scope.get('tid')==host.get('tid') and
            ns(scope['ts']) <= ns(host['ts']) < ns(scope['ts'])+ns(scope.get('dur',0)))


def _scope_owners(scopes, hosts, progress=None):
    """Query exact thread intervals, retaining original call order."""
    unique={id(host):host for host in hosts if host is not None}
    threads={(host.get('pid'),host.get('tid')) for host in unique.values()}
    if progress: progress('scope_index', intervals=len(scopes), unique_hosts=len(unique))
    index=ThreadScopeIndex((order,dict(scope,dur=scope.get('dur',0)))
                     for order,(_,scope) in enumerate(scopes)
                     if (scope.get('pid'),scope.get('tid')) in threads)
    output={}
    for number,(key,host) in enumerate(unique.items()):
        if progress and number % 65536 == 0:
            progress('owner_lookup', completed=number, total=len(unique))
        thread=(host.get('pid'),host.get('tid'))
        if thread in index.groups:
            labels=sorted(index.at(*thread,ns(host['ts'])))
            output[key]=[scopes[label][0]['sequence'] for label in labels]
    return output


def linked_events(trace, progress=None):
    events=trace['traceEvents']
    launches=defaultdict(list)
    external=defaultdict(list)
    for number,e in enumerate(()) if hasattr(events, 'hosts') else enumerate(events):
        if progress and number % 65536 == 0:
            progress('host_index', completed=number, total=len(events))
        if e.get('ph')!='X': continue
        a=e.get('args',{})
        if e.get('cat') in ('cuda_runtime','cuda_driver') and a.get('correlation') is not None:
            launches[str(a['correlation'])].append(e)
        if e.get('cat')=='cpu_op' and a.get('External id') is not None:
            external[str(a['External id'])].append(e)
    output=events.kernel_sink() if hasattr(events, 'kernel_sink') else []
    for i,e in events.device_events() if hasattr(events, 'device_events') else enumerate(events):
        if progress and i % 65536 == 0:
            progress('event_linking', completed=i, total=len(events))
        cat=e.get('cat','')
        if e.get('ph')!='X' or cat not in ('kernel','gpu_memcpy','gpu_memset'): continue
        a=e.get('args',{}); corr=a.get('correlation'); ex=a.get('External id')
        candidates=(events.hosts('correlation', corr) if hasattr(events, 'hosts') else launches.get(str(corr),[])) if corr is not None else []
        method='CUPTI correlation'
        if not candidates and ex is not None:
            candidates=events.hosts('external', ex) if hasattr(events, 'hosts') else external.get(str(ex),[])
            method='Kineto external id'
        host=candidates[0] if len(candidates)==1 else None
        kind='communication' if cat=='kernel' and 'nccl' in e.get('name','').lower() else ('compute' if cat=='kernel' else 'copy')
        output.append({'id':i,'name':e.get('name'),'kind':kind,'start_ns':ns(e['ts']),
            'duration_ns':ns(e['dur']),'bytes':a.get('bytes'),'host':host,'method':method,
            'connection_id':str(corr) if corr is not None else None,'external_id':ex,'args':a})
    return output


def attribute(trace,calls, *, events=None, progress=None):
    if hasattr(trace['traceEvents'], 'kernel_sink'):
        return _indexed_attribute(trace, calls, events=events, progress=progress)
    markers=defaultdict(list)
    for number,e in enumerate(trace['traceEvents']):
        if progress and number % 65536 == 0:
            progress('layer_markers', completed=number, total=len(trace['traceEvents']))
        if e.get('ph')=='X' and e.get('cat')!='gpu_user_annotation' and e.get('name','').startswith('flagperf/layer/'):
            markers[e['name']].append(e)
    scopes=[(c,markers[c['marker']][0]) for c in calls if len(markers[c['marker']])==1]
    events=linked_events(trace, progress) if events is None else events
    owners=_scope_owners(scopes,(e['host'] for e in events), progress)
    by_sequence=defaultdict(list)
    for number,e in enumerate(events):
        if progress and number % 65536 == 0:
            progress('attribution', completed=number, total=len(events))
        host=e.pop('host')
        e['owners']=list(owners.get(id(host),[])) if host is not None else []
        e['attribution_status']=('unknown' if host is None else 'correlated' if e['owners'] else 'outside_selected_layers')
        for sequence in set(e['owners']): by_sequence[sequence].append(e)
    complete=bool(events) and len(scopes)==len(calls) and all(e['attribution_status']!='unknown' for e in events)
    rows=[]
    for number,c in enumerate(calls):
        if progress and number % 1024 == 0:
            progress('layer_summary', completed=number, total=len(calls))
        row={k:v for k,v in c.items() if k not in ('host_start_ns','host_end_ns')}
        own=by_sequence[c['sequence']]
        row['events']={}
        for kind in ('compute','communication','copy'):
            selected=[e for e in own if e['kind']==kind]
            row['events'][kind]={'count':len(selected),
                'duration_sum_ns':sum(e['duration_ns'] for e in selected) if selected or complete else None,
                'duration_union_ns':union_ns([(e['start_ns'],e['start_ns']+e['duration_ns']) for e in selected]) if selected or complete else None,
                'bytes':sum(e['bytes'] for e in selected) if selected and all(type(e['bytes']) is int for e in selected) else None}
        row['attribution_status']='correlated' if own else 'not_observed'
        rows.append(row)
    known=sum(e['attribution_status']!='unknown' for e in events)
    return {'status':'completed' if complete else 'partial','rows':rows,'events':events,
        'coverage':{'device_events':len(events),'associated_events':known,'unknown_events':len(events)-known,
            'outside_selected_events':sum(e['attribution_status']=='outside_selected_layers' for e in events),
            'fraction':known/len(events) if events else None,'expected_markers':len(calls),'matched_markers':len(scopes)},
        'method':'unique CUPTI correlation / Kineto external ID to host launch, same-thread enclosing module marker; no nearest-time matching',
        'source_semantics':'CUDA device kernels; NCCL kernels are not CANN collective elapsed spans'}


def _indexed_attribute(trace, calls, *, events=None, progress=None):
    """Same attribution rules, with event bodies and ownership kept in SQLite."""
    markers = defaultdict(list)
    if progress: progress('layer_markers')
    for event in trace['traceEvents'].markers('flagperf/layer/'):
        markers[event['name']].append(event)
    scopes = [(call, markers[call['marker']][0]) for call in calls if len(markers[call['marker']]) == 1]
    index = ThreadScopeIndex((order, dict(scope, dur=scope.get('dur', 0)))
                             for order, (_, scope) in enumerate(scopes))
    events = linked_events(trace, progress) if events is None else events
    known, outside = 0, 0
    for number, event in enumerate(events):
        if progress and number % 65536 == 0: progress('attribution', completed=number, total=len(events))
        host = event.pop('host')
        labels = sorted(index.at(host.get('pid'), host.get('tid'), ns(host['ts']))) if host is not None else []
        event['owners'] = [scopes[label][0]['sequence'] for label in labels]
        event['attribution_status'] = 'unknown' if host is None else 'correlated' if labels else 'outside_selected_layers'
        known += host is not None
        outside += event['attribution_status'] == 'outside_selected_layers'
        events.update(event)
    events.finish_owners()
    complete = bool(events) and len(scopes) == len(calls) and known == len(events)
    rows = []
    for number, call in enumerate(calls):
        if progress and number % 1024 == 0: progress('layer_summary', completed=number, total=len(calls))
        row = {k: v for k, v in call.items() if k not in ('host_start_ns', 'host_end_ns')}
        own = list(events.owned(call['sequence']))
        row['events'] = {}
        for kind in ('compute', 'communication', 'copy'):
            selected = [event for event in own if event['kind'] == kind]
            row['events'][kind] = {'count': len(selected),
                'duration_sum_ns': sum(event['duration_ns'] for event in selected) if selected or complete else None,
                'duration_union_ns': union_ns([(event['start_ns'], event['start_ns']+event['duration_ns']) for event in selected]) if selected or complete else None,
                'bytes': sum(event['bytes'] for event in selected) if selected and all(type(event['bytes']) is int for event in selected) else None}
        row['attribution_status'] = 'correlated' if own else 'not_observed'
        rows.append(row)
    if progress: progress('event_shards', records=len(events))
    stored = events.publish()
    return {'status': 'completed' if complete else 'partial', 'rows': rows, 'events': stored,
        'coverage': {'device_events': len(events), 'associated_events': known, 'unknown_events': len(events)-known,
                    'outside_selected_events': outside, 'fraction': known/len(events) if events else None,
                    'expected_markers': len(calls), 'matched_markers': len(scopes)},
        'method': 'unique CUPTI correlation / Kineto external ID to host launch, same-thread enclosing module marker; no nearest-time matching',
        'source_semantics': 'CUDA device kernels; NCCL kernels are not CANN collective elapsed spans'}


def analyze(folder,state,export=None):
    trace,source=load(folder)
    result=attribute(trace,state['rows']);result['source']=source
    return result


def communication(folder,rank):
    trace,source=load(folder)
    calls=read_json(folder/'collectives.json')
    expected=read_json(folder/'result.json')['expected_model_collectives']
    return _communication(trace, source, calls, expected, rank, linked_events(trace))


def _communication(trace, source, calls, expected, rank, kernels, progress=None):
    markers=defaultdict(list)
    raw = trace['traceEvents']
    candidates = raw.markers('flagperf/collective/') if hasattr(raw, 'markers') else raw
    for number,e in enumerate(candidates):
        if progress and number % 65536 == 0:
            progress('communication_markers', completed=number, total=len(trace['traceEvents']))
        if e.get('ph')=='X' and e.get('cat')!='gpu_user_annotation' and e.get('name','').startswith('flagperf/collective/'):
            markers[e['name']].append(e)
    comm=[e for e in kernels if e['kind']=='communication']
    scopes=[({'sequence':index},markers[call['marker']][0])
            for index,call in enumerate(calls) if len(markers[call['marker']])==1]
    owners=_scope_owners(scopes,(e['host'] for e in comm), progress)
    by_call=defaultdict(list)
    for event in comm:
        for index in owners.get(id(event['host']),[]): by_call[index].append(event)
    # NCCL may use an external-id CPU owner inside a marker while launching
    # on a separate communication thread. Index that exact fallback as well.
    fallback=[(call,marker) for call,marker in scopes if not by_call[call['sequence']]]
    if fallback:
        ids_by_call=defaultdict(set)
        index=ThreadScopeIndex((number,dict(marker,dur=marker.get('dur',0)))
                               for number,(_,marker) in enumerate(fallback))
        for event in trace['traceEvents']:
            if event.get('ph')!='X' or event.get('cat')!='cpu_op': continue
            external=event.get('args',{}).get('External id')
            if external is not None:
                for number in index.at(event.get('pid'),event.get('tid'),ns(event['ts'])):
                    ids_by_call[fallback[number][0]['sequence']].add(external)
        by_external=defaultdict(list)
        for event in comm: by_external[event['external_id']].append(event)
        for call,_ in fallback:
            index=call['sequence']
            by_call[index]=sorted((event for external in ids_by_call[index]
                                   for event in by_external[external]),key=lambda event:event['id'])
    issues=[];rows=[];used=[];categories={}
    if len(calls)!=expected: issues.append('logical collective count mismatch')
    base=trace.get('baseTimeNanoseconds',0)
    for index,call in enumerate(calls):
        if progress and index % 1024 == 0:
            progress('communication_summary', completed=index, total=len(calls))
        ms=markers[call['marker']]
        marker=ms[0] if len(ms)==1 else None
        own=by_call[index]
        used.extend(e['id'] for e in own)
        valid=bool(own)
        for e in own:
            args=e['args']
            count=args.get('In msg nelems')
            if count is not None and int(count)!=call['elements']: valid=False
        if not valid: issues.append('unattributed/payload-mismatched collective '+str(call['sequence']))
        duration=sum(e['duration_ns'] for e in own) if valid else None
        row=dict(call,attribution_status='correlated' if valid else 'unknown',
            host_api_start_ns=base+ns(marker['ts']) if marker else None,
            host_api_scope_ns=ns(marker['dur']) if marker else None,
            connection_id=own[0]['connection_id'] if len(own)==1 else None,
            connection_ids=[e['connection_id'] for e in own],
            device_start_ns=min(e['start_ns'] for e in own) if own else None,
            device_elapsed_ns=duration, nccl_kernel_duration_ns=duration,
            time={},source='CUPTI NCCL kernel duration; not collective end-to-end elapsed')
        rows.append(row)
        group=categories.setdefault(call['category'],{'collectives':0,'logical_payload_bytes':0,
            'nccl_kernel_ms_sum':0,'device_elapsed_ms_sum':None,'wait_ms_sum':None,
            'synchronization_ms_sum':None,'transit_time_info_ms_sum':None,'idle_ms_sum':None})
        group['collectives']+=1;group['logical_payload_bytes']+=call['logical_bytes']
        if duration is None: group['nccl_kernel_ms_sum']=None
        elif group['nccl_kernel_ms_sum'] is not None: group['nccl_kernel_ms_sum']+=duration/1e6
    if len(used)!=len(set(used)) or set(used)!={e['id'] for e in comm}:
        issues.append('ambiguous or unassigned NCCL kernels')
    ci=[(e['start_ns'],e['start_ns']+e['duration_ns']) for e in comm]
    if hasattr(kernels, 'intervals'):
        from runtime.trace_utils import merged_intervals
        communication_intervals=merged_intervals(ci)
        position=0;compute_union=overlap=0
        def merged():
            current=None
            for start,end in kernels.intervals('compute'):
                if end<start: raise ValueError('negative interval')
                if current is not None and start<=current[1]: current[1]=max(current[1],end)
                else:
                    if current is not None: yield current
                    current=[start,end]
            if current is not None: yield current
        for start,end in merged():
            compute_union += end-start
            while position<len(communication_intervals) and communication_intervals[position][1]<=start: position+=1
            i=position
            while i<len(communication_intervals) and communication_intervals[i][0]<end:
                a,b=communication_intervals[i];overlap+=max(0,min(end,b)-max(start,a));i+=1
    else:
        compute=[(e['start_ns'],e['start_ns']+e['duration_ns']) for e in kernels if e['kind']=='compute']
        compute_union,overlap=union_ns(compute),overlap_ns(ci,compute)
    return {'status':'partial' if issues else 'completed','rank':rank,'issues':issues,'events':rows,
        'observed_collectives':len(calls),'expected_collectives':expected,'attributed_collectives':sum(r['attribution_status']=='correlated' for r in rows),
        'categories':categories,'intervals':{'communication_union_ms':union_ns(ci)/1e6,
            'compute_union_ms':compute_union/1e6,'communication_compute_overlap_ms':overlap/1e6,
            'scope':'rank-local CUDA kernel intervals, not communication critical-path overhead'},
        'links':[],'link_status':'not_collected','sources':{'trace':source,'logical':'collectives.json'},
        'backend':'nccl','timing_source':'CUPTI kernels','host_clock_base_ns':base}


def has_capture(folder):
    return bool(trace_files(folder))


def analyze_bundle(folder, state, *, rank=None, communication_only=False, progress=None, associate_layers=False):
    """One load/link pass shared by layer attribution and NCCL interpretation."""
    trace,source=load(folder, progress)
    kernels=linked_events(trace, progress)
    communication_result=None
    if rank is not None:
        if progress: progress('communication')
        communication_result=_communication(trace, source, read_json(folder/'collectives.json'),
            state['expected_model_collectives'], rank, kernels, progress)
    if communication_only:
        return communication_result
    # Layer attribution removes host objects. Communication has already consumed
    # them; all returned dictionaries retain the historical public schema.
    result=attribute(trace, state['rows'], events=kernels, progress=progress)
    result['source']=source
    if communication_result is not None:
        result['communication']=communication_result
        if associate_layers:
            if progress: progress('collective_layer_association')
            from runtime.layer_trace import associate_collectives
            # Reuse normalized kernel objects while the index is alive. Reading
            # freshly compressed shards here would decode every kernel again.
            associate_collectives({'rows': result['rows'], 'events': kernels,
                                   'communication': communication_result})
    return result
