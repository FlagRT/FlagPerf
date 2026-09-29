# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Offline interpretation of sealed communication observations; never a preview gate."""
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import json
import re
from runtime.common import read_json, write_json


def ns(value): return int(Decimal(str(value))*1000)


def union_ns(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end < start: raise ValueError('negative interval')
        if merged and start <= merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
        else: merged.append([start,end])
    return sum(b-a for a,b in merged)


def overlap_ns(left,right):
    return union_ns([(max(a,c),min(b,d)) for a,b in left for c,d in right if max(a,c)<min(b,d)])


def flatten_communication(data):
    rows=[]
    for step,groups in data.items():
        for kind in ['collective','p2p']:
            for name,row in groups.get(kind,{}).items():
                if name == 'Total Op Info': continue
                rows.append({'step':step,'kind':kind,'name':name,**row})
    return rows


def matrix_totals(data,rank):
    """Use only total view, source-owned directed edges. LOCAL is separate."""
    rows=[]
    for step,groups in data.items():
        for kind in ['collective','p2p']:
            for name,edges in groups.get(kind,{}).items():
                if '-total@' not in name: continue
                for edge,values in edges.items():
                    match=re.fullmatch(r'(\d+)-(\d+)',edge)
                    if not match: raise ValueError(f'unknown matrix edge: {edge}')
                    src,dst=map(int,match.groups())
                    if src != rank: continue
                    size=values.get('Transit Size(MB)');duration=values.get('Transit Time(ms)')
                    if size is None or duration is None: raise ValueError('matrix missing transit size/time')
                    rows.append({'step':step,'group_collective':name,'src_rank':src,'dst_rank':dst,
                        'transport':values.get('Transport Type','unknown'),'transit_size_mb':size,
                        'transit_time_ms':duration,'source':'communication_matrix.json/*-total@group',
                        'volume_scope':'CANN reported directed link task volume; not wire counter'})
    return rows


def analyze_rank(folder,rank):
    folder=Path(folder)
    issues=[]
    result=read_json(folder/'result.json')
    calls=read_json(folder/'collectives.json')
    profile_folder=folder/'profiler'
    recovery_path=folder/'profiler-recovery/recovery.json'
    if recovery_path.is_file():
        recovery=read_json(recovery_path)
        if recovery.get('status')=='completed':
            from analysis.layer_trace import capture_hashes
            source=folder/recovery['source']
            if capture_hashes(source)!=recovery['source_sha256']:
                raise ValueError('raw profiler capture changed after independent export')
            profile_folder=folder/'profiler-recovery/capture'
    def one(name):
        files=list(profile_folder.rglob(name))
        if len(files)!=1: raise ValueError(f'{name}: expected one capture, got {len(files)}')
        return read_json(files[0]),str(files[0].relative_to(folder))
    comm,comm_path=one('communication.json')
    matrix,matrix_path=one('communication_matrix.json')
    trace,trace_path=one('trace_view.json')
    events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
    metadata={name:{e['pid']:e.get('args',{}) for e in events if e.get('ph')=='M' and e.get('name')==name}
              for name in ['process_name','process_labels','process_sort_index']}
    def same_host_process(marker,event):
        pid,parent=event.get('pid'),marker.get('pid')
        if pid==parent: return True
        # CANN 9 TraceViewManager.get_format_pid: high PID bits, category, host device 31.
        # Require the trace's own CANN/CPU/category metadata, not merely matching timestamps.
        index=metadata['process_sort_index'].get(pid,{}).get('sort_index')
        return (type(pid) is int and type(parent) is int and type(index) is int and 0<=index<32
                and metadata['process_name'].get(pid,{}).get('name')=='CANN'
                and metadata['process_labels'].get(pid,{}).get('labels')=='CPU'
                and pid==((parent<<10)|(index<<5)|31)
                and event.get('args',{}).get('Thread Id')==marker.get('tid'))
    ops=flatten_communication(comm)
    markers={e['name']:e for e in events if e.get('ph')=='X' and e.get('name','').startswith('flagperf/collective/')}
    hccl=[e for e in events if e.get('ph')=='X' and e.get('name','').lower().startswith('hcom_')]
    # CPU marker -> enqueue/dequeue correlation -> CANN launch connection -> HCCL.
    by_name={e['name']:e for e in hccl}
    enqueues=[e for e in events if e.get('name')=='Enqueue@HcclAllreduce']
    dequeues={e.get('args',{}).get('correlation_id'):e for e in events
              if e.get('name')=='Dequeue@HcclAllreduce'}
    launches=[e for e in events if e.get('name')=='Node@launch' and
              str(e.get('args',{}).get('item_id','')).lower().startswith('hcom_allreduce')]
    def contains(parent,child):
        return (ns(parent['ts']) <= ns(child['ts']) and
                ns(child['ts'])+ns(child['dur']) <= ns(parent['ts'])+ns(parent['dur']))
    mapped={}
    for call in calls:
        marker=markers.get(call['marker'])
        if marker is None: continue
        candidates=[e for e in enqueues if e.get('pid')==marker.get('pid') and
                    e.get('tid')==marker.get('tid') and contains(marker,e)]
        correlation=None
        method='marker_enqueue_dequeue_cann_connection'
        if len(candidates)==1:
            correlation=candidates[0].get('args',{}).get('correlation_id')
            dequeue=dequeues.get(correlation)
            if dequeue is None: continue
            candidates=[e for e in launches if e.get('tid')==dequeue.get('tid') and contains(dequeue,e)]
        elif not candidates:
            # TASK_QUEUE_ENABLE=0 may launch synchronously inside this exact CPU scope.
            method='marker_direct_cann_connection'
            candidates=[e for e in launches if same_host_process(marker,e) and
                        e.get('tid')==marker.get('tid') and contains(marker,e)]
            if len(candidates)==1 and candidates[0].get('pid')!=marker.get('pid'):
                method='marker_encoded_pid_cann_connection'
        else:
            continue
        if len(candidates)!=1: continue
        connection=candidates[0].get('args',{}).get('connection_id')
        candidates=[e for e in hccl if e.get('args',{}).get('connection_id')==connection]
        if len(candidates)!=1: continue
        event=candidates[0]
        if int(event.get('args',{}).get('count',-1))!=call['elements']: continue
        if event['name'] in mapped:
            raise ValueError('multiple collective markers resolve to one HCCL event')
        mapped[event['name']]=dict(call,correlation_id=correlation,connection_id=connection,attribution_method=method)
    expected=result['expected_model_collectives']
    if not (len(calls)==len(ops)==len(hccl)==len(markers)==len(mapped)==expected
            and len(by_name)==len(hccl)
            and {o['name'].split('@')[0] for o in ops}==set(by_name)):
        issues.append('collective count/correlation/payload coverage incomplete')
    ordered=sorted(ops,key=lambda o:mapped.get(o['name'].split('@')[0],{}).get('sequence',10**9))
    rows=[]
    for index,op in enumerate(ordered):
        name=op['name'].split('@')[0]
        event=by_name.get(name)
        call=mapped.get(name,{})
        valid=bool(call)
        row={'hccl_op':op['name'],'rank':rank,'category':call.get('category','unattributed'),
             'module':call.get('module'),'logical_bytes':call.get('logical_bytes'),
             'batch_index':call.get('batch_index'),'cycle':call.get('cycle'),
             'backend':call.get('backend','unknown'),
             'attribution_method':call.get('attribution_method','unmatched'),
             'attribution_status':'correlated' if valid else 'unknown',
             'time':op.get('Communication Time Info',{}),
             'bandwidth_by_transport':op.get('Communication Bandwidth Info',{})}
        if event:
            row['device_start_ns']=ns(event['ts']);row['device_elapsed_ns']=ns(event['dur'])
        if valid:
            row['host_api_scope_ns']=ns(markers[call['marker']]['dur'])
            row['host_api_start_ns']=ns(markers[call['marker']]['ts'])
            row['correlation_id']=call['correlation_id']
            row['connection_id']=call['connection_id']
            row['sequence']=call['sequence']
        rows.append(row)
    links=matrix_totals(matrix,rank)
    if not any(r['src_rank']!=r['dst_rank'] and r['transit_size_mb']>0 for r in links):
        issues.append('no positive inter-rank link volume')
    required=['Elapse Time(ms)','Transit Time(ms)','Wait Time(ms)','Synchronization Time(ms)','Idle Time(ms)']
    if any(any(k not in r['time'] for k in required) for r in rows):
        issues.append('communication time fields missing')
    categories={}
    for row in rows:
        group=categories.setdefault(row['category'],{'collectives':0,'logical_payload_bytes':0,
                 'device_elapsed_ms_sum':0,'wait_ms_sum':0,'synchronization_ms_sum':0,
                 'transit_time_info_ms_sum':0,'idle_ms_sum':0})
        group['collectives']+=1
        if row['logical_bytes'] is None: group['logical_payload_bytes']=None
        elif group['logical_payload_bytes'] is not None: group['logical_payload_bytes']+=row['logical_bytes']
        for dest,source in [('device_elapsed_ms_sum','Elapse Time(ms)'),('wait_ms_sum','Wait Time(ms)'),
                            ('synchronization_ms_sum','Synchronization Time(ms)'),
                            ('transit_time_info_ms_sum','Transit Time(ms)'),('idle_ms_sum','Idle Time(ms)')]:
            value=row['time'].get(source)
            if value is None: group[dest]=None
            elif group[dest] is not None: group[dest]+=value
    comm_intervals=[(r['device_start_ns'],r['device_start_ns']+r['device_elapsed_ns']) for r in rows if 'device_start_ns' in r]
    # Only accelerator compute kernels; exclude HCCL AICPU/SDMA and host wrappers.
    compute=[]
    for e in events:
        task=e.get('args',{}).get('Task Type','')
        if e.get('ph')=='X' and task in ['AI_CORE','MIX_AIC','MIX_AIV','AI_VECTOR_CORE']:
            compute.append((ns(e['ts']),ns(e['ts'])+ns(e['dur'])))
    interval={'communication_union_ms':union_ns(comm_intervals)/1e6,
              'compute_union_ms':union_ns(compute)/1e6 if compute else None,
              'communication_compute_overlap_ms':overlap_ns(comm_intervals,compute)/1e6 if compute else None,
              'scope':'same rank, same profiled window; not critical-path overhead'}
    if any(r['category']=='unattributed' for r in rows): issues.append('unattributed model communications')
    return {'status':'partial' if issues else 'completed','rank':rank,'issues':issues,
            'observed_collectives':len(rows),'expected_collectives':expected,
            'attributed_collectives':sum(r['category']!='unattributed' for r in rows),
            'categories':categories,'intervals':interval,'links':links,'events':rows,
            'sources':{'communication':comm_path,'matrix':matrix_path,'trace':trace_path}}


def analyze(root,world_size):
    root=Path(root);ranks={}
    for rank in range(world_size):
        try:ranks[str(rank)]=analyze_rank(root/f'rank-{rank}',rank)
        except Exception as error:ranks[str(rank)]={'status':'partial','rank':rank,'issues':[str(error)]}
    rank_skew=[]
    if all(r['status']=='completed' for r in ranks.values()):
        lists=[r['events'] for r in ranks.values()]
        if len({len(v) for v in lists})!=1:
            ranks['0']['status']='partial';ranks['0']['issues'].append('rank collective counts differ')
        else:
            for rows in zip(*lists):
                keys=['sequence','cycle','batch_index','module','logical_bytes']
                if any(any(r.get(k)!=rows[0].get(k) for k in keys) for r in rows):
                    ranks['0']['status']='partial';ranks['0']['issues'].append('rank collective pairing differs');break
                starts=[r['host_api_start_ns'] for r in rows]
                rank_skew.append({'sequence':rows[0]['sequence'],'module':rows[0]['module'],
                                  'host_entry_skew_ns':max(starts)-min(starts),
                                  'scope':'same-host trace CPU markers; diagnostic, not causal proof'})
    result={'status':'completed' if all(r['status']=='completed' for r in ranks.values()) else 'partial',
            'rank_entry_skew':rank_skew,
            'ranks':ranks,'scope':'independent profiled replay; never added to unprofiled latency',
            'aggregation_rules':{'logical_payload':'per rank; replicated TP inputs counted once for model throughput',
                'link_volume':'source-owned directed matrix total rows; LOCAL separate; do not add bandwidth aliases',
                'time':'wait/synchronization overlap; do not add them or sum ranks as wall latency',
                'attribution':'marker through task queue or verified same-thread direct launch to CANN connection ID; checked payload and counts'}}
    write_json(root/'communication.json',result)
    return result
