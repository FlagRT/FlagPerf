"""CANN communication fields and explicit PID/connection correlation."""
from pathlib import Path
from bisect import bisect_left, bisect_right
from collections import defaultdict
import re
import math
from runtime.common import read_json
from runtime.trace_utils import ns, union_ns, overlap_ns


# ProcessGroupHCCL::allgather TensorList pads equal-sized inputs on pre-950 SoCs.
# This is a runtime buffer contract, not a tolerance for mismatched payloads.
_ALIGNED_GATHER_COMMIT='94f8a8e6b523d7ba553e1b80d5b5248478391526'
_CANN_DTYPES={'torch.bfloat16':('BFP16',2),'torch.float16':('FP16',2),
              'torch.float32':('FP32',4),'torch.float64':('FP64',8),
              'torch.int8':('INT8',1),'torch.uint8':('UINT8',1),
              'torch.int32':('INT32',4),'torch.int64':('INT64',8)}


def payload_evidence(call,event,runtime):
    args=event.get('args',{})
    count=args.get('count')
    if type(count) is not int or count<0: return None
    expected=call['elements']
    dtype=_CANN_DTYPES.get(call.get('dtype'))
    if dtype is not None:
        if args.get('data_type')!=dtype[0] or call['logical_bytes']!=expected*dtype[1]: return None
    relation='logical_input'
    if (call.get('collective')=='all_gather' and call.get('backend')=='hccl'
            and runtime.get('torch_npu_version')=='2.10.0'
            and runtime.get('torch_npu_git_version')==_ALIGNED_GATHER_COMMIT
            and re.fullmatch(r'Ascend910(?:_\w+)?',runtime.get('device_name',''))):
        outputs=call.get('output_tensors',[])
        if dtype is None or not outputs or any(
                t.get('elements')!=expected or t.get('dtype')!=call.get('dtype') for t in outputs): return None
        shape=outputs[0].get('shape')
        if (not isinstance(shape,list) or any(type(n) is not int or n<0 for n in shape)
                or math.prod(shape)!=expected or any(t.get('shape')!=shape for t in outputs)): return None
        if call.get('output_logical_bytes')!=len(outputs)*call['logical_bytes']: return None
        expected=((call['logical_bytes']+511)//512)*512//dtype[1]
        relation='torch_npu_hccl_equal_tensorlist_512_byte_alignment'
    if count!=expected: return None
    return {'backend_elements':count,'backend_input_buffer_bytes':count*dtype[1] if dtype else None,
            'backend_input_padding_bytes':count*dtype[1]-call['logical_bytes'] if dtype else None,
            'payload_relation':relation,
            'buffer_scope':'per-rank runtime input buffer; not wire traffic'}


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


class _ContainedEvents:
    """Exact containment queries within a keyed host-thread event stream.

    Keep every candidate (including duplicates): ambiguous correlations must
    remain ambiguous. Convert timestamps once, not once per collective match.
    """
    def __init__(self, entries):
        groups = defaultdict(list)
        for key, event in entries:
            start = ns(event['ts'])
            groups[key].append((start, start + ns(event['dur']), event))
        self.groups = {}
        for key, rows in groups.items():
            rows.sort(key=lambda row: row[0])
            self.groups[key] = ([row[0] for row in rows], rows,
                                max((start-end for start, end, _ in rows), default=0))

    def within(self, key, parent):
        group = self.groups.get(key)
        if group is None:
            return []
        starts, rows, backwards = group
        start = ns(parent['ts'])
        end = start + ns(parent['dur'])
        left = bisect_left(starts, start)
        right = bisect_right(starts, end + max(0, backwards))
        return [event for _, finish, event in rows[left:right] if finish <= end]


def analyze_rank(folder,rank, *, trace_data=None):
    folder=Path(folder)
    issues=[]
    result=read_json(folder/'result.json')
    calls=read_json(folder/'collectives.json')
    capture_path=folder/'trace-capture.json'
    runtime=read_json(capture_path).get('collective_runtime',{}) if capture_path.is_file() else {}
    profile_folder=folder/'profiler'
    recovery_path=folder/'profiler-recovery/recovery.json'
    if not recovery_path.exists() and any(
            not list(profile_folder.rglob(name))
            for name in ('communication.json', 'communication_matrix.json', 'trace_view.json')):
        # CANN may finish collection but lose its asynchronous export workers.
        # Reuse the same bounded copy-only recovery as layer attribution.
        from vendors.ascend.trace import recover_trace
        recover_trace(folder)
    if recovery_path.is_file():
        recovery=read_json(recovery_path)
        if recovery.get('status')=='completed':
            from vendors.ascend.trace import capture_hashes
            source=folder/recovery['source']
            if capture_hashes(source)!=recovery['source_sha256']:
                raise ValueError('raw profiler capture changed after independent export')
            profile_folder=folder/'profiler-recovery/capture'
    def one(name, supplied=None):
        files=list(profile_folder.rglob(name))
        if len(files)!=1: raise ValueError(f'{name}: expected one capture, got {len(files)}')
        path = str(files[0].relative_to(folder))
        if supplied is not None and supplied[1] == path:
            return supplied[0], path
        from vendors.ascend.trace import read_trace
        return read_trace(files[0]),path
    comm,comm_path=one('communication.json')
    matrix,matrix_path=one('communication_matrix.json')
    # A separately recovered export may differ from the layer capture. Share
    # the parsed trace only when both analyses selected the exact same source.
    trace,trace_path=one('trace_view.json', trace_data)
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
    api_names={'all_reduce':'HcclAllreduce','all_gather':'HcclAllgather','all_gather_into_tensor':'HcclAllgather'}
    operation_names={'all_reduce':'hcom_allreduce','all_gather':'hcom_allgather','all_gather_into_tensor':'hcom_allgather'}
    enqueues=[e for e in events if e.get('name') in {'Enqueue@'+n for n in api_names.values()}]
    dequeues={(e.get('name'),e.get('args',{}).get('correlation_id')):e for e in events
              if e.get('name') in {'Dequeue@'+n for n in api_names.values()}}
    launches=[e for e in events if e.get('name')=='Node@launch' and
              str(e.get('args',{}).get('item_id','')).lower().startswith(tuple(operation_names.values()))]
    enqueue_index = _ContainedEvents(
        ((e.get('name'), e.get('pid'), e.get('tid')), e) for e in enqueues)
    operations = set(operation_names.values())
    launch_index = _ContainedEvents(
        ((operation, e.get('tid')), e) for e in launches for operation in operations
        if str(e.get('args', {}).get('item_id', '')).lower().startswith(operation))
    connections = defaultdict(list)
    for event in hccl:
        for operation in operations:
            if event.get('name', '').lower().startswith(operation):
                connections[(operation, event.get('args', {}).get('connection_id'))].append(event)
    mapped={}
    for call in calls:
        marker=markers.get(call['marker'])
        if marker is None: continue
        collective=call.get('collective','all_reduce')
        if collective not in api_names: continue
        candidates=enqueue_index.within(
            ('Enqueue@'+api_names[collective], marker.get('pid'), marker.get('tid')), marker)
        correlation=None
        method='marker_enqueue_dequeue_cann_connection'
        if len(candidates)==1:
            correlation=candidates[0].get('args',{}).get('correlation_id')
            dequeue=dequeues.get(('Dequeue@'+api_names[collective],correlation))
            if dequeue is None: continue
            candidates=launch_index.within((operation_names[collective], dequeue.get('tid')), dequeue)
        elif not candidates:
            # TASK_QUEUE_ENABLE=0 may launch synchronously inside this exact CPU scope.
            method='marker_direct_cann_connection'
            candidates=[e for e in launch_index.within((operation_names[collective], marker.get('tid')), marker)
                        if same_host_process(marker,e)]
            if len(candidates)==1 and candidates[0].get('pid')!=marker.get('pid'):
                method='marker_encoded_pid_cann_connection'
        else:
            continue
        if len(candidates)!=1: continue
        connection=candidates[0].get('args',{}).get('connection_id')
        candidates=connections.get((operation_names[collective], connection), [])
        if len(candidates)!=1: continue
        event=candidates[0]
        payload=payload_evidence(call,event,runtime)
        if payload is None: continue
        if event['name'] in mapped:
            raise ValueError('multiple collective markers resolve to one HCCL event')
        mapped[event['name']]=dict(call,**payload,correlation_id=correlation,connection_id=connection,attribution_method=method)
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
             **{k:call.get(k) for k in ('collective','input_logical_bytes','output_logical_bytes','phase','generation_step','kv_length_before',
                                      'backend_elements','backend_input_buffer_bytes','backend_input_padding_bytes','payload_relation','buffer_scope')},
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
