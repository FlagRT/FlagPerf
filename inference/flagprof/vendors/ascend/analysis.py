"""CANN analysis projection; legacy event taxonomy stays byte-compatible."""
import json
import re
from decimal import Decimal
from pathlib import Path

COMPUTE = {'AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV','AI_CPU',
           'KERNEL_AICORE','KERNEL_AIVEC','KERNEL_MIX_AIC','KERNEL_MIX_AIV','KERNEL_AICPU','KERNEL_AIV'}
COPY = {'SDMA_SQE','MEMCPY','MEMCPY_ASYNC','MEMSET'}
WAIT = {'WRITE_VALUE_SQE','PLACE_HOLDER_SQE','TASK_TIMEOUT_SET','STREAM_SWITCH','STREAM_ACTIVE'}
ARGMAX = 'aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2'


def activity(event, context, label):
    if event.get('ph') != 'X' or context.get(event.get('pid')) not in ('Ascend Hardware','Communication'):
        return None
    args=event.get('args',{});name=event.get('name','');task=args.get('Task Type')
    collective='allreduce' if 'allreduce' in name.lower() else 'allgather' if 'allgather' in name.lower() else None
    high=name.startswith('hcom_')
    auxiliary=name in ('allreduceAicpuKernel','allgatherAicpuKernel')
    # Engine type AICPU describes where a task runs, not its collective role.
    if task in COPY:category='memcpy'
    elif task in WAIT or str(task).startswith(('NOTIFY_','EVENT_')):category='wait'
    elif (high or auxiliary) and task in (None,'AI_CPU','KERNEL_AICPU'):category='comm'
    elif task in COMPUTE:category='compute'
    else:category='unknown'
    conflict=bool((high or auxiliary) and category!='comm')
    match=re.fullmatch(r'NPU (\d+)',label or '')
    return dict(category=category,dev=int(match.group(1)) if match else None,
                stream=args.get('Physic Stream Id',event.get('tid')),task_type=task,
                representation='collective' if high else 'auxiliary_kernel' if auxiliary else 'task',
                collective=collective,step_marker=name==ARGMAX,
                classification_conflict=conflict,
                classification_basis='registered_collective_name' if category=='comm' else 'Task Type')


def clock(directory, capture):
    """Explain exported time without applying a second counter conversion."""
    from flagprof.offline.limits import read_json
    root=Path(directory)/'sealed-raw';manifest=read_json(root/'committed.json')
    sources=[];start=None;end=None;host=None;device=None;boundaries={}
    for name,info in manifest['files'].items():
        path=root/name;base=path.name
        if base.startswith('profiler_info') and base.endswith('.json'):
            value=read_json(path);start=value.get('start_info');end=value.get('end_info')
        elif base in ('host_start.log','dev_start.log.0','dev_start.log.1'):
            values=dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
            values={k.strip():v.strip() for k,v in values.items()}
            if base=='host_start.log':host=values
            else:device=values
        elif re.fullmatch(r'(start|end)_info(?:\.\d+)?',base) and re.fullmatch(r'device_\d+',path.parent.name):
            value=read_json(path);dev=int(path.parent.name.split('_')[1]);kind='start' if base.startswith('start') else 'end'
            boundaries.setdefault(dev,{})[kind]=dict(value=value,source=dict(path='sealed-raw/'+name,sha256=info['sha256']))
        else:continue
        sources.append(dict(path='sealed-raw/'+name,sha256=info['sha256']))
    windows=[];transforms=[]
    for dev,pair in sorted(boundaries.items()):
        if set(pair)!={'start','end'}:continue
        a,b=pair['start']['value'],pair['end']['value']
        if not all((a.get('collectionTimeBegin'),a.get('clockMonotonicRaw'),b.get('clockMonotonicRaw'))):continue
        # Match the versioned exporter, including its float offset and decimal
        # microsecond quantization. Input events are already on this timeline.
        offset=str(float(a['collectionTimeBegin'])-float(a['clockMonotonicRaw'])/1000)
        first=int((Decimal(a['clockMonotonicRaw'])/1000+Decimal(offset)).quantize(Decimal('.001'))*1000)
        last=first+int(b['clockMonotonicRaw'])-int(a['clockMonotonicRaw'])
        if last>first:
            sources_pair=[pair['start']['source'],pair['end']['source']]
            windows.append(dict(kind='official_capture',dev=dev,start_ns=first,end_ns=last,
                boundary_source=sources_pair,
                boundary_rule='CANN local_time_offset from collectionTimeBegin_us and clockMonotonicRaw_ns; end on same anchored clock'))
            transforms.append(dict(method='raw_offset',dev=dev,offset_us=offset,
                raw_start_ns=int(a['clockMonotonicRaw']),raw_end_ns=int(b['clockMonotonicRaw']),
                sources=sources_pair,formula='export_us=quantize(raw_ns/1000+offset_us,0.001)',
                interpretation='CANN 9.0 InfoConfReader._load_local_time_offset/trans_into_local_time',
                offset_rounding_ns=str(Decimal(offset)*1000-(Decimal(a['collectionTimeBegin'])*1000-Decimal(a['clockMonotonicRaw'])))))
    parameters=dict(start_info=start,end_info=end,host=host,device=device)
    available=bool(start and start.get('syscnt_enable') and start.get('freq') and
                   start.get('start_cnt') is not None and start.get('start_monotonic') is not None)
    return dict(status='available' if available else 'unavailable',clock_valid=False,
        source_domain='official_export_ns',transform_status='sourced' if available else 'unavailable',
        transform_applied=False,export_already_transformed=True,
        transform_formula='monotonic_ns=(counter-start_cnt)*1000/freq_MHz+start_monotonic',
        alignment_status='official_export_internal',project_host_alignment='unavailable',
        uncertainty_ns=None,uncertainty_status='unavailable',
        reason='export clock conversion available; device synchronization error bound is not exported',
        parameters=parameters,sources=sources,capture_windows=windows,project_transforms=transforms,
        implementation_basis='CANN 9.0 InfoConfReader._load_local_time_offset/trans_into_local_time; source SHA 195d542a7b7b46f62fedde0ed7de1e6cdfc56d8d54b967fb80a942d91d0c17b5')
