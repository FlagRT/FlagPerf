# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Offline device-event attribution; timestamp proximity alone is never a link."""
from collections import defaultdict
import shutil
import sys
from runtime.common import read_json, write_json, file_hash, execute
from analysis.communication import ns, union_ns


def attribute(events, calls):
    process = {e['pid']: e.get('args', {}).get('name') for e in events
               if e.get('ph') == 'M' and e.get('name') == 'process_name'}
    markers = defaultdict(list)
    for e in events:
        if e.get('ph') == 'X' and e.get('name', '').startswith('flagperf/layer/'):
            markers[e['name']].append(e)
    scopes = [(c, markers[c['marker']][0]) for c in calls if len(markers[c['marker']]) == 1]
    batches = [e for e in events if e.get('ph') == 'X' and e.get('name') == 'flagperf/model_batch']
    flows = defaultdict(list)
    for e in events:
        if e.get('ph') in ('s', 'f'):
            flows[(e.get('cat'), str(e.get('id')))].append(e)
    sources = defaultdict(list)
    for pair in flows.values():
        start = [e for e in pair if e['ph'] == 's']
        end = [e for e in pair if e['ph'] == 'f']
        if len(start) == len(end) == 1:
            finish = end[0]
            sources[(finish.get('pid'), finish.get('tid'), ns(finish['ts']))].append(start[0])
    def contains(scope, source):
        same_process = source.get('pid') == scope.get('pid')
        # Verified CANN host records encode the original PID and original Thread Id.
        if not same_process:
            pid = source.get('pid')
            same_process = (process.get(pid) == 'CANN' and type(pid) is int
                            and pid >> 10 == scope.get('pid')
                            and source.get('args', {}).get('Thread Id', source.get('tid')) == scope.get('tid'))
        return (same_process and source.get('tid') == scope.get('tid') and
                ns(scope['ts']) <= ns(source['ts']) < ns(scope['ts']) + ns(scope['dur']))
    connections = defaultdict(list)
    for e in events:
        if e.get('name') == 'Node@launch' and 'connection_id' in e.get('args', {}):
            connections[e['args']['connection_id']].append(e)
    records, auxiliary = [], []
    control_events = 0
    for index, event in enumerate(events):
        if event.get('ph') != 'X':
            continue
        category = process.get(event.get('pid'))
        name = event.get('name', '')
        if event.get('args', {}).get('Task Type') in ('PROFILING_ENABLE', 'PROFILING_DISABLE'):
            control_events += 1
            continue
        # Communication sub-tasks remain in the raw trace, not double counted as copies.
        if category == 'Communication' and name.lower().startswith('hcom_'):
            kind = 'communication'
        elif category == 'Ascend Hardware':
            task = event.get('args', {}).get('Task Type')
            if name == 'allreduceAicpuKernel':
                kind = 'communication_detail'
            elif task and (task.startswith(('NOTIFY_', 'EVENT_', 'WRITE_VALUE_')) or task == 'SDMA_SQE'):
                kind = 'queue_or_transfer_task'
            elif 'memcpy' in name.lower():
                kind = 'copy'
            elif task in (None, 'AI_CORE', 'AI_VECTOR_CORE', 'MIX_AIC', 'MIX_AIV', 'AI_CPU'):
                kind = 'compute'
            else:
                kind = 'other_device_task'
        else:
            continue
        start, duration = ns(event['ts']), ns(event['dur'])
        src = sources.get((event.get('pid'), event.get('tid'), start), [])
        direct = connections.get(event.get('args', {}).get('connection_id'), [])
        if len(direct) == 1: src = [*src, direct[0]]
        alternatives = []
        for source in src:
            owners = tuple(sorted(c['sequence'] for c, marker in scopes if contains(marker, source)))
            if owners or any(contains(b, source) for b in batches):
                alternatives.append(owners)
        unique = set(alternatives)
        owners = list(next(iter(unique))) if len(unique) == 1 else []
        status = ('correlated' if owners else 'outside_selected_layers') if len(unique) == 1 else 'unknown'
        args = event.get('args', {})
        size = args.get('size(Byte)', args.get('bytes')) if kind == 'copy' else None
        target = records if kind in ('compute', 'communication', 'copy') else auxiliary
        target.append({'event_index': index, 'name': name, 'kind': kind, 'start_ns': start,
                        'duration_ns': duration, 'owners': owners, 'attribution_status': status,
                        'task_type': args.get('Task Type'),
                        'connection_id': args.get('connection_id'),
                        'bytes': int(size) if isinstance(size, (int, float)) else None,
                        'direction': args.get('Memcpy kind', args.get('copy_kind', 'unknown')) if kind == 'copy' else None})
    associated = [r for r in records if r['attribution_status'] != 'unknown']
    complete = bool(records) and len(associated) == len(records) and len(scopes) == len(calls)
    rows = []
    for call in calls:
        own = [e for e in records if call['sequence'] in e['owners']]
        row = {k: v for k, v in call.items() if k not in ('host_start_ns', 'host_end_ns')}
        row['events'] = {}
        for kind in ('compute', 'communication', 'copy'):
            selected = [e for e in own if e['kind'] == kind]
            row['events'][kind] = {'count': len(selected),
                'duration_sum_ns': sum(e['duration_ns'] for e in selected) if selected or complete else None,
                'duration_union_ns': union_ns([(e['start_ns'], e['start_ns']+e['duration_ns']) for e in selected]) if selected or complete else None,
                'bytes': sum(e['bytes'] for e in selected) if selected and all(e['bytes'] is not None for e in selected) else None}
        row['attribution_status'] = 'correlated' if own else 'not_observed'
        rows.append(row)
    return {'status': 'completed' if complete else 'partial', 'rows': rows, 'events': records,
            'auxiliary_events': auxiliary,
            'auxiliary_coverage': {'tasks': len(auxiliary),
                'unknown_tasks': sum(e['attribution_status']=='unknown' for e in auxiliary),
                'status': 'partial' if any(e['attribution_status']=='unknown' for e in auxiliary) else 'completed',
                'scope': 'queue/synchronization/low-level SDMA and communication implementation tasks; not added to compute or collective spans'},
            'coverage': {'device_events': len(records), 'associated_events': len(associated),
                         'unknown_events': len(records)-len(associated),
                         'outside_selected_events': sum(r['attribution_status']=='outside_selected_layers' for r in records),
                         'fraction': len(associated)/len(records) if records else None,
                         'expected_markers': len(calls), 'matched_markers': len(scopes),
                         'scope': 'compute kernels, collective spans and identifiable standalone memcpy; auxiliary tasks accounted separately'},
            'excluded_profiler_control_events': control_events,
            'method': 'paired flow endpoints or unique CANN connection, then same-host-thread module scope; no nearest-time guessing'}


def associate_collectives(detail):
    """Connect already verified collectives to exact selected module invocations."""
    calls = {row['sequence']: row for row in detail.get('rows', [])}
    by_connection = defaultdict(list)
    for event in detail.get('events', []):
        if event['kind'] == 'communication' and event.get('connection_id') is not None:
            by_connection[event['connection_id']].append(event)
    for event in detail.get('communication', {}).get('events', []):
        matches = by_connection.get(event.get('connection_id'), [])
        event['layer_calls'] = []
        if len(matches) == 1:
            event['layer_attribution_status'] = matches[0]['attribution_status']
            for sequence in matches[0]['owners']:
                row = calls[sequence]
                if (row['cycle'], row['batch_index']) != (event.get('cycle'), event.get('batch_index')):
                    raise ValueError('collective/module batch identity mismatch')
                event['layer_calls'].append({k: row[k] for k in ('sequence','module','call_index','cycle','batch_index','input_shape')})
        else:
            event['layer_attribution_status'] = 'unknown'


def capture_hashes(source):
    return {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}


def recover_trace(folder, export=None):
    """One bounded export of a copy; never rerun inference or modify raw capture."""
    destination = folder/'profiler-recovery'
    manifest = destination/'recovery.json'
    if destination.exists():
        recovery = read_json(manifest) if manifest.is_file() else {
            'status': 'partial', 'reason': 'previous export did not seal recovery metadata'}
        return recovery
    sources = [p for p in (folder/'profiler').glob('*_ascend_pt')
               if (p/'FRAMEWORK/torch.op_range').is_file() and list(p.glob('PROF_*'))]
    if len(sources) != 1:
        return {'status': 'partial', 'reason': 'no unique Ascend raw capture for independent export'}
    source = sources[0]
    hashes = capture_hashes(source)
    destination.mkdir()
    recovery = {'status': 'partial', 'trigger': 'missing_final_trace',
                'source': str(source.relative_to(folder)), 'source_sha256': hashes,
                'new_device_execution': False, 'timeout_seconds': 300}
    write_json(manifest, recovery)
    try:
        copied = destination/'capture'/source.name
        shutil.copytree(source, copied)
        command = [sys.executable, '-u', '-c',
                   'import sys, torch_npu; torch_npu.profiler.profiler.analyse(sys.argv[1], max_process_number=1)',
                   str(copied)]
        recovery['process'] = (export(command, destination/'export', 300) if export else
                               execute(command, destination/'export', 300))
        recovery['source_unchanged'] = capture_hashes(source) == hashes
        traces = list(copied.rglob('trace_view.json'))
        if recovery['process']['exit_code'] == 0 and len(traces) == 1 and recovery['source_unchanged']:
            recovery.update(status='completed', trace=str(traces[0].relative_to(folder)))
        else:
            recovery['reason'] = 'independent export incomplete or original source changed; see export/run.log'
    except Exception as error:
        recovery['reason'] = str(error)
    write_json(manifest, recovery)
    return recovery


def analyze(folder, state, export=None):
    files = list((folder/'profiler').rglob('trace_view.json'))
    recovery = None
    if not files:
        recovery = recover_trace(folder, export=export)
        if recovery['status'] == 'completed':
            source = folder/recovery['source']
            if capture_hashes(source) != recovery['source_sha256']:
                raise ValueError('raw profiler capture changed after independent export')
            files = [folder/recovery['trace']]
    if len(files) != 1:
        return {'status': 'partial', 'reason': 'expected one Ascend trace_view.json; other trace formats are not attributed',
                'coverage': {}, 'rows': [], 'events': [], 'recovery': recovery}
    trace = read_json(files[0])
    result = attribute(trace.get('traceEvents', []) if isinstance(trace, dict) else trace, state['rows'])
    result['source'] = str(files[0].relative_to(folder))
    if recovery is not None: result['recovery'] = recovery
    return result
