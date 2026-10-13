"""Offline telemetry quality and phase association. No device calls."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

from runtime.common import read_json, write_json, file_hash
from runtime.monitor import read_events


def conflict_reason(row, definitions, entities, phases, aligned):
    if not definitions.get(row['metric'], {}).get('profiler_conflict'): return None
    profiles = [p for p in phases if p['action'] in ('profile', 'layer_profile')]
    if profiles and not aligned:
        return 'clock alignment unavailable for profiler conflict'
    if any(p['rank'] in entities.get(row['entity'], {}).get('ranks', []) and
           p['start_ns'] <= row['read_end_ns'] and p['end_ns'] >= row['read_start_ns'] for p in profiles):
        return 'profiler worker active; conservative exclusion'
    return None


def phases_from_events(events, end_ns):
    active, phases, errors = {}, [], []
    for event in events:
        worker = event['worker']
        if event['event'] not in ('worker_start', 'stage', 'worker_end'):
            continue
        prior = active.pop(worker, None)
        if prior:
            phases.append(dict(prior, end_ns=event['timestamp_ns']))
        if event['event'] == 'worker_end':
            continue
        context = prior or event
        active[worker] = dict(worker=worker, rank=event['rank'], start_ns=event['timestamp_ns'],
            stage=event.get('stage', event.get('action', 'worker')),
            action=context.get('action'), components=context.get('components', {}),
            repeat=context.get('repeat'), side=context.get('side'))
    for worker, prior in active.items():
        phases.append(dict(prior, end_ns=end_ns, incomplete=True))
        errors.append('missing worker_end: '+worker)
    return phases, errors


def summarize(root, collector_error=None):
    root = Path(root)
    folder = root/'monitor'
    inventory = read_json(folder/'inventory.json') if (folder/'inventory.json').exists() else {}
    events, errors = read_events(folder)
    rows = []
    if (folder/'samples.jsonl').exists():
        with (folder/'samples.jsonl').open() as stream:
            for index, line in enumerate(stream):
                try: row = json.loads(line)
                except ValueError:
                    errors.append(f'incomplete sample line {index+1}'); continue
                if row.get('kind') == 'sample': rows.append(row)
                else: errors.append(row.get('error', 'collector error'))
    end = max([r['read_end_ns'] for r in rows]+[e['timestamp_ns'] for e in events]+[0])
    phases, incomplete = phases_from_events(events, end)
    errors += incomplete
    if collector_error: errors.append(collector_error)
    if inventory.get('device_error'): errors.append(inventory['device_error'])
    clocks_valid = bool(events) and all(e.get('clock') == inventory.get('clock') for e in events)
    offsets = [abs(e['utc_ns']-e['timestamp_ns']-(inventory['utc_anchor_ns']-inventory['clock_anchor_ns']))
               for e in events if 'utc_ns' in e and 'utc_anchor_ns' in inventory]
    clocks_valid = clocks_valid and (not offsets or max(offsets) <= 50_000_000)
    if not clocks_valid: errors.append('clock alignment could not be verified within 50 ms')
    definitions = {m['name']: m for m in inventory.get('metrics', [])}
    entities = inventory.get('entities', {})
    grouped, associated = defaultdict(list), defaultdict(list)
    conflicts = missing = 0
    for row in rows:
        context = [i for i, p in enumerate(phases) if clocks_valid and
                   p['start_ns'] <= row['read_end_ns'] and p['end_ns'] >= row['read_start_ns'] and
                   (entities.get(row['entity'], {}).get('scope') == 'host' or
                    p['rank'] in entities.get(row['entity'], {}).get('ranks', [])) and
                   (entities.get(row['entity'], {}).get('scope') != 'process' or
                    p['worker'] == entities[row['entity']]['worker'])]
        row['phase_indices'] = context
        conflict = conflict_reason(row, definitions, entities, phases, clocks_valid)
        if conflict:
            row.update(value=None, status='unavailable', reason=conflict)
            conflicts += 1
        if row['status'] != 'ok': missing += 1
        grouped[(row['entity'], row['metric'])].append(row)
        for i in context: associated[(i, row['entity'], row['metric'])].append(row)

    def stats(items):
        values = [r['value'] for r in items if r['status'] == 'ok' and r['value'] is not None]
        times = sorted(r['read_start_ns'] for r in items)
        intervals = [(b-a)/1e6 for a,b in zip(times, times[1:])]
        return dict(samples=len(items), valid_samples=len(values),
                    sample_mean=mean(values) if values else None,
                    minimum=min(values) if values else None, maximum=max(values) if values else None,
                    median_poll_interval_ms=median(intervals) if intervals else None,
                    max_poll_gap_ms=max(intervals) if intervals else None)

    series = [dict(entity=e, metric=m, **stats(items)) for (e,m), items in sorted(grouped.items())]
    phase_stats = [dict(phase_index=i, entity=e, metric=m, **stats(items))
                   for (i,e,m), items in sorted(associated.items())]
    formal = [i for i,p in enumerate(phases) if p['action'] == 'performance' and p['stage'] == 'measurement']
    short = [i for i in formal if not any(s['phase_index'] == i and s['valid_samples'] >= 2 and
            entities.get(s['entity'], {}).get('scope') == 'chip' for s in phase_stats)]
    if short: errors.append('insufficient timeline samples in formal phases: '+str(short))
    device_ok = any(s['valid_samples'] and entities.get(s['entity'], {}).get('scope') in ('card','chip') for s in series)
    unexpected_missing = any(r['status'] != 'ok' and r.get('reason') not in
        ('first counter sample', 'process exited or PID access unavailable',
         'profiler worker active; conservative exclusion') for r in rows)
    status = 'failed' if not rows else 'partial' if errors or not device_ok or unexpected_missing else 'completed'
    result = dict(schema_version=1, enabled=True, status=status, provider=inventory.get('provider'),
        interval_ms=inventory.get('interval_ms'), samples=len(rows), unavailable_samples=missing,
        profiler_conflict_samples=conflicts, errors=errors, clock_aligned=clocks_valid,
        max_clock_offset_difference_ns=max(offsets) if offsets else None,
        association='query interval overlap with phase; unknown source windows prevent exact batch attribution',
        statistics_scope='sample statistics, not time-weighted means or cumulative batch-timer statistics',
        phases=phases, series=series, phase_statistics=phase_stats,
        insufficient_resolution_phases=short, inventory='monitor/inventory.json',
        raw_samples='monitor/samples.jsonl', summary='monitor/summary.json')
    write_json(folder/'summary.json', result)
    # Sealed inputs remain the authoritative evidence; derived views can be regenerated.
    files = [p for p in folder.rglob('*') if p.is_file() and p.name not in ('manifest.json', 'heartbeat.json')]
    write_json(folder/'manifest.json', {'schema_version': 1, 'files': {
        p.relative_to(root).as_posix(): file_hash(p) for p in files}})
    return result


def timeline_rows(root, summary):
    """Apply the same conservative conflict mask while reading immutable raw samples."""
    inventory = read_json(Path(root)/'monitor/inventory.json')
    definitions = {m['name']:m for m in inventory['metrics']}
    entities = inventory['entities']
    rows = []
    with (Path(root)/'monitor/samples.jsonl').open() as stream:
        for line in stream:
            try: row = json.loads(line)
            except ValueError: continue
            if row.get('kind') != 'sample': continue
            conflict = conflict_reason(row, definitions, entities, summary['phases'], summary['clock_aligned'])
            if conflict:
                row.update(value=None, status='unavailable', reason=conflict)
            rows.append(row)
    return inventory, rows
