"""Public monitor read adapter: historical sidecars and rolling sessions."""
from pathlib import Path
from .compat import descriptor
from .reference import read, resolve_reference


def status(run):
    run = Path(run)
    if run.is_file():
        ref,source,inventory=resolve_reference(run)
    elif (run/'session.json').exists():
        from .session import snapshot
        session = snapshot(run)
        inventory = session['inventory']
        ref = dict(version=2,session_id=session['id'],status=session['status'],
                   window=[session['started_raw_ns'],session.get('ended_raw_ns')],
                   devices=inventory.get('entities',[]))
        source = run
    elif (run/'inventory.json').exists() and not (run/'monitor-ref.json').exists():
        inventory = read(run/'inventory.json')
        ref = dict(status=inventory.get('status', 'unknown'),
                   window=[inventory.get('started_raw_ns'), inventory.get('ended_raw_ns')],
                   devices=inventory.get('entities', []))
        source = run
    else:
        ref, source, inventory = resolve_reference(run/'monitor-ref.json')
    inventory = dict(inventory or {})
    inventory['descriptors'] = [descriptor(d) for d in inventory.get('descriptors', [])]
    if source and (source/'session.json').exists() and ref.get('worker_proc_key'):
        from .retention import overlap
        from .subscriptions import target_id
        ident=target_id(ref['worker_proc_key']); shared=set(ref.get('shared_devices',[]))
        own={d['entity'] for d in ref['devices'] if d.get('status')=='available'}
        for kind in ('active','history'):
            for path in (source/'targets'/kind).glob('*.json'):
                try: other=read(path)
                except FileNotFoundError: continue
                if other.get('target_id')==ident or not overlap(ref['window'],other['window']): continue
                shared.update(own & {d['entity'] for d in other.get('devices',[]) if d.get('status')=='available'})
        ref['shared_devices']=sorted(shared)
    return dict(reference=ref, inventory=inventory, path=source)


def normalize(stored):
    event = stored.record
    result = dict(event)
    result['evidence_id'] = list(stored.identity)
    # The extension is interpreted only at this public boundary.
    result['telemetry'] = dict(event.get('ext', {}).get('telemetry', {}))
    return result


def minutes(run):
    """Versioned derived trends are separate from raw events and their coverage."""
    state=status(run);source=state['path']
    if source is None:return
    for path in sorted((source/'aggregates').glob('*.json')):
        try:row=read(path)
        except FileNotFoundError:continue  # Concurrent expiry of a derived trend.
        if row.get('schema')!='monitor-minute/1':raise ValueError('unsupported monitor aggregate')
        yield row


def window_coverage(state):
    """Segment-time coverage is distinct from per-metric sampling-slot coverage."""
    source=state['path']
    if source is None or not (source/'session.json').exists():return None
    from .session import load
    from flagprof.telemetry.contract import raw_clock
    session=load(source);begin,end=state['reference']['window'];end=end or session.get('ended_raw_ns') or raw_clock()
    intervals=sorted((max(begin,s['window'][0]),min(end,s['window'][1] or end)) for s in session['segments']
                     if s['window'][0]<end and (s['window'][1] or end)>begin and (source/'segments'/s['id']).exists())
    missing=[];cursor=begin
    for a,b in intervals:
        if a>cursor:missing.append([cursor,a])
        cursor=max(cursor,b)
    if cursor<end:missing.append([cursor,end])
    return dict(window=[begin,end],missing_segment_intervals=missing,
                recorded_interruptions=[g for g in session['gaps'] if begin<=g['ts']<=end],
                semantics='segment time coverage; sampling slots are evaluated separately')


def events(run, types=None, *, work=None, limits=None, report=None):
    from flagprof.store import StoreView
    state = status(run)
    report = report if report is not None else dict(errors=[])
    if state['path'] is None:
        return
    if (state['path']/'session.json').exists():
        from .retention import read_lease, overlap
        from .session import load
        session = load(state['path'])
        for segment in session['segments']:
            if not overlap(segment['window'],state['reference']['window']): continue
            try:
                with read_lease(state['path'],segment['id']) as directory, StoreView(directory) as source:
                    for stored in source.iter_events(types,limits=limits):
                        yield normalize(stored)
            except (OSError,ValueError) as error:
                report['errors'].append(dict(segment=segment['id'],reason=str(error),
                                             recorded_interruption=segment.get('reason')))
        return
    with StoreView(state['path']) as source:
        for stored in source.iter_events(types, limits=limits):
            yield normalize(stored)
