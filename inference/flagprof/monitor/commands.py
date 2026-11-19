"""Portable monitor dependency closure and verified release through public APIs."""
from contextlib import ExitStack
import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import uuid
from flagprof.delivery import publish_tree, verify, restore
from flagprof.delivery_refs import resolve
from flagprof.store import StoreView
from flagprof.store.manifest import atomic_json
from .consumer import replay
from .reference import read, resolve_reference
from .retention import overlap, read_lease, reserve
from .session import control_json, edit, load


def checkpoint_closure(root, state, selected):
    """Include earlier journal segments referenced by a replay starting state."""
    if 'diagnosis' not in state['config']:return selected,set()
    available={row['id']:row for row in state['segments']}
    sources={}
    for kind in ('active','history'):
        for path in (root/'targets'/kind).glob('*.json'):
            row=read(path);identity=row.get('diagnostic_identity')
            if identity:sources[identity['run_id']]=row
    wanted={row['id'] for row in selected};checked=set();needed_sources=set()
    while wanted-checked:
        ident=sorted(wanted-checked)[0];checked.add(ident)
        checkpoint=root/'checkpoints'/(ident+'.json')
        if not checkpoint.exists():raise ValueError('diagnostic archive lacks checkpoint: '+ident)
        pending=[read(checkpoint)]+[read(p) for p in (root/'findings'/ident).glob('*.json')]
        while pending:
            value=pending.pop()
            if isinstance(value,dict):pending.extend(value.values())
            elif isinstance(value,list):
                if len(value)==4 and value[0]=='event' and isinstance(value[1],str):
                    if value[1] in available:wanted.add(value[1])
                    elif value[1] in sources:needed_sources.add(value[1])
                    else:raise ValueError('diagnostic archive lacks referenced event source: '+value[1])
                else:pending.extend(value)
    return [row for row in state['segments'] if row['id'] in wanted],needed_sources


def publish(reference, destination, *, max_bytes=30 << 30):
    reference=Path(reference).resolve()
    if reference.is_dir():reference=reference/'monitor-ref.json'
    ref,root,inventory=resolve_reference(reference)
    if root is None:raise ValueError('monitor reference unavailable')
    if ref.get('version')!=2:
        return publish_tree([reference.parent,root],destination,max_bytes=max_bytes,reference=reference)
    from .reading import status
    ref=status(reference)['reference']
    if ref['window'][1] is None:raise ValueError('target must finish before delivery')
    state=load(root)
    segments=[s for s in state['segments'] if overlap(s['window'],ref['window'])]
    if not segments or any(s['status']!='closed' for s in segments):
        raise ValueError('target monitor segment must close before delivery')
    # All related Findings bring their pre/post evidence into the same closure.
    pins={k:v for k,v in state['pins'].items() if overlap(v['window'],ref['window'])}
    segments=[s for s in state['segments'] if overlap(s['window'],ref['window']) or
              any(overlap(s['window'],p['window']) for p in pins.values())]
    segments,diagnostic_sources=checkpoint_closure(root,state,segments)
    if any(s['status']!='closed' for s in segments):raise ValueError('incident post window still open')
    state=dict(state,segments=segments,pins=pins,status='archived',ended_raw_ns=max(s['window'][1] for s in segments),reserved_bytes=0)
    consumer=read(root/'consumer.json') if (root/'consumer.json').exists() else None
    amount=2*(len(json.dumps(state).encode())+len(json.dumps(consumer).encode())+reference.stat().st_size)+65536
    with edit(root) as live:reserve(root,live,amount)
    # Only metadata is copied. Immutable Stores stream directly to the public archive.
    stage=root/('.delivery-'+uuid.uuid4().hex);stage.mkdir()
    try:
        atomic_json(stage/'session.json',state)
        atomic_json(stage/'reference.json',dict(read(reference),shared_devices=ref.get('shared_devices',[])))
        sources=[reference if ref.get('kind')=='finding' else reference.parent,root/'session.json',root/'.state.lock']
        overrides={root/'session.json':stage/'session.json',reference:stage/'reference.json'}
        if consumer is not None:
            atomic_json(stage/'consumer.json',consumer)
            sources.append(root/'consumer.json');overrides[root/'consumer.json']=stage/'consumer.json'
        if (root/'rules.json').exists():sources.append(root/'rules.json')
        with ExitStack() as stack:
            for segment in segments:
                sources.append(stack.enter_context(read_lease(root,segment['id'])))
                checkpoint=root/'checkpoints'/(segment['id']+'.json')
                if checkpoint.exists():sources.append(checkpoint)
                findings=root/'findings'/segment['id']
                if findings.exists():sources.append(findings)
            for path in (root/'aggregates').glob('*.json'):
                minute=read(path)['minute']*60_000_000_000
                if any(overlap([minute,minute+60_000_000_000],s['window']) for s in segments):sources.append(path)
            for kind in ('active','history'):
                folder=root/'targets'/kind
                for path in folder.glob('*.json'):
                    target=read(path)
                    if overlap(target['window'],ref['window']) or target.get('diagnostic_identity',{}).get('run_id') in diagnostic_sources:
                        if kind=='active':raise ValueError('shared target still active')
                        sources.append(path)
                        if target.get('reference_base')=='session':
                            for relative in target.get('references',[]):
                                dependency=resolve(path, relative, base=root)
                                if target.get('diagnostic_identity',{}).get('run_id') in diagnostic_sources and not dependency.exists():
                                    raise ValueError('diagnostic archive lacks target reference: '+str(dependency))
                                if dependency.exists() and dependency!=reference:
                                    sources.append(dependency.parent if (dependency.parent/'manifest.json').exists() else dependency)
            return publish_tree(sources,destination,max_bytes=max_bytes,
                                file_overrides=overrides,reference=reference)
    finally:
        shutil.rmtree(stage)
        with edit(root) as live:live['reserved_bytes']=max(0,live['reserved_bytes']-amount)


def release(root, reference, delivery):
    from flagprof.offline.worker import on_disk
    return on_disk(root, _release, root, reference, delivery)


def _release(root, reference, delivery):
    root=Path(root).resolve()
    if str(reference).startswith('finding:'):
        import re
        ident=str(reference).split(':',1)[1]
        if not re.fullmatch('[0-9a-f]{32}',ident):raise ValueError('invalid Finding reference')
        reference=root/'references'/(ident+'.json')
    reference=Path(reference).resolve()
    ref,source,_=resolve_reference(reference)
    if source!=root:raise ValueError('reference belongs to another session')
    verification=verify(delivery)
    # A fresh, attributable directory is mandatory; restore never reuses an old tree.
    parent=Path(tempfile.gettempdir())
    directory=parent/('flagprof-release-'+uuid.uuid4().hex)
    restored=restore(delivery,directory,max_bytes=load(root)['config']['total_bytes'])
    try:
        mapped = Path(restored['mapping']['monitor-reference']) if 'monitor-reference' in restored['mapping'] else directory/'payload'/str(reference).lstrip('/')
        if not mapped.exists():raise ValueError('delivery does not contain target reference')
        restored_ref,restored_root,_=resolve_reference(mapped)
        if restored_ref.get('session_id')!=ref.get('session_id') or restored_root is None:
            raise ValueError('restored monitor identity mismatch')
        # Compare public records, including stable event IDs. Recorded capture
        # interruptions remain partial in both trees and are not repaired by restore.
        def fingerprint(target):
            digest=hashlib.sha256();count=0
            with StoreView(target) as view:
                for event in view.iter_monitor_events(('counter','health')):
                    digest.update(json.dumps(event,sort_keys=True,separators=(',',':')).encode());count+=1
                errors=[e['segment'] for e in view.monitor_coverage()['read_report']['errors']]
            return count,digest.hexdigest(),errors
        original=fingerprint(reference);reopened=fingerprint(mapped)
        if original!=reopened:raise ValueError('restored public evidence differs')
        count=reopened[0]
        if count==0:raise ValueError('restored monitor evidence empty')
        from .report import generate
        report=generate(restored_root,directory/'report')
        with StoreView(mapped) as view:coverage=view.monitor_coverage()
        restored_state=load(restored_root);live=load(root)
        selected=ref.get('pin_key') or 'target:'+str(ref.get('target_id'))
        if selected not in restored_state['pins']:raise ValueError('delivery target pin missing')
        # Preserve confirmed sharing before released history can be reclaimed.
        # The same attribution snapshot is already in the verified archive.
        if ref.get('worker_proc_key'):
            atomic_json(reference,dict(read(reference),shared_devices=restored_ref.get('shared_devices',[])))
        # Pin content must still match; a newer incident window cannot be released
        # by a stale archive. Only this target's pin is removed.
        with edit(root) as state:
            if state['pins'].get(selected)!=restored_state['pins'][selected]:
                raise ValueError('pin changed since archive')
            state['pins'].pop(selected)
            if selected.startswith('target:'):
                history=root/'targets/history'/(selected.split(':',1)[1]+'.json')
                if history.exists():
                    row=read(history);row['released_delivery']=str(Path(delivery).resolve())
                    control_json(root,state,history,row)
            # Identity remains in the verified delivery. Keep live history while
            # any remaining reference still needs it for overlap attribution.
            for history in (root/'targets/history').glob('*.json'):
                row=read(history)
                if row.get('released_delivery') and not any(overlap(row['window'],p['window']) for p in state['pins'].values()):
                    history.unlink()
        result=dict(status='released',reference=str(reference),delivery=str(Path(delivery).resolve()),
                    verified=verification,restored_events=count,public_evidence_sha256=reopened[1],
                    retained_partial_segments=reopened[2],public_report=report,reference_coverage=coverage)
        atomic_json(root/('release-'+uuid.uuid4().hex+'.json'),result)
    except BaseException:
        # Keep failed fresh restore for diagnosis, never unpin on a failed check.
        raise
    else:
        shutil.rmtree(directory)
        return result
