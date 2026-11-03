# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""SQL lookups preserve full rule unions; per-device sets spill to scratch."""
from .disk import Rows,key,unpack
from flagprof.correlate.linker import ref,registry_match
from flagprof.correlate.native import contains
from flagprof.vendors.ascend.format import queue_key_valid


def records(db,kind,k):return db.records(kind,k)


def device_hits(db,event,vendor):
    yield from db.edge_hits(event,vendor)


def host_hits(db,launch,vendor,native=False):
    prefix='cuda' if vendor in ('nvidia','thead') else 'cann'
    for n in db.contains(launch,'call' if native else 'project'):
        yield prefix+('.native_nesting' if native else '.same_thread'),n,()
    args=launch['ext']['chrome.args']
    if vendor in ('nvidia','thead'):
        ext=args.get('External id')
        if ext is not None:
            for source in records(db,'native_external' if native else 'project_external',(launch['host_pid'],str(ext))):
                owners=[source] if native or source['t']=='op' else db.contains(source,'project')
                for owner in owners:yield prefix+('.native_external_id' if native else '.external_id'),owner,() if native else (source,)
    else:
        # Project Dequeue interval treats missing duration as zero; native
        # ScopeIndex refuses missing duration. A separate query preserves this.
        mode='dequeue' if native else 'project_dequeue'
        if not native:
            launch=dict(launch)  # no point-duration substitution for project
        for dequeue in db.contains(launch,mode):
            corr=dequeue['ext']['chrome.args'].get('correlation_id')
            if queue_key_valid(corr):
                for enqueue in records(db,'enqueue',(dequeue['host_pid'],corr)):
                    for owner in db.contains(enqueue,'call' if native else 'project'):
                        yield prefix+('.native_enqueue' if native else '.enqueue_corr'),owner,(dequeue,enqueue)


def copyset(db,source,target):
    db.c.execute('INSERT OR IGNORE INTO scratch SELECT ?,k,id,value FROM scratch WHERE kind=?',(target,source))


def project(db,event,vendor,registry):
    db.clear();db.add('pevidence',event)
    if registry_match(event,registry):return dict(status='not_applicable',reason='excluded',via=[],evidence=db.refs('pevidence'))
    via=set();unknown=False
    for rule,launch,proof in device_hits(db,event,vendor):
        via.add(rule);db.add('launches',launch);db.add('pevidence',launch)
        for n in proof:db.add('pevidence',n)
    for launch in db.members('launches'):
        found=False
        for rule,owner,proof in host_hits(db,launch,vendor):
            found=True;via.add(rule);db.scalar('pcandidate',owner['corr']);db.add('pevidence',owner)
            for n in proof:db.add('pevidence',n)
        unknown |= not found
    count=db.count('pcandidate');sources=db.count('launches')
    def candidates():
        for value, in db.c.execute("SELECT value FROM scratch WHERE kind='pcandidate' ORDER BY corr_type(value),corr_num(value) COLLATE BIGINT,corr_str(value)"):
            yield unpack(value)
        if unknown:yield None
    result=dict(via=sorted(via),evidence=db.refs('pevidence'))
    if count>=2 or (count and unknown):result.update(status='ambiguous',reason='conflict',candidates=Rows(candidates))
    elif count==1:result.update(status='linked',corr=next(db.scalars('pcandidate')))
    else:
        reason='no_bridge' if sources else 'no_device_key'
        if event.get('ext',{}).get('source_missing'):reason='source_missing'
        if event.get('ext',{}).get('graph_replay'):reason='graph_replay'
        result.update(status='unlinked',reason=reason)
    result['segments']=dict(host_attr='ok' if result['status']=='linked' else 'broken',device_link='ok' if sources else 'broken')
    result['level']='A' if result['status']=='linked' else 'B+' if sources else 'C'
    return result


def native(db,event,vendor,registry,single_pid):
    db.clear();db.add('evidence',event);via=set();unknown=False
    for rule,launch,proof in device_hits(db,event,vendor):
        via.add(rule);db.add('launches',launch);db.add('evidence',launch)
        for n in proof:db.add('evidence',n)
    for launch in db.members('launches'):
        db.c.execute("DELETE FROM scratch WHERE kind IN ('host_candidates','host_owners')")
        db.add('evidence',launch)
        for rule,owner,proof in host_hits(db,launch,vendor,native=True):
            via.add(rule);db.add('host_candidates',owner);db.add('evidence',owner)
            for n in proof:db.add('evidence',n)
        db.innermost('host_candidates','host_owners');unknown |= db.count('host_owners')==0
        copyset(db,'host_owners','candidates')
    if vendor in ('nvidia','thead') and single_pid[0]:
        ext=event['ext']['chrome.args'].get('External id')
        if ext is not None:
            for n in records(db,'native_external',(single_pid[1],str(ext))):
                via.add('cuda.native_direct');db.add('direct',n);db.add('evidence',n)
    excluded=registry_match(event,registry);count=db.count('candidates')
    state='exempt' if excluded else 'conflicted' if count>1 or (count==1 and unknown) else 'owned' if count else 'unowned'
    owner=next(db.members('candidates')) if state=='owned' else None
    if owner:
        for n in db.contains(owner,'scope',strict=True):
            db.add('ancestors',n);db.add('evidence',n)
            if db.c.execute('SELECT is_module FROM norm WHERE id=?',(db.rid(n),)).fetchone()[0]:db.add('module_candidates',n)
    db.innermost('module_candidates','modules')
    for launch in db.members('launches'):
        for n in db.contains(launch,'scope'):
            if not db.c.execute('SELECT is_call FROM norm WHERE id=?',(db.rid(n),)).fetchone()[0]:
                db.add('contexts',n);db.add('evidence',n)
    def ancestors():
        for payload, in db.c.execute("SELECT n.payload FROM scratch s JOIN norm n ON n.id=s.id JOIN raw r ON r.id=n.id WHERE s.kind='ancestors' ORDER BY n.ts COLLATE BIGINT,n.dur COLLATE BIGINT DESC,r.stem_order,r.suffix"):
            yield unpack(payload)
    ordered=True;previous=None
    for n in ancestors():
        if previous is not None and not contains(previous,n,strict=True):ordered=False
        previous=n
    def candidates():
        yield from db.refs('candidates')
        if unknown and count:yield None
    modules=db.count('modules')
    return dict(trace_uid=event['trace_uid'],loc=event['loc'],category=event['category'],state=state,
        owner=ref(owner) if owner else None,candidates=Rows(candidates),launches=db.refs('launches'),
        direct_owners=db.refs('direct'),unknown_launch_owner=unknown,
        scope_chain=Rows(lambda:(ref(n) for n in ancestors())),scope_chain_ordered=ordered,
        module_scope=ref(next(db.members('modules'))) if modules==1 else None,module_candidates=db.refs('modules'),
        launch_scopes=db.refs('contexts'),via=sorted(via),evidence=db.refs('evidence'),
        reason='excluded' if excluded else 'conflict' if state=='conflicted' else None if owner else 'no_bridge' if db.count('launches') else 'no_device_key')
