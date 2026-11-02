# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Fixed-budget local relation worksets; overflow replays the full disk algorithm.

Lookups still cover the whole source. The budget bounds one device's candidates,
not the trace or a time window. No source candidate is dropped on overflow.
"""
from flagprof.correlate.linker import ref, identity, ref_sort, corr_sort, registry_match
from flagprof.correlate.native import contains, innermost
from .relations import device_hits, host_hits
from .disk import pack

class Spill(Exception):pass
class Budget:
    def __init__(self):self.nodes={};self.bytes=0;self.operations=0
    def add(self,group,node):
        self.operations+=1
        if self.operations>4096:raise Spill()
        k=identity(node)
        if k not in self.nodes:
            size=len(pack(node))
            if len(self.nodes)>=256 or self.bytes+size>1<<20:raise Spill()
            self.nodes[k]=node;self.bytes+=size
        group[k]=node
    def refs(self,group):return [ref(n) for n in sorted(group.values(),key=ref_sort)]


def project(db,event,vendor,registry):
    b=Budget();proof={};sources={};b.add(proof,event)
    if registry_match(event,registry):return dict(status='not_applicable',reason='excluded',via=[],evidence=b.refs(proof))
    via=set();unknown=False;candidates=set()
    for rule,launch,extra in device_hits(db,event,vendor):
        via.add(rule);b.add(sources,launch);b.add(proof,launch)
        for n in extra:b.add(proof,n)
    for launch in sources.values():
        found=False
        for rule,owner,extra in host_hits(db,launch,vendor):
            found=True;via.add(rule);candidates.add(owner['corr']);b.add(proof,owner)
            for n in extra:b.add(proof,n)
        unknown|=not found
    ordered=sorted(candidates,key=corr_sort);result=dict(via=sorted(via),evidence=b.refs(proof))
    if len(ordered)>=2 or (ordered and unknown):result.update(status='ambiguous',reason='conflict',candidates=ordered+([None] if unknown else []))
    elif ordered:result.update(status='linked',corr=ordered[0])
    else:
        reason='no_bridge' if sources else 'no_device_key'
        if event.get('ext',{}).get('source_missing'):reason='source_missing'
        if event.get('ext',{}).get('graph_replay'):reason='graph_replay'
        result.update(status='unlinked',reason=reason)
    result['segments']=dict(host_attr='ok' if result['status']=='linked' else 'broken',device_link='ok' if sources else 'broken')
    result['level']='A' if result['status']=='linked' else 'B+' if sources else 'C'
    return result


def native(db,event,vendor,registry,single_pid):
    b=Budget();proof={};sources={};candidates={};direct={};via=set();unknown=False;b.add(proof,event)
    for rule,launch,extra in device_hits(db,event,vendor):
        via.add(rule);b.add(sources,launch);b.add(proof,launch)
        for n in extra:b.add(proof,n)
    for launch in sources.values():
        owners={}
        for rule,owner,extra in host_hits(db,launch,vendor,native=True):
            via.add(rule);b.add(owners,owner);b.add(proof,owner)
            for n in extra:b.add(proof,n)
        chosen=innermost(list(owners.values()));unknown|=not chosen
        for n in chosen:b.add(candidates,n)
    if vendor=='nvidia' and single_pid[0]:
        ext=event['ext']['chrome.args'].get('External id')
        if ext is not None:
            for i in db.ids('native_external',(single_pid[1],str(ext))):
                n=db.node(i);via.add('cuda.native_direct');b.add(direct,n);b.add(proof,n)
    excluded=registry_match(event,registry);count=len(candidates)
    state='exempt' if excluded else 'conflicted' if count>1 or (count==1 and unknown) else 'owned' if count else 'unowned'
    owner=next(iter(candidates.values())) if state=='owned' else None
    ancestors={};module_candidates={};contexts={}
    if owner:
        for n in db.contains(owner,'scope',strict=True):
            b.add(ancestors,n);b.add(proof,n)
            if db.c.execute('SELECT is_module FROM norm WHERE id=?',(db.rid(n),)).fetchone()[0]:b.add(module_candidates,n)
    modules=innermost(list(module_candidates.values()))
    for launch in sources.values():
        for n in db.contains(launch,'scope'):
            if not db.c.execute('SELECT is_call FROM norm WHERE id=?',(db.rid(n),)).fetchone()[0]:b.add(contexts,n);b.add(proof,n)
    ordered=sorted(ancestors.values(),key=lambda n:(n['ts'],-n['dur'],ref_sort(n)))
    return dict(trace_uid=event['trace_uid'],loc=event['loc'],category=event['category'],state=state,owner=ref(owner) if owner else None,
        candidates=b.refs(candidates)+([None] if unknown and candidates else []),launches=b.refs(sources),direct_owners=b.refs(direct),
        unknown_launch_owner=unknown,scope_chain=[ref(n) for n in ordered],scope_chain_ordered=all(contains(a,c,strict=True) for a,c in zip(ordered,ordered[1:])),
        module_scope=ref(modules[0]) if len(modules)==1 else None,module_candidates=sorted([ref(n) for n in modules],key=ref_sort),
        launch_scopes=b.refs(contexts),via=sorted(via),evidence=b.refs(proof),
        reason='excluded' if excluded else 'conflict' if state=='conflicted' else None if owner else 'no_bridge' if sources else 'no_device_key')


def install():
    from . import relations
    for name in ('project','native'):
        old=getattr(relations,name);new=globals()[name]
        def run(*args,_old=old,_new=new,**kwargs):
            try:return _new(*args,**kwargs)
            except Spill:
                db=args[0];db.spills=getattr(db,'spills',0)+1
                return _old(*args,**kwargs)
        setattr(relations,name,run)
