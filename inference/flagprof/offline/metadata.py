# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Second candidate: metadata cardinality also spills to disk."""
from decimal import Decimal,InvalidOperation
import json
import re
from itertools import zip_longest,chain
from functools import cmp_to_key
from .disk import Rows,key,unpack,tokens
from .context import Taxonomy,Context,Names,NameMap
from flagprof.vendors import contracts as c


class ObjectRows:
    def __init__(self,factory):self.factory=factory
    def items(self):return self.factory()
    __json_items__=items


def setup(db):
    db.c.create_function('pstr',1,lambda p:str(unpack(p)))
    db.c.create_function('track_pid',1,lambda p:str(unpack(p)['pid']))
    db.c.create_function('track_tid',1,lambda p:str(unpack(p)['tid']))
    db.c.create_function('group_dim',1,lambda p:unpack(p)['dimension'])
    db.c.create_function('group_label',1,lambda p:str(unpack(p)['label']))
    db.c.create_function('group_reason',1,lambda p:unpack(p)['reason'])
    db.c.create_function('warning_order',1,lambda p:json.dumps(unpack(p),sort_keys=True))


def values(db,kind,order='pstr(payload)'):
    return (unpack(row[0]) for row in db.c.execute('SELECT payload FROM stage WHERE kind=? ORDER BY '+order,(kind,)))


class BoundedNames(Names):
    def __bool__(self):
        return self.db.c.execute('SELECT 1 FROM stage WHERE kind=? LIMIT 1',('names:'+str(self.pid),)).fetchone() is not None
    def __contains__(self,v):
        return self.db.c.execute('SELECT 1 FROM stage WHERE kind=? AND k=?',('names:'+str(self.pid),key(v))).fetchone() is not None


class BoundedNameMap(NameMap):
    def get(self,pid,default=None):return BoundedNames(self.db,key(pid).hex())


class BoundedContext(Context):
    def __init__(self,db):super().__init__(db);self.names=BoundedNameMap(db)
    @property
    def tracks(self):
        def rows():
            q="""SELECT s.payload FROM stage s JOIN
              (SELECT k,min(id) AS first FROM kv WHERE kind='track' GROUP BY k) t ON t.k=s.k
              WHERE s.kind='track' ORDER BY track_pid(s.payload),track_tid(s.payload),t.first"""
            for payload, in self.db.c.execute(q):
                item=unpack(payload);pid=item['pid']
                item['process_names']=Rows(lambda pid=pid:values(self.db,'names:'+key(pid).hex()))
                yield item
        return Rows(rows)


def ingest_context(db,taxonomy,i,e):
    roles=taxonomy.contract['roles']
    pid=e.get('pid');cat=e.get('cat');name=e.get('name','');db.addkey('track',(pid,e.get('tid')),i)
    if name=='process_name':
        value=e.get('args',{}).get('name');db.put('process_name',pid,value);db.put('names:'+key(pid).hex(),value,value)
    if name=='process_labels':db.put('labels',pid,e.get('args',{}).get('labels',''))
    if cat in roles['device_categories']:db.addkey('device_pid',pid,i)
    if cat=='cpu_op' or name.startswith(('Enqueue','Dequeue')):db.addkey('host_pid',pid,i)


def build_context(db,taxonomy):
    setup(db);db.flush_pending();ctx=BoundedContext(db);roles=taxonomy.contract['roles']
    for k,first in db.c.execute("SELECT k,min(id) FROM kv WHERE kind='track' GROUP BY k ORDER BY min(id)"):
        firstraw=db.raw(first);pid,tid=firstraw.get('pid'),firstraw.get('tid');pn=ctx.names.get(pid)
        count=0;device_structure=False;witness=False;host_cat=False
        allhost=bool(roles['host_categories']);allallowed=bool(roles['allowed_non_device_categories']);structural=[True]*len(roles['structural_records'])
        for e in db.track_raws((pid,tid)):
            if e.get('ph')!='X':continue
            count+=1;device_structure |= roles['device_identity_signal']=='category_or_device_stream' and taxonomy.is_device(e,ctx)
            audit=roles['audit_device_structure'];witness |= bool(audit) and all(k in e.get('args',{}) for k in audit['args_required'])
            host_cat |= e.get('cat') in roles['host_categories'];allhost &= e.get('cat') in roles['host_categories'];allallowed &= e.get('cat') in roles['allowed_non_device_categories']
            for j,r in enumerate(roles['structural_records']):structural[j] &= all(type(e.get(k)) is type(v) and e.get(k)==v for k,v in r['match'].items())
        declared=next((role for role,field in [('metadata','metadata_process_names'),('device','device_process_names'),('host','host_process_names')] if pn and pn<=set(roles[field])),None)
        witness=bool(count) and declared in ('host','metadata') and witness
        structure=next((r for i,r in enumerate(roles['structural_records']) if count and structural[i]),None)
        if not count:role,basis='metadata','no_complete_events'
        elif device_structure:role,basis='device','declared_device_structure'
        elif witness:role,basis='device','witnessed_device_structure_conflict'
        elif pn and pn<=set(roles['metadata_process_names']):role,basis='metadata','derived_summary_track'
        elif pn and pn<=set(roles['device_process_names']):role,basis='device','declared_device_process'
        elif pn and pn<=set(roles['host_process_names']):role,basis='host','declared_host_process'
        elif structure:role,basis=structure['role'],structure['id']
        elif allhost:role,basis='host','declared_host_categories'
        elif allallowed:role,basis='metadata','allowed_non_device_track'
        else:role,basis='unrecognized','no_declared_role'
        conflicts=[]
        if role=='device' and host_cat:conflicts.append('device_track_host_event')
        if (device_structure or witness) and declared is not None and declared!='device':conflicts.append('name_structure_conflict')
        db.put('track',(pid,tid),dict(pid=pid,tid=tid,role=role,basis=basis,complete_events=count,conflicts=conflicts));db.put('role',(pid,tid),role)
    return Taxonomy(taxonomy,ctx)


def new_scope_state():return dict(missing=0,examples=[],selected=0,mismatch=False,missing_arg=False)


def observe_scope(db,tax,vendor,e,state):
    if e.get('ph')=='X' and e.get('cat') in (tax.contract['roles']['host_categories'] if vendor=='nvidia' else ['cpu_op']):db.put('scope_host',e.get('pid'),True)
    if not tax.is_device(e,tax.ctx):return
    db.put('scope_device_pid',e.get('pid'),True)
    if len(state["examples"])<3:state["examples"].append(e['_linker_loc'])
    if vendor=='nvidia':v=e.get('args',{}).get('device')
    else:
        m=re.fullmatch(r'NPU (\d+)',db.get('labels',e.get('pid'),''));v=int(m.group(1)) if m else None
    valid=type(v) is int and v>=0
    if valid:db.put('scope_device',v,v)
    else:state["missing"]+=1
    if tax.ctx.role_index.get((e.get('pid'),e.get('tid')))=='device':
        label=str(v) if valid else 'unknown';db.put('scope_track',(label,e.get('pid'),e.get('tid')),label)
        state["selected"]+=1;args=e.get('args',{});state["mismatch"] |= 'device' in args and (type(e.get('pid')) is not int or type(args['device']) is not int or e.get('pid')!=args['device']);state["missing_arg"] |= 'device' not in args


def scope(db,tax,vendor,state=None):
    if state is None:
        state=new_scope_state()
        for e in db.raws():observe_scope(db,tax,vendor,e,state)
    missing,examples,selected,mismatch,missing_arg=(state[k] for k in ('missing','examples','selected','mismatch','missing_arg'))
    known=Rows(lambda:values(db,'scope_device','pstr(payload) COLLATE BIGINT'))
    n=db.c.execute("SELECT count(*) FROM stage WHERE kind='scope_device'").fetchone()[0]
    def tracks():
        for label,count in db.c.execute("SELECT pstr(payload),count(*) FROM stage WHERE kind='scope_track' GROUP BY pstr(payload) ORDER BY pstr(payload)"):yield label,count
    collision=bool(db.c.execute("SELECT 1 FROM stage a JOIN stage b ON a.k=b.k WHERE a.kind='scope_host' AND b.kind='scope_device_pid' LIMIT 1").fetchone())
    fields=['args.device','pid','cat'] if vendor=='nvidia' else ['process_labels.args.labels','pid','cat']
    facts=dict(device_count=n if not missing else None,device_ordinals=known if not missing else None,observed_devices=known,
        missing_device_identity=missing,host_device_pid_collision=collision,source=dict(kind='raw_trace',fields=fields,examples=examples),
        execution_track_count=sum(t['role']=='device' and bool(t['complete_events']) for t in tax.ctx.tracks),tracks_per_device=ObjectRows(tracks))
    if vendor=='nvidia':facts['device_pid_matches_device_arg']=False if mismatch else None if missing_arg or not selected else True
    return facts


def assess(db,tax,identities,ledger,vendor,uid,scope_state=None):
    contract=tax.contract;conditions=c.evaluate_conditions(contract['conditions'],c.identity_facts(identities))
    for row in conditions:
        row['source']={'hardware':identities.get('hardware',{}).get('source'),'collector':identities.get('collector',{}).get('source'),
          'actual':identities.get('collector',{}).get('configuration',{}),'sdk':identities.get('collector',{}).get('configuration',{}).get('actual',{}).get('sources',[]),
          'execution':identities.get('collector',{}).get('source'),'exporter':identities.get('exporter',identities.get('parser',{}).get('exporter',{}))}.get(row['path'].split('.')[0])
    tracks=tax.ctx.tracks;structural=c.structure_conditions(contract,tracks);conditions.extend(structural)
    facts=scope(db,tax,vendor,scope_state);sc=c.evaluate_conditions(contract['support_scope']['conditions'],dict(scope=facts))
    for row in sc:
        if isinstance(row['actual'],Rows):
            match=any(all(type(a) is type(b) and a==b for a,b in zip_longest(row['actual'],v,fillvalue=object())) for v in row['allowed'] if isinstance(v,list))
            row.update(state='pass' if match else 'fail',reason='value_match' if match else 'outside_verified_values')
    for row in sc:row['source']=facts['source']
    conditions.extend(sc);failure_count=0
    def fail(item):
        nonlocal failure_count
        db.put('failure',failure_count,item);failure_count+=1
    if vendor=='ascend':
        for item in tax.audit_dependencies(Rows(db.raws),tax.ctx):fail(item)
    else:
        allowed=set(contract['roles']['expected_process_names'])
        # Baseline's last process_name wins; raw names stay separately preserved.
        for k,payload in db.c.execute("SELECT k,payload FROM stage WHERE kind='process_name'"):
            if unpack(payload) not in allowed:db.put('unexpected_pid',unpack(k),str(unpack(k)))
        if db.c.execute("SELECT 1 FROM stage WHERE kind='unexpected_pid' LIMIT 1").fetchone():
            # This potentially large array is streamed via a separate failure view.
            fail(dict(id='device_process_label',reason='Device process label was not present in the verified export structure',pids_marker=True))
    for track in tracks:
        for kind in track['conflicts']:fail(dict(id=kind,reason='device structure conflicts with declared host/metadata evidence',pid=track['pid'],tid=track['tid'],count=track['complete_events']))
    fmt=tax.identity()
    def add(dimension,reason,e):
        label=tax.task_type(e) or e.get('cat') or '<absent>';k=(dimension,str(label),reason)
        g=db.get('unknown_group',k,dict(dimension=dimension,label=label,reason=reason,format=fmt,count=0,known_time_ns=0,dur_missing=0,examples=[]));g['count']+=1
        try:
            duration=Decimal(str(e.get('dur')))
            if not duration.is_finite() or duration<0:raise ValueError()
            g['known_time_ns']+=int(duration*1000)
        except (InvalidOperation,ValueError):g['dur_missing']+=1
        if len(g['examples'])<3:g['examples'].append(dict(loc=e['_linker_loc'],trace_uid=uid))
        db.put('unknown_group',k,g)
    for track in tracks:
        if track['role']=='unrecognized' and track['complete_events']:
            fail(dict(id='unrecognized_track',reason='unrecognized track contains complete events',pid=track['pid'],tid=track['tid'],count=track['complete_events']))
            for raw_id, in db.c.execute("SELECT r.id FROM kv k JOIN raw r ON r.id=k.id WHERE k.kind='track' AND k.k=? ORDER BY r.loctype,r.loc",(key((track['pid'],track['tid'])),)):
                e=db.raw(raw_id)
                if e.get('ph')=='X':add('track','unrecognized_track',e)
    for row in ledger:
        if row['category']!='unknown' and row['disposition']!='unknown':continue
        e=db.raw(db.rid(row))
        if row['category']=='unknown':add('category','unknown_category',e)
        if row['disposition']=='unknown':add('disposition',row['reason'] or 'unknown_disposition',e)
    def failures():
        for f in values(db,'failure','pstr(k) COLLATE BIGINT'):
            if f.pop('pids_marker',False):f['pids']=Rows(lambda:values(db,'unexpected_pid'))
            yield f
    def all_conditions():
        yield from conditions
        for f in failures():yield dict(id=f['id'],requirement='required',state='fail',reason=f['reason'],evidence=f)
    groups=Rows(lambda:values(db,'unknown_group','group_dim(payload),group_label(payload),group_reason(payload)'))
    warning_count=0;specials=[]
    for f in values(db,'failure','pstr(k) COLLATE BIGINT'):
        if f.pop('pids_marker',False):
            f['pids']=Rows(lambda:values(db,'unexpected_pid'));specials.append(f)
        else:db.put('warning',warning_count,f);warning_count+=1
    for r in structural+sc:
        if r['state']=='fail':
            f=dict(id=r['id'],reason=r['reason'],actual=r['actual'])
            if isinstance(r['actual'],Rows):specials.append(f)
            else:db.put('warning',warning_count,f);warning_count+=1
    for g in groups:
        if g['dimension']=='category':db.put('warning',warning_count,dict(id='unknown_category',label=g['label'],count=g['count'],known_time_ns=g['known_time_ns'],dur_missing=g['dur_missing']));warning_count+=1
    def warnings():
        # Match baseline json.dumps(sort_keys=True), including its whitespace
        # and ASCII escaping. Ordinary rows fit one source-record budget.
        def baseline_tokens(obj):
            if isinstance(obj,Rows):
                yield b'['
                for i,v in enumerate(obj):
                    if i:yield b', '
                    yield from baseline_tokens(v)
                yield b']'
            elif isinstance(obj,dict):
                yield b'{'
                for i,k in enumerate(sorted(obj)):
                    if i:yield b', '
                    yield json.dumps(k).encode();yield b': ';yield from baseline_tokens(obj[k])
                yield b'}'
            else:yield json.dumps(obj,sort_keys=True).encode()
        def cmp(a,b):
            for x,y in zip_longest(chain.from_iterable(baseline_tokens(a)),chain.from_iterable(baseline_tokens(b)),fillvalue=-1):
                if x!=y:return (x>y)-(x<y)
            return 0
        pending=iter(sorted(specials,key=cmp_to_key(cmp)));special=next(pending,None)
        for f in values(db,'warning','warning_order(payload)'):
            while special is not None and cmp(special,f)<=0:
                yield special;special=next(pending,None)
            yield f
        while special is not None:yield special;special=next(pending,None)
    return dict(format_version=1,format=fmt,status='outside' if failure_count else c.applicability(conditions),conditions=Rows(all_conditions),
        tracks=tracks,warnings=Rows(warnings),unknown_groups=groups,support_scope=dict(observed=facts,unchecked=contract['support_scope']['unchecked']),
        time_semantics='sum of source activity durations; overlapping; not wall-clock',configuration=identities.get('collector',{}).get('configuration',dict(status='unavailable',reason='not_recorded')))
