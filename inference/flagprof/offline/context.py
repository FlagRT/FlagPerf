# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Streaming taxonomy context and format assessment, preserving source order."""
from collections import Counter
from decimal import Decimal,InvalidOperation
import json
import re
from .disk import Rows,key,pack,unpack
from flagprof.vendors import contracts as c


class Names:
    def __init__(self,db,pid):self.db,self.pid=db,pid
    def __iter__(self):
        # process_name cardinality is handled on disk; duplicate names collapse.
        seen='names:'+str(self.pid)
        for row in self.db.c.execute("SELECT payload FROM stage WHERE kind=? ORDER BY k",(seen,)):yield unpack(row[0])
    def __contains__(self,v):return self.db.get('names:'+str(self.pid),v,False) is not False
    def __bool__(self):return next(iter(self),False) is not False
    def __le__(self,other):return all(n in other for n in self)
    def __eq__(self,other):return self<=other and sum(1 for _ in self)==len(other)


class NameMap:
    def __init__(self,db):self.db=db
    def get(self,pid,default=None):return Names(self.db,key(pid).hex())
    def __getitem__(self,pid):return self.get(pid)


class Roles:
    def __init__(self,db):self.db=db
    def get(self,k,default=None):return self.db.get('role',k,default)


class Context:
    def __init__(self,db):
        self.db=db;self.names=NameMap(db);self.host_records=HostSet(db);self.role_index=Roles(db)
    def get(self,pid,default=None):return self.db.get('process_name',pid,default)
    def __contains__(self,pid):return self.db.has('device_pid',pid)
    @property
    def tracks(self):return Rows(lambda:self.db.stage('track'))


class HostSet:
    def __init__(self,db):self.db=db
    def __contains__(self,pid):return self.db.has('host_pid',pid)


class Taxonomy:
    def __init__(self,original,context):self.original,self.ctx=original,context
    def __getattr__(self,name):return getattr(self.original,name)
    def context(self,events):return self.ctx


def build_context(db,taxonomy):
    ctx=Context(db);roles=taxonomy.contract['roles']
    for i,e in enumerate(db.raws()):
        pid=e.get('pid');cat=e.get('cat');name=e.get('name','')
        db.addkey('track',(pid,e.get('tid')),i)
        db.put('track_key',(pid,e.get('tid')),(pid,e.get('tid')))
        if name=='process_name':
            value=e.get('args',{}).get('name');db.put('process_name',pid,value)
            db.put('names:'+key(pid).hex(),value,value)
        if name=='process_labels':db.put('labels',pid,e.get('args',{}).get('labels',''))
        if cat in roles['device_categories']:db.addkey('device_pid',pid,i)
        if cat=='cpu_op' or name.startswith(('Enqueue','Dequeue')):db.addkey('host_pid',pid,i)
    for pid,tid in db.stage('track_key'):
        pn=set(ctx.names.get(pid));count=0;device_structure=False;witness=False;host_cat=False
        allhost=bool(roles['host_categories']);allallowed=bool(roles['allowed_non_device_categories'])
        structural=[True]*len(roles['structural_records'])
        for i in db.ids('track',(pid,tid)):
            e=db.raw(i)
            if e.get('ph')!='X':continue
            count+=1
            device_structure |= roles['device_identity_signal']=='category_or_device_stream' and taxonomy.is_device(e,ctx)
            audit=roles['audit_device_structure'];witness |= bool(audit) and all(k in e.get('args',{}) for k in audit['args_required'])
            host_cat |= e.get('cat') in roles['host_categories']
            allhost &= e.get('cat') in roles['host_categories'];allallowed &= e.get('cat') in roles['allowed_non_device_categories']
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
        item=dict(pid=pid,tid=tid,process_names=sorted(pn,key=str),role=role,basis=basis,complete_events=count,conflicts=conflicts)
        db.put('track',(pid,tid),item);db.put('role',(pid,tid),role)
    return Taxonomy(taxonomy,ctx)


def scope(db,tax,vendor):
    missing=0;examples=[];selected=0;mismatch=False;missing_arg=False
    for i,e in enumerate(db.raws()):
        if e.get('ph')=='X' and e.get('cat') in (tax.contract['roles']['host_categories'] if vendor in ('nvidia','thead','kunlunxin') else ['cpu_op']):db.put('scope_host',e.get('pid'),True)
        if not tax.is_device(e,tax.ctx):continue
        db.put('scope_device_pid',e.get('pid'),True)
        if len(examples)<3:examples.append(e['_linker_loc'])
        if vendor in ('nvidia','thead','kunlunxin'):v=e.get('args',{}).get('device')
        else:
            m=re.fullmatch(r'NPU (\d+)',db.get('labels',e.get('pid'),''));v=int(m.group(1)) if m else None
        valid=type(v) is int and v>=0
        if valid:db.put('scope_device',v,v)
        else:missing+=1
        if tax.ctx.role_index.get((e.get('pid'),e.get('tid')))=='device':
            label=str(v) if valid else 'unknown';db.put('scope_track',(label,e.get('pid'),e.get('tid')),label)
            selected+=1;args=e.get('args',{})
            mismatch |= 'device' in args and (type(e.get('pid')) is not int or type(args['device']) is not int or e.get('pid')!=args['device'])
            missing_arg |= 'device' not in args
    # Device/track cardinality is a control dimension; values are emitted from disk.
    known=sorted(db.stage('scope_device'))
    # Kept small in normal captures, independently challenged by metadata-scale tests.
    tracks=Counter(db.stage('scope_track'))
    collision=bool(db.c.execute("SELECT 1 FROM stage a JOIN stage b ON a.k=b.k WHERE a.kind='scope_host' AND b.kind='scope_device_pid' LIMIT 1").fetchone())
    fields=['args.device','pid','cat'] if vendor in ('nvidia','thead','kunlunxin') else ['process_labels.args.labels','pid','cat']
    facts=dict(device_count=len(known) if not missing else None,device_ordinals=known if not missing else None,
       observed_devices=known,missing_device_identity=missing,host_device_pid_collision=collision,
       source=dict(kind='raw_trace',fields=fields,examples=examples),
       execution_track_count=sum(t['role']=='device' and bool(t['complete_events']) for t in tax.ctx.tracks),
       tracks_per_device=dict(sorted(tracks.items())))
    if vendor in ('nvidia','thead','kunlunxin'):facts['device_pid_matches_device_arg']=False if mismatch else None if missing_arg or not selected else True
    return facts


def assess(db,tax,identities,ledger,vendor,uid):
    contract=tax.contract;conditions=c.evaluate_conditions(contract['conditions'],c.identity_facts(identities))
    for row in conditions:
        row['source']={'hardware':identities.get('hardware',{}).get('source'),
          'collector':identities.get('collector',{}).get('source'),'actual':identities.get('collector',{}).get('configuration',{}),
          'sdk':identities.get('collector',{}).get('configuration',{}).get('actual',{}).get('sources',[]),
          'execution':identities.get('collector',{}).get('source'),'exporter':identities.get('exporter',identities.get('parser',{}).get('exporter',{}))}.get(row['path'].split('.')[0])
    tracks=tax.ctx.tracks;structural=c.structure_conditions(contract,tracks);conditions.extend(structural)
    facts=scope(db,tax,vendor);sc=c.evaluate_conditions(contract['support_scope']['conditions'],dict(scope=facts))
    for row in sc:row['source']=facts['source']
    conditions.extend(sc)
    failures=tax.audit_dependencies(Rows(db.raws),tax.ctx)+c.role_conflicts(tracks)
    fmt=tax.identity()
    def add(dimension,reason,e):
        label=tax.task_type(e) or e.get('cat') or '<absent>';k=(dimension,str(label),reason)
        g=db.get('unknown_group',k,dict(dimension=dimension,label=label,reason=reason,format=fmt,count=0,known_time_ns=0,dur_missing=0,examples=[]))
        g['count']+=1
        try:
            duration=Decimal(str(e.get('dur')))
            if not duration.is_finite() or duration<0:raise ValueError()
            g['known_time_ns']+=int(duration*1000)
        except (InvalidOperation,ValueError):g['dur_missing']+=1
        if len(g['examples'])<3:g['examples'].append(dict(loc=e['_linker_loc'],trace_uid=uid))
        db.put('unknown_group',k,g)
    for track in tracks:
        if track['role']=='unrecognized' and track['complete_events']:
            failures.append(dict(id='unrecognized_track',reason='unrecognized track contains complete events',pid=track['pid'],tid=track['tid'],count=track['complete_events']))
            for raw_id, in db.c.execute("SELECT r.id FROM kv k JOIN raw r ON r.id=k.id WHERE k.kind='track' AND k.k=? ORDER BY r.loctype,r.loc",(key((track['pid'],track['tid'])),)):
                e=db.raw(raw_id)
                if e.get('ph')=='X':add('track','unrecognized_track',e)
    for row in ledger:
        e=db.raw(db.rid(row))
        if row['category']=='unknown':add('category','unknown_category',e)
        if row['disposition']=='unknown':add('disposition',row['reason'] or 'unknown_disposition',e)
    for failure in failures:conditions.append(dict(id=failure['id'],requirement='required',state='fail',reason=failure['reason'],evidence=failure))
    warnings=list(failures)+[dict(id=r['id'],reason=r['reason'],actual=r['actual']) for r in structural+sc if r['state']=='fail']
    # Ordering uses the original Python tuple ordering, not SQLite BLOB ordering.
    groups=sorted(db.stage('unknown_group'),key=lambda g:(g['dimension'],str(g['label']),g['reason']))
    for g in groups:
        if g['dimension']=='category':warnings.append(dict(id='unknown_category',label=g['label'],count=g['count'],known_time_ns=g['known_time_ns'],dur_missing=g['dur_missing']))
    ordered_tracks=Rows(lambda: sorted(db.stage('track'),key=lambda t:(str(t['pid']),str(t['tid']))))
    return dict(format_version=1,format=fmt,status=c.applicability(conditions),conditions=conditions,tracks=ordered_tracks,
        warnings=sorted(warnings,key=lambda r:json.dumps(r,sort_keys=True)),unknown_groups=groups,
        support_scope=dict(observed=facts,unchecked=contract['support_scope']['unchecked']),
        time_semantics='sum of source activity durations; overlapping; not wall-clock',
        configuration=identities.get('collector',{}).get('configuration',dict(status='unavailable',reason='not_recorded')))
