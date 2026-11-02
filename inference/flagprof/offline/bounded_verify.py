# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Independent bounded import validation, with production Reader.stream.

Large JSON documents are parsed into a bounded-cache SQLite tree. This avoids
silently omitting descriptor inventory, completion maps or large native arrays.
It is deliberately independent of the preparation scratch database.
"""
from pathlib import Path
import hashlib
import json
import os
import sqlite3
from .disk import JSONStream,key,pack,unpack,streaming,scratch_path
from flagprof.store.reader import Reader
from flagprof.store.manifest import json_bytes


def identity(path):
    s=Path(path).stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


class Spilled:
    def __init__(self,group):self.group=group

class Value:
    def __init__(self,value):self.value=value

class Document:
    """Disk rows for large arrays; no per-JSON-node SQL tree.

    Fixed envelope fields are bounded. Each large array item is independently
    bounded and JSON-decoded. All large arrays remain ordered disk iterators.
    """
    def __init__(self,path):
        self.c=sqlite3.connect(scratch_path(path))
        self.c.executescript("""PRAGMA cache_size=-2048; PRAGMA temp_store=FILE;
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE items(g INTEGER,n INTEGER,payload BLOB,PRIMARY KEY(g,n)) WITHOUT ROWID;
        CREATE TABLE sources(loc TEXT PRIMARY KEY,ordinal INTEGER,start INTEGER,end INTEGER);
        CREATE TABLE expected(seq INTEGER PRIMARY KEY,loc TEXT,ordinal INTEGER UNIQUE,seen INTEGER DEFAULT 0);
        CREATE TABLE ledger_population(loc TEXT PRIMARY KEY,disposition TEXT,category TEXT);
        CREATE TABLE ledger_outputs(ordinal INTEGER PRIMARY KEY,loc TEXT);
        CREATE TABLE inventory(loc TEXT PRIMARY KEY,category TEXT);
        CREATE TABLE native_population(loc TEXT PRIMARY KEY,state TEXT,reason TEXT,category TEXT);
        """)
        self.counter=0;self.group=0;self.documents={};self.last_handle=None;self.last_value=None
    def parse(self,path,jsonl=False):
        stream=JSONStream(path);before=identity(path)
        def array(jsonl=False):
            g=self.group;self.group+=1;n=0
            if not jsonl:
                stream.expect('[');stream.space()
                if stream.peek()==']':stream.take(1);return Spilled(g)
            while True:
                stream.space()
                if jsonl and not stream.peek():break
                row,_,_=stream.value()
                self.c.execute('INSERT INTO items VALUES(?,?,?)',(g,n,json_bytes(row)));n+=1
                if not jsonl:
                    stream.space()
                    if stream.peek()==']':stream.take(1);break
                    stream.expect(',')
            return Spilled(g)
        def envelope(path=()):
            stream.expect('{');result={};budget=0;stream.space()
            if stream.peek()=='}':stream.take(1);return result
            while True:
                k,_,_=stream.value()
                if not isinstance(k,str) or k in result:raise ValueError('duplicate/nonstring envelope key')
                if len(result)>=1024 or len(k.encode('utf-8'))>1024:raise ValueError('metadata key budget')
                stream.expect(':');q=path+(k,)
                if q in [('inventory',)]:v=envelope(q)
                elif q in [('inventory','records'),('device_ids',),('event_map',),('entries',),('unbound_calls',)]:v=array()
                else:
                    v,a,b=stream.value();budget+=b-a
                    if budget>4<<20:raise ValueError('metadata materialization limit')
                result[k]=v;stream.space()
                if stream.peek()=='}':stream.take(1);break
                stream.expect(',')
            return result
        try:
            value=array(True) if jsonl else envelope()
            stream.space()
            if stream.peek():raise ValueError('trailing document data')
            if before!=identity(path):raise ValueError('document changed during parse')
            root=self.counter;self.counter+=1;self.documents[root]=value
            return root,stream.sha.hexdigest(),stream.bytes
        finally:stream.close()
    def get(self,root,*path):
        if isinstance(root,Value):value=root.value
        elif isinstance(root,tuple):
            if root!=self.last_handle:
                self.last_value=json.loads(self.c.execute('SELECT payload FROM items WHERE g=? AND n=?',root).fetchone()[0]);self.last_handle=root
            value=self.last_value
        else:value=self.documents[root]
        for k in path:value=value[k]
        return value
    def children(self,root,*path):
        value=self.get(root,*path)
        if isinstance(value,Spilled):return (Value(json.loads(payload)) for payload, in self.c.execute('SELECT payload FROM items WHERE g=? ORDER BY n',(value.group,)))
        return (Value(v) for v in value)
    def walk(self,value):
        if isinstance(value,Spilled):
            for payload, in self.c.execute('SELECT payload FROM items WHERE g=? ORDER BY n',(value.group,)):
                yield from self.walk(json.loads(payload))
        elif isinstance(value,dict):
            yield value
            for v in value.values():yield from self.walk(v)
        elif isinstance(value,list):
            for v in value:yield from self.walk(v)
    def validate_refs(self,lo,hi,uid):
        return sum(self.validate_value_refs(self.documents[root],uid) for root in range(lo,hi))
    def validate_value_refs(self,value,uid):
        count=0
        for row in self.walk(value):
            if 'trace_uid' in row and 'loc' in row:
                if row['trace_uid']!=uid or self.c.execute('SELECT 1 FROM sources WHERE loc=?',(str(row['loc']),)).fetchone() is None:raise ValueError('unresolved or wrong-trace reference')
                count+=1
        return count
    def digest(self,root,*,omit=(),overrides=None):
        from .disk import tokens,Rows
        overrides=overrides or {}
        def convert(value,path=()):
            if path in overrides:return overrides[path]
            if isinstance(value,Spilled):return Rows(lambda:(json.loads(r[0]) for r in self.c.execute('SELECT payload FROM items WHERE g=? ORDER BY n',(value.group,))))
            if isinstance(value,dict):return {k:convert(v,path+(k,)) for k,v in value.items() if path+(k,) not in omit}
            return value
        h=hashlib.sha256()
        for b in tokens(convert(self.get(root))):h.update(b)
        return h.hexdigest()


def verify(run,db_path,*,expected_pid=None,expected_tid=None):
    doc=Document(db_path)
    try:return _verify(run,doc,expected_pid=expected_pid,expected_tid=expected_tid)
    finally:doc.c.close()


def verify_open(run,db_path):
    """Retain independently decoded metadata for the lifetime of one view."""
    doc=Document(db_path)
    try:return _verify(run,doc),doc
    except BaseException:
        doc.c.close()
        raise


def _verify(run,doc,*,expected_pid=None,expected_tid=None):
    from .limits import preflight_store
    run=Path(run);preflight_store(run);identities={};paths=list((run/'traces').glob('*.json'))
    if len(paths)!=1:raise ValueError('smoke verifier requires exactly one trace')
    descriptor_path=paths[0];d,dsha,_=doc.parse(descriptor_path);uid=doc.get(d,'trace_uid')
    doc.descriptor_root=d
    def path(relative,prefix):
        p=Path(relative);target=run/p
        if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!=prefix or target.is_symlink() or not target.resolve().is_relative_to((run/prefix).resolve()):raise ValueError('path escape')
        return target
    def keep(p):identities[str(p)]=identity(p)
    keep(descriptor_path)
    if doc.get(d,'format_version')!=1 or doc.get(d,'sha256')[:16]!=uid:raise ValueError('descriptor identity')
    capture=doc.get(d,'capture')
    if capture is not None:
        from .bundle import verify_registration
        verify_registration(run,capture,doc.get(d,'sha256'))
    cp=path(doc.get(d,'completion_path'),'imports');comp,csha,_=doc.parse(cp);keep(cp)
    if doc.get(comp,'descriptor_sha256')!=dsha or doc.get(comp,'trace_uid')!=uid or doc.get(comp,'writer_id')!=doc.get(d,'writer_id') or doc.get(comp,'format_version')!=1:raise ValueError('completion identity')
    roots={};derived={}
    for name in ('index','ledger','native','summary'):
        info=doc.get(d,'derived',name);p=path(info['path'],'derived');keep(p)
        root,sha,size=doc.parse(p,jsonl=p.suffix=='.jsonl');roots[name]=root;derived[name]=sha
        if sha!=info['sha256'] or size!=info['bytes'] or doc.get(comp,'derived_sha256',name)!=sha:raise ValueError('derived SHA/size')
    if 'analysis_intervals' in doc.get(d,'derived'):
        info=doc.get(d,'derived','analysis_intervals');p=path(info['path'],'derived');keep(p)
        from .disk import streaming
        sha=streaming.sha_file(p)
        if sha!=info['sha256'] or p.stat().st_size!=info['bytes'] or doc.get(comp,'derived_sha256','analysis_intervals')!=sha:
            raise ValueError('analysis intervals SHA/size')
        derived['analysis_intervals']=sha
    doc.summary_root=roots['summary']
    raw=path(doc.get(d,'raw_path'),'vendor_raw');keep(raw);scan=JSONStream(raw);ordinal=0
    idx=roots['index'];entries=iter(doc.children(idx,'entries'))
    try:
        for row,start,end in scan.records():
            e=doc.get(next(entries));loc=row.get('_linker_loc',ordinal)
            if e!=dict(loc=loc,ordinal=ordinal,start=start,end=end):raise ValueError('evidence offset/content mismatch')
            doc.c.execute('INSERT INTO sources VALUES(?,?,?,?)',(str(loc),ordinal,start,end));ordinal+=1
        if next(entries,None) is not None:raise ValueError('extra evidence entries')
        if scan.sha.hexdigest()!=doc.get(d,'sha256') or doc.get(idx,'raw_sha256')!=scan.sha.hexdigest() or doc.get(idx,'trace_uid')!=uid or doc.get(idx,'raw_bytes')!=scan.bytes:raise ValueError('raw/index identity')
    finally:scan.close()
    refs=0
    # Hash equality is not population conservation. Independently reconcile
    # every source, disposition, native row, output ordinal and stored event.
    for i in doc.children(roots['ledger']):
        refs+=doc.validate_value_refs(doc.get(i),uid)
        loc=str(doc.get(i,'loc'));disp=doc.get(i,'disposition');cat=doc.get(i,'category')
        if disp not in ('source_only','imported','unknown','unrepresentable') or cat not in (None,'compute','comm','memcpy','unknown'):
            raise ValueError('unknown ledger disposition/category')
        doc.c.execute('INSERT INTO ledger_population VALUES(?,?,?)',(loc,disp,cat));outputs=0
        for output in doc.children(i,'outputs'):
            doc.c.execute('INSERT INTO ledger_outputs VALUES(?,?)',(doc.get(output),loc));outputs+=1
        if bool(outputs)!=(disp=='imported'):raise ValueError('ledger disposition/output mismatch')
    population=doc.c.execute('SELECT count(*) FROM ledger_population').fetchone()[0]
    if population!=ordinal or population!=doc.get(d,'disposition','records') or population!=doc.get(roots['summary'],'disposition','records'):
        raise ValueError('ledger/source population mismatch')
    dispositions=dict(doc.c.execute('SELECT disposition,count(*) FROM ledger_population GROUP BY disposition'))
    if dispositions!=doc.get(d,'disposition','dispositions') or dispositions!=doc.get(roots['summary'],'disposition','dispositions'):
        raise ValueError('ledger disposition counts')
    # Descriptor inventory is also conserved against the independently scanned
    # source and ledger. A rehashed deletion must not become a valid descriptor.
    for i in doc.children(d,'inventory','records'):
        e=doc.get(i);loc=str(e['loc'])
        if e['trace_uid']!=uid:raise ValueError('inventory trace identity')
        row=doc.c.execute('SELECT category FROM ledger_population WHERE loc=?',(loc,)).fetchone()
        if row is None or row[0] is None or row[0]!=e['category']:raise ValueError('inventory ledger identity')
        doc.c.execute('INSERT INTO inventory VALUES(?,?)',(loc,e['category']))
    if doc.c.execute('SELECT 1 FROM ledger_population l LEFT JOIN inventory i USING(loc) WHERE l.category IS NOT NULL AND i.loc IS NULL LIMIT 1').fetchone():raise ValueError('inventory population mismatch')
    inventory_counts={k:0 for k in ('compute','comm','memcpy','unknown')}
    inventory_counts.update(dict(doc.c.execute('SELECT category,count(*) FROM inventory GROUP BY category')))
    if inventory_counts!=doc.get(d,'inventory','categories'):raise ValueError('inventory category counts')
    for i in doc.children(roots['native']):
        refs+=doc.validate_value_refs(doc.get(i),uid)
        loc=str(doc.get(i,'loc'));state=doc.get(i,'state');reason=doc.get(i,'reason');cat=doc.get(i,'category')
        if state not in ('owned','unowned','conflicted','exempt') or reason not in (None,'excluded','conflict','no_bridge','no_device_key'):
            raise ValueError('unknown native state/reason')
        row=doc.c.execute('SELECT category FROM ledger_population WHERE loc=?',(loc,)).fetchone()
        if row is None or row[0]!=cat:raise ValueError('native/ledger identity mismatch')
        doc.c.execute('INSERT INTO native_population VALUES(?,?,?,?)',(loc,state,str(reason),cat))
    states=dict(doc.c.execute('SELECT state,count(*) FROM native_population GROUP BY state'))
    reasons=dict(doc.c.execute('SELECT reason,count(*) FROM native_population GROUP BY reason'))
    if states!=doc.get(roots['summary'],'native_states') or reasons!=doc.get(roots['summary'],'native_reasons'):
        raise ValueError('native population/count mismatch')
    if doc.c.execute("SELECT 1 FROM ledger_population l LEFT JOIN native_population n USING(loc) WHERE l.disposition='imported' AND n.loc IS NULL LIMIT 1").fetchone():
        raise ValueError('imported device lacks native relation')
    attempted=enqueued=persisted=0;seqmin=seqmax=None;reasons=set()
    for i in doc.children(comp,'event_map'):
        e=doc.get(i)
        if e['trace_uid']!=uid or e['ordinal']!=attempted:raise ValueError('mapping ordinal/uid')
        attempted+=1;enqueued+=e['accepted'];persisted+=e['persisted']
        if e.get('reason'):reasons.add(e['reason'])
        if e.get('seq') is not None:
            seqmin=e['seq'] if seqmin is None else min(seqmin,e['seq']);seqmax=e['seq'] if seqmax is None else max(seqmax,e['seq'])
        if e['persisted']:doc.c.execute('INSERT INTO expected(seq,loc,ordinal) VALUES(?,?,?)',(e['seq'],str(e['loc']),e['ordinal']))
    if (attempted!=doc.get(comp,'attempted') or enqueued!=doc.get(comp,'enqueued') or persisted!=doc.get(comp,'persisted') or attempted-persisted!=doc.get(comp,'dropped') or doc.get(comp,'status')!='complete' or attempted!=persisted or doc.get(comp,'reasons')!=sorted(reasons) or doc.get(comp,'seq_range')!=([seqmin,seqmax+1] if seqmin is not None else None)):
        raise ValueError('completion population/status')
    if (doc.c.execute('SELECT count(*) FROM ledger_outputs').fetchone()[0]!=persisted or
        doc.c.execute('SELECT 1 FROM ledger_outputs l LEFT JOIN expected e USING(ordinal) WHERE e.seq IS NULL OR e.loc!=l.loc LIMIT 1').fetchone()):
        raise ValueError('ledger/completion output mismatch')
    count=0;h=hashlib.sha256();writer_id=doc.get(comp,'writer_id')
    from .memory import MemoryVerifier
    memory = MemoryVerifier(run,doc.c)
    with Reader(run,retain=False) as reader:
        for stored in reader.stream():
            memory.accept(stored)
            r=stored.record;ext=r.get('ext',{})
            if ext.get('chrome.trace_uid')!=uid:continue
            if stored.writer_id!=writer_id:raise ValueError('writer mapping')
            row=doc.c.execute('SELECT loc,seen FROM expected WHERE seq=?',(r['seq'],)).fetchone()
            if row is None or row[0]!=str(ext.get('chrome.loc')) or row[1]:raise ValueError('missing/duplicate/mismatched device mapping')
            doc.c.execute('UPDATE expected SET seen=1 WHERE seq=?',(r['seq'],))
            if expected_pid is not None:
                if r['pid']!=expected_pid or r['tid']!=expected_tid:raise ValueError('writer execution identity')
                r['pid']='<importer-process>';r['tid']='<importer-main-thread>'
            h.update(json_bytes(r)+b'\n');count+=1
        report=reader.report
    memory_result = memory.finish()
    if report['read_integrity']!='verified' or report['truncated']:raise ValueError('reader integrity: '+str(report))
    if count!=persisted or doc.c.execute('SELECT 1 FROM expected WHERE seen=0 LIMIT 1').fetchone():raise ValueError('missing device events')
    for name,old in identities.items():
        if identity(name)!=old:raise ValueError('file changed during verification')
    raw_sha=doc.get(d,'sha256');descriptor_sha=doc.digest(d,omit={('imported_at',)})
    completion_sha=doc.digest(comp,overrides={('descriptor_sha256',):descriptor_sha})
    doc.c.commit()
    return dict(status='complete',read_integrity='verified',count=count,events=h.hexdigest(),derived=derived,
                raw_sha=raw_sha,descriptor=descriptor_sha,completion=completion_sha,memory=memory_result,
                source_records=ordinal,checked_references=refs,reader_report=report)
