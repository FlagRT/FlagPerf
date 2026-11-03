# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Private experiment storage. SQLite is scratch, source bytes remain evidence."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sqlite3
from collections import OrderedDict

from . import streaming
JSONStream = streaming.JSONStream


from contextlib import contextmanager
from contextvars import ContextVar
_scratch = ContextVar('flagprof_offline_scratch', default=None)


@contextmanager
def scratch_scope(root, anchor):
    token=_scratch.set((Path(root).resolve(),Path(anchor).resolve()))
    try:yield
    finally:_scratch.reset(token)


def scratch_path(path):
    """Only rebuildable SQLite goes to configured local disk."""
    path=Path(path).resolve();config=_scratch.get()
    if config is not None:
        root,anchor=config
        if not path.is_relative_to(anchor):raise ValueError('scratch path outside work anchor')
        path=root/path.relative_to(anchor)
    path.parent.mkdir(parents=True,exist_ok=True);return path


def pack(value): return pickle.dumps(value, protocol=4)
def unpack(value): return pickle.loads(value)


def integer_order_key(value):
    """Exact numeric order as SQLite BLOB order, including signed large integers."""
    number = int(value)
    if number == 0:
        return b'\x01'
    digits = str(abs(number)).encode('ascii')
    magnitude = len(digits).to_bytes(8, 'big') + digits
    if number < 0:
        return b'\x00' + magnitude.translate(_COMPLEMENT)
    return b'\x02' + magnitude


_COMPLEMENT = bytes.maketrans(bytes(range(256)), bytes(reversed(range(256))))


def key(value):
    # Python's equality namespace: bool/int/integral floats compare equal.
    if isinstance(value,int):value=int(value)
    elif isinstance(value,float) and math.isfinite(value) and int(value)==value:value=int(value)
    if isinstance(value, (tuple,list)): value=tuple(key(v) for v in value)
    return pack(value)


class Rows:
    def __init__(self, factory): self.factory=factory
    def __iter__(self): return iter(self.factory())

class JSONFile:
    def __init__(self,path):self.path=Path(path)

class ArrayFile:
    """Canonical JSONL to JSON array without materializing even one long row."""
    def __init__(self,path):self.path=Path(path)


def plain(value):
    if isinstance(value,dict):return all(plain(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(plain(v) for v in value)
    return isinstance(value,(str,int,float,bool,type(None)))


def tokens(value):
    # C encoder for a bounded ordinary record; streaming envelopes keep Rows.
    if isinstance(value,(dict,list,tuple)):
        try:data=json.dumps(value,ensure_ascii=False,separators=(',',':'),sort_keys=True,allow_nan=True).encode()
        except TypeError:pass
        else:
            yield data
            return
    if isinstance(value,ArrayFile):
        yield b'['
        first=True;pending_separator=False
        with value.path.open('rb') as f:
            for block in iter(lambda:f.read(65536),b''):
                for i,part in enumerate(block.split(b'\n')):
                    if i:pending_separator=True
                    if part:
                        if pending_separator and not first:yield b','
                        pending_separator=False;first=False;yield part
        yield b']'
    elif isinstance(value,JSONFile):
        with value.path.open('rb') as f:
            size=value.path.stat().st_size
            if size:
                f.seek(-1,2)
                if f.read(1)==b'\n':size-=1
            f.seek(0)
            while size:
                block=f.read(min(size,65536));size-=len(block);yield block
    elif hasattr(value,'__json_items__'):
        yield b'{'
        for i,(k,item) in enumerate(value.__json_items__()):
            if i:yield b','
            yield json.dumps(k,ensure_ascii=False,separators=(',',':')).encode();yield b':'
            yield from tokens(item)
        yield b'}'
    elif isinstance(value, dict):
        yield b'{'
        for i,k in enumerate(sorted(value)):
            if i: yield b','
            yield json.dumps(k,ensure_ascii=False,separators=(',',':')).encode();yield b':'
            yield from tokens(value[k])
        yield b'}'
    elif isinstance(value,(list,tuple,Rows)):
        yield b'['
        for i,item in enumerate(value):
            if i:yield b','
            yield from tokens(item)
        yield b']'
    else:yield json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=True).encode()


def encoded(value, limit=4<<20):
    out=bytearray()
    for token in tokens(value):
        if len(out)+len(token)>limit:raise ValueError('max_output_event_bytes; no record omitted')
        out.extend(token)
    return bytes(out)


class Sink:
    def __init__(self,path,max_row_bytes=None):
        self.max_row_bytes=max_row_bytes
        self.path=Path(path);self.f=self.path.open('xb');self.sha=hashlib.sha256();self.size=0;self.buf=bytearray()
    def write(self,data):
        self.buf.extend(data)
        if len(self.buf)>=65536:self.drain()
    def drain(self):
        if self.buf:self.f.write(self.buf);self.sha.update(self.buf);self.size+=len(self.buf);self.buf.clear()
    def row(self,value):
        size=0
        for t in tokens(value):
            size+=len(t)
            if self.max_row_bytes is not None and size>self.max_row_bytes:raise ValueError('max_derived_record_bytes; source preserved')
            self.write(t)
        self.write(b'\n')
    def close(self):
        import os
        self.drain();self.f.flush();os.fsync(self.f.fileno());self.f.close()
        return dict(path=str(self.path),sha256=self.sha.hexdigest(),bytes=self.size)


def file_rows(path):
    with Path(path).open('rb') as f:
        while line:=f.readline((16<<20)+1):
            if len(line)>16<<20:raise ValueError('staged row budget')
            yield json.loads(line)


class ByteCache:
    def __init__(self,limit=2<<20):self.limit=limit;self.used=0;self.data=OrderedDict()
    def get(self,k):
        hit=self.data.get(k)
        if hit is not None:self.data.move_to_end(k);return hit[0]
        return None
    def put(self,k,v):
        cost=len(k)+len(v)+192
        old=self.data.pop(k,None)
        if old:self.used-=old[1]
        if cost>self.limit:return
        while self.data and self.used+cost>self.limit:
            _,(_,n)=self.data.popitem(last=False);self.used-=n
        self.data[k]=(v,cost);self.used+=cost


class DB:
    def __init__(self,path,existing=False):
        self.path=scratch_path(path);self.c=sqlite3.connect(self.path)
        self.source=None;self.cache=ByteCache();self.pending={};self.pending_bytes=0;self.pending_rows=0;self.ordinal_pending={};self.empty_modes=set()
        self.c.execute('PRAGMA cache_size=-2048');self.c.execute('PRAGMA temp_store=FILE')
        self.c.execute('PRAGMA journal_mode=OFF');self.c.execute('PRAGMA synchronous=OFF')
        self.c.create_collation('BIGINT',lambda a,b:(int(a)>int(b))-(int(a)<int(b)))
        self.c.create_function('integer_order_key',1,integer_order_key,deterministic=True)
        self.c.create_function('corr_type',1,lambda p:0 if isinstance(unpack(p),int) else 1)
        self.c.create_function('corr_num',1,lambda p:str(unpack(p)) if isinstance(unpack(p),int) else '0')
        self.c.create_function('corr_str',1,lambda p:'' if isinstance(unpack(p),int) else str(unpack(p)))
        if existing:return
        self.c.executescript('''
        CREATE TABLE raw(id INTEGER PRIMARY KEY,loc TEXT UNIQUE,stem TEXT,suffix TEXT,loctype TEXT,original BLOB,stamped BLOB,start INTEGER,end INTEGER,stem_order BLOB GENERATED ALWAYS AS (integer_order_key(stem)) STORED);
        CREATE INDEX raworder ON raw(stem_order,suffix);
        CREATE TABLE norm(id INTEGER PRIMARY KEY,payload BLOB,kind TEXT,category TEXT,host BLOB,thread BLOB,ts TEXT,dur TEXT,is_call INTEGER,is_annotation INTEGER,is_module INTEGER);
        CREATE INDEX normkind ON norm(kind,id);
        CREATE TABLE kv(kind TEXT,k BLOB,id INTEGER,PRIMARY KEY(kind,k,id)) WITHOUT ROWID;
        CREATE INDEX kv_id ON kv(kind,id,k);
        CREATE TABLE tracks(k BLOB PRIMARY KEY,payload BLOB);
        CREATE TABLE stage(kind TEXT,k BLOB,payload BLOB,PRIMARY KEY(kind,k)) WITHOUT ROWID;
        CREATE TABLE scratch(kind TEXT,k BLOB,id INTEGER,value BLOB,PRIMARY KEY(kind,k)) WITHOUT ROWID;
        CREATE TABLE source_stage(kind TEXT,id INTEGER,payload BLOB,PRIMARY KEY(kind,id)) WITHOUT ROWID;
        CREATE TABLE device_edges(src INTEGER,mode INTEGER,dst INTEGER,extra INTEGER);
        CREATE INDEX device_edges_src ON device_edges(src,mode,extra,dst);
        CREATE TABLE threads(id INTEGER PRIMARY KEY,k BLOB UNIQUE);
        CREATE TABLE time_points(t TEXT PRIMARY KEY,ordinal INTEGER,t_order BLOB GENERATED ALWAYS AS (integer_order_key(t)) STORED) WITHOUT ROWID;
        CREATE INDEX timeorder ON time_points(t_order);
        CREATE TABLE interval_inputs(id INTEGER PRIMARY KEY,lo TEXT,hi TEXT,gid INTEGER);
        CREATE VIRTUAL TABLE intervals USING rtree_i32(id,lo,hi,glo,ghi);
        ''')
    def flush_pending(self):
        for sql,rows in self.pending.items():self.c.executemany(sql,rows)
        self.pending.clear();self.ordinal_pending.clear();self.pending_rows=0;self.pending_bytes=0
    def enqueue(self,sql,row):
        cost=sum(len(v) if isinstance(v,bytes) else 4*len(v) if isinstance(v,str) else 32 for v in row)+192
        if self.pending_rows and (self.pending_rows>=128 or self.pending_bytes+cost>1<<20):self.flush_pending()
        self.pending.setdefault(sql,[]).append(row);self.pending_rows+=1;self.pending_bytes+=cost
    def commit(self):self.flush_pending();self.c.commit()
    def close(self):
        self.c.close()
        if self.source:self.source.close()
    def addkey(self,kind,k,i):self.enqueue('INSERT OR IGNORE INTO kv VALUES(?,?,?)',(kind,key(k),i))
    def ids(self,kind,k):
        self.flush_pending()
        return (r[0] for r in self.c.execute('SELECT id FROM kv WHERE kind=? AND k=? ORDER BY id',(kind,key(k))))
    def has(self,kind,k):
        kval=key(k)
        if any(r[0]==kind and r[1]==kval for r in self.pending.get('INSERT OR IGNORE INTO kv VALUES(?,?,?)',())):return True
        return self.c.execute('SELECT 1 FROM kv WHERE kind=? AND k=? LIMIT 1',(kind,kval)).fetchone() is not None
    def raw(self,i,original=False):
        ck=pack(('raw',i));data=self.cache.get(ck)
        if data is None:
            data=self.c.execute('SELECT original FROM raw WHERE id=?',(i,)).fetchone()[0]
            self.cache.put(ck,data)
        r=unpack(data)
        if not original:r.setdefault('_linker_loc',i)
        return r
    def track_raws(self,k):
        for i,payload in self.c.execute("SELECT r.id,r.original FROM kv k JOIN raw r ON r.id=k.id WHERE k.kind='track' AND k.k=? ORDER BY r.id",(key(k),)):
            r=unpack(payload);r.setdefault('_linker_loc',i);yield r
    def records(self,kind,k):
        return (unpack(r[0]) for r in self.c.execute('SELECT n.payload FROM kv k JOIN norm n ON n.id=k.id WHERE k.kind=? AND k.k=? ORDER BY k.id',(kind,key(k))))
    def ordered_stage(self,kind,order='r.stem_order,r.suffix'):
        self.flush_pending()
        return (unpack(r[0]) for r in self.c.execute('SELECT s.payload FROM source_stage s JOIN raw r ON r.id=s.id WHERE s.kind=? ORDER BY '+order,(kind,)))
    def add_interval(self,i,start,end,thread):
        """Stage exact endpoints; absolute nanoseconds must never enter R-tree floats."""
        if abs(start)>1e35 or abs(end)>1e35:raise ValueError('interval index numeric range; source preserved')
        self.c.execute('INSERT OR IGNORE INTO threads(k) VALUES(?)',(thread,))
        gid=self.c.execute('SELECT id FROM threads WHERE k=?',(thread,)).fetchone()[0]
        if gid>2147483647:raise ValueError('interval thread ordinal range; source preserved')
        lo,hi=str(min(start,end)),str(max(start,end))
        for t in (lo,hi):self.enqueue('INSERT OR IGNORE INTO time_points(t) VALUES(?)',(t,))
        self.enqueue('INSERT INTO interval_inputs VALUES(?,?,?,?)',(i,lo,hi,gid))
    def build_intervals(self):
        """Coordinate compression is disk ordered and bounded by the normal batch budget."""
        self.flush_pending()
        for ordinal,(t,) in enumerate(self.c.execute('SELECT t FROM time_points ORDER BY t_order')):
            if ordinal>2147483647:raise ValueError('interval time ordinal range; source preserved')
            self.enqueue('UPDATE time_points SET ordinal=? WHERE t=?',(ordinal,t))
        self.flush_pending()
        self.c.execute('''INSERT INTO intervals SELECT i.id,l.ordinal,h.ordinal,i.gid,i.gid
          FROM interval_inputs i CROSS JOIN time_points l ON l.t=i.lo
          CROSS JOIN time_points h ON h.t=i.hi''')
        self.c.commit()
    def time_bounds(self,t):
        # For a point between endpoints, floor bounds starts and ceil bounds ends.
        # This also supports callers querying points absent from the source.
        ck=pack(('time_bounds',str(t)));hit=self.cache.get(ck)
        if hit is not None:return unpack(hit)
        row=self.c.execute('SELECT t,ordinal FROM time_points WHERE t_order<=? ORDER BY t_order DESC LIMIT 1',(integer_order_key(t),)).fetchone()
        bounds=None if row is None else (row[1],row[1]+int(int(row[0])!=t))
        self.cache.put(ck,pack(bounds));return bounds
    def build_device_edges(self):
        self.flush_pending()
        self.c.execute("INSERT INTO device_edges SELECT d.id,1,l.id,NULL FROM kv d JOIN kv l ON l.kind='launch_key' AND l.k=d.k WHERE d.kind='device_launch'")
        self.c.execute("""INSERT INTO device_edges SELECT d.id,0,s.id,e.id FROM kv d
          CROSS JOIN kv e ON e.kind='flow_endpoint' AND e.k=d.k
          CROSS JOIN kv f ON f.kind='endpoint_flow' AND f.id=e.id
          CROSS JOIN kv s ON s.kind='flow_start' AND s.k=f.k WHERE d.kind='device_endpoint'""")
        for mode,predicate in [('project',"kind='op'"),('call','is_call=1'),('scope','is_call=1 OR is_annotation=1'),('annotation','is_annotation=1')]:
            if not self.c.execute('SELECT 1 FROM norm WHERE '+predicate+' LIMIT 1').fetchone():self.empty_modes.add(mode)
        if not self.c.execute("SELECT 1 FROM kv WHERE kind='dequeue' LIMIT 1").fetchone():self.empty_modes.update(('dequeue','project_dequeue'))
        self.c.commit()
    def edge_hits(self,event,vendor):
        prefix='cuda' if vendor in ('nvidia','thead') else 'cann'
        q='SELECT e.mode,n.payload,f.payload FROM device_edges e JOIN norm n ON n.id=e.dst LEFT JOIN norm f ON f.id=e.extra WHERE e.src=? ORDER BY e.mode,e.extra,e.dst'
        for mode,launch,extra in self.c.execute(q,(self.rid(event),)):
            yield prefix+('.flow' if mode==0 else '.correlation_id' if vendor in ('nvidia','thead') else '.connection_id'),unpack(launch),(unpack(extra),) if extra is not None else ()
    def node(self,i):
        ck=pack(('norm',i));data=self.cache.get(ck)
        if data is None:
            data=self.c.execute('SELECT payload FROM norm WHERE id=?',(i,)).fetchone()[0];self.cache.put(ck,data)
        return unpack(data)
    def rid(self,node):
        ck=pack(('rid',str(node['loc'])));data=self.cache.get(ck)
        if data is None:
            data=pack(self.c.execute('SELECT id FROM raw WHERE loc=?',(str(node['loc']),)).fetchone()[0]);self.cache.put(ck,data)
        return unpack(data)
    def raws(self):
        for i,payload in self.c.execute('SELECT id,original FROM raw ORDER BY id'):
            r=unpack(payload);r.setdefault('_linker_loc',i);yield r
    def nodes(self,where='1',params=()):
        self.flush_pending()
        return (unpack(r[0]) for r in self.c.execute('SELECT payload FROM norm n JOIN raw r ON n.id=r.id WHERE '+where+' ORDER BY r.stem_order,r.suffix',params))
    def put(self,kind,k,value):
        payload=pack(value)
        if kind in ('ledger','inventory','unbound'):
            self.enqueue('INSERT OR REPLACE INTO source_stage VALUES(?,?,?)',(kind,k,payload))
            self.ordinal_pending[kind,k]=payload
            return
        kval=key(k);ck=pack(('stage',kind,kval));cached=pack((True,payload))
        if self.cache.get(ck)==cached:return
        self.c.execute('INSERT OR REPLACE INTO stage VALUES(?,?,?)',(kind,kval,payload));self.cache.put(ck,cached)
    def get(self,kind,k,default=None):
        if kind in ('ledger','inventory','unbound'):
            payload=self.ordinal_pending.get((kind,k))
            if payload is not None:return unpack(payload)
            row=self.c.execute('SELECT payload FROM source_stage WHERE kind=? AND id=?',(kind,k)).fetchone()
            return unpack(row[0]) if row else default
        kval=key(k);ck=pack(('stage',kind,kval));hit=self.cache.get(ck)
        if hit is None:
            r=self.c.execute('SELECT payload FROM stage WHERE kind=? AND k=?',(kind,kval)).fetchone()
            hit=pack((r is not None,r[0] if r else None));self.cache.put(ck,hit)
        exists,payload=unpack(hit)
        return unpack(payload) if exists else default
    def stage(self,kind):return (unpack(r[0]) for r in self.c.execute('SELECT payload FROM stage WHERE kind=? ORDER BY k',(kind,)))
    def clear(self):self.c.execute('DELETE FROM scratch')
    def add(self,kind,value):
        i=self.rid(value)
        self.c.execute('INSERT OR IGNORE INTO scratch VALUES(?,?,?,NULL)',(kind,key(i),i))
    def scalar(self,kind,value):self.c.execute('INSERT OR IGNORE INTO scratch VALUES(?,?,NULL,?)',(kind,key(value),pack(value)))
    def count(self,kind):return self.c.execute('SELECT count(*) FROM scratch WHERE kind=?',(kind,)).fetchone()[0]
    def members(self,kind):
        return (unpack(r[0]) for r in self.c.execute('SELECT n.payload FROM scratch s JOIN norm n ON n.id=s.id JOIN raw r ON r.id=n.id WHERE s.kind=? ORDER BY r.stem_order,r.suffix',(kind,)))
    def refs(self,kind):
        return Rows(lambda: (dict(trace_uid=n['trace_uid'],loc=n['loc']) for n in self.members(kind)))
    def scalars(self,kind):return (unpack(r[0]) for r in self.c.execute('SELECT value FROM scratch WHERE kind=?',(kind,)))
    def contains(self,node,mode,strict=False):
        if mode in self.empty_modes:return
        from flagprof.correlate.native import contains
        from flagprof.vendors.rules_common import raw_interval,source_thread
        from flagprof.vendors.native_common import launch_interval
        if mode not in ('project','project_dequeue'):node=launch_interval(node) if mode in ('call','dequeue') else node
        t=node.get('ts')
        if t is None:return
        predicate={'project':"n.kind='op'",'call':'n.is_call=1','scope':'(n.is_call=1 OR n.is_annotation=1)',
                   'annotation':'n.is_annotation=1','dequeue':"EXISTS (SELECT 1 FROM kv k WHERE k.kind='dequeue' AND k.id=n.id)",
                   'project_dequeue':"EXISTS (SELECT 1 FROM kv k WHERE k.kind='dequeue' AND k.id=n.id)"}[mode]
        thread=key((node['host_pid'],node['tid']))
        gid=self.c.execute('SELECT id FROM threads WHERE k=?',(thread,)).fetchone()
        if gid is None:return
        bounds=self.time_bounds(t)
        if bounds is None:return
        query='SELECT n.payload FROM intervals q CROSS JOIN norm n ON n.id=q.id CROSS JOIN raw r ON r.id=n.id WHERE q.lo<=? AND q.hi>=? AND q.glo<=? AND q.ghi>=? AND n.thread=? AND '+predicate+' ORDER BY r.stem_order,r.suffix'
        for payload, in self.c.execute(query,(*bounds,gid[0],gid[0],thread)):
            n=unpack(payload)
            if mode in ('project','project_dequeue'):
                a,b=raw_interval(n);x,y=raw_interval(node)
                good=source_thread(n)==source_thread(node) and a<=x<b and y<=b
            else:good=n['loc']!=node['loc'] and contains(n,node,strict=strict)
            if good:yield n
    def innermost(self,kind,out):
        # A strict chain can be recognized by adjacent pairs in outer-to-inner
        # order. Crossings/equal bounds retain the complete set, never pruning.
        from flagprof.correlate.native import contains
        q='SELECT n.payload FROM scratch s JOIN norm n ON n.id=s.id JOIN raw r ON r.id=n.id WHERE s.kind=? ORDER BY n.ts COLLATE BIGINT,n.dur COLLATE BIGINT DESC,r.stem_order,r.suffix'
        last=None;chain=True
        for payload, in self.c.execute(q,(kind,)):
            n=unpack(payload)
            if last is not None and not contains(last,n,strict=True):chain=False
            last=n
        if last is not None and chain:self.add(out,last)
        else:self.c.execute('INSERT OR IGNORE INTO scratch SELECT ?,k,id,value FROM scratch WHERE kind=?',(out,kind))
