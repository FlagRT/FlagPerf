# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Supplemental official memory import with sealed sources and a bounded ledger."""
import importlib
from contextlib import nullcontext
import json
from pathlib import Path
from .checkpoints import fileinfo, publish
from flagprof.store.manifest import atomic_json
from flagprof.store._offline_writer import offline_writer


def adapter(vendor):
    if vendor not in ('ascend','nvidia','kunlunxin'):
        raise ValueError('unsupported memory vendor')
    return importlib.import_module('flagprof.vendors.'+vendor+'.memory')


def seal_artifacts(trace, destination, vendor, metadata):
    if not metadata.get('requested_configuration',{}).get('profile_memory'):
        return []
    from .bundle import copy_checked
    paths = adapter(vendor).artifacts(trace)
    for path in paths:
        copy_checked(path, destination/path.name,max_bytes=metadata.get('source_max_bytes',1<<30))
    return [p.name for p in paths]


def import_memory(directory, run, capture, *, writer=None, prepared=None):
    requested = capture['metadata'].get('requested_configuration',{}).get('profile_memory',False)
    state = dict(schema='official-memory/1', requested=requested, status='disabled',
                 records=0, normalized=0, excluded=0, sources=[],
                 clock_status='source_clock_only', layer_status='explicit_project_call_missing',
                 perturbation='enabled_unmeasured' if requested else 'disabled')
    state['configuration'] = capture['metadata'].get('identities',{}).get('collector',{}).get('configuration',{})
    if not requested:
        publish(run/'memory.json', state)
        return state
    root = Path(directory)/'sealed-trace/data'
    names = ['trace.json'] if capture['vendor'] in ('nvidia','kunlunxin') else capture.get('memory_files',[])
    state['sources'] = [dict(path='sealed-trace/data/'+name, **fileinfo(root/name)) for name in names]
    state['status'] = 'incomplete'
    atomic_json(run/'memory.json', state)
    ledger = run/'memory-ledger.jsonl'
    # A writer is closed before success. Receipts bind every accepted memory
    # record to the ledger; independent verification runs once at import time.
    owns_writer = writer is None
    with ledger.open('x') as output, (offline_writer(run) if owns_writer else nullcontext(writer)) as writer:
        if not writer.flush(120):raise ValueError('memory writer initial barrier failed')
        for source, name in zip(state['sources'], names):
            if prepared is not None and hasattr(__import__('flagprof.vendors.'+capture['vendor']+'.importer',fromlist=['memory_record']),'memory_record'):
                if 'memory' not in prepared.files:raise ValueError('prepared memory derivative missing')
                records=prepared.arrays('memory')
            else:records=adapter(capture['vendor']).records(root/name)
            for row in records:
                state['records'] += 1
                entry = dict(source=source['path'], **row)
                if row['status'] == 'normalized' and row['values']:
                    event = dict(t='counter', src='framework', dev=row['dev'], ts=None,
                        values=row['values'], ext={'memory':dict(component=row['component'],
                        source=source, loc=row['loc'], source_time_ns=row.get('source_time_ns'),
                        clock_status='source_clock_only', layer_status=row.get('layer_status','explicit_project_call_missing'),
                        native_owner=row.get('native_owner'), operation_status=row.get('operation_status'),
                        release_observed=row.get('release_observed'))})
                    receipt = writer.emit_with_receipt(event)
                    if not receipt.accepted:
                        raise ValueError('memory writer rejected a record')
                    entry.update(w=writer.writer_id, seq=receipt.seq)
                    state['normalized'] += 1
                    if state['normalized'] % min(128,writer.batch_limit)==0 and not writer.flush(120):
                        raise ValueError('memory writer barrier failed')
                else:
                    entry.update(status='excluded',reason=row.get('reason','no_occupancy_fields'))
                    state['excluded'] += 1
                output.write(json.dumps(entry, sort_keys=True)+'\n')
        if not writer.flush(120):raise ValueError('memory writer final barrier failed')
    state['ledger'] = dict(path=ledger.name, **fileinfo(ledger))
    if owns_writer:return finish_memory(run,state)
    return state


def finish_memory(run,state,*,verify=True):
    if not state['requested']:return state
    if verify:verify_memory(run, state)
    state['status'] = 'complete' if state['normalized'] else 'unavailable'
    if not state['normalized']:
        state['reason'] = 'official_memory_records_missing'
    atomic_json(run/'memory.json', state)
    return state


class MemoryVerifier:
    """Receipt checks share the existing production Reader traversal."""
    def __init__(self,run,db,state=None):
        import hashlib
        from .limits import read_json
        self.db=db;self.seen=0;run=Path(run)
        self.state=state if state is not None else (read_json(run/'memory.json') if (run/'memory.json').exists() else {})
        self.enabled=bool(self.state.get('requested'))
        db.execute('create table memory_expected(w integer,seq integer,payload text,primary key(w,seq))')
        if not self.enabled:return
        if self.state['ledger']['path']!='memory-ledger.jsonl':raise ValueError('memory ledger path')
        digest=hashlib.sha256();size=records=normalized=excluded=0
        with (run/'memory-ledger.jsonl').open('rb') as source:
            while line:=source.readline((16<<20)+1):
                if len(line)>16<<20:raise ValueError('memory ledger record byte budget')
                digest.update(line);size+=len(line);records+=1;row=json.loads(line)
                if row['status']=='normalized' and 'seq' in row:
                    normalized+=1
                    db.execute('insert into memory_expected values(?,?,?)',(row['w'],row['seq'],line))
                elif row['status']=='excluded':excluded+=1
                else:raise ValueError('memory ledger disposition')
        if ({'bytes':size,'sha256':digest.hexdigest()}!={k:self.state['ledger'][k] for k in ('bytes','sha256')} or
            [records,normalized,excluded]!=[self.state[k] for k in ('records','normalized','excluded')]):
            raise ValueError('memory ledger hash/population mismatch')
        self.sources={r['path']:r for r in self.state['sources']}
        registration=run/'plan-b-capture.json'
        if registration.exists():
            capture=read_json(registration)['capture']
            expected=({'trace.json':{k:capture['trace'][k] for k in ('bytes','sha256')}} if capture['vendor'] in ('nvidia','kunlunxin')
                      else capture.get('memory_sources',{}))
            if self.sources!={'sealed-trace/data/'+name:dict(path='sealed-trace/data/'+name,**info) for name,info in expected.items()}:
                raise ValueError('memory source/capture mismatch')

    def accept(self,stored):
        record=stored.record;meta=record.get('ext',{}).get('memory')
        if meta is None:return
        if not self.enabled:raise ValueError('unregistered memory record')
        found=self.db.execute('select payload from memory_expected where w=? and seq=?',(stored.writer_id,record['seq'])).fetchone()
        if not found:raise ValueError('unexpected memory record')
        row=json.loads(found[0])
        if (record['t']!='counter' or record['values']!=row['values'] or record['dev']!=row['dev'] or
            meta['source']!=self.sources.get(row['source']) or meta['loc']!=row['loc'] or meta['component']!=row['component'] or
            meta['source_time_ns']!=row.get('source_time_ns')):
            raise ValueError('memory ledger payload mismatch')
        self.db.execute('delete from memory_expected where w=? and seq=?',(stored.writer_id,record['seq']))
        self.seen+=1

    def finish(self):
        if self.enabled and (self.db.execute('select count(*) from memory_expected').fetchone()[0] or self.seen!=self.state['normalized']):
            raise ValueError('memory receipt conservation failed')
        return dict(records=self.seen,status='verified' if self.enabled else 'not_requested')


def verify_memory(run,state):
    """Standalone counter-only import; Plan B reuses bounded_verify instead."""
    import sqlite3
    import tempfile
    from flagprof.store import StoreView
    with tempfile.TemporaryDirectory(prefix='memory-verify-',dir=run.parent) as directory:
        db=sqlite3.connect(Path(directory)/'receipts.sqlite')
        try:
            verifier=MemoryVerifier(run,db,state)
            with StoreView(run) as view:
                for stored in view.iter_events(types=('counter',),work=directory):verifier.accept(stored)
            verifier.finish()
        finally:db.close()
