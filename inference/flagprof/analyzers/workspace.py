# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Owned disk indexes and replayable result tables for analysis consumers."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sqlite3
from flagprof.store import StoreView
from flagprof.store.analysis_read import ResourceLimit


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class Limits:
    cache_bytes: int = 64 << 20
    record_bytes: int = 16 << 20
    evidence_samples: int = 8
    html_rows: int = 100
    scratch_bytes: int = 4 << 30

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('positive integer analysis limits required')


class AnalysisSet:
    def __init__(self, view):
        self.view = view
        self.collection = None
        self.notes = []
        try:
            source = view.analysis_source()
            if source['status'] == 'available':
                self.collection = view.open_collection()
        except (ValueError, OSError, KeyError) as error:
            self.notes.append('collection unavailable: '+str(error))

    @classmethod
    def open(cls, view):
        return cls(view)

    def views(self):
        return [self.view] + ([self.collection] if self.collection else [])

    def close(self):
        if self.collection:
            self.collection.close()


class Table:
    def __init__(self, db, panel, table='rows'):
        self.db, self.panel, self.table = db, panel, table

    def __iter__(self):
        order = 'metric desc,key'
        for payload, in self.db.execute('select payload from '+self.table+' where panel=? order by '+order, (self.panel,)):
            yield json.loads(payload)

    def __len__(self):
        return self.db.execute('select count(*) from '+self.table+' where panel=?', (self.panel,)).fetchone()[0]


class Workspace:
    def __init__(self, view, directory, limits=None):
        self.view = view
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_relative_to(view.run.resolve()) or view.run.resolve().is_relative_to(self.directory):
            raise ValueError('analysis work must be outside the input Store')
        self.limits = limits or Limits()
        self.source = AnalysisSet.open(view)
        self.db = sqlite3.connect(self.directory/'analysis.sqlite')
        self.db.execute('pragma cache_size=-'+str(max(1, min(4096, self.limits.cache_bytes//8192))))
        self.db.executescript('''pragma journal_mode=OFF; pragma temp_store=FILE;
            create table events(run text,w integer,seq integer,type text,corr text,sid text,scope text,parent text,
                ts integer,dur integer,clk text,payload text,primary key(run,w,seq));
            create index by_corr on events(run,corr,type);
            create index by_scope on events(run,scope,type);
            create index by_parent on events(run,parent,type,ts);
            create table sigs(sid text primary key,payload text);
            create table rows(panel text,key text,payload text,metric real,primary key(panel,key));
            create index row_order on rows(panel,metric desc,key);
            create table members(panel text,key text,run text,corr text,primary key(panel,key,run,corr));
            create table native(trace text,loc text,owner text,payload text,primary key(trace,loc));
            create table native_seen(key text,trace text,loc text,primary key(key,trace,loc));
            create table opgroups(key text primary key, sid text,dev text,nested integer,decision text,
                calls integer,host integer,device integer,each_n integer,summary_n integer,samples text,corrs text,scopes text);
            create index op_sid on opgroups(sid,each_n);
            create table compilations(backend text,ver text,dur integer,payload text);
            create index compile_order on compilations(backend,ver,dur);
        ''')
        self.notes = list(self.source.notes)
        self.statuses = []
        self.read_reports = []
        self.limit_error = None
        self.count = 0
        self.native_available = False
        self.qualities = {}
        self.legacy_summary_calls=0

    def check_budget(self):
        self.db.commit()
        total = sum(p.stat().st_size for p in self.directory.rglob('*') if p.is_file())
        if total > self.limits.scratch_bytes:
            raise ResourceLimit('resource_limit: analysis scratch_bytes')

    def ingest(self):
        for view in self.source.views():
            state=view.probe_status(limits=asdict(self.limits))
            self.statuses.extend(state['attachments'])
            try:
                for stored in view.iter_events(work=self.directory, limits=asdict(self.limits)):
                    r = dict(stored.record)
                    if r['t']=='opsum' and state['schema_version']=='1.0':
                        self.legacy_summary_calls+=r['n']
                    if stored.source is not None:
                        r['_source'] = stored.source
                    if r['t'] == 'sig':
                        payload = encode(r)
                        previous = self.db.execute('select payload from sigs where sid=?', (r['sid'],)).fetchone()
                        if previous and previous[0] != payload:
                            raise ValueError('conflicting signature across analysis sources')
                        self.db.execute('insert or ignore into sigs values(?,?)', (r['sid'], payload))
                    elif 'seq' in r:
                        self.db.execute('insert into events values(?,?,?,?,?,?,?,?,?,?,?,?)',
                            (stored.run_id, stored.writer_id, r['seq'], r['t'], str(r.get('corr', 0)), r.get('sid'),
                             str(r.get('scope', 0)), str(r.get('parent', 0)), r.get('ts'), r.get('dur'), r.get('clk'), encode(r)))
                    self.count += 1
                    if self.count % 256 == 0:
                        self.check_budget()
                self.read_reports.append(view.analysis_read_report)
            except (ResourceLimit, ValueError, OSError) as error:
                self.limit_error = str(error)
                self.notes.append(str(error))
        try:
            for r in self.view.iter_native_relations(work=self.directory, limits=asdict(self.limits)):
                self.qualities[r['trace_uid']]=r.get('source_state',{})
                self.db.execute('insert into native values(?,?,?,?)', (r['trace_uid'], str(r['loc']),
                    encode(r.get('owner')), encode(r)))
                self.native_available |= r.get('state') == 'owned'
                self.count += 1
                if self.count % 256 == 0:
                    self.check_budget()
        except (ResourceLimit, ValueError, OSError) as error:
            self.notes.append('native unavailable: '+str(error))
            self.native_available=False
        self.check_budget()

    def probe(self, name):
        rows = [s['probes'][name] for s in self.statuses if name in s.get('probes', {})]
        if any(r['status']=='available' for r in rows):
            return next(r for r in rows if r['status']=='available')
        return dict(status='unavailable', reason=rows[0].get('reason') if rows else 'probe_record_missing')

    def events(self, types):
        placeholders = ','.join('?' for _ in types)
        for run, w, seq, payload in self.db.execute('select run,w,seq,payload from events where type in ('+placeholders+') order by run,w,seq', tuple(types)):
            yield run, w, seq, json.loads(payload)

    def signature(self, sid):
        row = self.db.execute('select payload from sigs where sid=?', (sid,)).fetchone()
        return json.loads(row[0]) if row else None

    def event(self, run, corr, kind='op'):
        row = self.db.execute('select payload from events where run=? and corr=? and type=?', (run, str(corr), kind)).fetchone()
        return json.loads(row[0]) if row else None

    def row(self, panel, key, value):
        metric=next((value.get(k) for k in ('host_ns','inclusive_ns','total_ns','device_work_ns','calls') if value.get(k) is not None),0)
        self.db.execute('insert into rows values(?,?,?,?) on conflict(panel,key) do update set payload=excluded.payload,metric=excluded.metric',
                        (panel, str(key), encode(value),metric))

    def table(self, panel):
        return Table(self.db, panel)

    def coverage(self):
        return dict(limits=asdict(self.limits), clock_valid=dict(status='unavailable'),
                    legacy_summary_calls=self.legacy_summary_calls,
                    events_processed=self.count, resource_limit=self.limit_error,
                    read_reports=self.read_reports, analysis_notes=self.notes,
                    interpretation=self.qualities)

    def partial(self):
        return bool(self.limit_error or self.source.notes or self.legacy_summary_calls or any(report.get('read_integrity')!='verified' or
            any(w.get('dropped') or w.get('tail_unknown') for w in report.get('writers',{}).values()) for report in self.read_reports))

    def close(self):
        self.source.close()
        self.db.close()
