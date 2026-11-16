# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Disk-backed presentation records. No device libraries or whole-Store lists."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from flagprof.analyzers.workspace import Limits, encode
from flagprof.offline.streaming import JSONStream
from flagprof.store.analysis_read import ResourceLimit


class Rows:
    def __init__(self, model, section):
        self.model, self.section = model, section

    def __iter__(self):
        for row, in self.model.db.execute(
                'select payload from report_rows where section=? order by ordinal', (self.section,)):
            yield json.loads(row)

    def __len__(self):
        return self.model.db.execute(
            'select count(*) from report_rows where section=?', (self.section,)).fetchone()[0]


class Model:
    def __init__(self, directory, limits=None):
        self.limits = limits if isinstance(limits,Limits) else Limits(**(limits or {}))
        self.path = Path(directory) / 'report-view.sqlite'
        self.db = sqlite3.connect(self.path)
        self.db.execute('pragma journal_mode=OFF')
        self.db.execute('pragma temp_store=FILE')
        self.db.execute('pragma cache_size=-'+str(max(1, min(4096, self.limits.cache_bytes//1024))))
        self.db.execute('create table report_rows(section text,ordinal integer,payload text,'
                        'primary key(section,ordinal))')
        self.sections = []
        self.meta = dict(schema='flagprof-result-view/1', title='推理运行结果',
                         facts=[], findings=[], notices=[], links=[])
        self._counts = {}

    def add(self, section, row):
        payload = encode(row)
        if len(payload.encode()) > self.limits.record_bytes:
            raise ResourceLimit('report row exceeds record_bytes')
        n = self._counts.get(section, 0)
        self.db.execute('insert into report_rows values(?,?,?)', (section, n, payload))
        self._counts[section] = n+1
        if n % 256 == 0:
            self.db.commit()
            if self.path.stat().st_size > self.limits.scratch_bytes:
                raise ResourceLimit('report presentation exceeds scratch_bytes')

    def rows(self, section):
        return Rows(self, section)

    def section(self, ident, title, columns, *, description='', status='ok', kind='table'):
        value = dict(id=ident, title=title, columns=list(columns), description=description,
                     status=status, kind=kind)
        self.sections.append(value)
        return value

    def finding(self, title, detail, section=None, level='observation', *, row_index=None):
        self.meta['findings'].append(dict(title=title, detail=detail, section=section,
                                          level=level, row_index=row_index))

    def save(self, path):
        with Path(path).open('w', encoding='utf-8') as stream:
            stream.write(encode(dict(type='page', value=self.meta))+'\n')
            for section in self.sections:
                stream.write(encode(dict(type='section', value=section))+'\n')
                for row in self.rows(section['id']):
                    stream.write(encode(dict(type='row', section=section['id'], value=row))+'\n')

    def load(self, path):
        with Path(path).open(encoding='utf-8') as stream:
            while True:
                line = stream.readline(self.limits.record_bytes+1)
                if not line:
                    break
                if len(line.encode()) > self.limits.record_bytes:
                    raise ResourceLimit('report snapshot record exceeds record_bytes')
                item = json.loads(line)
                if item['type'] == 'page':
                    if item['value'].get('schema') != 'flagprof-result-view/1':
                        raise ValueError('unsupported report presentation schema')
                    self.meta = item['value']
                elif item['type'] == 'section':
                    self.sections.append(item['value'])
                elif item['type'] == 'row':
                    self.add(item['section'], item['value'])
                else:
                    raise ValueError('unknown report presentation record')

    def close(self):
        self.db.close()


@contextmanager
def opened_model(work=None, limits=None):
    if work:
        Path(work).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='flagprof-report-', dir=work) as folder:
        model = Model(folder, limits)
        try:
            yield model
        finally:
            model.close()


TABLE_KEYS = frozenset((
    'data', 'traceEvents', 'pair_statuses', 'rank_execution', 'phases', 'metric_comparisons', 'headline', 'groups', 'op_groups', 'layer_groups', 'introduced',
    'introduced_routes', 'amplification', 'streams', 'device_calls', 'series',
    'transfers', 'allocations', 'stages', 'energy', 'memory', 'restrictions',
    'frequency', 'steps', 'kv', 'calls', 'missing_ops', 'inventory_v1_view',
    'unscoped_by_tid', 'minute_trends', 'request_windows', 'rank_counts',
    'findings', 'evidence', 'progress_coverage', 'samples', 'alignment',
    'sources', 'comparisons', 'window_comparisons', 'ranks',
))


def read_document(path, model):
    """Read legacy result.json with large record arrays spooled to SQLite."""
    parser = JSONStream(path, limit=model.limits.record_bytes)
    serial = 0
    model._document_counter=getattr(model,"_document_counter",0)+1
    document_id=model._document_counter

    def value(path=()):
        nonlocal serial
        parser.space()
        if parser.peek() == '{':
            parser.expect('{')
            result = {}
            parser.space()
            if parser.peek() == '}':
                parser.take(1)
                return result
            while True:
                key, _, _ = parser.value()
                if not isinstance(key, str) or key in result:
                    raise ValueError('invalid or duplicate report object key')
                parser.expect(':')
                result[key] = value(path+(key,))
                parser.space()
                if parser.peek() == '}':
                    parser.take(1)
                    return result
                parser.expect(',')
        if (parser.peek() == '[' and path and path[-1] in TABLE_KEYS and
                (len(path) == 1 or path[0] == 'results' and
                 (len(path) == 3 or len(path) == 4 and path[2] == 'coverage') or
                 path[0] == 'diagnosis' and len(path) == 2)):
            serial += 1
            ident = '_input_'+str(document_id)+'_'+str(serial)
            parser.expect('[')
            parser.space()
            if parser.peek() != ']':
                while True:
                    row, _, _ = parser.value()
                    model.add(ident, row)
                    parser.space()
                    if parser.peek() == ']':
                        break
                    parser.expect(',')
            parser.expect(']')
            return model.rows(ident)
        return parser.value()[0]
    try:
        result = value()
        parser.space()
        if parser.peek():
            raise ValueError('trailing report input')
        return result
    finally:
        parser.close()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            value.update(part)
    return value.hexdigest()
