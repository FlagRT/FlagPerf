"""Strict, disk-backed analysis reader; the snapshot Reader API stays intact."""
from collections.abc import MutableMapping
import ast
import json
from pathlib import Path
import sqlite3
from .reader import Reader, StoredRecord, StoreReadError


class ResourceLimit(ValueError):
    pass


class DiskMap(MutableMapping):
    def __init__(self, db, namespace):
        self.db, self.namespace = db, namespace

    def __getitem__(self, key):
        row = self.db.execute('select value from state where ns=? and key=?', (self.namespace, repr(key))).fetchone()
        if row is None:
            raise KeyError(key)
        return StoredRecord(**json.loads(row[0]))

    def __setitem__(self, key, value):
        self.db.execute('insert or replace into state values(?,?,?)',
                        (self.namespace, repr(key), json.dumps(value.as_dict())))

    def __delitem__(self, key):
        self.db.execute('delete from state where ns=? and key=?', (self.namespace, repr(key)))

    def __iter__(self):
        for row in self.db.execute('select key from state where ns=? order by key', (self.namespace,)):
            yield ast.literal_eval(row[0])

    def __len__(self):
        return self.db.execute('select count(*) from state where ns=?', (self.namespace,)).fetchone()[0]


class DiskSet:
    def __init__(self, db, namespace):
        self.db, self.namespace = db, namespace

    def add(self, value):
        self.db.execute('insert or ignore into members values(?,?)', (self.namespace, value))

    def __contains__(self, value):
        return self.db.execute('select 1 from members where ns=? and value=?', (self.namespace, value)).fetchone() is not None


class DiskGaps:
    def __init__(self, db, writer):
        self.db, self.writer = db, writer

    def append(self, pair):
        self.db.execute('insert into gaps values(?,?,?)', (self.writer, *pair))

    def __iter__(self):
        yield from self.db.execute('select start,end from gaps where writer=? order by start', (self.writer,))


class LimitedFile:
    def __init__(self, handle, limit):
        self.handle, self.limit = handle, limit

    def readline(self):
        line = self.handle.readline(self.limit+1)
        if len(line) > self.limit:
            raise ResourceLimit('resource_limit: max_record_bytes')
        return line

    def __getattr__(self, name):
        return getattr(self.handle, name)


class AnalysisReader(Reader):
    def __init__(self, run, db, *, record_bytes=16 << 20, cache_bytes=64 << 20, scratch_bytes=4 << 30):
        run = Path(run)
        # Bound metadata before the compatibility reader loads its manifests.
        paths = [run/'manifest.json', *(run/'writers').glob('*.json')]
        if len(paths) > 257 or any(p.stat().st_size > record_bytes for p in paths) or sum(p.stat().st_size for p in paths)>cache_bytes//4:
            raise ResourceLimit('resource_limit: reader metadata')
        super().__init__(run, retain=False)
        self.db = db
        self.record_bytes, self.scratch_bytes = record_bytes, scratch_bytes
        self.db.execute('pragma cache_size=-'+str(max(1, min(2048, cache_bytes//8192))))
        self.db.execute('pragma temp_store=FILE')
        self.db.executescript('''create table if not exists state(ns text,key text,value text,primary key(ns,key));
            create table if not exists members(ns text,value text,primary key(ns,value));
            create table if not exists gaps(writer integer,start integer,end integer);''')
        self._records = DiskMap(db, 'records')
        self._checked = 0

    def _error(self, shard, reason):
        # This public strict iterator stops at the first integrity failure.
        raise StoreReadError(dict(read_integrity='failed', errors=[dict(writer_id=shard.writer_id,
            shard_index=shard.index, record_index=shard.read_lines, reason=reason)]))

    def _consume_iter(self, shard):
        if len(self._shards) > 4096:
            raise ResourceLimit('resource_limit: reader shards')
        shard.signatures = DiskSet(self.db, 'sigs:'+str(shard.writer_id)+':'+str(shard.index))
        self._drop_reports.setdefault(shard.writer_id, DiskMap(self.db, 'drops:'+str(shard.writer_id)))
        self._gaps.setdefault(shard.writer_id, DiskGaps(self.db, shard.writer_id))
        yield from super()._consume_iter(shard)

    def _open(self, shard):
        result = super()._open(shard)
        if result:
            shard.handle = LimitedFile(shard.handle, self.record_bytes)
        return result

    def _decode(self, raw, shard):
        self._checked += 1
        if self._checked % 256 == 0:
            self.db.commit()
            if self.db.execute('pragma page_count').fetchone()[0]*self.db.execute('pragma page_size').fetchone()[0] > self.scratch_bytes:
                raise ResourceLimit('resource_limit: reader scratch_bytes')
        return super()._decode(raw, shard)
