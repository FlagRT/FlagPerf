"""Task-local indexed layer observations for exact, bounded group reductions."""
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sqlite3
import tempfile


class Rows(Mapping):
    def __init__(self, index, side): self.index, self.side = index, side
    def __len__(self): return self.index.db.execute('SELECT count(*) FROM rows WHERE side=?', (self.side,)).fetchone()[0]
    def __iter__(self):
        for key, in self.index.db.execute('SELECT identity FROM rows WHERE side=? ORDER BY position', (self.side,)):
            yield tuple(json.loads(key))
    def __getitem__(self, key):
        row = self.index.db.execute('SELECT body FROM rows WHERE side=? AND identity=?', (self.side,json.dumps(key))).fetchone()
        if row is None: raise KeyError(key)
        return json.loads(row[0])
    def __setitem__(self, key, row):
        group = (row['module'],str(row['rank']),str(row['input_shape']),row.get('phase'),row.get('generation_step'),row.get('kv_length_before'))
        call = (row['repeat'],row['cycle'],row['batch_index'],row['module'],row['call_index'])
        self.index.db.execute('INSERT INTO rows(side,identity,grouping,call,body) VALUES (?,?,?,?,?)',
            (self.side,json.dumps(key),json.dumps(group),json.dumps(call),json.dumps(row)))
    def items(self):
        for key,body in self.index.db.execute('SELECT identity,body FROM rows WHERE side=? ORDER BY position', (self.side,)):
            yield tuple(json.loads(key)),json.loads(body)
    def values(self):
        for _,body in self.items(): yield body


class Group(Sequence):
    def __init__(self, index, field, value, side, count):
        self.index,self.field,self.value,self.side,self.count=index,field,value,side,count
    def __len__(self): return self.count
    def __iter__(self):
        for body, in self.index.db.execute('SELECT body FROM rows WHERE '+self.field+'=? AND side=? ORDER BY position',(self.value,self.side)):
            yield json.loads(body)
    def __getitem__(self, key): return list(self)[key]


class Index:
    def __init__(self):
        from runtime.tensor_io import work_directory
        root=work_directory(); root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='layers-',dir=root)
        self.db=sqlite3.connect(str(Path(self.temp.name)/'rows.sqlite'))
        self.db.execute('PRAGMA cache_size=-16384'); self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('CREATE TABLE rows(position INTEGER PRIMARY KEY,side TEXT,identity TEXT,grouping TEXT,call TEXT,body TEXT,UNIQUE(side,identity))')
        self.db.execute('CREATE INDEX by_group ON rows(grouping,side,position)')
        self.db.execute('CREATE INDEX by_call ON rows(call,side,position)')
    def side(self, name): return Rows(self,name)
    def groups(self):
        # Group metadata is small; observations stay in SQLite.
        groups={}
        for group,side,count in self.db.execute('SELECT grouping,side,count(*) FROM rows GROUP BY grouping,side ORDER BY min(position)'):
            groups.setdefault(tuple(json.loads(group)),{})[side]=Group(self,'grouping',group,side,count)
        return groups
    def calls(self, side):
        for key,count in self.db.execute('SELECT call,count(*) FROM rows WHERE side=? GROUP BY call ORDER BY min(position)',(side,)):
            yield tuple(json.loads(key)),Group(self,'call',key,side,count)
    def close(self):
        self.db.close(); self.temp.cleanup()
    def __del__(self):
        try: self.close()
        except Exception: pass
