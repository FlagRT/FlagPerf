"""Disk-backed grouping for exact statistics over large record artifacts."""
from collections.abc import Sequence
import json
from pathlib import Path
import sqlite3
import tempfile


class Index:
    def __init__(self, rows, fields):
        from runtime.tensor_io import work_directory
        root = work_directory(); root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='groups-', dir=root)
        self.db = sqlite3.connect(str(Path(self.temp.name)/'groups.sqlite'))
        self.db.execute('PRAGMA cache_size=-16384')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('CREATE TABLE rows (position INTEGER PRIMARY KEY, grouping TEXT, body TEXT)')
        self.counts = {}
        for position, row in enumerate(rows):
            key = tuple(str(row[field]) for field in fields)
            self.counts[key] = self.counts.get(key, 0) + 1
            self.db.execute('INSERT INTO rows VALUES (?, ?, ?)',
                            (position, json.dumps(key), json.dumps(row, ensure_ascii=False)))
        self.db.execute('CREATE INDEX grouped ON rows (grouping, position)')
        self.db.commit()

    def groups(self):
        return {key: Group(self, key, count) for key, count in self.counts.items()}

    def __del__(self):
        try:
            self.db.close(); self.temp.cleanup()
        except Exception: pass


class Group(Sequence):
    def __init__(self, index, key, count):
        self.index, self.key, self.count = index, json.dumps(key), count

    def __len__(self): return self.count

    def __iter__(self):
        for body, in self.index.db.execute('SELECT body FROM rows WHERE grouping=? ORDER BY position', (self.key,)):
            yield json.loads(body)

    def __getitem__(self, index):
        if isinstance(index, slice): return list(self)[index]
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        row = self.index.db.execute('SELECT body FROM rows WHERE grouping=? ORDER BY position LIMIT 1 OFFSET ?',
                                    (self.key, index)).fetchone()
        return json.loads(row[0])
