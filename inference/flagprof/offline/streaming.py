# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded UTF-8 JSON framing, hashing and verified JSONL pagination."""
from pathlib import Path
import codecs
import hashlib
import json
import sqlite3

class JSONStream:
    """Incremental UTF-8 decode + stdlib JSON values; bounded by max value size.

    Framing explicitly validates the top-level object/array, separators and EOF.
    raw_decode retries may be expensive for a single large record: measured,
    not represented as a native streaming parser or a production implementation.
    """
    def __init__(self, path, block=65536, limit=4*1024*1024):
        self.f = Path(path).open('rb')
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.json = json.JSONDecoder()
        self.text, self.pos, self.offset = '', 0, 0
        self.block, self.limit, self.eof = block, limit, False
        self.sha, self.bytes = hashlib.sha256(), 0
        self.max_buffer_chars = 0
    def close(self):
        self.f.close()
    def more(self):
        self.text = self.text[self.pos:]
        self.pos = 0
        raw = self.f.read(self.block)
        self.sha.update(raw)
        self.bytes += len(raw)
        self.eof = not raw
        self.text += self.decoder.decode(raw, final=self.eof)
        self.max_buffer_chars = max(self.max_buffer_chars, len(self.text))
    def take(self, n):
        value = self.text[self.pos:self.pos+n]
        self.pos += n
        self.offset += len(value.encode('utf-8'))
        return value
    def peek(self):
        while self.pos == len(self.text) and not self.eof:
            self.more()
        return self.text[self.pos:self.pos+1]
    def space(self):
        while self.peek() and self.peek() in ' \t\r\n':
            self.take(1)
    def expect(self, ch):
        self.space()
        if self.peek() != ch:
            raise ValueError('expected ' + repr(ch))
        self.take(1)
    def value(self):
        self.space()
        start = self.offset
        while True:
            try:
                value, end = self.json.raw_decode(self.text, self.pos)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                if len(self.text[self.pos:].encode('utf-8')) > self.limit:
                    raise ValueError('max_value_bytes')
                self.more()
                continue
            # raw_decode can accept a numeric prefix, e.g. 1 from 1e-3
            # when the exponent is split at a block boundary.
            if (end == len(self.text) or self.text[end] not in ' \t\r\n,:]}') and not self.eof:
                if len(self.text[self.pos:].encode('utf-8')) > self.limit:
                    raise ValueError('max_value_bytes')
                self.more()
                continue
            raw = self.take(end-self.pos)
            if len(raw.encode('utf-8')) > self.limit:
                raise ValueError('max_value_bytes')
            return value, start, self.offset
    def records(self):
        def array():
            self.expect('[')
            self.space()
            if self.peek() == ']':
                self.take(1)
                return
            while True:
                row, start, end = self.value()
                if not isinstance(row, dict):
                    raise ValueError('record is not an object')
                yield row, start, end
                self.space()
                ch = self.peek()
                if ch == ']':
                    self.take(1)
                    return
                self.expect(',')
        self.space()
        if self.peek() == '[':
            yield from array()
        else:
            self.expect('{')
            found = False
            self.space()
            if self.peek() == '}':
                raise ValueError('missing traceEvents')
            while True:
                key, _, _ = self.value()
                if not isinstance(key, str):
                    raise ValueError('object key is not string')
                self.expect(':')
                if key == 'traceEvents':
                    if found:
                        raise ValueError('duplicate traceEvents')
                    found = True
                    yield from array()
                else:
                    self.value()
                self.space()
                if self.peek() == '}':
                    self.take(1)
                    break
                self.expect(',')
            if not found:
                raise ValueError('missing traceEvents')
        self.space()
        if self.peek():
            raise ValueError('trailing data')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def stat_identity(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


class JSONLPage:
    def __init__(self, source, db_path, expected_sha):
        self.source = Path(source)
        self.index_path = Path(db_path)
        before = stat_identity(source)
        self.db = sqlite3.connect(str(db_path))
        self.db.execute('PRAGMA cache_size=-2048')
        self.db.execute('CREATE TABLE pages(ordinal INTEGER PRIMARY KEY, start INTEGER, size INTEGER)')
        h, offset, count = hashlib.sha256(), 0, 0
        with self.source.open('rb') as f:
            i = 0
            while True:
                line = f.readline(4*1024*1024+1)
                if not line:
                    break
                if len(line)>4*1024*1024:
                    raise ValueError('max_line_bytes')
                json.loads(line)
                h.update(line)
                self.db.execute('INSERT INTO pages VALUES(?,?,?)', (i, offset, len(line)))
                offset += len(line)
                count += 1
                i += 1
        if h.hexdigest() != expected_sha or before != stat_identity(source):
            self.db.close()
            raise ValueError('integrity mismatch')
        self.db.commit()
        self.db.close()
        self.index_identity = stat_identity(db_path)
        self.index_sha = sha_file(db_path)
        if self.index_identity != stat_identity(db_path):
            raise ValueError('index changed during verify')
        self.db = sqlite3.connect(self.index_path.resolve().as_uri() + '?mode=ro', uri=True)
        self.db.execute('PRAGMA cache_size=-2048')
        self.identity, self.count = before, count
    def check_identity(self):
        if stat_identity(self.source) != self.identity:
            raise ValueError('source changed after verify')
        if stat_identity(self.index_path) != self.index_identity:
            raise ValueError('index changed after verify')
    def page(self, offset, limit):
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=1000:
            raise ValueError('page limit')
        self.check_identity()
        spans=list(self.db.execute('SELECT start,size FROM pages WHERE ordinal>=? ORDER BY ordinal LIMIT ?', (offset, limit)))
        for i, (start, size) in enumerate(spans):
            if start < 0 or size <= 0 or start+size > self.identity[2]:
                raise ValueError('invalid index range')
            if i and spans[i-1][0]+spans[i-1][1] != start:
                raise ValueError('noncontiguous page range')
        total=sum(size for _,size in spans)
        if total>4*1024*1024:
            raise ValueError('max_page_bytes')
        result = []
        with self.source.open('rb') as f:
            if spans:
                f.seek(spans[0][0])
                result=[json.loads(line) for line in f.read(total).splitlines()]
        if len(result) != len(spans):
            raise ValueError('page row count mismatch')
        self.check_identity()
        return result
    def close(self):
        self.db.close()
