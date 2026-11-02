"""Observe configured log bytes after the session baseline, source by source."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import re


def validate_logs(logs, patterns):
    return [pattern for pattern in patterns if re.search(pattern, logs)]


class LogObserver:
    def __init__(self, paths, patterns, reader=None):
        self.patterns = patterns
        self.reader = reader
        self.sources = [dict(path=str(p), offset=0, size=0, prefix=b'', pending=b'', failed=None)
                        for p in paths]
        if reader is not None:
            self.sources.append(dict(path='<test-reader>', offset=0, size=0, prefix=b'', pending=b'', failed=None))

    def _open(self, source):
        if source['path']=='<test-reader>':
            return io.BytesIO(self.reader().encode('utf8'))
        return Path(source['path']).open('rb')

    def start(self):
        for source in self.sources:
            try:
                with self._open(source) as stream:
                    stream.seek(0, 2)
                    source['offset'] = source['size'] = stream.tell()
                    stream.seek(0)
                    source['prefix'] = stream.read(min(4096, source['size']))
            except Exception:
                pass  # The first observation records missing/unreadable sources.

    def check(self, *, final=False):
        results = []
        for source in self.sources:
            begin = source['offset']-len(source['pending'])
            row = dict(path=source['path'], begin=begin, end=begin, sha256=None)
            try:
                with self._open(source) as stream:
                    stream.seek(0, 2)
                    size = stream.tell()
                    stream.seek(0)
                    prefix = stream.read(len(source['prefix']))
                    if source['failed'] or size<source['size'] or prefix!=source['prefix']:
                        source['failed'] = 'rotated_or_rewritten'
                        raise ValueError(source['failed'])
                    stream.seek(source['offset'])
                    data = source['pending']+stream.read(size-source['offset'])
                    # Fix a prefix once the initially empty/missing source grows.
                    if not source['prefix'] and size:
                        stream.seek(0)
                        source['prefix'] = stream.read(min(4096, size))
                source['offset'] = source['size'] = size
                stop = len(data) if final else data.rfind(b'\n')+1
                segment, source['pending'] = data[:stop], data[stop:]
                hits = validate_logs(segment.decode('utf8', errors='replace'), self.patterns)
                row.update(end=begin+len(segment), sha256=hashlib.sha256(segment).hexdigest(),
                           status='error' if hits else 'clean', matched_patterns=hits)
            except FileNotFoundError:
                row.update(status='unreadable', reason='missing')
            except Exception as error:
                row.update(status='unreadable', reason=source['failed'] or type(error).__name__+': '+str(error))
            results.append(row)
        status = max((r['status'] for r in results), key={'clean':0,'unreadable':1,'error':2}.get) if results else 'no_source'
        return dict(status=status, sources=results,
                    reason='no configured source' if not results else 'new bytes observed after export; not event-time attribution')
