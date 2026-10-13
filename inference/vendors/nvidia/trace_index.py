"""Task-local CUDA indexes: fast memory admission, bounded SQLite fallback."""
from collections import OrderedDict, defaultdict
import gzip
import json
from pathlib import Path
import sqlite3
import tempfile

from runtime.stream_json import trace_events


class Events:
    def __init__(self, path, workdir, progress=None):
        workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='cuda-index-', dir=workdir)
        self.db = sqlite3.connect(str(Path(self.temporary.name)/'events.sqlite'))
        self.db.execute('PRAGMA journal_mode=OFF')
        self.db.execute('PRAGMA synchronous=OFF')
        self.db.execute('PRAGMA cache_size=-16384')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('CREATE TABLE events (id INTEGER PRIMARY KEY, cat TEXT, ph TEXT, name TEXT, correlation TEXT, external TEXT, body TEXT)')
        self.count = 0
        self.folder, self.metadata = workdir.parent, {}
        self.cache, self.cache_bytes = OrderedDict(), 0
        opener = gzip.open if Path(path).suffix == '.gz' else open
        try:
            with opener(path, 'rt', encoding='utf-8') as stream:
                for index, event in enumerate(trace_events(stream, self.metadata)):
                    args = event.get('args', {})
                    corr, external = args.get('correlation'), args.get('External id')
                    cat, phase, name = event.get('cat'), event.get('ph'), event.get('name', '')
                    # Preserve original event IDs. Unused raw events remain in the
                    # immutable trace, but need no duplicate SQLite body.
                    needed = phase == 'X' and (
                        cat in ('kernel', 'gpu_memcpy', 'gpu_memset') or
                        (cat in ('cuda_runtime', 'cuda_driver') and corr is not None) or
                        (cat == 'cpu_op' and external is not None) or
                        (cat != 'gpu_user_annotation' and name.startswith(('flagperf/layer/', 'flagperf/collective/'))))
                    if needed:
                        self.db.execute('INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)',
                            (index, cat, phase, name,
                             str(corr) if corr is not None else None,
                             str(external) if external is not None else None,
                             json.dumps(event, ensure_ascii=False, separators=(',', ':'))))
                    self.count += 1
                    if index % 65536 == 0:
                        if progress:
                            raw_position = stream.buffer.fileobj.tell() if Path(path).suffix == '.gz' else stream.buffer.tell()
                            progress('loading', records=self.count, bytes_read=raw_position, total_bytes=Path(path).stat().st_size)
                        self.db.commit()
            if progress: progress('indexing', records=self.count, bytes_read=Path(path).stat().st_size)
            self.db.execute('CREATE INDEX correlation ON events (correlation, cat)')
            self.db.execute('CREATE INDEX external ON events (external, cat)')
            self.db.commit()
        except BaseException:
            self.close()
            raise

    def __len__(self): return self.count

    def __iter__(self):
        for body, in self.db.execute('SELECT body FROM events ORDER BY id'):
            yield json.loads(body)

    def device_events(self):
        for index, body in self.db.execute("SELECT id, body FROM events WHERE ph='X' AND cat IN ('kernel','gpu_memcpy','gpu_memset') ORDER BY id"):
            yield index, json.loads(body)

    def markers(self, prefix):
        for body, in self.db.execute("SELECT body FROM events WHERE ph='X' AND (cat IS NULL OR cat!='gpu_user_annotation') AND name GLOB ? ORDER BY id", (prefix+'*',)):
            yield json.loads(body)

    def kernel_sink(self):
        return Kernels(self)

    def hosts(self, field, value):
        if field not in ('correlation', 'external'): raise ValueError('invalid host index')
        key = (field, str(value))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        cats = "('cuda_runtime','cuda_driver')" if field == 'correlation' else "('cpu_op')"
        raw = self.db.execute("SELECT body FROM events WHERE ph='X' AND "+field+'=? AND cat IN '+cats+' ORDER BY id LIMIT 2', (str(value),)).fetchall()
        values = [json.loads(row[0]) for row in raw]
        size = sum(len(row[0].encode('utf-8'))*4 for row in raw) + 256
        while self.cache and self.cache_bytes + size > 32*1024*1024:
            _, (_, removed) = self.cache.popitem(last=False); self.cache_bytes -= removed
        if size <= 32*1024*1024:
            self.cache[key] = (values, size); self.cache_bytes += size
        return values

    def close(self):
        self.db.close()
        self.temporary.cleanup()

    def __del__(self):
        try: self.close()
        except Exception: pass


class Kernels:
    def __init__(self, trace):
        self.trace, self.db, self.count = trace, trace.db, 0
        self.db.execute('CREATE TABLE kernels (id INTEGER PRIMARY KEY, body TEXT, kind TEXT, start INTEGER, end INTEGER)')
        self.db.execute('CREATE TABLE owners (sequence INTEGER, id INTEGER)')

    def append(self, row):
        self.db.execute('INSERT INTO kernels VALUES (?, ?, ?, ?, ?)', (row['id'], json.dumps(row, separators=(',', ':')), row['kind'], row['start_ns'], row['start_ns']+row['duration_ns']))
        self.count += 1

    def __len__(self): return self.count

    def __iter__(self):
        for body, in self.db.execute('SELECT body FROM kernels ORDER BY id'):
            yield json.loads(body)

    def update(self, row):
        self.db.execute('UPDATE kernels SET body=? WHERE id=?', (json.dumps(row, separators=(',', ':')), row['id']))
        self.db.executemany('INSERT INTO owners VALUES (?, ?)', ((owner, row['id']) for owner in set(row['owners'])))

    def finish_owners(self):
        self.db.execute('CREATE INDEX by_owner ON owners (sequence, id)')
        self.db.commit()

    def owned(self, sequence):
        for body, in self.db.execute('SELECT kernels.body FROM owners JOIN kernels USING(id) WHERE sequence=? ORDER BY id', (sequence,)):
            yield json.loads(body)

    def intervals(self, kind):
        yield from self.db.execute('SELECT start,end FROM kernels WHERE kind=? ORDER BY start,end', (kind,))

    def publish(self):
        from runtime.artifacts import Store, storage_root
        return Store(storage_root(self.trace.folder/'attribution.json')).records(self)


def memory_budget():
    """Conservative admission budget, respecting host and cgroup headroom."""
    available = 0
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'): available = int(line.split()[1])*1024
        for limit_path, used_path in (
            ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
            ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')):
            if Path(limit_path).is_file():
                limit = Path(limit_path).read_text().strip()
                if limit != 'max': available = min(available, max(0, int(limit)-int(Path(used_path).read_text())))
    except (OSError, ValueError):
        return 0
    return min(32*1024**3, available//4)


def open_events(path, workdir, progress=None, *, budget=None):
    budget = memory_budget() if budget is None else budget
    # Compressed input has no reliable bounded uncompressed size (ISIZE wraps).
    estimate = max(512*1024**2, Path(path).stat().st_size*6)
    backend = 'memory' if Path(path).suffix != '.gz' and estimate <= budget else 'sqlite'
    if progress: progress('index_admission', backend=backend, estimated_bytes=estimate, budget_bytes=budget)
    return (MemoryEvents if backend == 'memory' else Events)(path, workdir, progress)


class MemoryEvents:
    """One bulk decode, selected events and direct correlation maps.

    Keep the first two hosts per ID: unique versus ambiguous is the existing
    matching rule. Device IDs and event order remain those of the raw trace.
    """
    def __init__(self, path, workdir, progress=None):
        self.folder = Path(workdir).parent
        self.metadata, self.count = {}, 0
        self.selected, self.devices = [], []
        self.by_correlation, self.by_external = {}, {}
        self.layer_markers, self.collective_markers = [], []
        # A single C-decoder call for the event array shares field-name strings
        # across events. Per-event raw_decode loses that sharing and adds Python
        # dispatch/GC cost. Admission reserves memory before this bulk decode.
        from runtime.stream_json import Reader
        import gc
        enabled = gc.isenabled()
        gc.disable()
        try:
            with open(path, encoding='utf-8') as stream:
                reader = Reader(stream)
                reader.buffer, reader.eof = stream.read(), True
                reader.take('{'); events = None
                if reader.peek() != '}':
                    while True:
                        key = reader.value()
                        if not isinstance(key, str): raise ValueError('invalid JSON object key')
                        reader.take(':'); value = reader.value()
                        if key == 'traceEvents':
                            if events is not None: raise ValueError('duplicate traceEvents')
                            if not isinstance(value, list): raise ValueError('invalid CUDA trace events')
                            events = value
                        else: self.metadata[key] = value
                        if reader.peek() == '}': break
                        reader.take(',')
                reader.take('}')
                if reader.peek() or events is None: raise ValueError('trailing JSON content or missing traceEvents')
                del reader
            for index, event in enumerate(events):
                if not isinstance(event, dict): raise ValueError('invalid CUDA trace event')
                self.count += 1
                cat, phase, name = event.get('cat'), event.get('ph'), event.get('name', '')
                if phase == 'X':
                    args = event.get('args', {})
                    corr, external = args.get('correlation'), args.get('External id')
                    needed = False
                    if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
                        self.devices.append((index, event)); needed = True
                    if cat in ('cuda_runtime', 'cuda_driver') and corr is not None:
                        bucket = self.by_correlation.setdefault(str(corr), [])
                        if len(bucket) < 2: bucket.append(event)
                        needed = True
                    if cat == 'cpu_op' and external is not None:
                        bucket = self.by_external.setdefault(str(external), [])
                        if len(bucket) < 2: bucket.append(event)
                        needed = True
                    if cat != 'gpu_user_annotation':
                        if name.startswith('flagperf/layer/'):
                            self.layer_markers.append(event); needed = True
                        if name.startswith('flagperf/collective/'):
                            self.collective_markers.append(event); needed = True
                    if needed: self.selected.append(event)
                if progress and index % 65536 == 0:
                    progress('loading', backend='memory', records=self.count,
                             bytes_read=Path(path).stat().st_size, total_bytes=Path(path).stat().st_size)
        finally:
            if enabled: gc.enable()

    def __len__(self): return self.count
    def __iter__(self): return iter(self.selected)
    def device_events(self): return iter(self.devices)
    def markers(self, prefix):
        if prefix == 'flagperf/layer/': return iter(self.layer_markers)
        if prefix == 'flagperf/collective/': return iter(self.collective_markers)
        return (e for e in self.selected if e.get('cat') != 'gpu_user_annotation' and e.get('name','').startswith(prefix))
    def hosts(self, field, value):
        if field not in ('correlation','external'): raise ValueError('invalid host index')
        return (self.by_correlation if field == 'correlation' else self.by_external).get(str(value), [])
    def kernel_sink(self): return MemoryKernels(self)


class MemoryKernels:
    def __init__(self, trace):
        self.trace, self.rows, self.owners = trace, [], defaultdict(list)
    def append(self, row): self.rows.append(row)
    def __len__(self): return len(self.rows)
    def __iter__(self): return iter(self.rows)
    def update(self, row):
        for owner in set(row['owners']): self.owners[owner].append(row)
    def finish_owners(self): pass
    def owned(self, sequence): return iter(self.owners.get(sequence, ()))
    def intervals(self, kind):
        return iter(sorted((r['start_ns'],r['start_ns']+r['duration_ns']) for r in self.rows if r['kind']==kind))
    def publish(self):
        from runtime.artifacts import Store, storage_root
        return Store(storage_root(self.trace.folder/'attribution.json')).records(self.rows)
