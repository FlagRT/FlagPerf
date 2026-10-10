"""Portable, content-addressed record storage. No accelerator dependencies.

The storage version is independent of measurement schemas. JSONL preserves record
order and Python's JSON number representation; missing fields are never filled.
"""
from collections.abc import Sequence
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

STORAGE_VERSION = 1
SHARD_BYTES = 32 * 1024 * 1024
CACHE_BYTES = 256 * 1024 * 1024
CONTEXT_KEYS = frozenset(('module', 'input_shape', 'sample_ids', 'ids', 'phase', 'rank', 'repeat'))


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'),
                      default=lambda item: list(item) if isinstance(item, Records) else _unsupported(item)).encode('utf-8')


def _unsupported(value):
    raise TypeError('not JSON serializable: ' + type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            stream.write(encoded(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def storage_root(document):
    document = Path(document).resolve()
    for parent in document.parents:
        if parent.name in ('core', 'package', 'package.incomplete'):
            return parent
        if (parent/'.artifact-root').is_file() or (parent/'effective.yaml').is_file() or (parent/'source-snapshot.json').is_file() or (parent/'manifest.json').is_file():
            return parent
    for parent in document.parents:
        if (parent/'analysis-job.json').is_file(): return parent
    return document.parent


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.directory = self.root/'artifacts'
        self.catalog = self.directory/'catalog.json'

    @contextmanager
    def locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory/'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def register(self, artifact_id, entry):
        with self.locked():
            catalog = json.loads(self.catalog.read_text()) if self.catalog.exists() else {'storage_version': STORAGE_VERSION, 'objects': {}}
            previous = catalog['objects'].get(artifact_id)
            if previous is not None and previous != entry:
                raise ValueError('artifact identity conflict')
            catalog['objects'][artifact_id] = entry
            atomic_json(self.catalog, catalog)

    def records(self, rows, *, shard_bytes=SHARD_BYTES):
        if shard_bytes <= 0:
            raise ValueError('shard size must be positive')
        self.directory.mkdir(parents=True, exist_ok=True)
        shards, contexts, context_ids, pending, size, count = [], [], {}, [], 0, 0
        context_size = 0

        def flush():
            nonlocal pending, size, contexts, context_ids, context_size
            if not pending:
                return
            header = encoded({'contexts': contexts}) + b'\n'
            fd, temporary = tempfile.mkstemp(prefix='.shard-', dir=self.directory)
            try:
                with os.fdopen(fd, 'wb') as raw:
                    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=1) as compressed:
                        compressed.write(header)
                        for line in pending:
                            compressed.write(line)
                path = Path(temporary)
                h = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(chunk)
                sha = h.hexdigest()
                destination = self.directory/(sha + '.jsonl.gz')
                path.replace(destination)
                shards.append({'path': destination.name, 'sha256': sha, 'bytes': destination.stat().st_size,
                               'records': len(pending), 'uncompressed_bytes': len(header) + size})
            finally:
                Path(temporary).unlink(missing_ok=True)
            pending, contexts, context_ids, size = [], [], {}, 0
            context_size = 0

        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('record artifacts require mapping rows')
            context = {key: value for key, value in row.items() if key in CONTEXT_KEYS}
            key = encoded(context)
            # Flush before assigning the context ID in the new shard.
            body = {key: value for key, value in row.items() if key not in CONTEXT_KEYS}
            body_bytes = encoded(body)
            estimate = len(body_bytes) + len(key) + 64
            if pending and size + context_size + estimate > shard_bytes:
                flush()
            if key not in context_ids:
                context_ids[key] = len(contexts)
                contexts.append(context)
                context_size += len(key) + 1
            line = b'[' + str(context_ids[key]).encode() + b',' + body_bytes + b']\n'
            pending.append(line)
            size += len(line)
            count += 1
        flush()
        entry = {'type': 'records', 'records': count, 'shards': shards,
                 'bytes': sum(s['bytes'] for s in shards), 'dependencies': []}
        artifact_id = hashlib.sha256(encoded(entry)).hexdigest()
        self.register(artifact_id, entry)
        return Records(self, artifact_id, count)

    def entry(self, artifact_id):
        catalog = json.loads(self.catalog.read_text())
        if catalog.get('storage_version') != STORAGE_VERSION:
            raise ValueError('unsupported artifact storage version')
        entry = catalog['objects'][artifact_id]
        if hashlib.sha256(encoded(entry)).hexdigest() != artifact_id:
            raise ValueError('artifact catalog digest mismatch')
        return entry


class Records(Sequence):
    """Replayable lazy records. Iteration holds only one decoded record at a time."""
    def __init__(self, store, artifact_id, count):
        self.store, self.artifact_id, self.count = store, artifact_id, count
        if type(count) is not int or count < 0:
            raise ValueError('invalid artifact record count')

    def __len__(self):
        return self.count

    def __iter__(self):
        entry = self.store.entry(self.artifact_id)
        if entry.get('type') != 'records' or entry['records'] != self.count:
            raise ValueError('artifact type/count mismatch')
        total = 0
        for shard in entry['shards']:
            name = shard['path']
            if Path(name).name != name:
                raise ValueError('unsafe artifact path')
            path = self.store.directory/name
            if path.is_symlink():
                raise ValueError('artifact must not be a symlink')
            from runtime.verification import file_hash, identity
            before = path.stat()
            if before.st_size != shard['bytes'] or file_hash(path) != shard['sha256']:
                raise ValueError('artifact shard digest mismatch: ' + name)
            with path.open('rb') as raw:
                if identity(os.fstat(raw.fileno())) != identity(before):
                    raise ValueError('artifact shard digest mismatch: ' + name)
                raw.seek(0)
                count = 0
                with gzip.GzipFile(fileobj=raw, mode='rb') as stream:
                    contexts = json.loads(stream.readline())['contexts']
                    for line in stream:
                        index, body = json.loads(line)
                        if type(index) is not int or not 0 <= index < len(contexts) or not isinstance(body, dict):
                            raise ValueError('invalid record encoding')
                        if contexts[index].keys() & body.keys():
                            raise ValueError('record context collision')
                        yield dict(contexts[index], **body)
                        count += 1
                after = os.fstat(raw.fileno())
                if identity(before) != identity(after) or identity(path.stat()) != identity(before):
                    raise ValueError('artifact changed while reading')
                if count != shard['records']:
                    raise ValueError('artifact shard record count mismatch')
                total += count
        if total != self.count:
            raise ValueError('artifact record count mismatch')

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step < 0:
                return list(self)[index]
            return [row for i, row in enumerate(self) if start <= i < stop and (i-start) % step == 0]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return next(row for i, row in enumerate(self) if i == index)

    def __eq__(self, other):
        if not isinstance(other, (Sequence, Records)) or len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self, other))

    def __deepcopy__(self, memo):
        return self


class RecordWriter:
    """Incremental capture sink; only the current shard remains in memory."""
    def __init__(self, store, target_bytes=SHARD_BYTES):
        self.store, self.target_bytes = store, target_bytes
        self.pending, self.shards, self.bytes, self.count = [], [], 0, 0

    def append(self, row):
        size = len(encoded(row)) + 1
        if self.pending and self.bytes + size > self.target_bytes:
            self.flush()
        self.pending.append(row); self.bytes += size; self.count += 1

    def extend(self, rows):
        for row in rows: self.append(row)

    def flush(self):
        if self.pending:
            records = self.store.records(self.pending, shard_bytes=self.target_bytes)
            self.shards.extend(self.store.entry(records.artifact_id)['shards'])
            self.pending, self.bytes = [], 0

    def finish(self):
        self.flush()
        entry = {'type': 'records', 'records': self.count, 'shards': self.shards,
                 'bytes': sum(s['bytes'] for s in self.shards), 'dependencies': []}
        artifact_id = hashlib.sha256(encoded(entry)).hexdigest()
        self.store.register(artifact_id, entry)
        return Records(self.store, artifact_id, self.count)


def pack(value, document, *, threshold=1024, store=None):
    """Replace large record arrays by portable references, preserving old readers via unpack."""
    store = store or Store(storage_root(document))
    if isinstance(value, Records):
        if value.store.root != store.root:
            value = store.records(value)
        return {'$records': {'artifact_id': value.artifact_id, 'type': 'records', 'records': len(value),
                             'catalog': os.path.relpath(store.catalog, Path(document).parent)}}
    if isinstance(value, list):
        if len(value) >= threshold and all(isinstance(row, dict) for row in value):
            return pack(store.records(value), document, threshold=threshold, store=store)
        return [pack(item, document, threshold=threshold, store=store) for item in value]
    if isinstance(value, dict):
        return {key: pack(item, document, threshold=threshold, store=store) for key, item in value.items()}
    return value


def unpack(value, document):
    if isinstance(value, dict):
        if set(value) == {'$records'}:
            ref = value['$records']
            catalog = (Path(document).parent/ref['catalog']).resolve()
            root = catalog.parent.parent
            if catalog.name != 'catalog.json' or catalog.parent.name != 'artifacts' or not Path(document).resolve().is_relative_to(root):
                raise ValueError('unsafe artifact catalog reference')
            if ref.get('type') != 'records':
                raise ValueError('unsupported artifact reference type')
            return Records(Store(root), ref['artifact_id'], ref['records'])
        return {key: unpack(item, document) for key, item in value.items()}
    if isinstance(value, list):
        return [unpack(item, document) for item in value]
    return value
