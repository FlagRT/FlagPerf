"""Task-scoped hash receipts, reusable only on kernel-enforced read-only inputs."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import contextmanager

SESSION_ENV = 'FLAGPERF_VERIFICATION_SESSION'


def finish():
    os.environ.pop(SESSION_ENV, None)
    os.environ.pop('FLAGPERF_TASK_WORKSPACE', None)
    from runtime.document_reader import clear
    clear()


@contextmanager
def task(root):
    previous = {key: os.environ.get(key) for key in (SESSION_ENV, 'FLAGPERF_TASK_WORKSPACE')}
    start(root)
    try: yield
    finally:
        from runtime.document_reader import clear
        clear()
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def readonly(path):
    return bool(os.statvfs(path).f_flag & os.ST_RDONLY)


def start(root):
    """Explicit new task; inherited receipts must never start another task."""
    directory = Path(root)/'.work'/('verification-' + uuid.uuid4().hex)
    directory.mkdir(parents=True, mode=0o700)
    session = directory/'receipts.sqlite'
    with sqlite3.connect(session) as db:
        db.execute('CREATE TABLE receipts (identity TEXT PRIMARY KEY, sha256 TEXT NOT NULL)')
        db.execute('CREATE TABLE counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL)')
        db.execute('CREATE TABLE validations (kind TEXT, manifest TEXT, PRIMARY KEY(kind,manifest))')
    session.chmod(0o600)
    os.environ[SESSION_ENV] = str(session)
    os.environ['FLAGPERF_TASK_WORKSPACE'] = str(Path(root)/'.work')
    return session


def file_hash(path):
    path = Path(path)
    session = os.environ.get(SESSION_ENV)
    if session and not Path(session).is_file():
        finish(); session = None
    reusable = session is not None and readonly(path)
    # The database is a private task-owned output, never accepted from a reference.
    connection = sqlite3.connect(session, timeout=600) if reusable else None
    try:
        with path.open('rb') as stream:
            before = identity(os.fstat(stream.fileno()))
            key = json.dumps(before)
            if connection is not None:
                connection.execute('BEGIN IMMEDIATE')
                cached = connection.execute('SELECT sha256 FROM receipts WHERE identity=?', (key,)).fetchone()
                # Linux timestamps may have a coarse clock tick even though stat
                # exposes nanoseconds. Never reuse receipts near a write's tick.
                settled = time.time_ns() - max(before[3], before[4]) >= 2_000_000_000
                if cached and settled and identity(path.stat()) == before and readonly(path):
                    connection.execute("INSERT INTO counters VALUES ('hash_reuses',1) ON CONFLICT(name) DO UPDATE SET value=value+1")
                    connection.commit()
                    return cached[0]
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                digest.update(chunk)
            if identity(os.fstat(stream.fileno())) != before or identity(path.stat()) != before:
                raise ValueError('file changed during verification: ' + str(path))
            value = digest.hexdigest()
            if connection is not None and readonly(path):
                connection.execute('INSERT OR REPLACE INTO receipts VALUES (?, ?)', (key, value))
                connection.commit()
            if session:
                counter = connection or sqlite3.connect(session, timeout=600)
                try:
                    for name, amount in (('full_hashes', 1), ('hashed_bytes', before[2])):
                        counter.execute('INSERT INTO counters VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value', (name, amount))
                    counter.commit()
                finally:
                    if counter is not connection: counter.close()
            return value
    finally:
        if connection is not None:
            connection.close()


def statistics():
    session = os.environ.get(SESSION_ENV)
    if not session or not Path(session).is_file(): return {}
    with sqlite3.connect(session, timeout=600) as db:
        return dict(db.execute('SELECT name,value FROM counters'))


def semantic_validation(root, manifest, kind, *, complete=False):
    """Called only AFTER manifest integrity and every member digest passed.

    Reuse structural validation for identical sealed bytes; this never bypasses
    the physical-file checks on a subsequent load. Execution freezes source code
    and each offline task starts a fresh session.
    """
    session = os.environ.get(SESSION_ENV)
    if not session or not Path(session).is_file() or not readonly(root): return False
    if any(not readonly(Path(root)/name) for name in manifest['files']): return False
    identifier = manifest['oracle_id' if kind == 'oracle' else 'baseline_id']
    with sqlite3.connect(session, timeout=600) as db:
        db.execute('CREATE TABLE IF NOT EXISTS validations (kind TEXT, manifest TEXT, PRIMARY KEY(kind,manifest))')
        if complete:
            db.execute('INSERT OR IGNORE INTO validations VALUES (?,?)', (kind,identifier))
            return False
        found = db.execute('SELECT 1 FROM validations WHERE kind=? AND manifest=?', (kind,identifier)).fetchone() is not None
        if found:
            db.execute("INSERT INTO counters VALUES ('semantic_validation_reuses',1) ON CONFLICT(name) DO UPDATE SET value=value+1")
        return found
