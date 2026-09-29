# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Small evidence and lifecycle utilities, independent of device packages."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def identity_differences(expected, actual, prefix=''):
    """Exact differing field paths; no implicit identity relaxation."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            name = prefix + '.' + str(key) if prefix else str(key)
            if key not in expected or key not in actual:
                differences.append(name)
            else:
                differences.extend(identity_differences(expected[key], actual[key], name))
        return differences
    return [] if expected == actual else [prefix]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


ANALYSIS_FILES = frozenset({
    'analysis/metrics.py', 'analysis/assessment.py', 'analysis/communication.py',
    'reporting/model.py', 'reporting/parallel.py', 'reporting/components.py',
    'analysis/__init__.py', 'reporting/__init__.py',
})


def source_snapshot():
    paths = list(ROOT.glob('*.py'))
    for folder in ['runtime', 'vendors', 'engines', 'models', 'analysis', 'reporting']:
        paths.extend((ROOT / folder).rglob('*.py'))
    files = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(set(paths))}
    execution = {k:v for k,v in files.items() if k not in ANALYSIS_FILES}
    analysis = {k:v for k,v in files.items() if k in ANALYSIS_FILES}
    return {'execution': execution, 'analysis': analysis,
            'execution_source_key': digest(execution), 'analysis_key': digest(analysis),
            'snapshot_key': digest(files)}


def source_identity():
    return source_snapshot()['execution']


def assert_source_snapshot(expected):
    if expected is None:
        return  # Private unit-test callers; production always seals a snapshot.
    actual = source_snapshot()
    if actual != expected:
        fields = identity_differences(expected, actual)
        raise ValueError('source changed during run: ' + ', '.join(fields))


def seal_sources(root):
    path = Path(root)/'source-snapshot.json'
    snapshot = read_json(path) if path.exists() else source_snapshot()
    assert_source_snapshot(snapshot)
    write_json(path, snapshot)
    # Preserve the exact readable sources, not only their fingerprints.
    for name, expected in {**snapshot['execution'], **snapshot['analysis']}.items():
        target = Path(root)/'source-code'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            payload = (ROOT/name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError('source changed while archiving: '+name)
            target.write_bytes(payload)
        if file_hash(target) != expected:
            raise ValueError('sealed source archive mismatch: '+name)
    assert_source_snapshot(snapshot)
    return snapshot


class Progress:
    def __init__(self):
        self.phase = 'starting'
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        def heartbeat():
            while not self.stop.wait(5):
                print(f'[progress] {self.phase}', file=sys.stderr, flush=True)
        self.thread = threading.Thread(target=heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def execute(argv, output, timeout, env=None):
    """Bound a fresh process group; never retry a failed device context in place."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'command.json', argv)
    started = time.monotonic()
    with (output / 'run.log').open('w') as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        timed_out = False
        cleanup_seconds = 0.0
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            cleanup_started = time.monotonic()
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            cleanup_seconds = time.monotonic()-cleanup_started
            code = 124
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    state = {'exit_code': code, 'timed_out': timed_out, 'cleanup_seconds':cleanup_seconds,
             'wall_seconds_diagnostic_only': time.monotonic() - started}
    write_json(output / 'exit.json', state)
    return state
