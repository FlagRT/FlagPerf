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
    from runtime.artifacts import pack
    stored = pack(value, path)
    with temporary.open('w') as stream:
        json.dump(stored, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)
    if path.name == 'stage.json' and os.environ.get('FLAGPERF_MONITOR_ROOT'):
        from runtime.monitor_events import emit
        emit(path.parent, 'stage', stage=value.get('stage', 'unknown'))


def read_json(path):
    path = Path(path).resolve()
    from runtime.artifacts import unpack
    if Path(path).stat().st_size > 32*1024*1024:
        from runtime.document_reader import read
        return unpack(read(path), path)
    with Path(path).open() as stream:
        return unpack(json.load(stream), path)


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
    from runtime.verification import file_hash as verified_hash
    return verified_hash(path)


ANALYSIS_FILES = frozenset({
    'runtime/layer_index.py', 'runtime/record_index.py', 'runtime/checkpoints.py', 'runtime/document_reader.py',
    'runtime/stream_json.py', 'runtime/reference_storage.py', 'runtime/offline.py',
    'runtime/monitor_analysis.py', 'runtime/monitor_report.py',
    'runtime/metric_quality.py', 'runtime/measurement_quality.py',
    'runtime/retention.py',
    'runtime/metrics.py', 'runtime/assessment.py', 'runtime/communication_analysis.py',
    'runtime/reporting.py', 'runtime/tp_reporting.py', 'runtime/stack_reporting.py',
    'runtime/layer_trace.py', 'runtime/layer_reporting.py', 'runtime/report_replay.py',
    'runtime/layer_analysis.py',
    'runtime/generation_analysis.py', 'runtime/generation_reporting.py',
    'runtime/cuda_trace.py', 'runtime/performance_comparison.py',
    'runtime/layer_experience.py',
    'runtime/oracle_analysis.py', 'runtime/oracle_reporting.py',
    'runtime/trace_utils.py', 'runtime/trace_dispatch.py',
    'runtime/cuda_analysis_job.py',
    'vendors/ascend/trace.py', 'vendors/ascend/communication.py', 'vendors/nvidia/trace.py',
})


def source_snapshot(cfg=None, scope=None):
    paths = list(ROOT.glob('*.py'))
    for folder in ['runtime', 'vendors', 'inference_engine', 'benchmarks', 'components']:
        paths.extend((ROOT / folder).rglob('*.py'))
    paths.extend((ROOT / 'registry').rglob('*.json'))
    paths.extend((ROOT / 'vendors').rglob('*.json'))
    paths.extend((ROOT / 'benchmarks').rglob('*.json'))
    declared_analysis, declared_execution = set(), set()
    declared_components = set()
    for path in (ROOT / 'registry').rglob('*.json'):
        entry = read_json(path)
        declared_analysis.update(entry.get('analysis_sources', []))
        declared_execution.update(entry.get('sources', []))
        for sources in entry.get('component_sources', {}).values():
            declared_components.update(sources)
    # Descriptors may seal non-code assets such as licenses and provenance.
    # Apply the same repository-local path checks as adapter discovery.
    from runtime.adapters import local_file
    paths.extend(local_file(name) for name in declared_analysis | declared_execution | declared_components)
    files = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(set(paths))}
    if declared_analysis & declared_execution:
        raise ValueError('sources declared as both execution and analysis: '+', '.join(sorted(declared_analysis & declared_execution)))
    analysis_only = ANALYSIS_FILES | (declared_analysis - declared_execution)
    execution = {k:v for k,v in files.items() if k not in analysis_only}
    analysis = {k:v for k,v in files.items() if k in ANALYSIS_FILES or k in declared_analysis or k.startswith('registry/vendors/')}
    audit = dict(execution)
    if cfg is not None:
        from runtime.adapters import source_scope
        scope = cfg.get('_source_scope') or source_scope(cfg)
    if scope is not None:
        missing = set(scope) - set(files)
        if missing:
            raise ValueError('source scope contains missing files: ' + ', '.join(sorted(missing)))
        execution = {k: v for k, v in execution.items() if k in scope}
    result = {'execution': execution, 'analysis': analysis,
            'execution_source_key': digest(execution), 'analysis_key': digest(analysis),
            'snapshot_key': digest(files)}
    if scope is not None:
        from runtime.adapters import SCHEME
        result.update(identity_scheme=SCHEME, scope=sorted(scope), audit_execution=audit)
    return result


def source_identity(cfg=None):
    return source_snapshot(cfg)['execution']


def assert_source_snapshot(expected):
    if expected is None:
        return  # Explicit offline callers may omit it; production always seals a snapshot.
    actual = source_snapshot(scope=expected.get('scope'))
    if actual != expected:
        fields = identity_differences(expected, actual)
        raise ValueError('source changed during run: ' + ', '.join(fields))


def seal_sources(root, cfg=None):
    path = Path(root)/'source-snapshot.json'
    snapshot = read_json(path) if path.exists() else source_snapshot(cfg)
    assert_source_snapshot(snapshot)
    write_json(path, snapshot)
    # Preserve the exact readable sources, not only their fingerprints.
    for name, expected in {**snapshot.get('audit_execution', snapshot['execution']), **snapshot['analysis']}.items():
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
