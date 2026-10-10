"""Bounded CUDA post-processing; attempts and raw captures remain inspectable."""
import json
import gc
import math
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid

from runtime.common import ROOT, file_hash, read_json
from runtime.artifacts import pack, unpack, atomic_json

CONTROL_BYTES = 1024 * 1024


class PublishedResult(dict):
    """Current-job result whose canonical JSON was already atomically written."""
    def __init__(self, value, path):
        super().__init__(value)
        self.published_at = Path(path).resolve()


def _write(path, value):
    """Stream large event maps and publish only a fully written JSON document."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(pack(value, path, threshold=1 if path.name == 'analysis-result.json' else 1024),
                  stream, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def _publish(source, destination):
    # Rebase only references; the control document contains no event arrays.
    atomic_json(destination, pack(read_json(source), destination, threshold=1))


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


class _Progress:
    def __init__(self, attempt):
        self.path = attempt / 'progress.json'
        self.started = self.changed = time.monotonic()
        self.last_write = 0.0
        self.stage = None
        self.timings = {}
        self.peak_temporary_bytes = 0

    def __call__(self, stage, **counts):
        now = time.monotonic()
        if stage != self.stage:
            if self.stage is not None:
                self.timings[self.stage] = self.timings.get(self.stage, 0) + now - self.changed
            self.stage, self.changed = stage, now
            force = True
        else:
            force = False
        if force or now - self.last_write >= 0.5:
            self.last_write = now
            counters = {}
            try:
                counters = {key: int(value) for key, value in
                            (line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())}
            except OSError:
                pass
            folder = self.path.parent.parent.parent
            self.peak_temporary_bytes = max(self.peak_temporary_bytes,
                sum(path.stat().st_size for path in (folder/'.work').glob('cuda-index-*/*.sqlite') if path.is_file()))
            _write(self.path, dict(stage=stage, elapsed_seconds=now-self.started,
                peak_rss_bytes=_rss_bytes(), phase_seconds=dict(self.timings),
                process_io=counters, peak_temporary_index_bytes=self.peak_temporary_bytes, **counts))


def _worker(request_path):
    request_path = Path(request_path)
    attempt = request_path.parent
    progress = _Progress(attempt)
    # This isolated worker processes acyclic JSON data. Reference counting frees
    # it normally; cyclic scans of millions of live trace objects add substantial
    # latency. Restore the caller's GC setting even on failure.
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        progress('request_loading')
        request = read_json(request_path)
        folder = Path(request['folder'])
        state = request['state']
        if state is None:
            state = read_json(folder/'result.json')
        from vendors.nvidia.trace import analyze_bundle
        result = analyze_bundle(folder, state, rank=request['rank'],
            communication_only=request['communication_only'], progress=progress,
            associate_layers=request['rank'] is not None and not request['communication_only'])
        if not isinstance(result, dict) or result.get('status') not in ('completed', 'partial'):
            raise ValueError('CUDA analysis returned an invalid result')
        progress('writing')
        # result.json is reserved for capture workers throughout the repository.
        result_path = attempt/'analysis-result.json'
        _write(result_path, result)
        if result_path.stat().st_size > CONTROL_BYTES:
            raise ValueError('CUDA analysis control message exceeds 1 MiB')
        # Validate all produced shards before publication, inside the deadline.
        progress('validating')
        from runtime.artifacts import Records
        def validate(value):
            if isinstance(value, Records):
                # Records decode plain JSON mappings; nested artifact objects
                # occur only in the control document. Iteration itself verifies
                # every digest, encoding, count and before/after file identity.
                for _ in value: pass
            elif isinstance(value, dict):
                for child in value.values(): validate(child)
            elif isinstance(value, list):
                for child in value: validate(child)
        compact = read_json(result_path)
        validate(compact)
        if not request['communication_only']:
            progress('publishing')
            _publish(result_path, folder/('.attribution-'+attempt.name+'.incomplete'))
        progress('completed')
        return 0
    except Exception as error:
        traceback.print_exc()
        try:
            progress('failed')
            _write(attempt/'error.json', {'error': str(error), 'exception_type': type(error).__name__})
        except Exception:
            pass
        return 1
    finally:
        if gc_enabled:
            gc.enable()


def _stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader may exit before a blocked descendant. Kill the group even
        # when wait() has already reaped its leader.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _command(request):
    return [sys.executable, '-B', '-u', '-m', 'runtime.cuda_analysis_job', '--worker', str(request)]


def _heartbeat(attempt, process, started):
    progress = read_json(attempt/'progress.json') if (attempt/'progress.json').exists() else {'stage': 'starting'}
    progress['elapsed_seconds'] = time.monotonic() - started
    # JSON loading and writing cannot emit callbacks. Observe the worker during
    # those phases as well; VmHWM is a peak, unlike the current resident size.
    try:
        for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
            if line.startswith('VmHWM:'):
                progress['peak_rss_bytes'] = max(progress.get('peak_rss_bytes', 0), int(line.split()[1])*1024)
                break
    except OSError:
        pass
    _write(attempt/'heartbeat.json', progress)
    return progress


def _partial(reason, rank, communication_only):
    communication = {'status': 'partial', 'rank': rank, 'issues': [reason], 'events': []}
    if communication_only:
        return communication
    result = {'status': 'partial', 'reason': reason, 'rows': [], 'events': [], 'coverage': {}}
    if rank is not None:
        result['communication'] = communication
    return result


def run(folder, state=None, *, rank=None, communication_only=False, timeout=900):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('CUDA analysis timeout must be positive and finite')
    folder = Path(folder).resolve()
    attempt = folder / 'analysis-attempts' / ('cuda-' + uuid.uuid4().hex)
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    request = attempt/'request.json'
    metadata = {'schema_version': 1, 'status': 'running', 'timeout_seconds': timeout,
                'attempt': str(attempt.relative_to(folder)), 'rank': rank,
                'communication_only': communication_only, 'new_device_execution': False,
                'python_version': sys.version,
                'analysis_sources': {name: file_hash(ROOT/name) for name in (
                    'runtime/common.py', 'runtime/cuda_analysis_job.py', 'runtime/layer_trace.py',
                    'vendors/nvidia/trace.py', 'vendors/nvidia/scope_index.py', 'vendors/nvidia/trace_index.py',
                    'runtime/artifacts.py', 'runtime/stream_json.py', 'runtime/verification.py',
                    'vendors/ascend/trace.py', 'runtime/trace_utils.py')}}
    process = None
    try:
        _write(attempt/'status.json', metadata)
        _write(folder/'analysis-job.json', metadata)
        _write(request, {'folder': str(folder), 'state': state, 'rank': rank,
                         'communication_only': communication_only})
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        with (attempt/'worker.log').open('w') as log:
            process = subprocess.Popen(_command(request), cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            metadata['pid'] = process.pid
            _write(attempt/'status.json', metadata)
            _write(folder/'analysis-job.json', metadata)
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f'CUDA analysis timed out after {timeout:g} seconds')
                try:
                    process.wait(timeout=min(5, remaining))
                    break
                except subprocess.TimeoutExpired:
                    progress = _heartbeat(attempt, process, started)
                    print('[analysis] ' + json.dumps(progress, ensure_ascii=False), file=sys.stderr, flush=True)
        if process.returncode:
            error = read_json(attempt/'error.json') if (attempt/'error.json').exists() else {}
            raise RuntimeError(error.get('error', f'CUDA analysis worker exited {process.returncode}'))
        result_path = attempt/'analysis-result.json'
        # A malicious/crashed child cannot force an unbounded parent JSON read.
        descriptor = os.open(result_path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as control:
            if not stat.S_ISREG(os.fstat(control.fileno()).st_mode):
                raise ValueError('CUDA analysis control must be a regular file')
            payload = control.read(CONTROL_BYTES + 1)
        if len(payload) > CONTROL_BYTES:
            raise ValueError('CUDA analysis control message exceeds 1 MiB')
        result = unpack(json.loads(payload), result_path)
        if time.monotonic() - started > timeout:
            raise TimeoutError(f'CUDA analysis timed out after {timeout:g} seconds')
        if not isinstance(result, dict) or result.get('status') not in ('completed', 'partial'):
            raise ValueError('CUDA analysis worker returned an invalid result')
        status = result['status']
        if result.get('communication', {}).get('status', 'completed') != 'completed':
            status = 'partial'
        metadata.update(status=status, exit_code=process.returncode)
        if not communication_only:
            staged = folder/('.attribution-'+attempt.name+'.incomplete')
            staged.replace(folder/'attribution.json')
        return result if communication_only else PublishedResult(result, folder/'attribution.json')
    except Exception as error:
        metadata.update(status='partial', reason=str(error), exception_type=type(error).__name__)
        return _partial(str(error), rank, communication_only)
    except BaseException:
        metadata.update(status='partial', reason='analysis coordinator interrupted')
        raise
    finally:
        if process is not None:
            _stop(process)
            metadata['exit_code'] = process.returncode
        metadata['elapsed_seconds'] = time.monotonic() - started
        if (attempt/'progress.json').exists():
            metadata['progress'] = read_json(attempt/'progress.json')
        _write(attempt/'status.json', metadata)
        _write(folder/'analysis-job.json', metadata)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('private CUDA analysis worker requires a request file')
    raise SystemExit(_worker(sys.argv[2]))
