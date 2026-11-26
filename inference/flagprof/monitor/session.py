"""Versioned node metadata, bounded configuration and serialized mutations."""
from contextlib import contextmanager
import fcntl
import json
import os
import shutil
from pathlib import Path
import uuid
from flagprof.store.manifest import atomic_json, proc_key
from flagprof.telemetry.contract import raw_clock


DEFAULTS = dict(interval_ms=1000, source='auto', max_targets=256,
    segment_seconds=300, segment_bytes=32 << 20, total_bytes=8 << 30,
    control_reserve_bytes=256 << 20, raw_bytes=3 << 30, raw_seconds=86400,
    aggregate_bytes=1 << 30, aggregate_seconds=30*86400, min_free_bytes=2 << 30,
    pre_seconds=60, post_seconds=30, update_seconds=60, clear_observations=2,
    max_segments=8192, max_history=16384, queue_capacity=512,
    restart_window_s=600, restart_backoff_s=[1, 2, 4], max_record_bytes=65536)


def configuration(path=None):
    data = json.loads(Path(path).read_text()) if path else {}
    diagnosis = data.pop('diagnosis', {})
    if set(data)-set(DEFAULTS):
        raise ValueError('unknown watch configuration: '+str(sorted(set(data)-set(DEFAULTS))))
    result = dict(DEFAULTS, **data)
    for key, value in result.items():
        if key in ('source', 'restart_backoff_s'):
            continue
        if type(value) not in (int, float) or value < 0 or value != value or value == float('inf'):
            raise ValueError('invalid watch configuration: '+key)
    if not 100 <= result['interval_ms'] <= 5000 or not 1 <= result['max_targets'] <= 256:
        raise ValueError('watch interval/target limit')
    for key in ('max_targets','max_segments','max_history','queue_capacity','max_record_bytes'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError('positive integer required: '+key)
    if result['source'] not in ('auto','dcmi','dsmi','nvml','npu-smi','nvidia-smi'):
        raise ValueError('unsupported telemetry source')
    if result['restart_backoff_s'] != [1, 2, 4]:
        raise ValueError('restart contract is 1, 2, 4 seconds')
    if result['restart_window_s'] != 600:
        raise ValueError('restart window contract is 600 seconds')
    if result['control_reserve_bytes'] >= result['total_bytes']:
        raise ValueError('control reserve must fit inside total budget')
    if diagnosis is not None:
        from .business import DEFAULTS as DIAG_DEFAULTS
        if not isinstance(diagnosis, dict) or set(diagnosis)-set(DIAG_DEFAULTS):
            raise ValueError('invalid diagnosis configuration')
        effective = dict(DIAG_DEFAULTS, **diagnosis)
        if any(type(v) not in (int, float) or not 0 < v < float('inf') for v in effective.values()):
            raise ValueError('diagnosis thresholds must be positive and finite')
        result['diagnosis'] = effective
    return result


@contextmanager
def locked(root, *, shared=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.state.lock').open('rb' if shared else 'a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load(root):
    return json.loads((Path(root)/'session.json').read_text())


def snapshot(root):
    from flagprof.store.manifest import process_alive
    state=load(root)
    state['persisted_status']=state['status']
    if state['status'] in ('starting','running'):
        state['supervisor_alive']=process_alive(state.get('supervisor'))
        state['worker_alive']=process_alive(state.get('worker_process'))
        if state['supervisor_alive'] is False:state['status']='unavailable'
    return state


@contextmanager
def edit(root):
    with locked(root):
        state = load(root)
        try:
            yield state
        finally:
            control_json(root, state, Path(root)/'session.json', state)


def control_json(root, state, path, value):
    """Caller holds the state lock. Control writes also reserve atomic-copy space."""
    amount=2*len(json.dumps(value,allow_nan=False).encode())+4096
    config=state['config']
    if (owned_size(root)+state['reserved_bytes']+amount>config['total_bytes'] or
            shutil.disk_usage(root).free<config['min_free_bytes']+amount):
        from .retention import CapacityError
        raise CapacityError('control_capacity')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    atomic_json(Path(path),value)


def initialize(root, vendor, devices, config):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with locked(root):
        if (root/'session.json').exists():
            state = load(root)
            if state['boot_id'] != proc_key()['boot_id']:
                raise ValueError('boot changed: use a new session root and recalibrate')
            if 'diagnosis' not in state['config']:
                from .business import DEFAULTS as DIAG_DEFAULTS
                if config.get('diagnosis') == DIAG_DEFAULTS:
                    config = {k:v for k,v in config.items() if k != 'diagnosis'}
            if (state['vendor'], state['devices'], state['config']) != (vendor, devices, config):
                raise ValueError('session configuration changed: use a new root')
            if state.get('ended_raw_ns'):
                state['gaps'].append(dict(ts=state['ended_raw_ns'],window=[state['ended_raw_ns'],raw_clock()],reason='service_restart'))
            state.update(status='starting', supervisor=proc_key(), ended_raw_ns=None)
        else:
            state = dict(schema='monitor-session/1', id=uuid.uuid4().hex, vendor=vendor,
                devices=devices, config=config, boot_id=proc_key()['boot_id'], supervisor=proc_key(),
                started_raw_ns=raw_clock(), ended_raw_ns=None, status='starting', segments=[],
                gaps=[], inventory={}, restarts=[], pins={}, reserved_bytes=0)
        atomic_json(root/'session.json', state)
    return state


def owned_size(root):
    """Count regular files only; sockets and symlinks are not owned payload."""
    total = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory)/name
            try:
                if path.is_file() and not path.is_symlink(): total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total
