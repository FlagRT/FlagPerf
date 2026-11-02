"""Explicit single-host TP workers, each owning its attachment and Store."""
import json
from contextlib import contextmanager
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

from .store.manifest import atomic_json


def worker(settings, rank):
    from .offline.capture import execute_target
    from .runtime import attach
    from .store import create_run
    root = Path(settings['root'])/('rank-'+str(rank))
    root.mkdir()
    state = dict(rank=rank, group_id=settings['group_id'], status='running', run=None)
    code = 1
    handle = None
    try:
        import importlib
        device = importlib.import_module('flagprof.vendors.'+settings['vendor']+'.execution').initialize(rank)
        os.environ['FLAGPROF_DEVICE'] = device
        args = settings['command'][1:]
        while args and args[0] in ('-B','-u'):
            args = args[1:]
        if settings['preset'] in ('perf','modules'):
            from .offline.capture import record_final
            code = record_final(settings['command'], store=root/'delivery', vendor=settings['vendor'],
                                preset=settings['preset'], config=settings['config'], rank=rank,
                                timeout=settings['timeout'], capture_only=True,
                                source_max_bytes=settings['source_max_bytes'],
                                capture_filesystem=settings.get('capture_filesystem','ext4'))
            state['capture'] = str(next((root/'delivery').glob('capture-*')).relative_to(settings['root']))
        else:
            if settings['preset'] != 'none':
                run = create_run(root/'delivery', identity=dict(group_id=settings['group_id'], rank=rank))
                state['run'] = str(run.relative_to(settings['root']))
                handle = attach(run, settings['vendor'], preset=settings['preset'], config=settings['config'], rank=rank)
                if handle.state['status'] != 'running':
                    raise RuntimeError('worker attachment did not start')
            execute_target(args)
            code = 0
    except BaseException as error:
        state['error'] = type(error).__name__+': '+str(error)
        traceback.print_exc()
    finally:
        if handle:
            result = handle.close(normal=code==0, returncode=code)
            if result['status'] != 'closed':
                code = code or 2
        state.update(status='complete' if code==0 else 'failed', returncode=code)
        atomic_json(root/'worker.json', state)
    return code


def launch(command, *, store, nproc_per_node=2, comm_backend='hccl', preset='perf',
           config=None, timeout=1800, source_max_bytes=1 << 30,
           capture_only=False, capture_filesystem='ext4'):
    vendors = {'hccl':'ascend','flagcx':'ascend','nccl':'nvidia'}
    if nproc_per_node != 2 or comm_backend not in vendors:
        raise ValueError('launch supports single-host TP2 HCCL/FlagCX/NCCL')
    vendor = vendors[comm_backend]
    if capture_only and preset not in ('perf','modules'):
        raise ValueError('capture-only requires an official profiler preset')
    if capture_filesystem != 'ext4' and not capture_only:
        raise ValueError('overlay requires explicit capture-only')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive finite timeout required')
    if not command or not Path(command[0]).name.startswith('python'):
        raise ValueError('launch requires a direct Python target')
    from .offline.workspace import check_disk
    from .offline.checkpoints import source_budget
    source_budget(source_max_bytes)
    root = Path(store).resolve()
    if capture_only:
        from .offline.workspace import check_capture_disk
        check_capture_disk(root,vendor,capture_filesystem)
    else:
        check_disk(root)
    root.mkdir(parents=True, exist_ok=False)
    group_id = uuid.uuid4().hex
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    settings = dict(schema='flagprof-launch/1', group_id=group_id, root=str(root), command=command,
                    world_size=2, backend=comm_backend, preset=preset, config=config or {},
                    timeout=timeout, source_max_bytes=source_max_bytes, vendor=vendor,
                    capture_only=capture_only,capture_filesystem=capture_filesystem)
    atomic_json(root/'launch.json', settings)
    state = dict(schema='flagprof-group/1', group_id=group_id, world_size=2, backend=comm_backend,
                 status='running', ranks=[], stages=[])
    @contextmanager
    def stage(name, rank):
        row=dict(stage=name,rank=rank,start_monotonic_ns=time.monotonic_ns(),status='running')
        state['stages'].append(row)
        atomic_json(root/'group-progress.json',state)
        try:
            yield
            row['status']='complete'
        except BaseException as error:
            row.update(status='failed',error=type(error).__name__+': '+str(error))
            raise
        finally:
            row['elapsed_seconds']=(time.monotonic_ns()-row['start_monotonic_ns'])/1e9
            atomic_json(root/'group-progress.json',state)
    children, logs = [], []
    def worker_state(rank):
        path=root/('rank-'+str(rank))/'worker.json'
        row=dict(rank=rank,group_id=group_id,status='missing',run=None)
        if path.exists():
            try:row=json.loads(path.read_text())
            except (OSError,ValueError) as error:
                row.update(status='unreadable',error=type(error).__name__+': '+str(error))
        row['process_returncode']=children[rank].returncode if rank<len(children) else None
        return row
    deadline = time.monotonic()+timeout
    previous = {}
    stopped = []
    def stop(signum, frame):
        stopped.append(signum)
        for child in children:
            if child.poll() is None:
                try:os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:pass
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, stop)
        for rank in range(2):
            env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2',
                       MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), FLAGPROF_COMM_BACKEND=comm_backend,
                       FLAGPROF_GROUP_ID=group_id, FLAGPROF_LAUNCH_WORKER='1')
            env.pop('FLAGPROF_RECORD_CONFIG', None)
            log = (root/('rank-'+str(rank)+'.log')).open('xb')
            logs.append(log)
            children.append(subprocess.Popen([command[0], '-B', '-m', 'flagprof.launch',
                                               str(root/'launch.json'), str(rank)], env=env,
                                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while any(child.poll() is None for child in children):
            if stopped or time.monotonic() >= deadline or any(child.poll() not in (None,0) for child in children):
                stop(signal.SIGTERM, None)
                break
            time.sleep(.25)
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state['device_workers_exited_monotonic_ns'] = time.monotonic_ns()
        atomic_json(root/'group-progress.json',state)
        for rank, child in enumerate(children):
            state['ranks'].append(worker_state(rank))
        if stopped or any(row['status'] != 'complete' or row['process_returncode'] != 0 for row in state['ranks']):
            state['status'] = 'failed'
            return 2
        if capture_only:
            state.update(status='captured',vendor=vendor,capture_filesystem=capture_filesystem)
            return 0
        # All device workers have exited. Finalize and import one rank at a time.
        for row in state['ranks']:
            if row.get('capture'):
                from .offline.capture import finish_capture
                from .offline.service import import_bundle
                capture = root/row['capture']
                with stage('official_analysis',row['rank']):
                    finish_capture(capture, timeout=max(1, deadline-time.monotonic()))
                with stage('first_import',row['rank']):
                    result = import_bundle(capture, store=root/('rank-'+str(row['rank']))/'runs',
                                           work=root/('import-work-'+str(row['rank'])),
                                           scratch=root/('scratch-'+str(row['rank'])))
                row['run'] = str(Path(result['run']).relative_to(root))
                row['import_status'] = result['status']
                if result['status'] != 'complete':
                    raise ValueError('rank import incomplete')
            elif row.get('run'):
                from .store._proxy_lifecycle import record_wait, deadline_scope
                from .store import finalize
                due, timed_out = record_wait(root/row['run'], deadline=deadline)
                with stage('collection_finalize',row['rank']), deadline_scope(due):
                    result = finalize(root/row['run'])
                if timed_out or result['status'] != 'complete':
                    raise ValueError('rank Store finalization incomplete')
        state['status'] = 'complete'
        return 0
    except BaseException as error:
        state.update(status='failed', error=type(error).__name__+': '+str(error))
        stop(signal.SIGTERM, None)
        raise
    finally:
        # A launch/import exception must not leave a rank process running.
        if any(child.poll() is None for child in children):
            stop(signal.SIGTERM, None)
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait()
        saved={row['rank']:row for row in state['ranks']}
        state['ranks']=[saved[rank] if rank in saved else worker_state(rank) for rank in range(nproc_per_node)]
        if 'device_workers_exited_monotonic_ns' not in state and children:
            state['device_workers_exited_monotonic_ns'] = time.monotonic_ns()
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        atomic_json(root/'group.json', state)


if __name__ == '__main__':
    raise SystemExit(worker(json.loads(Path(sys.argv[1]).read_text()), int(sys.argv[2])))
