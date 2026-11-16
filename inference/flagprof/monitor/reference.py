import json
import os
from pathlib import Path
from flagprof.store.manifest import atomic_json, proc_key
from flagprof.telemetry.contract import raw_clock


def read(path):
    return json.loads(Path(path).read_text())


def relocate(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.exists():
        return
    data = read(source)
    for key in ('monitor_path', 'session_path', 'root'):
        if data.get(key):
            data[key] = os.path.relpath((source.parent/data[key]).resolve(), destination.parent)
    atomic_json(destination, data)


def bind(path, *, root=None, topology=None, monitor=None, devices=None, worker=None, finish=False):
    path = Path(path)
    data = read(path) if path.exists() else dict(version=1, monitor_run_id=None, monitor_path=None,
        window=[raw_clock(), None], devices=[], worker_proc_key=None, status='absent',
        monitor_run_status='absent', reclaim=None, errors=[])
    if root is not None:
        data['root'] = os.path.relpath(Path(root).resolve(), path.parent)
    if topology is not None:
        data['topology'] = topology
    if monitor is not None and data.get('topology') != 'agent':
        data.update(monitor_run_id=Path(monitor).name, monitor_path=os.path.relpath(Path(monitor).resolve(), path.parent))
    if devices is not None:
        data['devices'] = devices
    if worker is not None:
        if data['worker_proc_key'] is None:
            data['window'][0] = raw_clock()
        data['worker_proc_key'] = worker
    if finish:
        data['window'][1] = raw_clock()
    atomic_json(path, data)
    if data.get('topology') == 'agent':
        from .subscriptions import request
        if data.get('worker_proc_key') and data.get('session_id'):
            try:
                result = request(path.parent/data['root'],dict(action='finish' if finish else 'register',
                    session_id=data['session_id'],reference=data,path=str(path.resolve())))
                data.update(target_id=result['target_id'],subscription_status=result['status'])
            except (OSError, ValueError) as error:
                data.update(status='unavailable',subscription_status='capacity' if 'monitor capacity:' in str(error) else 'unavailable',reason=str(error))
            atomic_json(path,data)
        return data
    if data.get('root') and data.get('worker_proc_key'):
        key = data['worker_proc_key']
        target = (path.parent/data['root'])/'targets'/('{pid_ns}-{pid}-{proc_start_ticks}.json'.format(**key))
        target.parent.mkdir(parents=True, exist_ok=True)
        registration = dict(data, reference=os.path.relpath(path.resolve(), target.parent))
        for name in ('root', 'monitor_path', 'session_path'):
            if data.get(name):
                registration[name] = os.path.relpath((path.parent/data[name]).resolve(), target.parent)
        atomic_json(target, registration)
    return data


def resolve_reference(path):
    path = Path(path)
    if not path.exists():
        return dict(status='absent', reason='reference_missing'), None, None
    ref = read(path)
    if ref.get('version') == 2:
        from .session import load
        try:
            root = (path.parent/ref['session_path']).resolve()
            state = load(root)
            if state['id'] != ref['session_id'] or state['boot_id'] != (ref.get('worker_proc_key') or {}).get('boot_id',state['boot_id']):
                raise ValueError('monitor session identity conflict')
            status = 'running' if state['status']=='running' else 'complete' if state['status'] in ('stopped','archived') else 'partial'
            if ref.get('subscription_status') in ('capacity','unavailable'):status='unavailable'
            return dict(ref,status=status,session_status=state['status']),root,state['inventory']
        except (OSError,ValueError,KeyError) as error:
            return dict(ref,status='absent',reason=str(error)),None,None
    candidates = []
    if ref.get('monitor_run_id') and ref.get('monitor_path'):
        candidates = [(path.parent/ref['monitor_path']).resolve()]
    elif ref.get('root'):
        devices = {d['entity'] for d in ref['devices'] if d.get('status') == 'available'}
        boot = (ref.get('worker_proc_key') or {}).get('boot_id')
        begin, end = ref['window']; end = end or raw_clock()
        for candidate in (path.parent/ref['root']/'monitor').glob('*'):
            try:
                inventory = read(candidate/'inventory.json')
                a = inventory['started_raw_ns']; b = inventory.get('ended_raw_ns') or raw_clock()
                available = {d['entity'] for d in inventory['entities']}
                if boot and boot == inventory.get('boot_id') and devices and devices <= available and a < end and b > begin:
                    candidates.append(candidate.resolve())
            except (OSError, ValueError, KeyError):
                continue
    if len(candidates) != 1:
        return dict(ref, status='absent', reason='ambiguous' if candidates else 'monitor_missing',
                    candidates=[p.name for p in candidates]), None, None
    candidate = candidates[0]
    try:
        inventory = read(candidate/'inventory.json')
        manifest = read(candidate/'manifest.json')
    except (OSError, ValueError) as error:
        return dict(ref, status='absent', reason='monitor_unreadable', errors=ref.get('errors', [])+[str(error)]), None, None
    if (ref.get('monitor_run_id') and candidate.name != ref['monitor_run_id']) or (
            (ref.get('worker_proc_key') or {}).get('boot_id') and inventory.get('boot_id') != ref['worker_proc_key']['boot_id']):
        return dict(ref, status='absent', reason='monitor_identity_conflict'), None, None
    monitor_status = inventory.get('status', manifest['status'])
    begin, end = ref['window']; end = end or raw_clock()
    available = {d['entity'] for d in inventory.get('entities', [])}
    devices = {d['entity'] for d in ref['devices'] if d.get('status') == 'available'}
    status = ('failed' if monitor_status == 'failed' else 'running' if monitor_status == 'running' else
              'complete' if monitor_status == 'complete' else 'partial')
    if status == 'complete' and (inventory['started_raw_ns'] > begin or
            inventory.get('ended_raw_ns', 0) < end or not devices or not devices <= available):
        status = 'partial'
    return dict(ref, status=status, monitor_run_status=monitor_status,
                monitor_run_id=candidate.name, monitor_path=os.path.relpath(candidate, path.parent)), candidate, inventory
