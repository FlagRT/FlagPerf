"""Coordinator-owned inventory for retention, without repeated whole-run walks."""
from pathlib import Path
from runtime.common import read_json, write_json, file_hash


def register(root, destination, world_size=1):
    root, destination = Path(root), Path(destination)
    path = root/'capture-index.json'
    index = read_json(path) if path.exists() else {'storage_version': 1, 'complete': False, 'workers': {}}
    folders = [destination]
    if world_size > 1: folders += [destination/f'rank-{rank}' for rank in range(world_size)]
    for folder in folders:
        state = folder/'result.json'
        if not state.is_file(): continue
        name = state.relative_to(root).as_posix()
        index['workers'][name] = {'type': 'worker_state', 'sha256': file_hash(state), 'bytes': state.stat().st_size,
            'purpose': 'capture evidence and retention ownership',
            'diagnostics': [(folder/part).relative_to(root).as_posix() for part in ('profiler', 'profiler-recovery')]}
    write_json(path, index)


def finish(root):
    path = Path(root)/'capture-index.json'
    if path.exists():
        index = read_json(path); index['complete'] = True; write_json(path, index)


def paths(root):
    root = Path(root)
    path = root/'capture-index.json'
    if not path.is_file(): return None
    index = read_json(path)
    if index.get('storage_version') != 1 or index.get('complete') is not True: return None
    workers, directories = [], {root/'cache'}
    for name, row in index['workers'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('unsafe capture inventory member')
        workers.append(root/relative)
        for directory in row['diagnostics']:
            relative = Path(directory)
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('unsafe capture inventory directory')
            directories.add(root/relative)
    return workers, sorted(directories)
