"""Front-door budgets for optional bounded import and views."""
from pathlib import Path
import json
import os

METADATA_BYTES = 4 << 20


def read_json(path, limit=METADATA_BYTES):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError('metadata budget or symlink: ' + str(path))
    with path.open('rb') as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError('metadata budget')
    return json.loads(data)


def walk_files(root, *, max_files, max_entries):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('directory type')
    # Iterative traversal: both the frontier and total entries have a cap.
    pending = [root]
    entries = files = 0
    while pending:
        with os.scandir(pending.pop()) as it:
            for entry in it:
                entries += 1
                if entries > max_entries:
                    raise ValueError('directory entry budget')
                if entry.is_symlink():
                    raise ValueError('directory symlink')
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files += 1
                    if files > max_files:
                        raise ValueError('file budget')
                    yield Path(entry.path)
                else:
                    raise ValueError('nonregular source file')


def preflight_store(run):
    run = Path(run)
    writers = shards = traces = 0
    for path in walk_files(run, max_files=256, max_entries=512):
        rel = path.relative_to(run)
        if len(rel.parts) == 1 and path.suffix == '.json':
            if path.stat().st_size > METADATA_BYTES:
                raise ValueError('run metadata budget')
        if rel.parts[0] == 'writers' and path.name.startswith('w-'):
            writers += 1
            if writers > 16:
                raise ValueError('writer budget')
            shards += len(read_json(path)['shards'])
            if shards > 64:
                raise ValueError('shard budget')
        if rel.parts[0] == 'traces' and path.suffix == '.json':
            traces += 1
            if traces > 1:
                raise ValueError('optional Plan B requires one trace')
        if rel.parts[0] == 'captures':
            raise ValueError('legacy capture layout not supported by bounded Plan B view')
    if traces != 1:
        raise ValueError('optional Plan B requires one trace')
