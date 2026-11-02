# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Owned ephemeral indexes, durable checkpoints, and explicit disk placement."""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import uuid


def locations(store, work=None, scratch=None, *, key='query'):
    if not key or Path(key).name != key or key in ('.', '..'):
        raise ValueError('invalid workspace key')
    store = Path(store).resolve()
    work = Path(work).resolve() if work else store.parent/'work'/key
    scratch = Path(scratch or os.environ.get('FLAGPROF_SCRATCH', '/var/tmp/flagprof-offline-'+str(os.getuid()))).resolve()
    return work, scratch


def filesystem_type(path):
    """Resolve the most specific Linux mount, including bind mounts.

    statfs's ext-family magic cannot distinguish ext4 from ext2/ext3. Inspect
    mountinfo instead, without creating the prospective output directory.
    """
    path = Path(path).resolve()
    selected = None
    depth = -1
    with Path('/proc/self/mountinfo').open() as mounts:
        for line in mounts:
            left, right = line.rstrip('\n').split(' - ', 1)
            encoded = left.split()[4]
            mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), encoded))
            if path.is_relative_to(mount) and len(mount.parts) >= depth:
                selected = right.split()[0]
                depth = len(mount.parts)
    if selected is None:
        raise OSError('cannot determine filesystem for ' + str(path))
    return selected


def check_disk(path):
    """Require the accepted ext4 workspace before making any artifacts."""
    kind = filesystem_type(path)
    if kind != 'ext4':
        raise ValueError('Plan B requires ext4: ' + str(Path(path).resolve()) +
                         ' is on ' + kind + '; select an ext4 directory')


def check_capture_disk(path, vendor, filesystem='ext4'):
    """Explicit NVIDIA capture-only admission; import/query remain ext4."""
    if filesystem == 'ext4':return check_disk(path)
    if filesystem != 'overlay' or vendor != 'nvidia' or filesystem_type(path) != 'overlay':
        raise ValueError('capture-only overlay admission requires NVIDIA on actual overlay')


def check_layout(store, work, scratch):
    paths = dict(store=Path(store).resolve(), work=Path(work).resolve(),
                 scratch=Path(scratch).resolve())
    for name, path in paths.items():
        for other, second in paths.items():
            if name < other and (path.is_relative_to(second) or second.is_relative_to(path)):
                raise ValueError(name + ' and ' + other + ' must be separate directory trees')
    for path in paths.values():
        check_disk(path)


def check_capture_layout(capture, *outputs):
    capture = Path(capture).resolve()
    check_disk(capture)
    for output in outputs:
        output = Path(output).resolve()
        if capture.is_relative_to(output) or output.is_relative_to(capture):
            raise ValueError('capture and output must be separate directory trees')


@contextmanager
def owned_scratch(root):
    root=Path(root).resolve();check_disk(root)
    root.mkdir(parents=True, exist_ok=True)
    own=root/('attempt-'+uuid.uuid4().hex);own.mkdir()
    try:yield own
    except BaseException:
        # Keep failure evidence; a new attempt never opens this partial database.
        raise
    else:shutil.rmtree(own)


@contextmanager
def query_view(run, *, work=None, scratch=None):
    from .service import lease, _check_scratch
    from .disk import scratch_scope
    from .finish import BoundedView
    store = Path(run).resolve().parent
    work,scratch=locations(store,work,scratch,key='query-'+Path(run).name)
    _check_scratch(scratch,store,work)
    with lease(work), owned_scratch(scratch) as own, scratch_scope(own,work):
        with BoundedView(run,work/('query-'+uuid.uuid4().hex),cache_root=work) as view:
            yield view
