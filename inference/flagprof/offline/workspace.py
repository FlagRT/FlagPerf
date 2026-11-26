"""Private operation directories on the user's selected volume."""
from contextlib import contextmanager
import errno
import fcntl
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import uuid


def private_name():
    return '.flagprof-' + str(os.getuid())


def is_artifact_directory(path):
    """Identify a Store, capture, report or workflow directory that owns its contents."""
    markers = ('resolved-config.json', 'plan-b-capture.json', 'record-status.json', 'group.json', 'result.json')
    return (any((path / name).is_file() for name in markers)
            or ((path / 'manifest.json').is_file() and (path / 'writers').is_dir()))


@contextmanager
def directory_lock(path, *, nonblocking=False):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(fd, flags)
        yield fd
    finally:
        os.close(fd)


def private_root(volume):
    volume = Path(volume).resolve()
    volume.mkdir(parents=True, exist_ok=True)
    root = volume / private_name()
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('private FlagPerf directory unavailable: ' + str(root))
    # An operation may itself be the output volume of a nested restore.
    # Serialize initialization on the namespace, independently of its parent lease.
    with directory_lock(root):
        marker = root / 'workspace-v1'
        if not marker.exists() and not any(root.iterdir()):
            marker.touch(mode=0o600)
        if marker.is_symlink() or not marker.is_file():
            raise ValueError('private FlagPerf directory is not initialized: ' + str(root))
    return root


def _discard_attempt(path):
    # A staged run is a sibling of the final run so its relative evidence
    # references are identical before and after publication.
    if (path / 'staged-run').is_file():
        stage = path.parents[2] / ('.flagprof-import-' + path.name)
        if stage.is_dir() and not stage.is_symlink():
            shutil.rmtree(stage)
    shutil.rmtree(path)


@contextmanager
def attempt(volume=None):
    root = private_root(volume if volume is not None else tempfile.gettempdir()) / 'tmp'
    root.mkdir(mode=0o700, exist_ok=True)
    with directory_lock(root):
        for old in root.iterdir():
            if re.fullmatch('[0-9a-f]{32}', old.name) and (old / 'attempt-v1').is_file():
                try:
                    with directory_lock(old, nonblocking=True):
                        _discard_attempt(old)
                except BlockingIOError:
                    continue
        own = root / uuid.uuid4().hex
        own.mkdir(mode=0o700)
        lock = directory_lock(own)
        fd = lock.__enter__()
        (own / 'attempt-v1').touch()
    try:
        yield own, fd
    finally:
        lock.__exit__(None, None, None)
        # A surviving worker inherits the lock if its parent was interrupted.
        with directory_lock(root):
            if own.exists():
                try:
                    with directory_lock(own, nonblocking=True):
                        _discard_attempt(own)
                except BlockingIOError:
                    pass


@contextmanager
def checkpoint(volume, key):
    root = private_root(volume) / 'checkpoints'
    root.mkdir(mode=0o700, exist_ok=True)
    with directory_lock(root):
        current = root / key
        current.mkdir(mode=0o700, exist_ok=True)
        disk = shutil.disk_usage(root)
        reserve = min(2 << 30, disk.total // 20)
        obsolete = time.time() - 30 * 86400
        for old in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime):
            if old == current or not re.fullmatch('[0-9a-f]{64}', old.name):
                continue
            if old.stat().st_mtime >= obsolete and shutil.disk_usage(root).free >= reserve:
                break
            try:
                with directory_lock(old, nonblocking=True):
                    shutil.rmtree(old)
            except BlockingIOError:
                continue
        # Wait on this key without blocking unrelated imports or pruning.
        fd = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        while True:
            fcntl.flock(fd, fcntl.LOCK_EX)
            # GC may have removed this empty key before it was locked.
            if current.exists() and os.path.samestat(os.fstat(fd), current.stat()):
                break
            with directory_lock(root):
                current.mkdir(mode=0o700, exist_ok=True)
                replacement = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            os.close(fd)
            fd = replacement
        os.utime(fd, None)
        yield current
    finally:
        os.close(fd)


@contextmanager
def query_view(run):
    from .finish import BoundedView
    with attempt() as (temp, _lease):
        with BoundedView(run, temp / 'view', cache_root=temp) as view:
            yield view


def publish_run(stage, target):
    import ctypes
    from .checkpoints import syncdir
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is not None:
        if rename(-100, os.fsencode(stage), -100, os.fsencode(target), 1) == 0:
            syncdir(target.parent)
            return
        error = ctypes.get_errno()
        if error not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            raise OSError(error, os.strerror(error), str(target))
    # Some mounted filesystems implement POSIX directory rename but not its
    # Linux no-replace flag. Serialize this account's publishers on the destination.
    with directory_lock(private_root(target.parent)):
        if os.path.lexists(target):
            raise FileExistsError(str(target))
        os.rename(stage, target)
        syncdir(target.parent)
