"""Linux flock leases: host boot + uid + vendor physical device/power domain.

Participating host samplers share one user runtime namespace.
These advisory leases do not constrain unrelated vendor tools or other users.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from flagprof.store.manifest import process_alive, proc_key


class Leases:
    def __init__(self):
        self.handles = []

    def acquire(self, keys, directory=None):
        if directory is None:
            runtime = Path('/run/user') / str(os.getuid())
            if not runtime.is_dir() or runtime.stat().st_uid != os.getuid():
                # The canonical host fallback must not vary with a caller's TMPDIR.
                runtime = Path('/tmp')
            directory = runtime / ('flagprof-monitor-' + str(os.getuid()))
        directory = Path(directory)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('lease directory must be private and owned by this user')
        try:
            for key in sorted(set(keys)):
                path = directory/(hashlib.sha256(key.encode()).hexdigest()+'.lock')
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                self.handles.append(fd)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                previous = os.read(fd,65536)
                owner = json.loads(previous).get('process') if previous else None
                if owner and owner['boot_id']==proc_key()['boot_id'] and process_alive(owner) is not False:
                    raise ValueError('previous sampler exit not confirmed')
        except BaseException:
            self.close()
            raise
        return self

    def owner(self, process):
        for fd in self.handles:
            data = json.dumps(dict(process=process)).encode()
            os.lseek(fd,0,os.SEEK_SET); os.write(fd,data); os.ftruncate(fd,len(data)); os.fsync(fd)

    def close(self):
        for fd in self.handles: os.close(fd)
        self.handles.clear()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def device_keys(state, inventory):
    prefix = state['boot_id']+':'+str(os.getuid())+':'+state['vendor']+':'
    names = set()
    for item in inventory['entities']:
        names.add(item['entity'])
        if item.get('power_domain'): names.add(item['power_domain'])
        # A shared board owns the complete power domain, even when one die is selected.
        names.update(item.get('members', []))
    return [prefix+name for name in names]
