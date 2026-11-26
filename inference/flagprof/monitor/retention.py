"""Rolling monitor Stores with conservative pre-write reservations and read leases."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
from flagprof.store import Writer, WriterConfig, create_run, finalize
from flagprof.store.manifest import atomic_json
from flagprof.telemetry.contract import raw_clock
from .session import edit, load, locked, owned_size


class CapacityError(ValueError): pass


def recover_closed(root):
    """Only call after confirming the old worker exited. Preserve failed Stores."""
    state = load(root)
    for segment in state['segments']:
        if segment['status'] != 'active': continue
        directory = Path(root)/'segments'/segment['id']
        manifest = finalize(directory)
        inventory = json.loads((directory/'inventory.json').read_text())
        end = raw_clock()
        inventory.update(status='partial',ended_raw_ns=end,reason='collector_exit')
        atomic_json(directory/'inventory.json',inventory)
        with edit(root) as current:
            row = next(s for s in current['segments'] if s['id']==segment['id'])
            current['reserved_bytes']=max(0,current['reserved_bytes']-row.get('reserved_bytes',0))
            row.update(status='closed',window=[row['window'][0],end],bytes=owned_size(directory),
                       store_status=manifest['status'],reason='collector_exit',reserved_bytes=0)


def overlap(a, b):
    return a[0] < (b[1] or float('inf')) and b[0] < (a[1] or float('inf'))


def pinned(state, segment):
    return any(overlap(segment['window'], pin['window']) for pin in state['pins'].values())


@contextmanager
def read_lease(root, ident):
    # The session lock closes the select/open race with GC. Locks are kernel
    # owned and disappear after a reader crash; PID based stale leases are unnecessary.
    with locked(root, shared=True):
        path = Path(root)/'segments'/ident
        stream = (path/'.read.lock').open('rb')
        fcntl.flock(stream, fcntl.LOCK_SH)
    try: yield path
    finally: stream.close()


def _segment_path(root, ident):
    parent = Path(root)/'segments'
    if not isinstance(ident, str) or Path(ident).name != ident or ident in ('', '.', '..') or parent.is_symlink():
        raise ValueError('invalid owned segment path')
    return parent/ident


def _check_owner(path, session_id):
    marker = path/'monitor-owner.json'
    if not stat.S_ISREG(marker.lstat().st_mode) or json.loads(marker.read_text())['session_id'] != session_id:
        raise ValueError('segment ownership mismatch')


def _idle(path):
    with os.fdopen(os.open(path/'.read.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb+') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('invalid segment lease')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    # The caller still holds the exclusive session lock, so no reader can enter.
    # An open, unlinked lock file would keep a hidden directory entry on dtfs.
    return True


def _finish_retirement(root, state):
    pending = state.get('retiring_segments', {})
    finished = []
    for ident, identity in pending.items():
        path = _segment_path(root, ident)
        try:
            info = path.lstat()
        except FileNotFoundError:
            # A previous collector removed the directory before acknowledging it.
            finished.append(ident)
            continue
        if not stat.S_ISDIR(info.st_mode) or [info.st_dev, info.st_ino] != identity:
            raise ValueError('retiring segment directory replaced')
        owner = path/'monitor-owner.json'
        try:
            _check_owner(path, state['id'])
        except FileNotFoundError:
            # The owner is removed last. rmdir can only finish an empty directory.
            path.rmdir()
        else:
            lease = path/'.read.lock'
            try:
                if not _idle(path):
                    continue
            except FileNotFoundError:
                pass  # A previous collector already closed and removed the lease.
            lease.unlink(missing_ok=True)
            for entry in path.iterdir():
                if entry == owner:
                    continue
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            owner.unlink()
            path.rmdir()
        finished.append(ident)
    if finished:
        directory = os.open(Path(root)/'segments', os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        for ident in finished:
            del pending[ident]
    if not pending:
        state.pop('retiring_segments', None)


def collect(root, state, now, *, incoming=0):
    """Run under edit(root), which excludes new read leases and pin changes."""
    _finish_retirement(root, state)
    config = state['config']
    ordinary = [s for s in state['segments'] if s['status']=='closed' and not pinned(state,s)]
    total = sum(s.get('bytes',0) for s in ordinary)
    selected = {}
    empty = set()
    for segment in ordinary:
        if now-segment['window'][1] <= config['pre_seconds']*1e9:
            continue
        if total+incoming <= config['raw_bytes'] and now-segment['window'][1] <= config['raw_seconds']*1e9:
            continue
        path = _segment_path(root, segment['id'])
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('owned segment missing or replaced')
        try:
            idle = _idle(path)
        except FileNotFoundError:
            idle = None  # Legacy Store without a lease file: readers cannot open it.
        if idle is False:
            continue
        try:
            _check_owner(path, state['id'])
        except FileNotFoundError:
            # A failed legacy rmtree left an empty directory. A second attempt
            # opened .read.lock with O_CREAT before checking the missing owner.
            # Forget that expired index entry but leave the actual files alone.
            residue = list(path.iterdir())
            lease = path/'.read.lock'
            if residue and not (idle and residue == [lease] and lease.lstat().st_size == 0):
                raise
            empty.add(segment['id'])
            total -= segment.get('bytes', 0)
            continue
        selected[segment['id']] = [info.st_dev, info.st_ino]
        total -= segment.get('bytes', 0)
    if selected or empty:
        # Publish the decision before touching files. A crash during a tree
        # removal can then be retried without advertising a half-deleted Store.
        remaining = [s for s in state['segments'] if s['id'] not in selected and s['id'] not in empty]
        pending = {**state.get('retiring_segments', {}), **selected}
        updated = dict(state, segments=remaining, retiring_segments=pending)
        # The data/control low-water marks are triggers for reclamation. Allow
        # its atomic metadata write to use the control reserve; real I/O errors
        # still abort before unlinking any newly selected segment.
        atomic_json(Path(root)/'session.json', updated)
        state.update(segments=remaining, retiring_segments=pending)
        _finish_retirement(root, state)
    return len(selected) + len(empty)


def reserve(root, state, amount):
    if amount < 0: raise ValueError('negative reservation')
    config = state['config']
    if shutil.disk_usage(root).free < config['min_free_bytes']+amount:
        raise CapacityError('filesystem_free_space')
    # Reserved bytes include the full open segment, its queue and close overhead.
    # Actual bytes are counted as well, deliberately conservative during writing.
    if owned_size(root)+state['reserved_bytes']+amount > config['total_bytes']-config['control_reserve_bytes']:
        raise CapacityError('node_capacity_pinned_or_active')
    state['reserved_bytes'] += amount


def publish_control(root, path, value):
    amount=2*len(json.dumps(value,allow_nan=False).encode())+4096
    with edit(root) as state:
        reserve(root,state,amount)
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        atomic_json(Path(path),value)
        state['reserved_bytes']-=amount


class RollingWriter:
    def __init__(self, root, inventory):
        self.root = Path(root); self.inventory = inventory
        self.writer = None; self.segment = None; self.held = 0; self.payload = 0
        self.config = load(root)['config']
        self.rotate()

    def rotate(self):
        self.close_segment()
        now = raw_clock()
        with edit(self.root) as state:
            collect(self.root, state, now)
            if len(state['segments']) >= self.config['max_segments']:
                raise CapacityError('segment_metadata_limit')
            # Bound headers/inventory/index/close metadata separately from event bytes.
            held = (1 << 20)+4*len(json.dumps(self.inventory).encode())
            reserve(self.root, state, held)
            run = create_run(self.root/'segments', identity=dict(kind='monitor',session_id=state['id']))
            atomic_json(run/'monitor-owner.json', dict(session_id=state['id']))
            (run/'.read.lock').touch()
            atomic_json(run/'inventory.json', dict(self.inventory,status='running',started_raw_ns=now,ended_raw_ns=None))
            segment = dict(id=run.name, window=[now,None], status='active', bytes=0,reserved_bytes=held)
            state['segments'].append(segment)
        self.segment, self.run, self.held, self.payload = segment, run, held, 0
        self.writer = Writer(run, kind='counter', config=WriterConfig(
            queue_capacity=self.config['queue_capacity'], compress=False,
            shard_bytes=self.config['segment_bytes'], shard_period_seconds=self.config['segment_seconds']))

    def emit(self, event):
        self.emit_many([event])

    def emit_many(self, events):
        """Reserve bounded batches before enqueueing any of their records."""
        batch=[];amount=0
        for event in events:
            size=len(json.dumps(event,allow_nan=False,separators=(',',':')).encode())
            if size>self.config['max_record_bytes']:raise CapacityError('monitor_record_limit')
            cost=size+2048
            if batch and (len(batch)>=min(64,self.config['queue_capacity']) or
                    amount+cost>self.config['segment_bytes']):
                self._emit_batch(batch,amount);batch=[];amount=0
            batch.append(event);amount+=cost
        if batch:self._emit_batch(batch,amount)

    def _emit_batch(self, events, amount):
        if self.payload and (raw_clock()-self.segment['window'][0]>=self.config['segment_seconds']*1e9 or
                self.payload+amount>self.config['segment_bytes']):
            self.rotate()
        with edit(self.root) as state:
            if not pinned(state,self.segment):
                ordinary=sum(s.get('bytes',0) for s in state['segments'] if s['status']=='closed' and not pinned(state,s))
                if ordinary+self.payload+amount>self.config['raw_bytes']:
                    collect(self.root,state,raw_clock(),incoming=self.payload+amount)
                    ordinary=sum(s.get('bytes',0) for s in state['segments'] if s['status']=='closed' and not pinned(state,s))
                    if ordinary+self.payload+amount>self.config['raw_bytes']:
                        raise CapacityError('raw_retention_capacity')
            reserve(self.root,state,amount)
            next(s for s in state['segments'] if s['id']==self.run.name)['reserved_bytes']+=amount
        self.held+=amount;self.payload+=amount
        for event in events:
            if not self.writer.emit(event):raise CapacityError('monitor_writer_queue')

    def close_segment(self):
        if self.writer is None: return
        self.writer.close(); self.writer = None
        end = raw_clock()
        atomic_json(self.run/'inventory.json',dict(self.inventory,status='complete',
            started_raw_ns=self.segment['window'][0],ended_raw_ns=end))
        manifest = finalize(self.run)
        with edit(self.root) as state:
            segment = next(s for s in state['segments'] if s['id']==self.run.name)
            segment.update(status='closed', window=[segment['window'][0],end],
                           bytes=owned_size(self.run), store_status=manifest['status'],reserved_bytes=0)
            state['reserved_bytes'] = max(0,state['reserved_bytes']-self.held)
        self.held = 0

    def close(self): self.close_segment()
