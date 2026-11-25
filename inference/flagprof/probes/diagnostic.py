"""Opt-in target observations. No device synchronization or agent calls on hooks."""
import os
import sys
import threading
from collections import deque
from copy import deepcopy
from .base import clock


class DiagnosticProbe:
    def __init__(self, attachment):
        self.handle = attachment
        self.writer = attachment.writer
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.active = {}
        self.calls = {}
        self.seen = deque(maxlen=128)
        self.identity = dict(run_id=self.writer.run_id, rank=attachment.rank,
                             group_id=attachment.state.get('group_id') or os.environ.get('FLAGPROF_GROUP_ID'),
                             proc_key=attachment.state['process'])
        self.period_ns = self.writer.config.heartbeat_period_ns
        self.old_hook = None
        self.thread = None
        self.dispatch_mode = None
        self.started = clock()

    def emit(self, action, *, kind='heartbeat', **data):
        if self.pid != os.getpid() or self.stop.is_set():
            return
        try:
            ok = self.writer.emit(dict(t='health', kind=kind,
                sev='error' if kind=='oom' else 'info', ts=clock(),
                detail=dict(scope='business', version=1, action=action,
                            identity=self.identity, started_ns=self.started, **data)))
            if ok is False:
                self.handle.state['diagnostic_status'] = 'writer_unavailable'
        except Exception as error:
            self.handle.state['diagnostic_status'] = 'unavailable:'+type(error).__name__

    def start(self):
        if self.handle.probe_manager.dispatch is None:
            try:
                from torch.utils._python_dispatch import TorchDispatchMode
            except ImportError:
                self.handle.state['diagnostic_dispatch']='unavailable'
            else:
                owner=self
                class Mode(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        try:
                            return func(*args, **(kwargs or {}))
                        except BaseException as error:
                            owner.exception(error)
                            raise
                        finally:
                            owner.progress()
                self.dispatch_mode=Mode()
                self.dispatch_mode.__enter__()
        self.old_hook = sys.excepthook
        def hook(typ, value, tb):
            self.exception(value)
            self.old_hook(typ, value, tb)
        self.hook = hook
        sys.excepthook = hook
        from flagprof.monitor.reference import read, bind
        from flagprof.store.manifest import atomic_json
        try:
            reference=read(self.handle.monitor_reference)
            reference['diagnostic_identity']=self.identity
            atomic_json(self.handle.monitor_reference,reference)
            bind(self.handle.monitor_reference,worker=self.identity['proc_key'])
        except (OSError,ValueError,KeyError) as error:
            self.handle.state['diagnostic_status']='registration_unavailable:'+str(error)
        self.emit('start', period_ns=self.period_ns)
        self.thread = threading.Thread(target=self._heartbeat, name='flagprof-diagnostic', daemon=True)
        self.thread.start()

    def _heartbeat(self):
        while not self.stop.is_set():
            with self.lock:
                active, calls = deepcopy(self.active), deepcopy(self.calls)
            self.emit('heartbeat', period_ns=self.period_ns, active=active, calls=calls)
            self.stop.wait(self.period_ns/1e9)

    def enter(self, token, name):
        with self.lock:
            if len(self.active) >= 1024:
                self.emit('gap', reason='active_scope_capacity')
                return
            self.active[token] = dict(name=name, thread=threading.get_ident(),
                since=clock(), last_progress=clock(), progress=0,
                eligible=name not in ('load', 'warmup') and not any(
                    r['thread']==threading.get_ident() and not r['eligible'] for r in self.active.values()))
        self.emit('enter', token=token, name=name)

    def progress(self):
        tid = threading.get_ident()
        with self.lock:
            now = clock()
            for row in self.active.values():
                if row['thread'] == tid:
                    row['progress'] += 1
                    row['last_progress'] = now

    def leave(self, token):
        self.progress()
        with self.lock:
            self.active.pop(token, None)
        self.emit('leave', token=token)

    def comm_enter(self, group, sequence, api, role, asynchronous):
        key = group+':'+str(sequence)
        with self.lock:
            if len(self.calls) >= 1024:
                self.emit('gap', reason='active_call_capacity')
                return None
            self.calls[key] = dict(group=group, sequence=sequence, api=api, role=role,
                asynchronous=asynchronous, since=clock(), members=[0,1])
            call = dict(self.calls[key])
        self.emit('comm_enter', key=key, call=call)
        return key

    def comm_leave(self, key, error):
        if key is None:
            return
        with self.lock:
            call = self.calls.pop(key, None)
        self.progress()
        self.emit('comm_return', key=key, call=call, error=error)

    def exception(self, error):
        # Match concrete runtime exception classes, not arbitrary RuntimeError text.
        torch = sys.modules.get('torch')
        classes = []
        for owner in (torch, getattr(torch, 'cuda', None), getattr(torch, 'npu', None)):
            cls = getattr(owner, 'OutOfMemoryError', None)
            if isinstance(cls, type):
                classes.append(cls)
        if not classes or not isinstance(error, tuple(classes)):
            return
        with self.lock:
            marker = '_flagprof_diagnostic_' + self.identity['run_id'].replace('-', '_')
            if getattr(error, marker, False):
                return
            setattr(error, marker, True)
            scopes = deepcopy(self.active)
        self.emit('oom', kind='oom', exception_type=type(error).__name__,
                  message=str(error)[:2048], active=scopes)

    def close(self):
        self.emit('finish')
        self.stop.set()
        if self.thread and self.thread.ident is not None:
            self.thread.join(timeout=2)
        if self.old_hook is not None and sys.excepthook is self.hook:
            sys.excepthook = self.old_hook
        if self.dispatch_mode is not None:
            self.dispatch_mode.__exit__(None,None,None)
        self.seen.clear()
