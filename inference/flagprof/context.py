"""Explicit scopes; no implicit thread, task, or fork propagation."""
from contextlib import contextmanager
from dataclasses import dataclass
import asyncio
import os
import threading
import weakref


@dataclass(eq=False)
class Scope:
    name: str
    corr: int
    seq: int = 0


class Context:
    def __init__(self, writer):
        self.writer = writer
        self.pid = os.getpid()
        self.closed = False
        self.local = threading.local()
        self.forward_count = 0
        self._forward_lock = threading.Lock()

    def probe_state(self):
        state = self._state()
        state.setdefault('frames', [])
        state.setdefault('modules', [])
        state.setdefault('root', None)
        return state

    def begin_forward(self, each_start, each_count):
        self._check()
        with self._forward_lock:
            index = self.forward_count
            self.forward_count += 1
        frame = dict(index=index, each=each_start <= index < each_start + each_count)
        self.probe_state()['root'] = frame
        return frame

    @contextmanager
    def probe_op(self, each, *, op_key=None):
        self._check()
        state = self.probe_state()
        parent = self.current_corr()
        corr = self.alloc_corr() if each else 0
        frame = dict(corr=corr, route=None, nested=bool(state['ops']),op_key=op_key,route_depth=0)
        scope = self.current_scope()
        fields = dict(corr=corr)
        if parent:
            fields['parent_corr'] = parent
        if scope:
            fields.update(scope=scope.corr, seq_in_scope=self.next_seq())
        state['ops'].append(corr)
        state['frames'].append(frame)
        try:
            yield fields, frame
        finally:
            if state['frames'][-1] is not frame:
                raise RuntimeError('probe frame ownership violated')
            state['frames'].pop()
            state['ops'].pop()

    def _state(self):
        if self.pid != os.getpid() or self.closed:
            return {"ops": [], "scopes": []}
        if not hasattr(self.local, 'tasks'):
            self.local.tasks = weakref.WeakKeyDictionary()
            self.local.sync = {"ops": [], "scopes": []}
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is None:
            return self.local.sync
        if task not in self.local.tasks:
            self.local.tasks[task] = {"ops": [], "scopes": []}
        return self.local.tasks[task]

    def _check(self):
        if self.pid != os.getpid() or self.closed:
            raise RuntimeError('context closed or inherited across fork')

    def alloc_corr(self):
        self._check()
        return self.writer.alloc_corr()

    def current_corr(self):
        ops = self._state()['ops']
        return ops[0] if ops else 0

    def current_scope(self):
        scopes = self._state()['scopes']
        return scopes[-1] if scopes else None

    def push_scope(self, name):
        self._check()
        token = Scope(name, self.alloc_corr())
        self._state()['scopes'].append(token)
        return token

    def pop_scope(self, token):
        self._check()
        stack = self._state()['scopes']
        if not stack or stack[-1] is not token:
            raise ValueError('scope must be popped by its owner in LIFO order')
        stack.pop()

    def next_seq(self):
        self._check()
        scope = self.current_scope()
        if scope is None:
            return None
        value = scope.seq
        scope.seq += 1
        return value

    @contextmanager
    def scope(self, name):
        token = self.push_scope(name)
        try:
            yield token
        finally:
            self.pop_scope(token)

    @contextmanager
    def op(self):
        parent = self.current_corr()
        corr = self.alloc_corr()
        stack = self._state()['ops']
        stack.append(corr)
        try:
            scope = self.current_scope()
            fields = dict(corr=corr)
            if parent:
                fields['parent_corr'] = parent
            if scope:
                fields.update(scope=scope.corr, seq_in_scope=self.next_seq())
            yield fields
        finally:
            if not stack or stack[-1] != corr:
                raise RuntimeError('operation stack ownership violated')
            stack.pop()

    def close(self):
        if self.pid == os.getpid():
            self.closed = True


def _context():
    from .runtime import current_attachment
    handle = current_attachment()
    if handle is None:
        raise RuntimeError('no current attachment')
    return handle.context


def alloc_corr(): return _context().alloc_corr()
def current_corr():
    from .runtime import current_attachment
    handle = current_attachment()
    return handle.context.current_corr() if handle else 0
def push_scope(name): return _context().push_scope(name)
def pop_scope(token): return _context().pop_scope(token)
def current_scope():
    from .runtime import current_attachment
    handle = current_attachment()
    return handle.context.current_scope() if handle else None
