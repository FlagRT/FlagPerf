"""Registered proxy identity, shared waiting deadlines and read-only status."""
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import os
from pathlib import Path
import signal
import time

from .manifest import load_json, process_alive, writer_fragments

PROXY_WAIT_SECONDS = 10
_DEADLINE = ContextVar('flagprof_proxy_deadline', default=None)


def attachments(run):
    paths = list((Path(run)/'attachments').glob('*/state.json'))
    if len(paths) > 16: raise ValueError('attachment budget')
    result = []
    for path in paths:
        value = load_json(path)
        if not isinstance(value, dict): raise ValueError('invalid attachment state')
        result.append(value)
    return result


def registered_proxies(run, states=None):
    states = attachments(run) if states is None else states
    fragments = {f['writer_id']: f for f in writer_fragments(Path(run))}
    result = []
    for state in states:
        proxy = state.get('writer_proxy')
        if not proxy: continue
        fragment = fragments.get(proxy.get('writer_id'))
        if fragment is None or fragment.get('proc_key') != state.get('process'):
            raise ValueError('proxy owner registration mismatch')
        if not isinstance(proxy.get('instance'), int) or not isinstance(proxy.get('process'), dict):
            raise ValueError('invalid proxy identity')
        result.append(dict(proxy, attachment_id=state['id']))
    return result


def effective_writer_state(fragment, proxies):
    if fragment['state'] == 'closed': return 'closed'
    if process_alive(fragment.get('proc_key')) is False:
        matching = [p for p in proxies if p['writer_id'] == fragment['writer_id']]
        if not any(process_alive(p['process']) is not False for p in matching): return 'lost'
    return fragment['state']


@contextmanager
def deadline_scope(deadline=None):
    previous = _DEADLINE.get()
    chosen = time.monotonic()+PROXY_WAIT_SECONDS if deadline is None else deadline
    chosen = min(chosen, previous) if previous is not None else chosen
    token = _DEADLINE.set(chosen)
    try: yield chosen
    finally: _DEADLINE.reset(token)


def wait_proxies(run, deadline):
    while any(process_alive(p['process']) is not False for p in registered_proxies(run)):
        left = deadline-time.monotonic()
        if left <= 0: raise TimeoutError('proxy_wait_timeout')
        time.sleep(min(.02, left))


@contextmanager
def lock_until(lock, deadline):
    while True:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            left = deadline-time.monotonic()
            if left <= 0: raise TimeoutError('finalize_lock_timeout')
            time.sleep(min(.01, left))
    try: yield
    finally: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def terminate_proxies(run):
    """Only record calls this; pidfds prevent killing a recycled PID."""
    for proxy in registered_proxies(run):
        key = proxy['process']
        if process_alive(key) is not True: continue
        if key['pid'] == os.getpid(): raise ValueError('proxy cannot be the wrapper')
        identity = load_json(Path(run)/'attachments'/proxy['attachment_id']/'proxy-identity.json')
        if identity['instance'] != proxy['instance'] or identity['fragment']['writer_id'] != proxy['writer_id']:
            raise ValueError('proxy instance registration mismatch')
        fd = os.pidfd_open(key['pid'])
        try:
            if process_alive(key) is not True: continue
            signal.pidfd_send_signal(fd, signal.SIGKILL)
        except ProcessLookupError: pass
        finally: os.close(fd)
    until = time.monotonic()+1
    while any(process_alive(p['process']) is True for p in registered_proxies(run)):
        if time.monotonic() >= until: raise TimeoutError('proxy_cleanup_timeout')
        time.sleep(.01)


def record_wait(run, *, deadline=None):
    chosen = min(time.monotonic()+PROXY_WAIT_SECONDS, deadline) if deadline is not None else time.monotonic()+PROXY_WAIT_SECONDS
    try:
        wait_proxies(run, chosen)
        return chosen, False
    except TimeoutError:
        terminate_proxies(run)
        return chosen, True
