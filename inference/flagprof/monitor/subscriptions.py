"""Bounded same-user Unix socket registration; targets never poll the agent."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
from flagprof.store.manifest import atomic_json
from flagprof.telemetry.contract import raw_clock
from .session import control_json, edit, load, locked

MAX_MESSAGE = 65536


@contextmanager
def socket_path(root):
    # /proc fd paths avoid AF_UNIX's 108-byte pathname limit and work through
    # container bind mounts without baking the host root path into a reference.
    fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield '/proc/self/fd/'+str(fd)+'/agent.sock'
    finally:
        os.close(fd)


def receive(sock):
    data = bytearray()
    while not data.endswith(b'\n'):
        part = sock.recv(min(4096, MAX_MESSAGE+1-len(data)))
        if not part: raise ValueError('incomplete agent message')
        data.extend(part)
        if len(data) > MAX_MESSAGE: raise ValueError('agent message limit')
    return json.loads(data)


def request(root, message):
    with socket.socket(socket.AF_UNIX) as client, socket_path(root) as path:
        client.settimeout(1)
        client.connect(path)
        payload = json.dumps(message, allow_nan=False).encode()+b'\n'
        if len(payload) > MAX_MESSAGE: raise ValueError('agent message limit')
        client.sendall(payload)
        result = receive(client)
        if result.get('status') in ('error', 'capacity', 'unavailable'):
            raise ValueError('monitor '+result['status']+': '+result.get('reason','unknown'))
        return result


def target_id(key):
    fields = {name:key[name] for name in ('boot_id','pid_ns','pid','proc_start_ticks')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:32]


class Server:
    def __init__(self, root):
        self.root = Path(root)
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(.02)
        path = self.root/'agent.sock'
        if path.exists(): path.unlink()  # Caller already holds the root lease.
        with socket_path(self.root) as address: self.socket.bind(address)
        path.chmod(0o600)
        self.socket.listen(32)

    def step(self):
        try: connection, _ = self.socket.accept()
        except socket.timeout: return
        with connection:
            connection.settimeout(.2)
            try:
                _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if uid != os.getuid(): raise ValueError('agent peer user mismatch')
                result = self.handle(receive(connection))
            except (OSError, ValueError, KeyError, TypeError) as error:
                result = dict(status='error', reason=str(error))
            try: connection.sendall(json.dumps(result).encode()+b'\n')
            except OSError: pass

    def handle(self, message):
        action = message.get('action')
        with edit(self.root) as state:
            if message.get('session_id') not in (None,state['id']):
                raise ValueError('session identity mismatch')
            if action == 'hello':
                if state['status'] != 'running':
                    return dict(status='unavailable', reason=state['status'])
                return dict(status='available', session_id=state['id'], boot_id=state['boot_id'],
                            inventory=state['inventory'])
            if action not in ('register','finish','exit'):
                raise ValueError('unknown agent action')
            data = message['reference']; key = data['worker_proc_key']
            if key['boot_id'] != state['boot_id']: raise ValueError('target boot mismatch')
            ident = target_id(key)
            client_path=Path(message['path'])
            client_root=(client_path.parent/(data.get('session_path') or data.get('root') or '')).resolve()
            reference_path=os.path.relpath(client_path,client_root if data.get('session_path') or data.get('root') else self.root)
            active = self.root/'targets/active'; history = self.root/'targets/history'
            active.mkdir(parents=True, exist_ok=True); history.mkdir(parents=True, exist_ok=True)
            path = active/(ident+'.json'); archive = history/path.name
            previous = json.loads((path if path.exists() else archive).read_text()) if path.exists() or archive.exists() else {}
            if action == 'register':
                if state['status'] != 'running': return dict(status='unavailable',reason=state['status'])
                if previous.get('window',[None,None])[1] is not None:
                    return dict(status='ended', target_id=ident)
                if not path.exists() and len(list(active.glob('*.json'))) >= state['config']['max_targets']:
                    return dict(status='capacity', reason='active_target_limit')
                if not archive.exists() and not path.exists() and len(list(history.glob('*.json'))) >= state['config']['max_history']:
                    return dict(status='capacity', reason='history_limit_requires_release')
                data = dict(previous, **data)
                data['target_id'] = ident
                refs = set(previous.get('references', [])); refs.add(reference_path)
                if len(refs) > 16: raise ValueError('target reference limit')
                data['references'] = sorted(refs)
                data['reference_base']='session'
                control_json(self.root,state,path,data)
                state['pins'].setdefault('target:'+ident, dict(window=data['window'], target_id=ident))
            else:
                if not previous: previous = dict(data, target_id=ident, references=[reference_path],reference_base='session')
                previous['window'][1] = previous['window'][1] or raw_clock()
                if action == 'exit':
                    receipt = message['exit']
                    if receipt.get('source') != 'subprocess.wait' or type(receipt.get('returncode')) is not int:
                        raise ValueError('exit requires launcher wait receipt')
                    if previous.get('exit') and previous['exit'] != receipt:
                        raise ValueError('conflicting exit receipt')
                    if previous.get('exit') == receipt:
                        return dict(status='ended',target_id=ident)
                    previous['exit'] = receipt
                    pending = self.root/'notifications'; pending.mkdir(exist_ok=True)
                    notification = pending/(ident+'.json')
                    if not notification.exists() and len(list(pending.glob('*.json'))) >= state['config']['queue_capacity']:
                        state.update(status='capacity',worker_error='exit_notification_queue')
                    else:
                        control_json(self.root,state,notification,dict(t='health',ts=raw_clock(),kind='process_exit',
                            sev='info' if receipt['returncode']==0 else 'error',detail=dict(
                                entity='proc:{pid_ns}:{pid}:{proc_start_ticks}'.format(**key),
                                target_id=ident,returncode=receipt['returncode'],source='subprocess.wait',
                                diagnostic_identity=previous.get('diagnostic_identity'),
                                termination_reason=receipt.get('termination_reason','target_exit'))))
                previous.setdefault('process_coverage','worker_not_observed')
                if action=='finish' and not previous.get('exit'):
                    pending=self.root/'notifications';pending.mkdir(exist_ok=True)
                    if len(list(pending.glob('*.json'))) < state['config']['queue_capacity']:
                        control_json(self.root,state,pending/(ident+'.json'),dict(t='health',ts=raw_clock(),kind='process_exit',sev='info',
                            detail=dict(entity='proc:{pid_ns}:{pid}:{proc_start_ticks}'.format(**key),
                                        target_id=ident,source='registration_end',exit_status='unknown')))
                control_json(self.root,state,archive,previous)
                path.unlink(missing_ok=True)
                state['pins'].setdefault('target:'+ident, dict(target_id=ident))['window'] = previous['window']
            return dict(status='registered' if action=='register' else 'ended', target_id=ident)

    def close(self):
        self.socket.close()
        (self.root/'agent.sock').unlink(missing_ok=True)
