import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .reference import bind, relocate, resolve_reference
from flagprof.store.manifest import atomic_json, proc_key


class Monitor:
    def __init__(self, root, reference, vendor, options, *, preset=None):
        from . import DEFAULT_MONITOR
        self.enabled = options.get('monitor', 'on' if preset == 'monitor' else DEFAULT_MONITOR) == 'on'
        self.path = Path(reference)
        self.child = None
        self.log = None
        self.closed = False
        self.owner = False
        self.root = Path(options.get('monitor_root') or root).resolve()
        self.vendor = vendor
        self.interval = options.get('monitor_interval_ms', 1000)
        self.topology = options.get('monitor_topology') or ('host' if vendor == 'ascend' else 'local')
        if not self.enabled:
            return
        shared = os.environ.get('FLAGPROF_MONITOR_REFERENCE')
        try:
            if shared:
                relocate(shared, self.path)
                self.root = (self.path.parent/json.loads(self.path.read_text())['root']).resolve()
            else:
                self.owner = True
                bind(self.path, root=self.root, topology=self.topology)
                if self.topology == 'agent':
                    from .subscriptions import request
                    response = request(self.root,dict(action='hello'))
                    data = json.loads(self.path.read_text())
                    data.update(version=2,session_id=response['session_id'],
                        session_path=data['root'],status='running',monitor_run_status='running')
                    atomic_json(self.path,data)
                if self.topology == 'local':
                    from flagprof.store import create_run
                    monitor = create_run(self.root/'monitor', identity=dict(kind='monitor', vendor=vendor))
                    bind(self.path, monitor=monitor)
                    self.log = (monitor/'logs/sidecar.log').open('wb')
                    env = dict(os.environ); env.pop('FLAGPROF_RECORD_CONFIG', None)
                    self.child = subprocess.Popen([sys.executable, '-B', '-S', '-m', 'flagprof.monitor',
                        '--root', str(self.root), '--run', str(monitor), '--vendor', vendor,
                        '--interval-ms', str(self.interval)], env=env, stdout=self.log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic()+max(2, 2*self.interval/1000)
                    while not (monitor/'first-sample.json').exists() and self.child.poll() is None and time.monotonic()<deadline:
                        time.sleep(.02)
                    data = json.loads(self.path.read_text())
                    data['pre_sample'] = 'available' if (monitor/'first-sample.json').exists() else 'no_pre_sample'
                    atomic_json(self.path, data)
            from flagprof.vendors.fingerprint import parser_identity
            data = json.loads(self.path.read_text())
            data['target_source_identity'] = parser_identity('telemetry','1')
            atomic_json(self.path, data)
            atexit.register(self.close)
        except (OSError, ValueError) as error:
            self.error(error)

    def error(self, error):
        try:
            data = json.loads(self.path.read_text())
            data.update(status='failed', errors=data.get('errors', [])+[str(error)])
            atomic_json(self.path, data)
        except (OSError, ValueError):
            pass

    def environment(self, env):
        if self.enabled:
            env['FLAGPROF_MONITOR_REFERENCE'] = str(self.path.resolve())

    def target_exit(self, receipt, worker=None):
        """Only the controller's wait result can establish a target exit code."""
        if not self.enabled or self.topology != 'agent': return
        try:
            from .subscriptions import request
            data = json.loads(self.path.read_text())
            if worker: data['worker_proc_key'] = worker
            if not data.get('worker_proc_key') or not data.get('session_id'): return
            request(self.root,dict(action='exit',session_id=data['session_id'],reference=data,
                path=str(self.path.resolve()),exit=receipt))
        except (OSError,ValueError) as error:
            self.error(error)

    def close(self):
        if not self.enabled or self.closed:
            return
        self.closed = True
        try:
            # Target interval ends before sidecar teardown.
            data = bind(self.path, finish=True)
            started = time.monotonic(); timed_out = False
            if self.child:
                self.child.terminate()
                try:
                    self.child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    timed_out = True
            data, _, _ = resolve_reference(self.path)
            data['reclaim'] = dict(wait_ms=(time.monotonic()-started)*1000, timed_out=timed_out)
            atomic_json(self.path, data)
            if self.log:
                self.log.close()
        except (OSError, ValueError) as error:
            self.error(error)
