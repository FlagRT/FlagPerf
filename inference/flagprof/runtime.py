# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Own a Writer and official session; importing remains a separate process."""
import os
import atexit
from pathlib import Path
import threading
import uuid
import weakref
from .context import Context
from .registry import Registry
from .store._online_writer import online_writer
from .store.manifest import atomic_json, proc_key
from .vendors.session import ProfilerSession, SessionConfig

_handles = []
_lock = threading.RLock()


def _fork():
    global _handles, _lock
    _handles = []
    _lock = threading.RLock()


os.register_at_fork(after_in_child=_fork)


def current_attachment():
    with _lock:
        live = [ref() for ref in _handles if ref() is not None and ref().pid == os.getpid() and not ref().closed]
        return live[-1] if live else None


def configuration(preset='perf', **options):
    from .probes.config import resolve
    options, _ = resolve(preset, options)
    values = dict(with_stack=preset == 'modules', with_modules=preset == 'modules')
    values.update(options)
    return SessionConfig(**values)


class Attachment:
    def __init__(self, run, vendor, *, preset='perf', config=None, registry=None, rank=0,
                 node_index=None, identity=None, backend_options=None, log_reader=None,capture_mode='final',capture_filesystem='ext4'):
        registry = registry or Registry()
        descriptor = registry.describe(vendor)
        if descriptor['kind'] != 'vendor':
            raise ValueError('attachment requires a vendor')
        if capture_mode not in ('final','windowed'):raise ValueError('unknown capture mode')
        if capture_mode=='final' and set(config or {}) & {'active','windows','log_files','invalid_policy'}:
            raise ValueError('window settings require capture_mode=windowed')
        if capture_mode == 'final' and preset in ('perf', 'modules'):
            from .offline.workspace import check_capture_disk
            check_capture_disk(run,descriptor['format_vendor'],capture_filesystem)
        self.capture_mode=capture_mode
        from .probes.config import resolve
        session_options, probe_options = resolve(preset, config)
        official = preset in ('perf', 'modules')
        config = configuration(preset, **session_options)
        self.pid = os.getpid()
        self.closed = False
        self.errors = []
        self.writer = self.context = self.session = None
        self.probe_manager = None
        self.run = Path(run).resolve()
        self.id = uuid.uuid4().hex
        self.path = self.run/'attachments'/self.id
        self.capture = self.run/'capture-staging'/self.id
        self.state = dict(id=self.id, process=proc_key(), vendor=vendor, format_vendor=descriptor['format_vendor'],
                          preset=preset, capture=str(self.capture), capture_relative=str(self.capture.relative_to(self.run)),
                          capture_mode=capture_mode,status='starting',
                          capture_filesystem=capture_filesystem,
                          capabilities=dict(enabled=['official_session', 'writer', 'explicit_context'],
                              unavailable=['module_probe', 'dispatch_probe', 'route_probe', 'compile_probe', 'eager_module_tree']),
                          import_status='pending_owner_exit')
        # Writer duplicate-run guard errors remain explicit, before any SDK action.
        # Normal atexit follows concurrent.futures shutdown. No deferred
        # compression jobs may be submitted then; offline import can compress.
        self.state['writer_configuration'] = dict(compress=False, reason='safe ordinary atexit handoff')
        self.state['writer_mode'] = 'process_proxy'
        def handshake(writer):
            self.state['writer_proxy'] = writer.registration()
            self._save()
        try:
            self.writer = online_writer(self.run, rank=rank, node_index=node_index,
                                        attachment_dir=self.path, handshake_callback=handshake)
            self.context = Context(self.writer)
        except Exception as error:
            self.state.update(status='failed', writer_failure=dict(reason='startup_failed:'+repr(error), ts=__import__('time').monotonic_ns()))
            self.errors.append('writer startup: '+repr(error))
            self._save()
            raise
        try:
            backend_options=dict(backend_options or {})
            if official:
                if capture_mode=='final' and descriptor['format_vendor']=='ascend':backend_options.setdefault('data_simplification',False)
                backend, implementation = registry.create(vendor, **backend_options)
                self.state['descriptor'] = implementation
                from .vendors.final_session import FinalSession
                self.session = (FinalSession if capture_mode=='final' else ProfilerSession)(backend, self.capture, config, identity=identity, log_reader=log_reader)
                if capture_mode=='final':self.session.meta['source_run_id']=self.writer.run_id
                self.state['status'] = 'running' if self.session.start() else 'failed'
            else:
                if descriptor['format_vendor']=='thead':
                    from .vendors.thead.session import hardware_identity
                if descriptor['format_vendor']=='kunlunxin':
                    from .vendors.kunlunxin.session import hardware_identity
                    self.state['hardware_identity']=hardware_identity()
                self.state.update(status='running', collector_status='not_requested', capture_status='not_requested')
                self.state['capabilities']['enabled'].remove('official_session')
            from .probes.manager import ProbeManager
            self.probe_manager = ProbeManager(self.context, probe_options, self.state, official=official,state_callback=self._save)
            self.probe_manager.install()
            self.state['capabilities']['unavailable'] = [name+'_probe' for name, value in self.state['probes'].items() if value['status']=='unavailable']
            self.state['capabilities']['enabled'].extend(name+'_probe' for name, value in self.state['probes'].items() if value['status']=='available')
            self._save()
            if self.state['status'] == 'failed':
                self.close()
        except Exception as error:
            self.errors.append(type(error).__name__ + ': ' + str(error))
            self.state['status'] = 'failed'
            self.close()
        if not self.closed:
            atexit.register(self._exit_close)
            with _lock:
                _handles[:] = [ref for ref in _handles if ref() is not None and not ref().closed]
                _handles.append(weakref.ref(self))

    def _save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.state['errors'] = list(self.errors) + (self.probe_manager.errors if self.probe_manager else [])
        if self.probe_manager:
            probes=self.state.get('probes',{})
            self.state['capabilities']['enabled']=[v for v in self.state['capabilities']['enabled'] if not v.endswith('_probe')]+[name+'_probe' for name,value in probes.items() if value['status']=='available']
            self.state['capabilities']['unavailable']=[name+'_probe' for name,value in probes.items() if value['status']=='unavailable']
        if self.writer:
            failure = self.writer.metrics().get('writer_failure')
            if failure:
                self.state['writer_failure'] = dict(reason=failure, ts=__import__('time').monotonic_ns())
        if self.session:
            self.state['collector_status'] = self.session.status
            self.state['collector_errors'] = list(self.session.errors)
        atomic_json(self.path/'state.json', self.state)

    def step(self):
        if self.pid != os.getpid() or self.closed:
            return False
        if self.session is None:
            return True
        result = self.session.step()
        if self.session.status == 'failed' or self.writer.metrics().get('writer_failure'):
            self.state['status'] = 'failed'
            self._save()
        return result

    def _exit_close(self):
        # Plain atexit cannot distinguish normal exit from an unhandled error.
        return self.close(normal=None,returncode=None)

    def close(self,*,normal=True,returncode=0):
        if self.pid != os.getpid():
            return dict(status='foreign_process')
        if self.closed:
            return self.state
        self.closed = True
        atexit.unregister(self._exit_close)
        try:
            if self.probe_manager:
                self.probe_manager.close()
            if self.session:
                if self.capture_mode=='final':self.session.close(normal=normal,returncode=returncode)
                else:self.session.close()
        except Exception as error:
            self.errors.append('session close: ' + repr(error))
        finally:
            if self.context:
                self.context.close()
            if self.writer:
                try:
                    self.writer.close()
                    self.errors.extend(self.writer.errors)
                except Exception as error:
                    self.errors.append('writer close: ' + repr(error))
            self.state['status'] = ('closed' if (self.session and self.session.status == 'closed' or self.state.get('capture_status')=='not_requested')
                                    and not self.errors else 'failed' if self.errors or
                                    not self.session or self.session.status == 'failed' else 'partial')
            try:
                self._save()
            except OSError as error:
                self.errors.append('state save: ' + repr(error))
        return self.state

    def register_model(self, model):
        if self.pid != os.getpid() or self.closed:
            raise RuntimeError('attachment closed or inherited across fork')
        if self.probe_manager:
            self.probe_manager.register_model(model)
            self._save()
        return model

    def bind_tensor_inputs(self, sample_ids, inputs, *, identity=None):
        if identity is not None:
            self.state['oracle_identity'] = dict(identity, boundaries=self.state.get('tensor_boundaries',[]))
            self._save()
        if self.probe_manager and self.probe_manager.tensor:
            self.probe_manager.tensor.bind(sample_ids, inputs)

    def bind_generation_step(self, sample_ids, inputs, step):
        """Bind a complete model advance to its sample IDs and step span."""
        self.context.probe_state()['generation'] = dict(step)
        self.bind_tensor_inputs(sample_ids, inputs)

    def record_tensor(self, boundary, tensor, *, step=None):
        if self.probe_manager and self.probe_manager.tensor:
            probe=self.probe_manager.tensor
            probe.safe(probe.capture, '<root>', boundary, tensor,
                       **(dict(counts=[step]*len(probe.samples)) if step is not None else {}))

    def __enter__(self): return self
    def __exit__(self, *exc):
        normal=exc[0] is None or (exc[0] is SystemExit and exc[1].code in (None,0))
        self.close(normal=normal,returncode=0 if normal else 1)


def attach(run, vendor, **options):
    return Attachment(run, vendor, **options)
