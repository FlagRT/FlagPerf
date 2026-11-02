# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Standard-library official-profiler session orchestration.

Capture files are independent of run/store layout until T8 Gate 1. The session
is the only manifest writer. A reader may infer lost windows but never repair
the manifest. Errors disable collection, not the caller's forward execution.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Callable, Mapping, Protocol
import uuid
from flagprof.store.manifest import atomic_json, environment_identity, json_bytes, proc_key
from flagprof.store.captures import inspect_capture
from .logs import LogObserver, validate_logs

@dataclass(frozen=True)
class SessionConfig:
    active: int = 5
    windows: int = 2
    record_shapes: bool = True
    with_stack: bool = False
    profile_memory: bool = False
    with_flops: bool = False
    with_modules: bool = False
    invalid_policy: str = 'stop'
    log_files: tuple[str, ...] = ()
    error_patterns: tuple[str,...] = (r'Traceback \(most recent call last\)', r'(?i)\b(error|failed|fatal)\b')

    def __post_init__(self) -> None:
        if type(self.active) is not int or type(self.windows) is not int or self.active<1 or self.windows<1:
            raise ValueError('active and windows must be positive integers')
        for name in ('record_shapes','with_stack','profile_memory','with_flops','with_modules'):
            if type(getattr(self,name)) is not bool:
                raise ValueError(name+' must be bool')
        if self.invalid_policy not in ('stop','continue'):
            raise ValueError('invalid_policy must be stop or continue')
        if isinstance(self.log_files, str) or not all(isinstance(p,str) and p for p in self.log_files):
            raise ValueError('log_files must contain nonempty paths')
        object.__setattr__(self, 'log_files', tuple(self.log_files))
        for pattern in self.error_patterns:
            re.compile(pattern)

class ProfilerBackend(Protocol):
    activity_source: str
    def conflict(self) -> str: ...
    def identity(self, config: SessionConfig) -> Mapping[str,Any]: ...
    def set_destination(self, path: Path) -> None: ...
    def create(self, config: SessionConfig, on_ready: Callable[[Any],None]) -> Any: ...
    def export(self, profiler: Any, path: Path) -> None: ...
    def validate(self, path: Path) -> Mapping[str,Any]: ...


def requested_configuration(config: SessionConfig, activities: list[str]) -> dict[str, Any]:
    """Options passed explicitly to the public profiler API."""
    return dict(activities=activities, record_shapes=config.record_shapes,
                with_stack=config.with_stack, profile_memory=config.profile_memory,
                with_flops=config.with_flops, with_modules=config.with_modules,
                schedule=dict(wait=0,warmup=0,active=config.active,repeat=config.windows))

_LOCK=threading.Lock()
_OWNERS: set[str]=set()

def _after_fork() -> None:
    global _LOCK,_OWNERS
    _LOCK=threading.Lock();_OWNERS=set()

os.register_at_fork(after_in_child=_after_fork)

def files_identity(path: Path) -> list[dict[str,Any]]:
    output=[]
    for file in sorted(path.rglob('*')):
        if file.is_symlink():
            raise ValueError('capture artifacts must not be symlinks')
        if not file.is_file():
            continue
        data=file.read_bytes()
        with file.open('rb') as stream:
            os.fsync(stream.fileno())
        output.append(dict(path=str(file.relative_to(path)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    return output

def seal_window(path: Path, validate: Callable[[Path],Mapping[str,Any]]) -> tuple[Path,list[dict],dict]:
    """Publish a verified copy; the SDK keeps owning only its working directory.

    A crash or IO failure leaves an unregistered staging directory and an
    exporting window. Never hardlink SDK files into the immutable snapshot.
    """
    before=files_identity(path)
    staging=path.parent/('.sealing-'+path.name+'-'+uuid.uuid4().hex)
    sealed=path.parent/('sealed-'+path.name)
    shutil.copytree(path,staging,symlinks=True)
    copied=files_identity(staging)
    if copied!=before:
        raise RuntimeError('capture changed while copying window')
    validity=dict(validate(staging))
    if files_identity(staging)!=before or files_identity(path)!=before:
        raise RuntimeError('capture changed while validating window')
    for directory in [p for p in staging.rglob('*') if p.is_dir()]+[staging]:
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    staging.rename(sealed)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return sealed,copied,validity

def validate_chrome(path: Path, pattern: str, device_record: Callable[[dict,dict],bool]) -> dict[str,Any]:
    matches=sorted(path.glob(pattern))
    if not matches:
        return dict(valid=False,reasons=['expected_trace_missing'],traces=[])
    counts=[]
    for file in matches:
        try:
            data=json.loads(file.read_bytes())
            events=data if isinstance(data,list) else data['traceEvents']
            if not isinstance(events,list) or not all(isinstance(e,dict) for e in events):
                raise ValueError('traceEvents is not an array of records')
            processes={e.get('pid'):e.get('args',{}).get('name') for e in events if e.get('name')=='process_name'}
            count=sum(device_record(e,processes) for e in events)
            if count==0:
                return dict(valid=False,reasons=['empty_device_track'],traces=counts)
            times=[e['ts'] for e in events if isinstance(e.get('ts'),(int,float))]
            ends=[e['ts']+(e.get('dur') or 0) for e in events if isinstance(e.get('ts'),(int,float))]
            counts.append(dict(path=str(file.relative_to(path)),records=len(events),device_records=count,
                               source_clock=dict(domain='chrome:'+hashlib.sha256(file.read_bytes()).hexdigest(),
                                                 unit='us',start=min(times) if times else None,
                                                 end=max(ends) if ends else None,clock_valid=False)))
        except (OSError,ValueError,TypeError,KeyError) as error:
            return dict(valid=False,reasons=['unparseable_trace: '+str(error)],traces=counts)
    return dict(valid=True,reasons=[],traces=counts)


class ProfilerSession:
    def __init__(self, backend: ProfilerBackend, directory: str | Path, config: SessionConfig | None = None,
                 identity: Mapping[str,Any] | None = None, *, log_reader: Callable[[],str] | None = None) -> None:
        self.backend=backend;self.directory=Path(directory);self.config=config or SessionConfig()
        self._identity=dict(identity or {})
        self._logs=LogObserver(self.config.log_files,self.config.error_patterns,log_reader)
        self.status='new';self.errors: list[str]=[];self._profiler=None;self._owns=False;self._pid=os.getpid()
        self._steps=0;self._window=None;self._closing=False;self._callback_error=None
        self.manifest: dict[str,Any]={}

    def _save(self) -> None:
        self.manifest.update(status=self.status,errors=list(self.errors))
        atomic_json(self.directory/'session.json',self.manifest)

    def _open_window(self) -> None:
        index=len(self.manifest['windows']);path=self.directory/f'window-{index:04d}'
        path.mkdir()
        self._window=dict(capture_id=uuid.uuid4().hex,index=index,path=path.name,state='collecting',
            step_begin=index*self.config.active,step_end=None,clock_valid=False,
            source_clocks=[],orchestration_clock=dict(domain='host_monotonic_ns',start=time.monotonic_ns(),end=None),artifacts=[])
        self.manifest['windows'].append(self._window)
        self.backend.set_destination(path)
        self._save()

    def _ready(self, profiler: Any) -> None:
        if self._window is None or self._window['state']!='collecting':
            return
        window=self._window;started=time.monotonic_ns()
        try:
            self._ready_work(profiler)
        finally:
            window['callback_work_ns']=time.monotonic_ns()-started
            window['callback_timing_scope']='export, snapshot copy, validation, artifact hash/fsync, manifest seal and next-window setup; excludes final timing-field publication'
            try:self._save()
            except Exception as error:
                self._callback_error=error
                raise

    def _ready_work(self, profiler: Any) -> None:
        if self._window is None or self._window['state']!='collecting':
            return
        try:
            window=self._window;path=self.directory/window['path']
            window['step_end']=self._steps
            window['orchestration_clock']['end']=time.monotonic_ns()
            window['state']='exporting';self._save()
            started=time.monotonic_ns();self.backend.export(profiler,path)
            window['export_ns']=time.monotonic_ns()-started
            started=time.monotonic_ns();sealed,artifacts,validity=seal_window(path,self.backend.validate)
            window['log_check']=self._logs.check()
            if window['log_check']['status']=='error':
                validity.update(valid=False,reasons=list(validity.get('reasons',[]))+['collector_log_error'])
            if self._steps-window['step_begin']!=self.config.active:
                validity.update(valid=False,reasons=list(validity.get('reasons',[]))+['incomplete_step_window'])
            window['working_path']=path.name
            window['path']=sealed.name
            window['artifacts']=artifacts
            window['source_clocks']=[t['source_clock'] for t in validity.get('traces',[]) if 'source_clock' in t]
            window['validity']=validity;window['state']='valid' if validity['valid'] else 'invalid'
            window['validation_ns']=time.monotonic_ns()-started
            self._save()  # durable before official profiler can start its next window
            if not validity['valid'] and self.config.invalid_policy=='stop':
                raise RuntimeError('invalid capture window: '+repr(validity['reasons']))
            if not self._closing and len(self.manifest['windows'])<self.config.windows:
                self._open_window()
        except Exception as error:
            self._callback_error=error
            raise  # SDK may suppress this; step() checks _callback_error too.

    def _fail(self, error: Exception) -> None:
        self.errors.append(type(error).__name__+': '+str(error));self.status='failed'
        if self.manifest:
            try:self._save()
            except OSError:pass

    def start(self) -> bool:
        if self.status!='new' or os.getpid()!=self._pid:
            return False
        with _LOCK:
            if self.backend.activity_source in _OWNERS:
                self._fail(RuntimeError('profiler session already owns this activity source'))
                return False
            _OWNERS.add(self.backend.activity_source);self._owns=True
        try:
            conflict=self.backend.conflict()
            if conflict=='conflict':
                raise RuntimeError('user profiler is already active')
            self.directory.mkdir(parents=True,exist_ok=False)
            identity=dict(self.backend.identity(self.config))
            prerequisites=identity.get('collector_prerequisites',[])
            if prerequisites:
                raise RuntimeError('unapproved collector prerequisites: '+repr(prerequisites))
            environment,redacted=environment_identity()
            cfg=dict(asdict(self.config),schedule=dict(wait=0,warmup=0,active=self.config.active,repeat=self.config.windows))
            self.manifest=dict(version='t8-capture-draft-1',session_id=uuid.uuid4().hex,process=proc_key(),
                activity_source=self.backend.activity_source,conflict_detection=conflict,
                identity=identity,execution=self._identity,environment=environment,redacted_env=redacted,
                config=cfg,config_sha256=hashlib.sha256(json_bytes(cfg)).hexdigest(),windows=[])
            self._logs.start()
            self.status='running';self._open_window()
            self._profiler=self.backend.create(self.config,self._ready)
            self._profiler.start()
            return True
        except Exception as error:
            self._fail(error);self.close();return False

    def step(self) -> bool:
        if self.status!='running' or os.getpid()!=self._pid:
            return False
        if self._steps>=self.config.active*self.config.windows:
            return False
        try:
            self._steps+=1
            self._profiler.step()
            if self._callback_error is not None:
                raise self._callback_error
            if self._steps%self.config.active==0:
                completed=self.manifest['windows'][self._steps//self.config.active-1]
                if completed['state']!='valid' and not (self.config.invalid_policy=='continue' and completed['state']=='invalid'):
                    raise RuntimeError('official profiler did not export scheduled window')
            return True
        except Exception as error:
            self._fail(error);self.close();return False

    def close(self) -> dict[str,Any]:
        if os.getpid()!=self._pid:
            return dict(status='foreign_process',errors=['inherited session cannot be closed by child'])
        if self._closing:
            return self.manifest or dict(status=self.status,errors=self.errors)
        self._closing=True
        if self.status=='new':
            self.status='closed'
        try:
            if self._profiler is not None:
                self._profiler.stop()
                if self._callback_error is not None and self.status!='failed':
                    self._fail(self._callback_error)
        except Exception as error:
            self._fail(error)
        finally:
            if self._owns:
                with _LOCK:_OWNERS.discard(self.backend.activity_source)
                self._owns=False
            if self.manifest:
                self.manifest['log_tail']=self._logs.check(final=True)
                if self.status!='failed':
                    self.status='closed' if len(self.manifest['windows'])==self.config.windows and all(w['state']=='valid' for w in self.manifest['windows']) else 'partial'
                try:self._save()
                except OSError as error:self._fail(error)
        return self.manifest or dict(status=self.status,errors=self.errors)

    def __enter__(self) -> ProfilerSession:
        self.start();return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def inspect_session(directory: str | Path, *, owner_alive: bool | None = None) -> dict[str,Any]:
    """Read-only crash inspection; explicit liveness can come from a launcher
    that waited for this exact process/container (including SIGKILL).
    """
    return inspect_capture(directory,owner_alive=owner_alive)
