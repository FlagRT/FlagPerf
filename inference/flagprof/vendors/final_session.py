"""One public SDK start/stop; sealing and analysis belong to the exited-owner worker."""
import os
from pathlib import Path
import resource
import time
from flagprof.store.manifest import atomic_json, proc_key
from . import session as ownership


class FinalSession:
    def __init__(self,backend,directory,config,identity=None,**unused):
        self.backend=backend;self.directory=Path(directory);self.config=config
        self.status='new';self.errors=[];self.profiler=None;self.owns=False
        self.pid=os.getpid();self.stopped=False
        self.meta=dict(origin='official-final',mode='final',normal_close=False,target_returncode=None,
                       process=proc_key(),start_calls=0,stop_calls=0,step_calls=0,compatibility_steps=0,
                       callbacks=0,capture_loss=dict(status='unknown',count=None),
                       coverage='official start through normal stop; SDK loss counters unavailable')
        self.identity=dict(identity or {})

    def save(self):
        self.meta.update(status=self.status,errors=list(self.errors),
                         peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        atomic_json(self.directory/'producer.json',self.meta)

    def start(self):
        if self.pid!=os.getpid() or self.status!='new':return False
        self.directory.mkdir(parents=True,exist_ok=False)
        raw=self.directory/'official';raw.mkdir()
        try:
            with ownership._LOCK:
                if self.backend.activity_source in ownership._OWNERS:raise RuntimeError('official profiler already owned')
                if self.backend.conflict()=='conflict':raise RuntimeError('external profiler conflict')
                ownership._OWNERS.add(self.backend.activity_source);self.owns=True
            measured=self.backend.identity(self.config)
            measured['collector'].get('requested_configuration',{}).pop('schedule',None)
            self.meta['measured_identity']=dict(self.identity,**measured)
            self.meta['requested_configuration']=measured['collector'].get('requested_configuration',{})
            self.meta['configuration']=dict(requested=dict(status='available',values=self.meta['requested_configuration']),
                                             actual=dict(status='unknown'),comparison=dict(status='unavailable'))
            self.profiler=self.backend.create_final(self.config,raw,self.ready)
            self.profiler.step=self.forbidden_step
            self.meta['start_calls']+=1;t=time.perf_counter();self.profiler.start()
            self.meta['start_s']=time.perf_counter()-t;self.status='running';self.save();return True
        except BaseException as error:
            self.errors.append(type(error).__name__+': '+str(error));self.status='failed'
            self.close(normal=False,returncode=None);return False

    def ready(self,profiler):self.meta['callbacks']+=1
    def forbidden_step(self,*args,**kwargs):
        self.meta['step_calls']+=1
        raise RuntimeError('final SDK step is forbidden')
    def step(self):
        if self.pid!=os.getpid() or self.status!='running':return False
        self.meta['compatibility_steps']+=1;return True

    def close(self,*,normal=True,returncode=0):
        if self.pid!=os.getpid():return
        if self.stopped:return
        self.stopped=True
        try:
            if self.profiler is not None and self.meta['start_calls']:
                self.meta['stop_calls']+=1;t=time.perf_counter();self.profiler.stop()
                self.meta['stop_s']=time.perf_counter()-t
                t=time.perf_counter();self.backend.export_final(self.profiler,self.directory/'official')
                self.meta['export_s']=time.perf_counter()-t
            if self.backend.activity_source=='npu' and self.meta['callbacks']!=1:
                raise ValueError('one final raw delivery required')
            self.meta['normal_close']=bool(normal is True and returncode==0 and not self.errors and
                self.meta['start_calls']==self.meta['stop_calls']==1 and self.meta['step_calls']==0)
            self.status='closed' if self.meta['normal_close'] else 'failed' if self.errors else 'partial'
        except BaseException as error:
            self.errors.append(type(error).__name__+': '+str(error));self.status='failed'
        finally:
            self.meta['target_returncode']=returncode
            if self.owns:
                with ownership._LOCK:ownership._OWNERS.discard(self.backend.activity_source)
                self.owns=False
            self.save()
