# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""One official session around a direct Python target, without business steps."""
from pathlib import Path
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
import uuid
from .checkpoints import fileinfo, seal, validate, publish, source_budget
from .limits import read_json, walk_files
from .bundle import finish_bundle, load_bundle
from flagprof.store.manifest import atomic_json, proc_key


def execute_target(arguments):
    if not arguments:raise ValueError('missing Python target')
    sys.argv = list(arguments)
    if arguments[0] == '-m':
        if len(arguments)<2:raise ValueError('missing module')
        sys.argv = arguments[1:];runpy.run_module(arguments[1],run_name='__main__',alter_sys=True)
    elif arguments[0] == '-c':
        if len(arguments)<2:raise ValueError('missing code')
        sys.argv = ['-c']+arguments[2:];exec(compile(arguments[1],'<string>','exec'),{'__name__':'__main__'})
    elif arguments[0].startswith('-'):
        raise ValueError('final mode supports script, -m or -c targets')
    else:
        script = Path(arguments[0]).resolve();sys.path.insert(0,str(script.parent))
        runpy.run_path(str(script),run_name='__main__')


def capture_target(directory, vendor, preset, arguments, config=None, capture_filesystem='ext4', rank=0):
    from flagprof.runtime import attach
    from flagprof.store.manifest import create_run
    import signal
    import traceback
    if os.environ.get('FLAGPROF_LAUNCH_WORKER') == '1':
        import importlib
        os.environ['FLAGPROF_DEVICE']=importlib.import_module('flagprof.vendors.'+vendor+'.execution').initialize(rank)
    directory=Path(directory)
    run=create_run(directory/'collection',identity=dict(route='final-capture',argv=list(arguments)))
    handle=attach(run,vendor,preset=preset,config=config,capture_mode='final',capture_filesystem=capture_filesystem,rank=rank)
    target_file=Path(arguments[0]).resolve() if arguments and not arguments[0].startswith('-') else None
    target_before=fileinfo(target_file) if target_file else None
    code=0;started=time.perf_counter();error=None
    previous={}
    def stop_signal(signum,frame):raise SystemExit(128+signum)
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,stop_signal)
        if handle.state['status']!='running':raise RuntimeError('official profiler failed to start: '+repr(handle.state))
        try:execute_target(arguments)
        except SystemExit as exc:code=exc.code if isinstance(exc.code,int) else 0 if exc.code is None else 1
        if target_file and fileinfo(target_file)!=target_before:raise ValueError('target source changed')
    except BaseException as exc:
        code=1;error=type(exc).__name__+': '+str(exc);traceback.print_exc()
    finally:
        premature=handle.closed
        handle.close(normal=code==0 and error is None,returncode=code)
        for sig,handler in previous.items():signal.signal(sig,handler)
        meta=read_json(handle.capture/'producer.json')
        meta.update(target_s=time.perf_counter()-started,target_before=target_before,
                    raw_path=str((handle.capture/'official').relative_to(directory)),
                    collection_run=str(run.relative_to(directory)),source_run_id=run.name,attachment=handle.id,
                    capture_filesystem=capture_filesystem,
                    closed_before_target_return=premature)
        if error:meta['target_error']=error
        if handle.state['status']!='closed' or premature:meta['normal_close']=False
        atomic_json(directory/'producer.json',meta)
    return code or (0 if meta['normal_close'] else 2)


def analyse_worker(directory):
    from torch_npu.profiler.profiler import analyse
    analyse(str(directory),max_process_number=1)


def check_analysis_log(path):
    """The pinned CANN analyser can report failed tasks while exiting zero."""
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            if 'Failed to export CANN' in line or ('Task [' in line and '] run failed' in line):
                raise ValueError('official offline analysis failed: '+line.strip()+'; sealed raw and analysis retained')


def finish_capture(directory, *, timeout=1800):
    import fcntl
    directory=Path(directory).resolve()
    from .workspace import check_disk
    check_disk(directory)
    with (directory/'.analysis.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _finish_capture(directory,timeout=timeout)


def _finish_capture(directory, *, timeout=1800):
    directory=Path(directory).resolve()
    if (directory/'capture.json').exists():return load_bundle(directory)
    state=read_json(directory/'capture-input.json')
    occurrence=state['occurrence'];raw=directory/'sealed-raw'
    before=validate(raw,dict(occurrence=occurrence,normal_close=True),'sealed-source',
                    source_max_bytes=source_budget(state['metadata'].get('source_max_bytes',1<<30)))
    if fileinfo(raw/'committed.json')!=state['raw_manifest']:raise ValueError('raw capture changed')
    metadata=state['metadata'];vendor=state['vendor'];t=time.perf_counter()
    if vendor=='ascend':
        working=directory/('analysis-'+uuid.uuid4().hex);shutil.copytree(raw/'data',working)
        with (directory/('analyse-'+working.name+'.log')).open('xb') as log:
            from .process import run_owned
            cp=run_owned([sys.executable,'-B','-m','flagprof.offline.capture','analyse',str(working)],stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if cp.returncode:raise ValueError('official offline analysis failed; sealed raw retained')
        check_analysis_log(directory/('analyse-'+working.name+'.log'))
        traces=[]
        for path in walk_files(working,max_files=4096,max_entries=8192):
            if path.name=='trace_view.json':traces.append(path)
            if len(traces)>1:raise ValueError('single-trace budget')
        if len(traces)!=1:raise ValueError('one analysed trace required')
        trace=traces[0]
    else:trace=raw/'data/trace.json'
    metadata['official_analysis_s']=time.perf_counter()-t
    validate(raw,before['binding'],'sealed-source')
    # Failed parse/seal attempts are retained. A later explicit retry must use
    # fresh derivative paths rather than overwriting committed artifacts.
    for name in ['trace-input','sealed-trace']:
        path=directory/name
        if path.exists():path.rename(directory/(name+'-failed-'+uuid.uuid4().hex))
    if (directory/'capture.json').exists():raise ValueError('capture already sealed')
    from .final_evidence import identities
    metadata['identities']=identities(directory,metadata,vendor,trace,occurrence,before)
    result=finish_bundle(directory,trace,vendor=vendor,metadata=metadata,occurrence=occurrence,raw_manifest=before)
    if vendor=='ascend':shutil.rmtree(working)
    return result


def seal_capture(directory,metadata,vendor,*,capture_filesystem='ext4'):
    directory=Path(directory).resolve()
    from .workspace import check_capture_disk
    check_capture_disk(directory,vendor,capture_filesystem)
    if metadata.get('normal_close') is not True or metadata.get('target_returncode')!=0:
        raise ValueError('abnormal capture cannot be certified')
    from .checkpoints import checked_relative
    raw=directory/checked_relative(metadata.get('raw_path','official'))
    occurrence=uuid.uuid4().hex;t=time.perf_counter()
    seal(raw,directory/'sealed-raw',occurrence,max_bytes=source_budget(metadata.get('source_max_bytes',1<<30)))
    metadata['raw_seal_s']=time.perf_counter()-t
    publish(directory/'capture-input.json',dict(occurrence=occurrence,vendor=vendor,metadata=metadata,
                                                raw_manifest=fileinfo(directory/'sealed-raw/committed.json')))


def record_final(command, *, store, work=None, scratch=None, vendor, preset='perf', config=None, timeout=1800,
                 capture_only=False,capture_filesystem='ext4',source_max_bytes=1<<30,rank=0):
    import math
    source_max_bytes=source_budget(source_max_bytes)
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('positive finite timeout required')
    if not command or not Path(command[0]).name.startswith('python'):
        raise ValueError('final mode requires a direct Python command')
    arguments=command[1:];flags=[]
    while arguments and arguments[0] in ('-B','-u'):
        flags.append(arguments[0]);arguments=arguments[1:]
    if not arguments:raise ValueError('missing Python target')
    from .service import _check_scratch
    from .workspace import locations,check_disk
    from .process import run_owned
    store=Path(store).resolve()
    capture_id='capture-'+uuid.uuid4().hex
    if capture_only:
        if vendor not in ('nvidia','ascend') or work is not None or scratch is not None:
            raise ValueError('capture-only requires a supported vendor without import work/scratch')
        from .workspace import check_capture_disk
        check_capture_disk(store,vendor,capture_filesystem)
    else:
        if capture_filesystem!='ext4':raise ValueError('overlay requires explicit capture-only')
        work,scratch=locations(store,work,scratch,key=capture_id)
        check_disk(store)
        _check_scratch(scratch,store/'runs',work)
    store.mkdir(parents=True,exist_ok=True)
    destination=store/capture_id;destination.mkdir()
    from flagprof.probes.config import resolve
    session_options, probe_options = resolve(preset, config)
    atomic_json(destination/'record-config.json', dict(preset=preset,capture_filesystem=capture_filesystem,rank=rank, config=dict(session_options,
        **{k:v for k,v in probe_options.items() if k != 'include_sha256'})))
    root=str(Path(__file__).resolve().parents[2]);env=dict(os.environ)
    env.pop('FLAGPROF_RECORD_CONFIG',None)
    env['PYTHONPATH']=os.pathsep.join([root]+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    started=time.perf_counter();deadline=time.monotonic()+timeout
    state=dict(schema='plan-b-record/1',status='incomplete',capture=str(destination),target_exit=None,
               importer_exit=None,import_completion='not_started',finalize='not_started',
               capture_completeness='unknown',throughput_gate='not_passed',performance_blocking=False)
    code=2
    try:
        with (destination/'target.log').open('xb') as log:
            cp=run_owned([command[0],'-B',*flags,'-m','flagprof.offline.capture','target',str(destination),vendor,preset,*arguments],
                         env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.01,deadline-time.monotonic()))
        state['target_exit']=dict(returncode=cp.returncode,timed_out=cp.timed_out,source='subprocess.wait')
        code=128-cp.returncode if cp.returncode<0 else cp.returncode
        from flagprof.store._proxy_lifecycle import record_wait, deadline_scope
        from flagprof.store.manifest import finalize
        collections=list((destination/'collection').glob('*/manifest.json'))
        if len(collections)>1:raise ValueError('one owned collection run required')
        for manifest in collections:
            proxy_deadline, proxy_timeout=record_wait(manifest.parent,deadline=deadline)
            with deadline_scope(proxy_deadline):
                collection_final=finalize(manifest.parent)
            state['collection_finalize']=collection_final['status']
            if proxy_timeout:
                state['writer_failure']='proxy_wait_timeout'
                code=code or 2
        if code:return code
        code=2
        metadata=read_json(destination/'producer.json');metadata['producer_exited_before_analysis']=True
        metadata['source_max_bytes']=source_max_bytes
        metadata['target_process_s']=time.perf_counter()-started
        seal_capture(destination,metadata,vendor,capture_filesystem=capture_filesystem)
        if capture_only:
            state.update(status='sealed',import_completion='deferred_to_ext4',capture_filesystem=capture_filesystem)
            code=0
            return code
        finish_capture(destination,timeout=max(.01,deadline-time.monotonic()))
        result=destination/'import-result.json'
        cmd=[sys.executable,'-B','-S','-m','flagprof','import',str(destination),'--store',str(store/'runs'),
             '--work',str(work),'--scratch',str(scratch)]
        with result.open('xb') as output,(destination/'importer.log').open('xb') as errors:
            child=run_owned(cmd,env=env,stdout=output,stderr=errors,timeout=max(.01,deadline-time.monotonic()))
        state['importer_exit']=child.returncode
        if child.returncode:
            code=128-child.returncode if child.returncode<0 else child.returncode
            return code
        summary=read_json(result);state.update(summary)
        state.update(capture=str(destination),import_completion=summary['status'],status=summary['status'])
        code=0 if summary['status']=='complete' else 2
        return code
    except (ValueError,OSError,RuntimeError) as error:
        state['error']=type(error).__name__+': '+str(error);return code
    finally:
        state['exit_code']=code;state['entry_total_s']=time.perf_counter()-started
        atomic_json(destination/'record-status.json',state)
        if state.get('run'):atomic_json(Path(state['run'])/'record-status.json',state)
        print(json.dumps(state,ensure_ascii=False))


if __name__=='__main__':
    if sys.argv[1]=='target':
        config_path=Path(sys.argv[2])/'record-config.json'
        settings=read_json(config_path) if config_path.exists() else {}
        raise SystemExit(capture_target(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5:],
            config=settings.get('config'),capture_filesystem=settings.get('capture_filesystem','ext4'),rank=settings.get('rank',0)))
    elif sys.argv[1]=='analyse':analyse_worker(sys.argv[2])
    else:raise ValueError('internal capture mode')
