# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Wrap a target with explicit handshake and distinct offline import."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import math
from .runtime import configuration
from .registry import Registry
from .store import create_run, finalize
from .store.manifest import atomic_json, proc_key


def record(command, *, store, vendor, preset='perf', config=None, descriptors=None,
           identity=None, backend_options=None,capture_mode='final',timeout=1800):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive finite record timeout required')
    if preset in ('coverage', 'perf-lite', 'accuracy', 'monitor'):
        capture_mode = 'windowed'
    if capture_mode=='final':
        if descriptors or identity or backend_options:
            raise ValueError('final record uses explicit vendor/preset; window/plugin settings require windowed mode')
        from .offline.capture import record_final
        return record_final(command,store=store,vendor=vendor,preset=preset,config=config,timeout=timeout)
    if capture_mode!='windowed':raise ValueError('unknown capture mode')
    if not command:
        raise ValueError('record needs a target command after --')
    config=dict(config or {})
    if 'log_files' in config:
        config['log_files']=tuple(str(Path(p).resolve()) for p in config['log_files'])
    from .probes.config import resolve
    session_config, probe_config = resolve(preset, config)
    if probe_config['diagnose'] == 'on':
        config.update({key:probe_config[key] for key in ('monitor','monitor_topology','monitor_root')})
    configuration(preset, **config)
    descriptor = Registry(descriptors).describe(vendor)
    if descriptor['kind'] != 'vendor':
        raise ValueError('record needs a vendor')
    run = create_run(store, identity=dict(argv=list(command), cwd=os.getcwd(), vendor=vendor, preset=preset)).resolve()
    from .monitor.lifecycle import Monitor
    monitor = Monitor(store, run/'monitor-ref.json', vendor, config, preset=preset)
    settings = dict(run='.', vendor=vendor, preset=preset, config=config or {}, descriptors=descriptors,
                    identity=identity, backend_options=backend_options, wrapper_process=proc_key(),
                    injection_policy='direct_python_target_only')
    atomic_json(run/'record-config.json', settings)
    state = dict(run=str(run), target_exit=None, importer_exit=None, import_completion='not_started', finalize='not_started')
    # Helpers run in the installed interpreter without target injection.
    original = dict(os.environ)
    original.pop('FLAGPROF_RECORD_CONFIG', None)
    root = str(Path(__file__).resolve().parent.parent)
    with tempfile.TemporaryDirectory(prefix='flagprof-injection-') as directory:
        Path(directory, 'sitecustomize.py').write_text('from flagprof._bootstrap import bootstrap\nbootstrap('+repr(directory)+')\n')
        env = dict(original)
        if preset in ('perf', 'modules') and descriptors is None:
            from .vendors import adapter
            prepare = getattr(adapter(vendor, 'execution'), 'prepare_capture_environment', None)
            if prepare:
                prepare(env)
        monitor.environment(env)
        env['PYTHONPATH'] = os.pathsep.join([directory, root] + ([original['PYTHONPATH']] if original.get('PYTHONPATH') else []))
        env['FLAGPROF_RECORD_CONFIG'] = str(run/'record-config.json')
        try:
            from .offline.process import run_owned
            target = run_owned(command, env=env, timeout=timeout)
            code = target.returncode
            target_key = target.process_key
            state['target_exit'] = dict(pid=target.pid, process=target_key, returncode=code,
                                       timed_out=target.timed_out, signal=target.signal, source='owned subprocess.wait')
        except OSError as error:
            code = 127
            state['target_exit'] = dict(returncode=code, error=repr(error))
        if state['target_exit'].get('source') == 'owned subprocess.wait':
            monitor.target_exit(dict(returncode=target.wait_returncode, source='subprocess.wait',
                termination_reason='launcher_timeout' if target.timed_out else
                    'external_signal' if target.signal else 'target_exit'), worker=target_key)
        monitor.close()
        handshakes = [json.loads(p.read_bytes()) for p in (run/'handshakes').glob('*')]
        attachments = [json.loads(p.read_bytes()) for p in (run/'attachments').glob('*/state.json')]
        state['handshakes'] = handshakes
        state['attachments'] = attachments
        from .store._proxy_lifecycle import record_wait
        proxy_deadline, proxy_timeout = record_wait(run)
        if proxy_timeout:
            state['writer_failure'] = 'proxy_wait_timeout'
        importer_env = dict(original)
        importer_env['PYTHONPATH'] = os.pathsep.join([root] + ([original['PYTHONPATH']] if original.get('PYTHONPATH') else []))
        with (run/'logs/importer.log').open('wb') as log:
            child = subprocess.run([sys.executable, '-B', '-m', 'flagprof._import_worker', str(run)],
                                   env=importer_env, stdout=log, stderr=subprocess.STDOUT)
        state['importer_exit'] = child.returncode
        result = run/'import-result.json'
        if result.exists():
            state['import_completion'] = json.loads(result.read_bytes())['status']
    from .store._proxy_lifecycle import deadline_scope
    with deadline_scope(proxy_deadline):
        final = finalize(run)
    state['finalize'] = final['status']
    key = state['target_exit'].get('process')
    root_handshake = bool(key and all(v is not None for v in key.values()) and
                         any(h['process']==key for h in handshakes))
    state['root_handshake'] = root_handshake
    matched = (len({h.get('attachment') for h in handshakes})==len(handshakes) and
               all(sum(a['id']==h.get('attachment') and a['process']==h['process']
                       for a in attachments)==1 for h in handshakes))
    state['handshake_attachments_verified'] = matched
    success = (not proxy_timeout and root_handshake and matched and
               all(h['status']=='running' for h in handshakes) and
               all(a['status']=='closed' for a in attachments) and
               state['importer_exit']==0 and state['import_completion']=='complete')
    state['status'] = 'complete' if success else 'incomplete'
    state['exit_code'] = (128-code if code < 0 else code) if code else (0 if success else 2)
    atomic_json(run/'record-status.json', dict(state, run='.'))
    print(json.dumps(dict(run=str(run), status=state['status'], exit_code=state['exit_code'])))
    return state['exit_code']
