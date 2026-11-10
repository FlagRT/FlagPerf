# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Called only by record's temporary sitecustomize."""
import importlib.machinery
import json
import os
from pathlib import Path
import sys


def chain_sitecustomize(injection):
    paths = [p for p in sys.path if Path(p or os.getcwd()).resolve() != Path(injection).resolve()]
    spec = importlib.machinery.PathFinder.find_spec('sitecustomize', paths)
    if spec is not None:
        # Execute in the actual module object so later imports see user globals.
        module = sys.modules['sitecustomize']
        module.__file__ = spec.origin
        module.__spec__ = spec
        module.__loader__ = spec.loader
        spec.loader.exec_module(module)


def bootstrap(injection):
    config_path = os.environ.get('FLAGPROF_RECORD_CONFIG')
    if not config_path:
        chain_sitecustomize(injection)
        return
    config = json.loads(Path(config_path).read_bytes())
    from .store.manifest import atomic_json, proc_key
    expected = config.get('wrapper_process')
    try:
        parent = proc_key(os.getppid())
    except OSError:
        return
    # SDK analysis helpers and ordinary spawned children inherit the env, but
    # do not enroll. Only a fully identified direct target may auto-attach.
    if not expected or not all(v is not None for v in parent.values()) or parent != expected:
        chain_sitecustomize(injection)
        return
    from .runtime import attach
    from .registry import Registry
    import uuid
    handshake = Path(config['run'])/'handshakes'/uuid.uuid4().hex
    handshake.parent.mkdir(exist_ok=True)
    state = dict(process=proc_key(), status='starting')
    atomic_json(handshake, state)
    try:
        from .components.compiler import select
        select(config['config'], cache_root=Path(config['run']).parent/'component-cache')
        chain_sitecustomize(injection)
        handle = attach(config['run'], config['vendor'], preset=config['preset'], config=config['config'],
                        registry=Registry(config.get('descriptors')), identity=config.get('identity'),
                        backend_options=config.get('backend_options'),capture_mode=config.get('capture_mode','windowed'))
        state.update(attachment=handle.id, status=handle.state['status'])
    except Exception as error:
        state.update(status='failed', error=repr(error))
    atomic_json(handshake, state)
    if (config['config'].get('flagtree') is not None or config['config'].get('flaggems') == 'on') and state['status'] != 'running':
        # Python otherwise ignores sitecustomize exceptions and runs the target.
        raise SystemExit('explicit component selection failed: '+str(state.get('error',state['status'])))
