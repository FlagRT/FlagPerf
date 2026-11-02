# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Derive facts only from artifacts sealed for this capture window (D30).

No requested option is an actual observation. This verifies provenance within
an explicitly trusted capture, not authenticity of arbitrary user metadata.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path

FLAGS=('record_shapes','with_stack','profile_memory','with_flops','with_modules')



def _consensus(values):
    if not values or any(v is None for v in values):return None
    return deepcopy(values[0]) if all(type(v) is type(values[0]) and v==values[0] for v in values) else None


def capture_identities(directory, session, window, trace_path, manifest_path, vendor):
    """directory is the copied capture; read only current registered artifacts."""
    directory=Path(directory)
    manifest=dict(path=manifest_path,sha256=hashlib.sha256((directory/'session.json').read_bytes()).hexdigest(),
                  window=window['index'],capture_id=window['capture_id'])
    artifacts={}
    for row in window['artifacts']:
        relative=Path(window['path'])/row['path']
        file=directory/relative
        if relative.is_absolute() or '..' in relative.parts or file.is_symlink() or not file.resolve().is_relative_to(directory.resolve()):
            raise ValueError('evidence path escapes capture')
        raw=file.read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('evidence artifact SHA mismatch')
        if row['path'] in artifacts:raise ValueError('duplicate artifact path')
        artifacts[row['path']]=(file,dict(path=str(Path(manifest_path).parent/relative),sha256=row['sha256'],window=window['index']))
    if trace_path not in artifacts:raise ValueError('unregistered trace evidence')
    trace_file,trace_source=artifacts[trace_path]
    document=json.loads(trace_file.read_bytes())
    identity=deepcopy(session['identity'])
    collector=identity.get('collector',{})
    configuration=deepcopy(window.get('validity',{}).get('configuration',{}))
    # A historical requested block is retained as a request only.
    configuration.pop('actual',None);configuration.pop('comparison',None)
    configuration.update(source=manifest_path,capture_id=window['capture_id'],window=window['index'])
    exporter=dict(source=trace_source)
    evidence=dict(producer='flagprof.vendors.evidence.capture_identities',manifest=manifest,
                  capture_completeness=dict(status='unchecked',reason='no scoped collector drop counter or sequence certificate'),
                  clock=dict(clock_valid=False,sources=[]),negative_evidence=[],observations=[],conflicts=[])
    from .format import load_adapter
    import importlib
    load_adapter(vendor)  # validate the registered rule family before resolving its producer
    producer=importlib.import_module(__package__+'.'+vendor+'.evidence')
    producer.derive(document,collector,trace_source,artifacts,configuration,exporter,evidence)
    requested=configuration.get('requested',{}).get('values',{}) or {}
    actual=configuration['actual'].get('values',{})
    mismatches=[dict(field=k,requested=v,actual=actual[k]) for k,v in requested.items()
                if k in actual and actual[k] is not None and (type(v) is not type(actual[k]) or v!=actual[k])]
    configuration['comparison']=dict(status='mismatch' if mismatches else 'unavailable',mismatches=mismatches,
        reason='only observed fields compared; absence is not agreement')
    return dict(hardware=dict(source=manifest,facts=identity.get('hardware',{})),
        collector=dict(source=manifest,facts=identity,config=session['config'],execution=session.get('execution',{}) if session.get('execution',{}).get('source') else {},
                       configuration=configuration,evidence=evidence),exporter=exporter)
