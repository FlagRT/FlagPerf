# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded official final configuration evidence, derived from sealed artifacts."""
from pathlib import Path
from copy import deepcopy
from .limits import read_json
from .streaming import JSONStream
from .checkpoints import fileinfo

FLAGS=('record_shapes','with_stack','profile_memory','with_flops','with_modules')

def trace_metadata(trace):
    stream=JSONStream(trace);values={};end_count=0;budget=0
    try:
        stream.space()
        if stream.peek()!='{':return {},0
        stream.expect('{')
        while True:
            stream.space()
            if stream.peek()=='}':stream.take(1);break
            key,_,_=stream.value();stream.expect(':')
            if key=='traceEvents':
                stream.expect('[');stream.space()
                if stream.peek()==']':stream.take(1)
                else:
                    while True:
                        row,_,_=stream.value()
                        if isinstance(row,dict) and row.get('name')=='Record Window End':end_count+=1
                        stream.space()
                        if stream.peek()==']':stream.take(1);break
                        stream.expect(',')
            else:
                value,a,b=stream.value();budget+=b-a
                if budget>4<<20:raise ValueError('official trace metadata budget')
                if key in (*FLAGS,'schemaVersion','deviceProperties','cupti_version','cuda_runtime_version','cuda_driver_version','traceName','baseTimeNanoseconds'):
                    values[key]=value
            stream.space()
            if stream.peek()=='}':stream.take(1);break
            stream.expect(',')
        stream.space()
        if stream.peek():raise ValueError('trailing official trace data')
    finally:stream.close()
    return values,end_count

def identities(directory,metadata,vendor,trace,occurrence,raw_manifest):
    directory=Path(directory);measured=deepcopy(metadata.get('measured_identity',{}))
    source=dict(kind='capture-envelope',artifact='capture.json',occurrence=occurrence)
    requested=metadata.get('requested_configuration',{})
    actual={};sources=[];conflicts=[];versions=[];exporter=dict(status='unknown')
    trace_values,end_count=trace_metadata(trace)
    if vendor=='ascend':
        for name,info in raw_manifest['files'].items():
            if Path(name).name.startswith('profiler_info') and Path(name).suffix=='.json':
                if len(sources)>=16:raise ValueError('official configuration cardinality budget')
                config=read_json(directory/'sealed-raw'/name)
                common=config.get('config',{}).get('common_config',{})
                extra=config.get('config',{}).get('experimental_config',{})
                value={k:common.get(k) for k in FLAGS}
                value.update(activities=[str(x).split('.')[-1] for x in common.get('activities',[])],
                             profiler_level=extra.get('_profiler_level'),data_simplification=extra.get('_data_simplification'))
                sources.append(dict(path='sealed-raw/'+name,sha256=info['sha256'],values=value))
                versions.append(config.get('cann_version'))
        for key in (*FLAGS,'activities','profiler_level','data_simplification'):
            values=[row['values'][key] for row in sources]
            actual[key]=values[0] if values and all(type(v) is type(values[0]) and v==values[0] for v in values) else None
        if sources:
            exporter=dict(source=[dict(path=r['path'],sha256=r['sha256']) for r in sources],
                          version=versions[0] if all(v==versions[0] for v in versions) else None,schema_version=None)
    else:
        info=fileinfo(trace);trace_source=dict(path='sealed-trace/data/trace.json',sha256=info['sha256'])
        version=measured.get('collector',{}).get('version')
        complete=(version in ('2.8.0+cu126','2.10.0+cu126')
            and type(trace_values.get('schemaVersion')) is int and trace_values['schemaVersion']==1
            and isinstance(trace_values.get('deviceProperties'),list) and bool(trace_values['deviceProperties'])
            and all(type(trace_values.get(k)) is int for k in ('cupti_version','cuda_runtime_version','cuda_driver_version'))
            and isinstance(trace_values.get('traceName'),str) and end_count>0)
        if complete:
            actual={k:True if type(trace_values.get(k)) is int and trace_values[k]==1 else False if k not in trace_values else None for k in FLAGS}
            sources=[dict(**trace_source,interpretation='version-bound Kineto '+version+' metadata',values=actual)]
        exporter=dict(source=trace_source,version=None,schema_version=trace_values.get('schemaVersion'))
    mismatches=[dict(field=k,requested=v,actual=actual[k]) for k,v in requested.items()
                if actual.get(k) is not None and (type(v) is not type(actual[k]) or v!=actual[k])]
    configuration=dict(requested=dict(status='available',values=requested,source=source),
        actual=dict(status='available' if sources else 'unknown',values=actual,sources=sources),
        comparison=dict(status='mismatch' if mismatches else 'match' if requested and all(k in actual and actual[k] is not None for k in requested) else 'partial' if sources else 'unavailable',
                        mismatches=mismatches,reason='only independently observed fields compared'))
    if mismatches:conflicts.extend(mismatches)
    evidence=dict(producer='flagprof.offline.final_evidence',configuration=configuration,
        capture_completeness=dict(status='unchecked',reason='SDK scoped loss counter unavailable'),
        clock=dict(clock_valid=False,status='unavailable'),
        observations=[dict(record_window_end_count=end_count,trace_metadata=trace_values)],conflicts=conflicts,negative_evidence=[])
    return dict(hardware=dict(source=source,facts=measured.get('hardware',{})),
                collector=dict(source=source,facts=measured,configuration=configuration,evidence=evidence),exporter=exporter)
