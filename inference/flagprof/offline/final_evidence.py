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
    from flagprof.vendors import adapter
    trace_values, end_count = trace_metadata(trace)
    actual, sources, exporter = adapter(vendor, 'evidence').final_configuration(
        directory, measured, trace, trace_values, end_count, raw_manifest)
    conflicts = []
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
    collector=dict(source=source,facts=measured,configuration=configuration,evidence=evidence)
    execution=metadata.get('execution',{})
    if execution.get('source'):
        collector['execution']=execution
    return dict(hardware=dict(source=source,facts=measured.get('hardware',{})),
                collector=collector,exporter=exporter)
