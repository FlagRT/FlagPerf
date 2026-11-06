# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Immutable final capture envelope and independent source registration."""
from pathlib import Path
import os
import shutil
import uuid
from .checkpoints import seal, validate, publish, fileinfo, source_budget
from .limits import read_json



def copy_checked(source, destination, *, max_bytes=1<<30):
    """Bounded-size, streaming snapshot; reject concurrent source mutation."""
    import hashlib
    import os
    max_bytes=source_budget(max_bytes)
    source,destination=Path(source),Path(destination)
    if source.is_symlink() or not source.is_file() or source.stat().st_size>max_bytes:
        raise ValueError('source file type or byte budget')
    before=fileinfo(source);total=0;digest=hashlib.sha256()
    with source.open('rb') as src,destination.open('xb') as dst:
        while block:=src.read(65536):
            total+=len(block)
            if total>max_bytes:raise ValueError('source byte budget')
            dst.write(block);digest.update(block)
        dst.flush();os.fsync(dst.fileno())
    if fileinfo(source)!=before or digest.hexdigest()!=before['sha256'] or fileinfo(destination)!=before:
        raise ValueError('source changed while snapshotting')

def make_bundle(raw, trace, destination, *, vendor, metadata, occurrence=None):
    if vendor not in ('nvidia', 'ascend', 'thead', 'kunlunxin'):
        raise ValueError('unsupported final vendor')
    if metadata.get('normal_close') is not True or metadata.get('target_returncode') != 0:
        raise ValueError('normal producer close required')
    destination = Path(destination).resolve()
    from .workspace import check_disk
    for path in (raw, trace, destination):
        check_disk(path)
    destination.mkdir(parents=True, exist_ok=False)
    occurrence = occurrence or uuid.uuid4().hex
    raw_manifest = seal(raw, destination/'sealed-raw', occurrence, max_bytes=source_budget(metadata.get('source_max_bytes',1<<30)))
    return finish_bundle(destination, trace, vendor=vendor, metadata=metadata,
                         occurrence=occurrence, raw_manifest=raw_manifest)


def finish_bundle(destination, trace, *, vendor, metadata, occurrence, raw_manifest):
    if metadata.get('normal_close') is not True or metadata.get('target_returncode')!=0:
        raise ValueError('normal producer close required')
    destination = Path(destination).resolve()
    trace = Path(trace)
    from .workspace import check_disk
    check_disk(destination)
    check_disk(trace)
    # Keep official memory derivatives beside the trace before analysis scratch
    # is removed. They share the immutable capture occurrence.
    trace_input = destination/'trace-input'
    trace_input.mkdir()
    limit=source_budget(metadata.get('source_max_bytes',1<<30))
    if raw_manifest.get('extra',{}).get('source_max_bytes',1<<30)!=limit:raise ValueError('raw source budget binding mismatch')
    copy_checked(trace,trace_input/'trace.json',max_bytes=limit)
    from .memory import seal_artifacts
    memory_files = seal_artifacts(trace, trace_input, vendor, metadata)
    sealed = seal(trace_input, destination/'sealed-trace', occurrence,max_bytes=limit)
    source = sealed['files']['data/trace.json']
    capture = dict(schema='plan-b-capture/1', occurrence=occurrence, vendor=vendor,
                   metadata=metadata, normal_close=True,
                   raw=dict(path='sealed-raw', manifest=fileinfo(destination/'sealed-raw/committed.json')),
                   trace=dict(path='sealed-trace', manifest=fileinfo(destination/'sealed-trace/committed.json'),
                              file='data/trace.json', **source))
    if memory_files:
        capture['memory_files'] = memory_files
        capture['memory_sources'] = {name:sealed['files']['data/'+name] for name in memory_files}
    publish(destination/'capture.json', capture)
    result=load_bundle(destination)
    shutil.rmtree(trace_input)
    from .checkpoints import syncdir
    syncdir(destination)
    return result


def load_bundle(directory):
    directory = Path(directory).resolve()
    from .workspace import check_disk
    check_disk(directory)
    capture = read_json(directory/'capture.json')
    if capture.get('schema') != 'plan-b-capture/1' or capture.get('normal_close') is not True:
        raise ValueError('capture protocol/normal close')
    if capture['vendor'] not in ('nvidia', 'ascend', 'thead', 'kunlunxin') or not isinstance(capture['occurrence'], str) or not capture['occurrence']:
        raise ValueError('capture identity')
    meta = capture['metadata']
    if meta.get('normal_close') is not True or meta.get('target_returncode') != 0:
        raise ValueError('capture did not close normally')
    for kind in ['raw', 'trace']:
        info = capture[kind]
        if info['path'] != 'sealed-' + kind:
            raise ValueError('capture path')
        stage = directory/info['path']
        if fileinfo(stage/'committed.json') != info['manifest']:
            raise ValueError('capture stage manifest changed')
        m = validate(stage, dict(occurrence=capture['occurrence'], normal_close=True), 'sealed-source',source_max_bytes=source_budget(meta.get('source_max_bytes',1<<30)))
        if kind == 'trace':
            if info['file'] != 'data/trace.json' or m['files'].get(info['file']) != {k: info[k] for k in ['bytes', 'sha256']}:
                raise ValueError('trace source binding')
            memory_files=capture.get('memory_files',[])
            if (len(memory_files)!=len(set(memory_files)) or
                not set(memory_files)<= {'memory_record.csv','operator_memory.csv'}):
                raise ValueError('official memory artifact names')
            for name in memory_files:
                if m['files'].get('data/'+name)!=capture.get('memory_sources',{}).get(name):
                    raise ValueError('memory source binding')
    return capture


def register(run, directory, capture):
    registration = dict(format_version=2, bundle_path=os.path.relpath(Path(directory).resolve(),Path(run).resolve()), capture=capture,
                        envelope=fileinfo(Path(directory)/'capture.json'))
    path = Path(run)/'plan-b-capture.json'
    publish(path, registration)
    return dict(occurrence=capture['occurrence'], registration_path=path.name,
                registration_sha256=fileinfo(path)['sha256'], source_sha256=capture['trace']['sha256'])


def verify_registration(run, reference, source_sha):
    if reference.get('registration_path') != 'plan-b-capture.json':
        raise ValueError('capture registration path')
    path = Path(run)/reference['registration_path']
    if fileinfo(path)['sha256'] != reference['registration_sha256']:
        raise ValueError('capture registration changed')
    registration = read_json(path)
    bundle = Path(registration['bundle_path'])
    if registration.get('format_version')==2:
        if bundle.is_absolute():raise ValueError('relative capture registration required')
        bundle=(Path(run)/bundle).resolve()
    if fileinfo(bundle/'capture.json') != registration['envelope']:
        raise ValueError('capture envelope changed')
    capture = load_bundle(bundle)
    if capture != registration['capture'] or capture['occurrence'] != reference['occurrence']:
        raise ValueError('capture occurrence changed')
    if capture['trace']['sha256'] != source_sha or reference['source_sha256'] != source_sha:
        raise ValueError('capture source SHA changed')
