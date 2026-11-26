# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Stream immutable evidence with portable paths and content-bound cross-root links."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import re
from .offline.workspace import attempt, publish_run

from .offline.checkpoints import fileinfo, identity, publish, syncdir


CHUNK = 65536


def _relative(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'payload':
        raise ValueError('delivery member path')
    return p


def _inventory(sources, max_bytes):
    roots = {Path(s).resolve(strict=True) for s in sources}
    if not roots or type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('sources and positive delivery budget required')
    from .delivery_refs import reference_key, resolve
    from .offline.limits import read_json
    files, directories, symlinks = set(), set(), {}
    references = {}
    sessions = set()
    pending = list(roots)
    total = 0
    while pending:
        path = pending.pop()
        if path in files or path in directories or path in symlinks:
            continue
        if re.fullmatch(r'\.flagprof-(?:[0-9]+|import-[0-9a-f]{32})', path.name):
            if path in roots:
                raise ValueError('operation workspaces are not deliverable')
            continue
        if path.is_symlink():
            symlinks[path] = path.resolve(strict=True)
        elif path.is_dir():
            directories.add(path)
            pending.extend(path.iterdir())
        elif path.is_file():
            total += path.stat().st_size
            if total > max_bytes:
                raise ValueError('delivery byte budget')
            files.add(path)
            if path.name == 'session.json':
                sessions.add(path.parent)
            if path.name == 'plan-b-capture.json':
                registration = read_json(path)
                raw = registration['bundle_path']
                capture = resolve(path, raw)
                if fileinfo(capture/'capture.json') != registration['envelope']:
                    raise ValueError('capture reference digest mismatch')
                references[path] = {reference_key(raw): capture}
                roots.add(capture)
                pending.append(capture)
            elif path.name == 'monitor-ref.json' or path.parent.name == 'references':
                if path.suffix == '.json':
                    data = read_json(path)
                    references[path] = {reference_key(data[k]): resolve(path, data[k])
                                        for k in ('session_path','root','monitor_path') if data.get(k)}
            elif path.parent.name in ('active','history') and path.parent.parent.name == 'targets' and path.suffix == '.json':
                data = read_json(path)
                if data.get('reference_base') == 'session':
                    references[path] = {reference_key(v): resolve(path, v, base=path.parents[2])
                                        for v in data.get('references', [])}
        else:
            raise ValueError('delivery source type: ' + str(path))

    roots = sorted(p for p in roots if not any(q in p.parents for q in roots))
    # Session metadata and selected segments share a logical session directory.
    # Other independent roots keep their leaf names, even across mount points.
    groups = set(roots) | sessions
    groups = sorted(p for p in groups if not any(q in p.parents for q in groups))
    slots = {p: Path('payload', 'object-' + str(i), p.name) for i, p in enumerate(groups)}
    def member(path):
        for ancestor in (path, *path.parents):
            if ancestor in slots:
                return (slots[ancestor] / path.relative_to(ancestor)).as_posix()
        return None
    included = files | directories | set(symlinks)
    for target in symlinks.values():
        if target not in included:
            raise ValueError('delivery link target outside selected sources: ' + str(target))
    # An implicit ancestor such as a session root is also a readable target.
    present = included | {a for p in included for a in p.parents if member(a)}
    edges = {member(p): {k: member(t) for k,t in refs.items() if t in present}
             for p,refs in references.items()}
    rows = [dict(source=str(p), path=member(p), bytes=p.stat().st_size, identity=identity(p))
            for p in sorted(files)]
    links = [dict(path=member(p), target=member(t)) for p,t in sorted(symlinks.items())]
    return ({p: member(p) for p in roots}, rows, links, sorted(member(p) for p in directories),
            {p: refs for p,refs in edges.items() if refs})


class _HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def publish_tree(sources, destination, *, max_bytes=200 << 30, file_overrides=None, reference=None):
    destination = Path(destination).absolute()
    if destination.exists():
        raise ValueError('delivery destination already exists')
    roots, rows, links, directories, references = _inventory(sources, max_bytes)
    overrides={str(Path(k).resolve()):Path(v).resolve(strict=True) for k,v in (file_overrides or {}).items()}
    unknown=set(overrides)-{row['source'] for row in rows}
    if unknown: raise ValueError('delivery override outside selected regular files')
    for row in rows:
        if row['source'] in overrides:
            original=row['source']; replacement=overrides[original]
            row.update(source=str(replacement),original_source=original,bytes=replacement.stat().st_size,
                       identity=identity(replacement))
    if sum(row['bytes'] for row in rows)>max_bytes: raise ValueError('delivery byte budget')
    if any(p == destination or p in destination.parents for p in roots):
        raise ValueError('delivery destination inside source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with attempt(destination.parent) as (temporary, _lease):
        stage = temporary/'output'
        stage.mkdir()
        with (stage/'payload.tar').open('xb') as output:
            with tarfile.open(fileobj=output, mode='w|', bufsize=CHUNK) as archive:
                archive.copybufsize = CHUNK
                for row in rows:
                    path = Path(row['source'])
                    if identity(path) != row['identity']:
                        raise ValueError('delivery source changed before copy')
                    info = tarfile.TarInfo(row['path'])
                    info.size = row['bytes']
                    info.mode = 0o600
                    with path.open('rb') as source:
                        reader = _HashedReader(source)
                        archive.addfile(info, reader)
                    row['sha256'] = reader.digest.hexdigest()
                    if identity(path) != row['identity']:
                        raise ValueError('delivery source changed during copy')
            output.flush()
            os.fsync(output.fileno())
        objects = {f'root-{i}': name for i, name in enumerate(roots.values())}
        for kind, marker in (('run','manifest.json'), ('capture','capture.json'), ('group','group.json')):
            matches = [name for p,name in roots.items() if (p/marker).is_file()]
            if len(matches) == 1:
                objects[kind] = matches[0]
        if reference is not None:
            selected = Path(reference).resolve(strict=True)
            owner = next((p for p in roots if selected == p or p in selected.parents), None)
            if owner is None:
                raise ValueError('delivery reference is outside selected objects')
            objects['monitor-reference'] = str(Path(roots[owner]) / selected.relative_to(owner))
        manifest = dict(schema='flagprof-delivery/2', max_bytes=max_bytes, objects=objects,
                        files=[{k:r[k] for k in ('path','bytes','sha256')} for r in rows], links=links,
                        directories=directories, references=references,
                        bytes=sum(r['bytes'] for r in rows), archive=fileinfo(stage/'payload.tar'))
        publish(stage/'manifest.json', manifest)
        verify(stage, require_complete=False)
        publish(stage/'COMPLETE.json', dict(manifest=fileinfo(stage/'manifest.json'),
                                           archive=manifest['archive']))
        publish_run(stage, destination)
    return dict(delivery=str(destination), files=len(rows), bytes=manifest['bytes'])


def _manifest(directory, require_complete=True):
    directory = Path(directory)
    if (directory/'manifest.json').stat().st_size > 64 << 20:
        raise ValueError('delivery manifest byte budget')
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema') not in ('flagprof-delivery/1', 'flagprof-delivery/2'):
        raise ValueError('delivery schema')
    if require_complete:
        complete = json.loads((directory/'COMPLETE.json').read_text())
        if complete != dict(manifest=fileinfo(directory/'manifest.json'), archive=manifest['archive']):
            raise ValueError('delivery completion binding')
    if fileinfo(directory/'payload.tar') != manifest['archive']:
        raise ValueError('delivery archive changed')
    rows = manifest['files']
    if (len({r['path'] for r in rows}) != len(rows) or
            sum(r['bytes'] for r in rows) != manifest['bytes'] or
            manifest['bytes'] > manifest['max_bytes']):
        raise ValueError('delivery size or duplicate member')
    for row in rows:
        _relative(row['path'])
    names = {r['path'] for r in rows}
    directories = {_relative(d).as_posix() for d in manifest.get('directories', [])}
    if directories & names or any(str(p) in names for n in names | directories for p in Path(n).parents):
        raise ValueError('delivery file overlaps directory')
    names.update(directories)
    for link in manifest.get('links', []):
        path, target = _relative(link['path']), _relative(link['target'])
        if str(path) in names or not any(target == Path(n) or target in Path(n).parents for n in names):
            raise ValueError('delivery link closure')
        if any(path in Path(n).parents for n in names):
            raise ValueError('delivery link overlaps member tree')
        names.add(str(path))
    if manifest['schema'] == 'flagprof-delivery/2':
        regular = {r['path'] for r in rows}
        present = names | {str(p) for n in names for p in Path(n).parents}
        for source, refs in manifest.get('references', {}).items():
            if source not in regular or any(not re.fullmatch('[0-9a-f]{64}', k) or
                                           str(_relative(v)) not in present for k,v in refs.items()):
                raise ValueError('delivery reference closure')
        for member in manifest['objects'].values():
            root = _relative(member)
            if not any(root == Path(n) or root in Path(n).parents for n in names):
                raise ValueError('delivery root missing from payload')
    return manifest


def _consume(directory, manifest, destination=None):
    expected = {r['path']: r for r in manifest['files']}
    seen = set()
    if destination is not None:
        for member in manifest.get('directories', []):
            (destination/_relative(member)).mkdir(parents=True, exist_ok=True)
    with tarfile.open(Path(directory)/'payload.tar', 'r|', bufsize=CHUNK) as archive:
        for member in archive:
            row = expected.get(member.name)
            if not member.isfile() or row is None or member.name in seen or member.size != row['bytes']:
                raise ValueError('unexpected delivery member')
            seen.add(member.name)
            target = None
            if destination is not None:
                path = destination/_relative(member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                target = path.open('xb')
            digest = hashlib.sha256()
            try:
                with archive.extractfile(member) as source:
                    for block in iter(lambda: source.read(CHUNK), b''):
                        digest.update(block)
                        if target:
                            target.write(block)
                if target:
                    target.flush()
                    os.fsync(target.fileno())
            finally:
                if target:
                    target.close()
            if digest.hexdigest() != row['sha256']:
                raise ValueError('delivery member SHA mismatch')
    if seen != set(expected):
        raise ValueError('delivery missing members')


def verify(directory, *, require_complete=True):
    manifest = _manifest(directory, require_complete)
    _consume(directory, manifest)
    return dict(status='verified', files=len(manifest['files']), bytes=manifest['bytes'])


def restore(directory, destination, *, max_bytes=40 << 30):
    manifest = _manifest(directory)
    if manifest['bytes'] > max_bytes:
        raise ValueError('restore byte budget')
    destination = Path(destination).absolute()
    if destination.exists():
        raise ValueError('restore requires a new directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with attempt(destination.parent) as (temporary, _lease):
        stage = temporary/'output'
        stage.mkdir()
        _consume(directory, manifest, stage)
        for link in manifest.get('links', []):
            path, target = stage/_relative(link['path']), stage/_relative(link['target'])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(os.path.relpath(target, path.parent))
        for parent, _, _files in os.walk(stage, topdown=False):
            syncdir(parent)
        if manifest['schema'] == 'flagprof-delivery/2':
            objects = manifest['objects']
        else:
            objects = {source: 'payload/' + source.lstrip('/') for source in manifest['sources']}
        mapping = {key: str(destination/_relative(value)) for key,value in objects.items()}
        publish(stage/'RESTORED.json', dict(schema=manifest['schema'], manifest=fileinfo(Path(directory)/'manifest.json'),
                                             objects=objects, references=manifest.get('references', {})))
        publish_run(stage, destination)
    return dict(status='restored', mapping=mapping, files=len(manifest['files']), bytes=manifest['bytes'])
