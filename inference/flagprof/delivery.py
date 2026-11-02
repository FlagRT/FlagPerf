# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Stream immutable delivery trees to an archive and restore their relative graph.

Absolute original paths are provenance in the index. Payload member names retain
their common filesystem hierarchy so relative Store/capture/blob links survive.
"""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import uuid

from .offline.checkpoints import fileinfo, identity, publish, syncdir
from .offline.workspace import check_disk


CHUNK = 65536


def _relative(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'payload':
        raise ValueError('delivery member path')
    return p


def _inventory(sources, max_bytes):
    roots = sorted({Path(s).resolve(strict=True) for s in sources})
    if not roots or type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('sources and positive delivery budget required')
    # Nested roots are already included, and must not make duplicate tar members.
    roots = [p for p in roots if not any(q != p and q in p.parents for q in roots)]
    rows = []
    links = []
    total = 0
    for root in roots:
        pending = [root]
        while pending:
            path = pending.pop()
            if path.is_symlink():
                target = path.resolve(strict=True)
                if not any(p == target or p in target.parents for p in roots):
                    raise ValueError('delivery link target outside selected sources: ' + str(path))
                links.append(dict(path='payload/' + str(path).lstrip('/'),
                                  target='payload/' + str(target).lstrip('/'),
                                  original_target=os.readlink(path)))
            elif path.is_dir():
                pending.extend(sorted(path.iterdir(), reverse=True))
            elif path.is_file():
                size = path.stat().st_size
                total += size
                if total > max_bytes:
                    raise ValueError('delivery byte budget')
                rows.append(dict(source=str(path), path='payload/' + str(path).lstrip('/'),
                                 bytes=size, identity=identity(path)))
            else:
                raise ValueError('delivery source type: ' + str(path))
    return roots, rows, links


class _HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def publish_tree(sources, destination, *, max_bytes=200 << 30):
    destination = Path(destination).absolute()
    if destination.exists():
        raise ValueError('delivery destination already exists')
    roots, rows, links = _inventory(sources, max_bytes)
    if any(p == destination or p in destination.parents for p in roots):
        raise ValueError('delivery destination inside source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + '.partial-' + uuid.uuid4().hex)
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
    manifest = dict(schema='flagprof-delivery/1', max_bytes=max_bytes,
                    sources=[str(p) for p in roots], files=rows, links=links,
                    bytes=sum(r['bytes'] for r in rows), archive=fileinfo(stage/'payload.tar'))
    publish(stage/'manifest.json', manifest)
    verify(stage, require_complete=False)
    publish(stage/'COMPLETE.json', dict(manifest=fileinfo(stage/'manifest.json'),
                                       archive=manifest['archive']))
    os.rename(stage, destination)
    syncdir(destination.parent)
    return dict(delivery=str(destination), files=len(rows), bytes=manifest['bytes'])


def _manifest(directory, require_complete=True):
    directory = Path(directory)
    if (directory/'manifest.json').stat().st_size > 64 << 20:
        raise ValueError('delivery manifest byte budget')
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema') != 'flagprof-delivery/1':
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
    for link in manifest.get('links', []):
        path, target = _relative(link['path']), _relative(link['target'])
        if str(path) in names or not any(target == Path(n) or target in Path(n).parents for n in names):
            raise ValueError('delivery link closure')
        if any(path in Path(n).parents for n in names):
            raise ValueError('delivery link overlaps member tree')
        names.add(str(path))
    return manifest


def _consume(directory, manifest, destination=None):
    expected = {r['path']: r for r in manifest['files']}
    seen = set()
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
    check_disk(destination)
    if destination.exists():
        raise ValueError('restore requires a new directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + '.partial-' + uuid.uuid4().hex)
    stage.mkdir()
    _consume(directory, manifest, stage)
    for link in manifest.get('links', []):
        path, target = stage/_relative(link['path']), stage/_relative(link['target'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(os.path.relpath(target, path.parent))
    for parent, _, _files in os.walk(stage, topdown=False):
        syncdir(parent)
    mapping = {source: str(destination/'payload'/source.lstrip('/')) for source in manifest['sources']}
    publish(stage/'RESTORED.json', dict(delivery=str(Path(directory).resolve()),
                                      manifest=fileinfo(Path(directory)/'manifest.json'), mapping=mapping))
    os.rename(stage, destination)
    syncdir(destination.parent)
    return dict(status='restored', mapping=mapping, files=len(manifest['files']), bytes=manifest['bytes'])
