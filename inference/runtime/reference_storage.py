"""Schema 4 core references and independently transportable evidence attachments."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import uuid

from runtime.common import digest, file_hash, read_json, write_json
from runtime.artifacts import atomic_json


def validate_attachment(core, evidence):
    core, evidence = Path(core).resolve(), Path(evidence).resolve()
    header = read_json(core/'manifest.json')
    if header.get('kind') == 'a100_performance':
        from runtime.performance_baseline import load_manifest
    else:
        from runtime.oracle import load_manifest
    m = load_manifest(core)
    value = read_json(evidence/'manifest.json')
    unsigned = {k: v for k, v in value.items() if k != 'attachment_id'}
    expected = m['evidence_attachment']['attachment_id']
    if value.get('attachment_id') != expected or digest(unsigned) != expected:
        raise ValueError('evidence attachment digest mismatch')
    identity = 'baseline_id' if m['kind'] == 'a100_performance' else 'oracle_id'
    if read_json(evidence/'core-binding.json') != {identity: m[identity], 'attachment_id': expected}:
        raise ValueError('evidence attachment belongs to another core')
    from runtime.oracle import safe_file
    for sha, info in value['objects'].items():
        path = safe_file(evidence, info['path'])
        if info['sha256'] != sha or path.stat().st_size != info['bytes'] or file_hash(path) != sha:
            raise ValueError('evidence object digest mismatch')
    for alias, sha in value['aliases'].items():
        if Path(alias).is_absolute() or any(part in ('', '.', '..') for part in alias.split('/')) or '\\' in alias:
            raise ValueError('unsafe evidence alias')
        if sha not in value['objects']: raise ValueError('missing evidence alias object')
    return value


def materialize_attachment(core, evidence, output):
    """Portable diagnostic view. Internal aliases share copied objects, never originals."""
    import os
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(Path(source).resolve()) or
                              Path(source).resolve().is_relative_to(output) for source in (core, evidence)):
        raise ValueError('diagnostic output must be new and non-overlapping')
    value = validate_attachment(core, evidence)
    output.mkdir(parents=True)
    canonical = {}
    for name, sha in value['aliases'].items():
        path = output/name; path.parent.mkdir(parents=True, exist_ok=True)
        if sha in canonical:
            path.symlink_to(os.path.relpath(canonical[sha], path.parent))
        else:
            info = value['objects'][sha]
            copy_verified(Path(evidence)/info['path'], path, {'bytes': info['bytes'], 'sha256': sha})
            canonical[sha] = path
    return output


def run_members(root):
    root = Path(root).resolve()
    def walk(path, parents):
        real = path.resolve(strict=True)
        if not real.is_relative_to(root):
            raise ValueError('external evidence link: ' + str(path))
        if real.is_dir():
            if real in parents:
                raise ValueError('evidence directory cycle')
            for child in sorted(path.iterdir()):
                if child.name not in ('cache', '__pycache__', '.git', '.work', '.lock', '.artifact-root') and not child.name.startswith('package'):
                    yield from walk(child, parents | {real})
        elif real.is_file():
            yield path.relative_to(root).as_posix(), path, None
        else:
            raise ValueError('unsupported evidence file: ' + str(path))
    yield from walk(root, set())


def copy_verified(source, target, expected=None):
    """Copy and hash in one pass; never link writable output to sealed evidence."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    h, count = hashlib.sha256(), 0
    with source.open('rb') as src, target.open('xb') as dst:
        for chunk in iter(lambda: src.read(4*1024*1024), b''):
            dst.write(chunk); h.update(chunk); count += len(chunk)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('reference source changed during copy: ' + str(source))
    info = {'bytes': count, 'sha256': h.hexdigest()}
    if expected is not None and info != expected:
        raise ValueError('reference copy digest mismatch: ' + str(source))
    return info


def attachment(destination, members):
    """Each logical path maps to a content object. Identical aliases have one copy."""
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    objects, aliases, physical = {}, {}, {}
    for name, path, expected in members:
        stat = path.stat(); key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        sha = expected['sha256'] if expected else physical.get(key)
        if sha not in objects:
            temporary = destination/'objects'/('copy-' + uuid.uuid4().hex)
            info = copy_verified(path, temporary, expected)
            sha = info['sha256']
            target = temporary.with_name(sha)
            if sha in objects:
                temporary.unlink()
            else:
                temporary.rename(target)
                objects[sha] = dict(info, path='objects/'+sha)
            physical[key] = sha
        aliases[name] = sha
    content = {'storage_version': 1, 'objects': objects, 'aliases': aliases}
    attachment_id = digest(content)
    atomic_json(destination/'manifest.json', dict(content, attachment_id=attachment_id))
    return {'attachment_id': attachment_id, 'objects': len(objects), 'logical_files': len(aliases),
            'bytes': sum(info['bytes'] for info in objects.values())}


def seal(core, manifest, evidence, members):
    info = attachment(evidence, members)
    manifest = deepcopy(manifest)
    identity = 'baseline_id' if manifest['kind'] == 'a100_performance' else 'oracle_id'
    old_id = manifest.pop(identity, None)
    old_schema = manifest['schema_version']
    manifest.update(schema_version=4, storage_version=1, protocol_schema_version=old_schema,
                    capabilities={'comparison': True,
                                  ('raw_profiler' if manifest['kind'] == 'a100_performance' else 'diagnostics'): 'requires_evidence_attachment'},
                    evidence_attachment=info)
    if old_id:
        manifest['source_reference'] = {'id': old_id, 'schema_version': old_schema}
    manifest['files'] = {p.relative_to(core).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                         for p in sorted(Path(core).rglob('*')) if p.is_file() and p.name not in ('manifest.json', '.lock')}
    manifest[identity] = digest(manifest)
    atomic_json(Path(core)/'manifest.json', manifest)
    atomic_json(Path(evidence)/'core-binding.json', {identity: manifest[identity], 'attachment_id': info['attachment_id']})
    return manifest


def seal_accuracy(package, manifest):
    package = Path(package)
    members = []
    keep = {'provenance/identity.json', 'provenance/effective.yaml', 'provenance/source-snapshot.json'}
    for path in sorted((package/'provenance').rglob('*')):
        if path.is_file() and path.relative_to(package).as_posix() not in keep:
            members.append((path.relative_to(package).as_posix(), path, None))
    evidence = package.parent/'package-evidence.incomplete'
    # Attachment creation needs the original stage files. Remove only these newly
    # generated duplicates after successful copying, before sealing the core.
    staged_members = list(members)
    members.extend(('capture/'+name, path, expected) for name, path, expected in run_members(package.parent))
    info = attachment(evidence, members)
    for _, path, _ in staged_members: path.unlink()
    identity = manifest.pop('oracle_id', None)
    schema = manifest['schema_version']
    manifest.update(schema_version=4, storage_version=1, protocol_schema_version=schema,
        capabilities={'comparison': True, 'diagnostics': 'requires_evidence_attachment'}, evidence_attachment=info)
    manifest['files'] = {p.relative_to(package).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                         for p in sorted(package.rglob('*')) if p.is_file() and p.name not in ('manifest.json', '.lock')}
    manifest['oracle_id'] = digest(manifest)
    atomic_json(package/'manifest.json', manifest)
    atomic_json(evidence/'core-binding.json', {'oracle_id': manifest['oracle_id'], 'attachment_id': info['attachment_id']})
    return manifest


def _repack(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('repack output must be a new non-overlapping directory')
    header = read_json(source/'manifest.json')
    if header.get('schema_version') == 4:
        raise ValueError('reference already uses schema 4')
    if header.get('kind') == 'a100_performance':
        from runtime.performance_baseline import load_manifest
    elif header.get('kind') == 'a100_accuracy':
        from runtime.oracle import load_manifest
    else:
        raise ValueError('unsupported reference kind')
    from runtime.verification import identity
    initial = {name: identity((source/name).stat()) for name in header.get('files', {})}
    manifest = load_manifest(source)
    if any(identity((source/name).stat()) != stamp for name, stamp in initial.items()):
        raise ValueError('source member changed during validation')
    source_digest = file_hash(source/'manifest.json')
    stage = output.with_name(output.name+'.incomplete-'+uuid.uuid4().hex)
    stage.mkdir(parents=True)
    core = stage/'core'; core.mkdir()
    members = []
    try:
        for name, info in manifest['files'].items():
            # Exact source code, raw profiler and capture logs live in the attachment.
            keep = (name.startswith(('prepared/', 'outputs/')) or name in (
                'inputs.jsonl', 'metrics.json', 'environment.json', 'provenance/identity.json',
                'provenance/source-snapshot.json', 'provenance/effective.yaml',
                'evidence/source-snapshot.json', 'evidence/effective.yaml'))
            if keep:
                if name == 'metrics.json':
                    write_json(core/name, read_json(source/name))
                else:
                    copy_verified(source/name, core/name, info)
            else:
                members.append((name, source/name, info))
        new = deepcopy(manifest)
        new['migration'] = {'source_manifest_sha256': source_digest, 'historical_code_executed': False}
        sealed = seal(core, new, stage/'evidence', members)
        load_manifest(core)
        if any(identity((source/name).stat()) != stamp for name, stamp in initial.items()):
            raise ValueError('source member changed during migration')
        if file_hash(source/'manifest.json') != source_digest:
            raise ValueError('source manifest changed during migration')
        stage.rename(output)
        return sealed
    except BaseException:
        # Leave the explicitly incomplete directory for diagnosing write failures.
        raise


def repack(source, output):
    """Budget before hashing/copying; all derived decoding stays by the output."""
    import tempfile
    from runtime.preflight import capacity
    from runtime.verification import task
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('repack output must be a new non-overlapping directory')
    header = read_json(source/'manifest.json')
    unique = {info['sha256']: info['bytes'] for info in header.get('files', {}).values()}
    required = sum(unique.values()) + 256*1024*1024
    if required > capacity(output)['usable_bytes']:
        raise ValueError('insufficient capacity for reference migration and temporary records')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name+'-validation-', dir=output.parent) as workspace:
        with task(workspace):
            return _repack(source, output)
