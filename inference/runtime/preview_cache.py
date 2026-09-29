# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Copy sealed Triton groups between identical preview identities; never share writers."""
from pathlib import Path
import shutil
import tempfile
import time

from runtime.common import read_json, write_json, file_hash


def cache_directory(root, stack):
    return Path(root)/'cache'/('tree-'+stack.get('flagtree', 'off')+'-cx-'+stack.get('flagcx', 'off'))/'triton'


def local_file(root, relative):
    name = Path(relative)
    path = root/name
    if name.is_absolute() or '..' in name.parts or path.is_symlink():
        raise ValueError('external cache member')
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('missing or external cache member')
    return path


class CacheLedger:
    def __init__(self, root, key, stack):
        self.root = Path(root).resolve()
        self.directory = cache_directory(self.root, stack)
        self.key = key
        self.groups = {}
        self.fingerprints = {}

    def seal(self):
        """Called after worker exit; incomplete groups are never exported."""
        started = time.monotonic()
        present = set()
        for index in sorted(self.directory.glob('*/__grp__*.json')):
            relative = index.relative_to(self.directory).as_posix()
            present.add(relative)
            try:
                local_file(self.directory, relative)
                children = read_json(index).get('child_paths')
                if not isinstance(children, dict) or not children:
                    raise ValueError('empty or unsupported cache group')
                names = []
                for child in children.values():
                    path = Path(child)
                    if not path.is_absolute():
                        raise ValueError('unsupported relative group reference')
                    name = path.relative_to(self.directory).as_posix()
                    local_file(self.directory, name)
                    names.append(name)
                paths = [index, *(self.directory/n for n in names)]
                fingerprint = [(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths]
                if self.fingerprints.get(relative) == fingerprint:
                    continue
                entry = {'index_sha256': file_hash(index),
                         'files': {n: file_hash(self.directory/n) for n in names}}
                # Do not commit a group which changed while being hashed.
                after = [(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths]
                if after != fingerprint:
                    self.groups.pop(relative, None)
                    continue
                self.groups[relative] = entry
                self.fingerprints[relative] = fingerprint
            except (OSError, ValueError, TypeError):
                self.groups.pop(relative, None)
                self.fingerprints.pop(relative, None)
        for relative in set(self.groups)-present:
            self.groups.pop(relative, None)
            self.fingerprints.pop(relative, None)
        return {'schema_version': 1, 'identity_key': self.key,
                'cache_relative': self.directory.relative_to(self.root).as_posix(),
                'groups': dict(self.groups)}, time.monotonic()-started

    def import_from(self, source, manifest, enabled=True):
        started = time.monotonic()
        report = {'status': 'disabled' if not enabled else 'not_available',
                  'imported_groups': 0, 'copied_bytes': 0, 'skipped': []}
        if not enabled or not manifest:
            report['seconds'] = time.monotonic()-started
            return report
        try:
            if manifest.get('schema_version') != 1 or manifest.get('identity_key') != self.key:
                raise ValueError('cache identity/schema mismatch')
            relative = Path(manifest['cache_relative'])
            expected = self.directory.relative_to(self.root)
            if relative != expected:
                raise ValueError('cache environment namespace mismatch')
            source_directory = Path(source).resolve()/relative
            if not source_directory.resolve().is_relative_to(Path(source).resolve()):
                raise ValueError('external cache directory')
            self.directory.mkdir(parents=True, exist_ok=True)
            for name, entry in manifest.get('groups', {}).items():
                try:
                    index = local_file(source_directory, name)
                    if not index.name.startswith('__grp__') or file_hash(index) != entry['index_sha256']:
                        raise ValueError('cache index digest mismatch')
                    children = read_json(index)['child_paths']
                    if not isinstance(children, dict) or not children:
                        raise ValueError('unsupported cache group')
                    translated = {}
                    for child_name, original in children.items():
                        member = Path(original).relative_to(source_directory).as_posix()
                        if member not in entry['files']:
                            raise ValueError('unsealed cache member')
                        translated[child_name] = str(self.directory/member)
                    if set(entry['files']) != {str(Path(p).relative_to(self.directory)) for p in translated.values()}:
                        raise ValueError('cache member list mismatch')
                    # Staging has independent inodes; no source file can be modified by a new worker.
                    with tempfile.TemporaryDirectory(prefix='.preview-import-', dir=self.directory) as temporary:
                        stage = Path(temporary)
                        for member, expected_hash in entry['files'].items():
                            original = local_file(source_directory, member)
                            if file_hash(original) != expected_hash:
                                raise ValueError('cache member digest mismatch')
                            target = stage/member
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(original, target)
                            if file_hash(target) != expected_hash:
                                raise ValueError('cache member changed during copy')
                            existing = self.directory/member
                            if existing.exists() and file_hash(local_file(self.directory, member)) != expected_hash:
                                raise ValueError('destination cache conflict')
                        # Verify the source index again before committing its relocated copy.
                        if file_hash(index) != entry['index_sha256']:
                            raise ValueError('cache index changed during copy')
                        for member in entry['files']:
                            target = self.directory/member
                            target.parent.mkdir(parents=True, exist_ok=True)
                            report['copied_bytes'] += (stage/member).stat().st_size
                            (stage/member).replace(target)
                        write_json(self.directory/name, {'child_paths': translated})
                    report['imported_groups'] += 1
                except (OSError, ValueError, TypeError, KeyError) as error:
                    report['skipped'].append({'group': name, 'reason': str(error)})
            report['status'] = 'imported' if report['imported_groups'] else 'cold_fallback'
        except (OSError, ValueError, TypeError, KeyError) as error:
            report['status'] = 'cold_fallback'
            report['skipped'].append({'reason': str(error)})
        report['seconds'] = time.monotonic()-started
        return report
