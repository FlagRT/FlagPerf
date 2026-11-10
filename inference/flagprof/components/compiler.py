# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Select a compiler before imports; observe its actual source and JIT use."""
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from functools import lru_cache


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def bundle_path(config):
    value = config.get('vendor_compiler') or os.environ.get('FLAGPROF_VENDOR_COMPILER')
    return Path(value).resolve() if value else None


def validate_bundle(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    files = manifest.get('files', {})
    if not files or not manifest.get('source_image') or not (root/'triton/__init__.py').is_file():
        raise ValueError('vendor compiler requires a sealed bundle manifest and triton package')
    actual = {p.relative_to(root).as_posix(): file_sha(p) for p in root.rglob('*')
              if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    if actual != files:
        raise ValueError('vendor compiler bundle differs from its sealed manifest')
    return file_sha(root/'manifest.json')


def distribution_entry(name):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    return Path(dist.locate_file('triton/__init__.py')).resolve()


def select(config, *, cache_root=None):
    """Called in the executing process before optional dependencies import."""
    requested = config.get('flagtree')
    if requested is None:
        return
    if requested not in ('off', 'on'):
        raise ValueError('flagtree requires off or on')
    root = bundle_path(config)
    if requested == 'off':
        if root is None:
            raise ValueError('FlagTree off requires --vendor-compiler or FLAGPROF_VENDOR_COMPILER')
        validate_bundle(root)
        expected = root/'triton/__init__.py'
    else:
        expected = distribution_entry('flagtree')
        if expected is None or not expected.is_file():
            raise ValueError('FlagTree on requires an installed FlagTree distribution')
    loaded = sys.modules.get('triton')
    if loaded is not None and Path(loaded.__file__).resolve() != expected.resolve():
        raise ValueError('compiler already imported from a conflicting source; use a fresh worker')
    if loaded is None:
        # Prepend only the chosen package root. Do not remove unrelated paths.
        location = str(expected.parent.parent)
        sys.path.insert(0, location)
        spec = importlib.util.find_spec('triton')
        if spec is None or Path(spec.origin).resolve() != expected.resolve():
            raise ValueError('compiler selection did not resolve to the requested provider')
    if cache_root is not None:
        key = hashlib.sha256(str(expected.resolve()).encode()).hexdigest()[:16]
        base = Path(os.environ.get('FLAGPROF_COMPONENT_CACHE') or cache_root)/('compiler-'+requested+'-'+key)
        for variable, name in (('TRITON_CACHE_DIR','triton'), ('FLAGGEMS_CACHE_DIR','flaggems'),
                               ('TORCHINDUCTOR_CACHE_DIR','inductor')):
            os.environ[variable] = str(base/name)


def identity():
    loaded = sys.modules.get('triton')
    spec = None if loaded is not None else importlib.util.find_spec('triton')
    if loaded is None and spec is None:
        return dict(status='not_installed', provider=None)
    import triton
    path = Path(triton.__file__).resolve()
    return dict(_identity(str(path), triton.__version__))


@lru_cache(maxsize=4)
def _identity(module_path, version):
    path = Path(module_path)
    tree = distribution_entry('flagtree')
    provider = 'flagtree' if tree == path else 'vendor'
    root = path.parent
    digest = hashlib.sha256()
    for member in sorted(root.rglob('*')):
        if member.is_file() and member.suffix in ('.py', '.so') and '__pycache__' not in member.parts:
            digest.update(member.relative_to(root).as_posix().encode())
            digest.update(bytes.fromhex(file_sha(member)))
    return dict(status='available', provider=provider, version=version,
                module_path=str(path), source_sha256=digest.hexdigest())


class JITObservation:
    """Counts actual Python JIT entries, including runtimes without listeners."""
    def __init__(self, state):
        self.state = state
        self.original = self.wrapper = self.kind = None

    def install(self):
        from triton.runtime.jit import JITFunction
        self.kind = JITFunction
        self.original = JITFunction.run
        owner = self
        def observed(kernel, *args, **kwargs):
            result = owner.original(kernel, *args, **kwargs)
            key = 'warmup_calls' if kwargs.get('warmup', False) else 'run_calls'
            owner.state[key] = owner.state.get(key, 0)+1
            return result
        self.wrapper = observed
        JITFunction.run = observed

    def close(self):
        if self.kind is not None and self.kind.run is self.wrapper:
            self.kind.run = self.original


def source_identity(package):
    spec = importlib.util.find_spec(package)
    if spec is None:
        raise ValueError('selected component is not installed: '+package)
    root = Path(spec.origin).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.so'):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(bytes.fromhex(file_sha(path)))
    return digest.hexdigest()
