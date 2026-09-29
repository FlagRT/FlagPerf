# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Ascend compiler selection before imports; no installation or process-global swap."""
import os
from pathlib import Path
from runtime.common import read_json, file_hash, digest

def selection(cfg):
    return cfg.get('_stack', {'flaggems':'off','flagtree':'off','flagcx':'off'})


def validate_bundle(folder):
    root = Path(folder)
    if not (root/'manifest.json').is_file():
        raise ValueError('vendor compiler bundle is missing; run tools/prepare_vendor_compiler.py --image YOUR_VENDOR_IMAGE')
    manifest = read_json(root/'manifest.json')
    source = manifest.get('source_image', '')
    if (manifest.get('schema_version') != 1 or not isinstance(source, str)
            or not source.startswith('sha256:') or len(source) != 71
            or any(c not in '0123456789abcdef' for c in source[7:])
            or not isinstance(manifest.get('files'), dict) or not manifest['files']):
        raise ValueError('vendor compiler manifest requires a resolved source image ID and file hashes')
    actual = {str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*'))
              if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    if actual != manifest['files']:
        raise ValueError('vendor compiler files differ from sealed manifest; prepare again')
    return digest(manifest)


def worker_environment(cfg, root):
    env = os.environ.copy()
    stack = selection(cfg)
    if cfg['runtime']['vendor'] == 'ascend':
        if stack['flagtree'] == 'off':
            folder = cfg['vendors']['ascend']['vendor_compiler']
            validate_bundle(folder)
            env['PYTHONPATH'] = folder + os.pathsep + env.get('PYTHONPATH','')
        # Disjoint namespaces even when both providers expose the module 'triton'.
        cache = Path(root)/'cache'/('tree-'+stack['flagtree']+'-cx-'+stack['flagcx'])
        env['TRITON_CACHE_DIR'] = str(cache/'triton')
        env['TORCHINDUCTOR_CACHE_DIR'] = str(cache/'inductor')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def compiler_identity(cfg):
    import triton
    path = Path(triton.__file__).resolve()
    stack = selection(cfg)
    root = path.parent
    provider = 'flagtree' if stack['flagtree']=='on' else 'vendor'
    if cfg['runtime']['vendor'] == 'ascend':
        vendor_root = Path(cfg['vendors']['ascend']['vendor_compiler']).resolve()
        if (provider == 'vendor') != path.is_relative_to(vendor_root):
            raise ValueError('loaded compiler does not match the requested provider: '+str(path))
        if provider == 'flagtree':
            import importlib.metadata as md
            distribution = md.distribution('flagtree')
            expected = Path(distribution.locate_file('triton/__init__.py')).resolve()
            if path != expected: raise ValueError('FlagTree import came from an unexpected location')
    files = {str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in ['.py','.so'] and '__pycache__' not in p.parts}
    return {'provider':provider,'import_version':triton.__version__,
            'module_path':str(path),'source_sha256':digest(files),
            'bundle_manifest_sha256':file_hash(Path(cfg['vendors']['ascend']['vendor_compiler'])/'manifest.json')
                if cfg['runtime']['vendor']=='ascend' and provider=='vendor' else None}


def communication_backend(cfg):
    return 'flagcx' if selection(cfg)['flagcx']=='on' else 'hccl'
