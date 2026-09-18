#!/usr/bin/env python3
"""P800 candidate qualification; static audit never imports accelerator packages.

PR0 only: not a Base driver and not an automatic candidate promotion tool.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREFIX = Path('/root/miniconda/envs/python310_torch29_cuda')
PACKAGES = ('torch', 'torch-xray', 'torch-xmlir', 'xmlir', 'xtorch-ops',
            'torch-plugin', 'flag-gems', 'flagtree', 'triton', 'flagcx',
            'vllm', 'vllm-plugin-fl')
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'XPU_VISIBLE_DEVICES', 'XPU_EVENT_KL3_ENABLE',
            'FLAGCX_ADAPTOR', 'GEMS_VENDOR', 'USE_FLAGGEMS', 'CONDA_PREFIX',
            'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'TORCH_DEVICE_BACKEND_AUTOLOAD')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def validate_manifest(manifest, observed=None):
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported manifest schema')
    if manifest.get('vendor') != 'kunlunxin' or manifest.get('chip') != 'P800':
        raise ValueError('wrong vendor/chip')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', manifest.get('image_id', '')):
        raise ValueError('immutable image ID required')
    if manifest.get('validated') is not False or manifest.get('release_stage') != 'candidate':
        raise ValueError('PR0 must remain candidate; cannot promote itself')
    if manifest.get('platform') != 'linux/amd64':
        raise ValueError('expected linux/amd64')
    if observed is not None:
        if isinstance(observed, list):
            if len(observed) != 1:
                raise ValueError('inspect must describe exactly one image')
            observed = observed[0]
        for field, expected in [('Id', manifest['image_id']), ('Architecture', 'amd64')]:
            if observed.get(field) != expected:
                raise ValueError(f'image identity mismatch: {field}')
        if set(observed.get('RepoDigests') or []) != set(manifest['repo_digests']):
            raise ValueError('RepoDigests changed; review provenance before use')


def run_command(argv, timeout=20):
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout)
    return {'argv': argv, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def validate_packages(packages, locked):
    return [f'{name}: expected {version}, got {packages.get(name, {}).get("version")}'
            for name, version in locked.items()
            if packages.get(name, {}).get('version') != version]


def static_audit():
    # Explicit search path: no site processing and no importlib.find_spec on a package.
    site = PREFIX / 'lib/python3.10/site-packages'
    search_paths = {site}
    for path in list(site.glob('*.pth')) + list(site.glob('*.egg-link')):
        for line in path.read_text(errors='replace').splitlines():
            if line.strip() and not line.startswith(('#', 'import ', 'import\t')):
                candidate = (site / line.strip()).resolve()
                if candidate.is_dir():
                    search_paths.add(candidate)
    distributions = {}
    for dist in metadata.distributions(path=[str(p) for p in sorted(search_paths)]):
        name = dist.metadata.get('Name', '')
        if normalize(name) not in {normalize(p) for p in PACKAGES}:
            continue
        item = {'name': name, 'version': dist.version,
                'metadata_path': str(dist._path)}
        direct = dist.read_text('direct_url.json')
        if direct:
            data = json.loads(direct)
            # Retain local/editable provenance, never remote URL credentials.
            item['editable'] = data.get('dir_info', {}).get('editable', False)
            if data.get('url', '').startswith('file://'):
                item['local_source'] = data['url'][7:]
        distributions[normalize(name)] = item
    pth = []
    for path in sorted(site.glob('*.pth')):
        pth.append({'path': str(path), 'sha256': sha256(path),
                    'lines': path.read_text(errors='replace').splitlines()})
    modules = {}
    for name in ('torch', 'torch_xmlir', 'torch_xray', 'xtorch_ops', 'flagcx', 'flag_gems', 'triton'):
        candidates = [p / n for p in sorted(search_paths) for n in (name, name + '.py')]
        modules[name] = [str(p) for p in candidates if p.exists()]
    xmlir = site / 'torch_xmlir'
    source_files = []
    for path in [site/'triton/__init__.py', site/'torch_xmlir/__init__.py',
                 Path('/env/FlagCX/plugin/torch/flagcx/__init__.py'),
                 Path('/env/FlagCX/plugin/torch/_build_config.py')]:
        if path.is_file():
            item = {'path': str(path), 'sha256': sha256(path)}
            for statement in ast.parse(path.read_text()).body:
                if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in statement.targets):
                    if isinstance(statement.value, ast.Constant):
                        item['literal_version'] = statement.value.value
            source_files.append(item)
    editable_sources = []
    for source in (Path('/env/FlagCX'), Path('/env/xvllm-plugin-FL')):
        if source.is_dir():
            editable_sources.append({'path': str(source),
                                     'git_head': run_command(['git', '-c', f'safe.directory={source}', '-C', str(source), 'rev-parse', 'HEAD']),
                                     'git_status': run_command(['git', '-c', f'safe.directory={source}', '-C', str(source), 'status', '--short'])})
    versions = []
    libraries = []
    if xmlir.is_dir():
        for path in sorted(xmlir.rglob('*')):
            if not path.is_file():
                continue
            if (path.name.lower() in ('version.txt', 'version', 'version.json', '_versions.txt')
                    or path.name.endswith('.version')) and path.stat().st_size < 32768:
                versions.append({'path': str(path), 'sha256': sha256(path),
                                 'text': path.read_text(errors='replace')})
            if path.name in ('libbkcl.so', 'libxpurt.so', 'libxpucuda.so', 'libcudart.so'):
                libraries.append({'path': str(path), 'resolved': str(path.resolve()),
                                  'sha256': sha256(path), 'ldd': run_command(['ldd', str(path)])})
    required = {'torch': '2.9.', 'torch-xray': None, 'xmlir': None}
    missing = [name for name in required if name not in distributions]
    errors = [f'missing required distribution metadata: {name}' for name in missing]
    if 'torch' in distributions and not distributions['torch']['version'].startswith('2.9.'):
        errors.append('unexpected torch version')
    unresolved = [item['path'] for item in libraries
                  if item['ldd']['returncode'] or 'not found' in item['ldd']['stdout']]
    return {'python': platform.python_version(), 'executable': sys.executable,
            'inventory_errors': errors, 'unresolved_ldd_paths': unresolved,
            'library_qualification': 'pending runtime loader verification',
            'metadata_search_paths': [str(p) for p in sorted(search_paths)],
            'source_files': source_files, 'editable_sources': editable_sources,
            'architecture': platform.machine(), 'site_processing_disabled': bool(sys.flags.no_site),
            'torchrun': {'path': str(PREFIX / 'bin/torchrun'),
                         'exists': (PREFIX / 'bin/torchrun').is_file()},
            'packages': distributions, 'module_files': modules, 'pth_files': pth,
            'vendor_version_files': versions, 'libraries': libraries,
            'environment': {key: os.environ.get(key) for key in ENV_KEYS},
            'visible_device_nodes': sorted(str(p) for p in Path('/dev').glob('xpu*'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--inspect-json', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = {'schema_version': 1, 'scope': 'static-audit', 'validated': False,
              'release_stage': 'candidate', 'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        manifest = json.loads((ROOT / 'image-manifest.json').read_text())
        observed = json.loads(args.inspect_json.read_text()) if args.inspect_json else None
        validate_manifest(manifest, observed)
        if not args.static:
            raise ValueError('use --static here; hardware stages are in hardware_probe.py')
        if not sys.flags.no_site:
            raise ValueError('static audit requires python -S (use container_bootstrap.sh)')
        result['audit'] = static_audit()
        lock = json.loads((ROOT / 'stack.lock.yaml').read_text())
        result['audit']['inventory_errors'].extend(validate_packages(result['audit']['packages'], lock['packages']))
        result['status'] = 'failed' if result['audit']['inventory_errors'] else 'passed'
        result['status_meaning'] = 'inventory collection only; hardware and library qualification pending'
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
