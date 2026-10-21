# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Run under -S: check identity before executing any environment site hook."""
import ast
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import time

CPU_PHASES = ('reference', 'reference-check', 'diagnose')
PROFILE = Path(__file__).parent / 'profiles/p800-xpytorch29.json'
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'XPU_VISIBLE_DEVICES', 'XPU_VISIBLE_DEVICE',
            'CUDA_DEVICE_ORDER', 'USE_FLAGGEMS', 'XRAY_DEBUG_ENABLE', 'DISABLE_XPYTORCH',
            'TORCH_DEVICE_BACKEND_AUTOLOAD', 'TRITON_XPU_ARCH', 'TRITON_BACKEND',
            'TRITON_XPU_DEVICE', 'FLAGTREE_BACKEND', 'GEMS_VENDOR', 'LD_PRELOAD',
            'LD_LIBRARY_PATH', 'XPU_FORCE_SHARED_DEVICE_CONTEXT', 'CUDART_DUMMY_REGISTER',
            'CUDART_MODULE_LOADING', 'CUDA_DEVICE_MAX_CONNECTIONS', 'PYTHONPATH',
            'TRITON_INTERPRET', 'TRITON_ALWAYS_COMPILE', 'XPU_ENABLE_PROFILER_TRACING',
            'XPU_CUPTI_ENABLE_DEVICE')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_helper(name):
    path = Path(__file__).resolve().parents[3] / 'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate' / (name + '.py')
    spec = util.spec_from_file_location('p800_static_' + name, path)
    module = util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_identity(profile, python, executable, packages, triton, local):
    if python != profile['python']:
        raise RuntimeError('P800 Python version differs from profile')
    if not local and Path(executable).resolve() != Path(profile['conda_prefix'], 'bin/python').resolve():
        raise RuntimeError('P800 Docker interpreter differs from profile')
    errors = base_helper('verify_runtime').validate_packages(packages, profile['packages'])
    if errors or triton != profile['triton_literal_version']:
        raise RuntimeError('P800 package lock mismatch: ' + repr(errors) + '; Triton=' + str(triton))


def configure(phase):
    if not sys.flags.no_site:
        raise RuntimeError('P800 worker requires -S before site hooks')
    if any(n == 'torch' or n.startswith(('torch.', 'torch_xmlir', 'flag_gems')) for n in sys.modules):
        raise RuntimeError('P800 bootstrap started after backend import')
    began = time.monotonic()
    profile = json.loads(PROFILE.read_text())
    local = os.environ.get('P800_OPERATION_EXECUTION') == 'local'
    prefix = Path(sys.prefix)
    site_path = prefix / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    search = [str(site_path)]
    # Non-executable path entries (including editable packages); never exec .pth
    # lines in a CPU worker. Device hooks run only after metadata has been checked.
    for path in sorted(site_path.glob('*.pth')):
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith(('#', 'import ', 'import\t')):
                candidate = (site_path / line.strip()).resolve()
                if candidate.is_dir(): search.append(str(candidate))
    helper = base_helper('verify_runtime')
    packages = {helper.normalize(d.metadata.get('Name', '')): {'version': d.version}
                for d in metadata.distributions(path=search)}
    literal = None
    triton_path = site_path / 'triton/__init__.py'
    for node in ast.parse(triton_path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets):
            if isinstance(node.value, ast.Constant): literal = node.value.value
    validate_identity(profile, platform.python_version(), sys.executable, packages, literal, local)
    if os.environ.get('TRITON_INTERPRET', '0') != '0' or os.environ.get('TRITON_ALWAYS_COMPILE', '0') != '0':
        raise RuntimeError('P800 execution does not accept Triton interpret/always-compile overrides')
    sys.path.extend(p for p in search if p not in sys.path)
    cpu = phase in CPU_PHASES
    if not local:
        for key in ('XPU_VISIBLE_DEVICES','XPU_VISIBLE_DEVICE','TRITON_XPU_DEVICE'):
            os.environ.pop(key, None)
    os.environ.update(USE_FLAGGEMS='0', XRAY_DEBUG_ENABLE='0', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
                      DISABLE_XPYTORCH='1' if cpu else '0')
    if not cpu:
        import site
        site.main()
    return {'python': platform.python_version(), 'executable': str(Path(sys.executable).resolve()),
            'profile_sha256': digest(PROFILE), 'packages': {k:packages[k] for k in profile['packages']},
            'triton_literal_version': literal, 'cpu_isolated': cpu,
            'site_hooks_executed': not cpu, 'phase': phase,
            'hook_hashes': {p.name:digest(p) for p in sorted(site_path.glob('*.pth'))},
            'environment': {k:os.environ.get(k) for k in ENV_KEYS},
            'bootstrap_seconds': time.monotonic() - began}


def main():
    phase = sys.argv[sys.argv.index('--phase') + 1]
    root = Path(sys.argv[sys.argv.index('--root') + 1])
    control = root / sys.argv[sys.argv.index('--serve') + 1] if '--serve' in sys.argv else root
    try:
        record = configure(phase)
        (control / 'bootstrap.json').write_text(json.dumps(record, indent=2) + '\n')
        os.environ['P800_OPERATION_BOOTSTRAP_RECORD'] = str(control / 'bootstrap.json')
    except Exception as exc:
        (control / 'bootstrap-error.json').write_text(json.dumps({'type':type(exc).__name__, 'error':str(exc)}) + '\n')
        raise
    runpy.run_path(str(Path(__file__).parents[2] / 'runtime/worker.py'), run_name='__main__')


if __name__ == '__main__':
    main()
