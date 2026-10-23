"""Load the PPU image's site packages after the -S identity boundary."""
import json
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import runpy
import sys

PROFILE = Path(__file__).parent / 'profiles/torch210-ppu.json'
CPU_PHASES = ('reference', 'reference-check', 'diagnose')


def checked_preload(phase, profile):
    """Only the locked Asight target may preload the image's glibc pthread."""
    preload = os.environ.get('LD_PRELOAD', '')
    if not preload:
        return {}
    expected = profile.get('asight_preload', {})
    if (phase != 'profile-target' or os.environ.get('FLAGPERF_PPU_COUNTER_TARGET') != '1'
            or preload != 'libpthread.so.0' or not expected.get('path')
            or not expected.get('sha256')):
        raise RuntimeError('PPU does not accept an unreviewed LD_PRELOAD')
    paths = {line.split(maxsplit=5)[5] for line in Path('/proc/self/maps').read_text().splitlines()
             if len(line.split(maxsplit=5)) == 6 and 'libpthread.so' in line.split(maxsplit=5)[5]}
    if paths != {expected['path']}:
        raise RuntimeError('Asight pthread preload mapped path differs from locked image')
    observed = hashlib.sha256(Path(expected['path']).read_bytes()).hexdigest()
    if observed != expected['sha256']:
        raise RuntimeError('Asight pthread preload hash differs from locked image')
    return {'name': preload, 'path': expected['path'], 'sha256': observed}


def configure(phase):
    if not sys.flags.no_site:
        raise RuntimeError('PPU worker requires -S before site hooks')
    if any(n == 'torch' or n.startswith(('torch.', 'flag_gems')) for n in sys.modules):
        raise RuntimeError('PPU bootstrap started after backend import')
    profile = json.loads(PROFILE.read_text())
    if platform.python_version() != profile['python']:
        raise RuntimeError('PPU Python version differs from runtime profile')
    if any(os.environ.get(key, '0') != '0' for key in ('TRITON_INTERPRET', 'TRITON_ALWAYS_COMPILE')):
        raise RuntimeError('PPU requires real execution, not interpreted or compile-only overrides')
    preload = checked_preload(phase, profile)
    site_path = Path(sys.prefix) / ('lib/python%d.%d/site-packages' % sys.version_info[:2])
    search = [str(site_path)]
    for path in sorted(site_path.glob('*.pth')):
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith(('#', 'import ', 'import\t')):
                candidate = (site_path / line.strip()).resolve()
                if candidate.is_dir():
                    search.append(str(candidate))
    packages = {d.metadata.get('Name', '').lower().replace('_', '-'): d.version
                for d in metadata.distributions(path=search)}
    if any(packages.get(name) != version for name, version in profile['packages'].items()):
        raise RuntimeError('PPU package versions differ from locked Operation profile')
    sys.path.extend(p for p in search if p not in sys.path)
    cpu = phase in CPU_PHASES
    os.environ.update(USE_FLAGGEMS='0', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        import site
        site.main()
    return {'python': platform.python_version(), 'executable': str(Path(sys.executable).resolve()),
            'phase': phase, 'cpu_isolated': cpu, 'site_hooks_executed': not cpu,
            'reviewed_preload': preload,
            'profile_sha256': hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
            'packages': {k: packages[k] for k in profile['packages']},
            'hook_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(site_path.glob('*.pth'))}}


def main():
    phase = sys.argv[sys.argv.index('--phase') + 1]
    root = Path(sys.argv[sys.argv.index('--root') + 1])
    control = root / sys.argv[sys.argv.index('--serve') + 1] if '--serve' in sys.argv else root
    control.mkdir(parents=True, exist_ok=True)
    try:
        record = configure(phase)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'error': str(exc)}
        if phase == 'profile-target' and os.environ.get('LD_PRELOAD'):
            error['ld_preload'] = os.environ['LD_PRELOAD']
        (control / 'bootstrap-error.json').write_text(json.dumps(error) + '\n')
        raise
    (control / 'bootstrap.json').write_text(json.dumps(record, indent=2) + '\n')
    os.environ['PPU_OPERATION_BOOTSTRAP_RECORD'] = str(control / 'bootstrap.json')
    runpy.run_path(str(Path(__file__).parents[2] / 'runtime/worker.py'), run_name='__main__')


if __name__ == '__main__':
    main()
