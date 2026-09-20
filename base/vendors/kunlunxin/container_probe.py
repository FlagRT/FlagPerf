"""Bounded identity/tensor probe. Site hooks run only after host gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import faulthandler
import json
import os
from pathlib import Path
import stat
import sys
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from base.vendors.kunlunxin.reuse import mapping, runtime
from base.vendors.kunlunxin.provider import binding_records


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def validate_context(context, environ, nodes, now=None):
    require(context.get('schema_version') == 1 and context.get('kind') == 'benchmark-preflight', 'invalid host context')
    require(context.get('probe_mode') in ('identity', 'timeout-check'), 'invalid probe mode')
    require(context.get('allow_candidate_runtime') is True, 'candidate runtime needs explicit authorization')
    require(isinstance(context.get('run_id'), str) and context['run_id'], 'missing run identity')
    require(type(context.get('timeout')) is int and 30 <= context['timeout'] <= 180, 'invalid probe timeout')
    require(context.get('reservation_reference'), 'reservation reference required')
    start = datetime.fromisoformat(context['reservation_start'].replace('Z', '+00:00'))
    end = datetime.fromisoformat(context['reservation_end'].replace('Z', '+00:00'))
    require(start.tzinfo and end.tzinfo, 'reservation timezone required')
    require(start <= (now or datetime.now(timezone.utc)) < end, 'outside reservation')
    devices = context['host']['devices']
    mapping.validate_device_set(devices)
    require(len(devices) == 1, 'PR2 real probe is single-device only')
    require(set(nodes) == {d['container_node'] for d in devices} | {'/dev/xpuctrl'}, 'unexpected device nodes')
    require(environ.get('CUDA_VISIBLE_DEVICES') == '0', 'incorrect selected-subset visibility')
    require(environ.get('XPU_VISIBLE_DEVICES') is None and environ.get('XPU_EVENT_KL3_ENABLE') is None, 'unqualified XPU environment')
    require(environ.get('USE_FLAGGEMS') == '0', 'native probe requires USE_FLAGGEMS=0')
    return devices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output
    result = {'schema_version': 1, 'status': 'failed', 'started_monotonic_ns': time.monotonic_ns()}
    try:
        require(sys.flags.no_site, 'probe must start with Python -S')
        raw = args.context.read_bytes()
        context = json.loads(raw)
        context_hash = hashlib.sha256(raw).hexdigest()
        result.update(run_id=context['run_id'], context_sha256=context_hash)
        devices = validate_context(context, os.environ, [str(p) for p in Path('/dev').glob('xpu*')])
        for device in devices:
            info = Path(device['container_node']).lstat()
            require(stat.S_ISCHR(info.st_mode) and os.minor(info.st_rdev) == device['device_minor'] and
                    os.major(info.st_rdev) == device['device_major'], 'node major/minor mismatch')
        require(stat.S_ISCHR(Path('/dev/xpuctrl').lstat().st_mode), 'control node is not character device')
        manifest = json.loads((runtime.ROOT / 'image-manifest.json').read_text())
        runtime.validate_manifest(manifest, context['image_identity'])
        require(Path(sys.executable).resolve() == (runtime.PREFIX / 'bin/python').resolve(), 'wrong runtime Python')
        if not args.worker:
            audit = runtime.static_audit()
            save(output / 'runtime-audit.json', audit)
            require(audit['python'] == json.loads((runtime.ROOT / 'stack.lock.yaml').read_text())['python'], 'Python version drift')
            require(not audit['inventory_errors'], 'runtime inventory audit failed')
            require(not runtime.validate_packages(audit['packages'], json.loads((runtime.ROOT / 'stack.lock.yaml').read_text())['packages']), 'package identity drift')
            # Keep accelerator initialization out of PID 1, as in PR0. The host
            # owns the shorter absolute watchdog and removes the whole container.
            worker = subprocess.run([sys.executable, '-S', str(Path(__file__).resolve()),
                                     '--context', str(args.context), '--output', str(output), '--worker'],
                                    timeout=context['timeout'] + 5, check=False)
            return worker.returncode
        # Only after all side-effect-free checks may executable .pth hooks run.
        import site
        site.main()
        import torch
        import torch_xmlir
        require(torch.cuda.is_available(), 'P800 backend unavailable')
        observed = [str(torch.cuda.get_device_properties(i).uuid).lower() for i in range(torch.cuda.device_count())]
        records = binding_records(devices, observed)  # reject whole set before any allocation
        bindings = {'schema_version': 1, 'run_id': context['run_id'], 'context_sha256': context_hash,
                    'observed_uuids': observed, 'bindings': records}
        save(output / 'runtime-bindings.json', bindings)
        binding = records[0]
        index = binding['framework_logical_id']
        faulthandler.dump_traceback_later(40, repeat=True)
        print('checkpoint: UUID set verified; selecting device', flush=True)
        torch.cuda.set_device(index)
        device = torch.device(binding['framework_device_name'])
        print('checkpoint: creating four-element tensor', flush=True)
        tensor = torch.ones(4, device=device)
        print('checkpoint: synchronizing device', flush=True)
        torch.cuda.synchronize(device)
        require(tensor.device == device and tensor.cpu().tolist() == [1.0] * 4, 'wrong device or tensor readback')
        print('checkpoint: tensor readback passed', flush=True)
        del tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        faulthandler.cancel_dump_traceback_later()
        libraries = []
        for line in Path('/proc/self/maps').read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].startswith('/') and '.so' in parts[5] and any(s in parts[5] for s in ('xpu', 'xmlir', 'xcudart', 'bkcl')):
                libraries.append(parts[5])
        result.update(status='passed', pid=os.getpid(), local_rank=0, binding=binding,
                      torch_version=torch.__version__, extension=str(torch_xmlir.__version__),
                      device_name=torch.cuda.get_device_name(index), tensor_readback=[1.0] * 4,
                      loaded_libraries=[{'path': p, 'sha256': runtime.sha256(p)} for p in sorted(set(libraries)) if Path(p).is_file()],
                      scope='identity and four-element native tensor; no performance/fallback claim')
        observation = {'schema_version': 1, 'kind': 'probe-observation', 'run_id': context['run_id'],
                       'device_id': binding['resource_key'], 'started_monotonic_ns': time.monotonic_ns()}
        save(output / 'probe.json', result)
        save(output / 'probe-observation.json', observation)
        seconds = context['timeout'] + 30 if context['probe_mode'] == 'timeout-check' else 16
        time.sleep(seconds)
        observation['finished_monotonic_ns'] = time.monotonic_ns()
        save(output / 'probe-observation.json', observation)
        require(context['probe_mode'] == 'identity', 'timeout watchdog failed to stop bounded wait')
    except Exception as exc:
        result.update(status='failed', error=str(exc), error_type=type(exc).__name__)
    result['finished_monotonic_ns'] = time.monotonic_ns()
    save(output / 'probe.json', result)
    print(json.dumps(result))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
