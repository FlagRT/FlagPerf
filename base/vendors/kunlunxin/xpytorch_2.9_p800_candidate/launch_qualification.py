#!/usr/bin/env python3
"""Host-only, bounded single-card PR0 launcher. Requires confirmed card/window."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from verify_runtime import ROOT, validate_manifest


def command(argv, timeout=20):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return {'argv': argv, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def cleanup_command(argv, timeout=20):
    try:
        return command(argv, timeout)
    except Exception as exc:
        return {'argv': argv, 'returncode': None, 'stdout': '', 'stderr': str(exc)}


def checked(argv, timeout=20):
    result = command(argv, timeout)
    if result['returncode']:
        raise RuntimeError(json.dumps(result))
    return result['stdout']


def normalize_bdf(value):
    domain, bus, device = value.lower().split(':')
    if int(domain, 16) > 65535:
        raise ValueError('unsupported PCI domain')
    return f'{int(domain, 16):04x}:{bus}:{device}'


def selected_device(raw, card):
    matches = []
    for line in raw.splitlines():
        fields = shlex.split(line)
        if not fields:
            continue
        if len(fields) < 22:
            raise ValueError('unexpected xpu-smi machine format')
        if int(fields[1]) == card:
            if fields[21] != 'P800 OAM':
                raise ValueError('selected device is not P800 OAM')
            matches.append({'physical_id': card, 'pci_bdf': normalize_bdf(fields[0]),
                            'serial': fields[3], 'used_memory_mib': int(fields[17]),
                            'total_memory_mib': int(fields[18]), 'utilization_percent': int(fields[19])})
    if len(matches) != 1:
        raise ValueError('selected card missing or duplicated')
    return matches[0]


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=int, choices=range(8), required=True)
    parser.add_argument('--reservation-end', required=True, help='ISO time with timezone')
    parser.add_argument('--reservation-reference', required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--container-timeout', type=int, default=360)
    args = parser.parse_args()
    end = datetime.fromisoformat(args.reservation_end.replace('Z', '+00:00'))
    now = datetime.now(timezone.utc)
    if end.tzinfo is None or (end-now).total_seconds() < args.container_timeout+30:
        raise ValueError('reservation must cover container timeout and cleanup')
    if not 60 <= args.container_timeout <= 600:
        raise ValueError('container timeout must be 60..600 seconds')
    if not args.reservation_reference.strip():
        raise ValueError('reservation reference required')
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/'image-manifest.json').read_text())
    repo = ROOT.parents[3]
    profile = '/workspace/FlagPerf/'+str(ROOT.relative_to(repo))
    docker = ['sudo', '-n', 'docker']
    name = 'zhiyu-p800-pr0-'+uuid.uuid4().hex[:12]
    cid = None
    summary = {'status': 'preflight', 'container_name': name, 'card': args.card,
               'validated': False, 'reservation_end': args.reservation_end}
    lock_fd = os.open(f'/tmp/flagperf-p800-pr0-card-{args.card}.lock',
                      os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o666)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inspected = json.loads(checked(docker+['image', 'inspect', manifest['image_id']]))[0]
        image = {key: inspected.get(key) for key in ('Id', 'RepoDigests', 'Architecture', 'Created')}
        validate_manifest(manifest, image)
        save(result_dir/'image-inspect.json', image)
        preflight = command(['xpu-smi', '-m'])
        save(result_dir/'preflight.json', preflight)
        if preflight['returncode']:
            raise RuntimeError('xpu-smi preflight failed')
        selected = selected_device(preflight['stdout'], args.card)
        if selected['used_memory_mib'] or selected['utilization_percent']:
            raise RuntimeError('selected card is occupied')
        users = command(['sudo', '-n', 'fuser', f'/dev/xpu{args.card}'])
        save(result_dir/'device-users.json', users)
        if users['returncode'] != 1 or users['stdout'].strip() or users['stderr'].strip():
            raise RuntimeError('selected device has open handles or occupancy check failed')
        binding = dict(schema_version=1, host_physical_id=args.card,
                       container_node=f'/dev/xpu{args.card}', logical_device=0,
                       pci_bdf=selected['pci_bdf'], serial=selected['serial'],
                       reservation_start=now.isoformat(), reservation_end=end.isoformat(),
                       reservation_reference=args.reservation_reference)
        save(result_dir/'binding.json', binding)
        save(result_dir/'code-identity.json', {
            'head': checked(['git', '-C', str(repo), 'rev-parse', 'HEAD']).strip(),
            'status': checked(['git', '-C', str(repo), 'status', '--short'])})
        argv = docker+['create', '--name', name, '--label', 'owner=zhiyu', '--label', 'task=p800-pr0',
                      '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                      '--read-only', '--pids-limit', '512', '--shm-size', '128m',
                      '--tmpfs', '/tmp:rw,nosuid,size=512m',
                      '--tmpfs', '/root/.cache:rw,nosuid,size=512m',
                      '--device', f'/dev/xpu{args.card}:/dev/xpu{args.card}',
                      '--device', '/dev/xpuctrl:/dev/xpuctrl',
                      '--env', f'CUDA_VISIBLE_DEVICES={args.card}',
                      '--env', 'XPU_EVENT_KL3_ENABLE=', '--env', 'USE_FLAGGEMS=0',
                      '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'XDG_CACHE_HOME=/tmp/cache',
                      '--env', 'TRITON_CACHE_DIR=/tmp/triton',
                      '--mount', f'type=bind,src={repo},dst=/workspace/FlagPerf,readonly',
                      '--mount', f'type=bind,src={result_dir},dst=/results',
                      '--entrypoint', '/bin/bash', manifest['image_id'], profile+'/container_bootstrap.sh',
                      '--allow-candidate', '--inspect-json', '/results/image-inspect.json',
                      '--binding-json', '/results/binding.json', '--output-dir', '/results/probes']
        save(result_dir/'launch-command.json', argv)
        cid = checked(argv).strip()
        actual = json.loads(checked(docker+['inspect', cid]))[0]
        devices = actual['HostConfig']['Devices']
        expected = {f'/dev/xpu{args.card}', '/dev/xpuctrl'}
        if actual['Image'] != manifest['image_id'] or actual['HostConfig']['Privileged']:
            raise RuntimeError('container image or privilege mismatch')
        if {d['PathOnHost'] for d in devices} != expected or any(d['PathOnHost'] != d['PathInContainer'] for d in devices):
            raise RuntimeError('container mapping mismatch')
        save(result_dir/'container-binding.json', {'id': cid, 'image': actual['Image'], 'devices': devices,
                                                  'privileged': actual['HostConfig']['Privileged']})
        checked(docker+['start', cid])
        deadline = time.monotonic()+args.container_timeout
        summary['status'] = 'running'
        with (result_dir/'telemetry.jsonl').open('w') as telemetry:
            while True:
                state = json.loads(checked(docker+['inspect', '--format', '{{json .State}}', cid]))
                if not state['Running']:
                    summary.update(status='exited', exit_code=state['ExitCode'], state=state)
                    break
                sample = command(['xpu-smi', '-i', str(args.card), '-q'], timeout=10)
                sample['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
                telemetry.write(json.dumps(sample)+'\n')
                telemetry.flush()
                if time.monotonic() >= deadline or datetime.now(timezone.utc) >= end:
                    summary['status'] = 'timeout'
                    break
                time.sleep(1)
    except Exception as exc:
        summary.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        if cid:
            summary['stop'] = cleanup_command(docker+['stop', '-t', '5', cid], timeout=15)
            save(result_dir/'container-logs.json', cleanup_command(docker+['logs', cid]))
            summary['remove'] = cleanup_command(docker+['rm', '-f', cid])
            remaining = cleanup_command(docker+['ps', '-a', '--no-trunc', '--filter', f'id={cid}', '--format', '{{.ID}}'])
            summary['container_absent'] = remaining['returncode'] == 0 and not remaining['stdout'].strip()
        save(result_dir/'postflight.json', cleanup_command(['xpu-smi', '-m']))
        save(result_dir/'host-summary.json', summary)
        os.close(lock_fd)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get('exit_code') == 0 and summary.get('container_absent') else 1


if __name__ == '__main__':
    raise SystemExit(main())
