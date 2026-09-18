#!/usr/bin/env python3
"""Host-only, bounded single-card PR0 launcher. Requires confirmed card/window."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import re
from pathlib import Path
import shlex
import stat
import subprocess
import time
import uuid

from verify_runtime import ROOT, sha256, validate_manifest


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


def device_identity(raw, expected_bdf):
    bdf = re.search(r'^XPU\s+([0-9a-fA-F:.]+)\s*$', raw, re.M)
    minor = re.search(r'^\s*Minor Number\s*:\s*(\d+)\s*$', raw, re.M)
    uid = re.search(r'^\s*XPU UUID\s*:\s*GPU-([0-9a-fA-F-]+)\s*$', raw, re.M)
    if not bdf or not minor or not uid or normalize_bdf(bdf[1]) != expected_bdf:
        raise ValueError('query identity missing or PCI mismatch')
    if int(minor[1]) not in range(8):
        raise ValueError('unexpected device minor')
    return {'host_device_node': f'/dev/xpu{minor[1]}', 'device_minor': int(minor[1]),
            'uuid': uid[1].lower()}


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=int, choices=range(8), required=True)
    parser.add_argument('--reservation-end', required=True, help='ISO time with timezone')
    parser.add_argument('--reservation-reference', required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--container-timeout', type=int, default=360)
    parser.add_argument('--profile-route', action='store_true', help='compare native FP32 kernel launch counters with a control')
    args = parser.parse_args()
    # The runtime enumerates the sole mapped node as logical device zero.
    visibility = 0
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
    # Container root has no DAC_OVERRIDE after cap-drop; grant only the host
    # result group write access rather than adding broad capabilities.
    result_dir.chmod(0o770)
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
        query = command(['xpu-smi', '-i', str(args.card), '-q'])
        save(result_dir/'selected-device-query.json', query)
        if query['returncode']:
            raise RuntimeError('selected device identity query failed')
        identity = device_identity(query['stdout'], selected['pci_bdf'])
        host_node = identity['host_device_node']
        node_stat = Path(host_node).stat()
        if not stat.S_ISCHR(node_stat.st_mode) or os.minor(node_stat.st_rdev) != identity['device_minor']:
            raise ValueError('host node minor does not match query')
        container_card = identity['device_minor']
        users = command(['sudo', '-n', 'fuser', host_node])
        users['processes'] = []
        for token in users['stdout'].split():
            pid = int(token)
            try:
                comm = Path(f'/proc/{pid}/comm').read_text().strip()
            except FileNotFoundError:
                comm = None
            users['processes'].append({'pid': pid, 'comm': comm})
        save(result_dir/'device-users.json', users)
        # Read-only xpu-smi sampling briefly opens nodes too. A vanished process
        # or known management sampler is not an active workload reservation.
        if users['returncode'] not in (0, 1) or any(p['comm'] not in (None, 'xpu-smi', 'xpu_smi') for p in users['processes']):
            raise RuntimeError('selected device has open handles or occupancy check failed')
        if users['returncode'] == 1 and users['stderr'].strip():
            raise RuntimeError('device handle check reported an error')
        binding = dict(schema_version=1, host_physical_id=args.card,
                       container_node=f'/dev/xpu{container_card}', logical_device=0,
                       cuda_visible_devices=str(visibility),
                       xpu_visible_devices=None,
                       pci_bdf=selected['pci_bdf'], serial=selected['serial'],
                       reservation_start=now.isoformat(), reservation_end=end.isoformat(),
                       reservation_reference=args.reservation_reference)
        binding.update(identity)
        save(result_dir/'binding.json', binding)
        save(result_dir/'code-identity.json', {
            'head': checked(['git', '-C', str(repo), 'rev-parse', 'HEAD']).strip(),
            'status': checked(['git', '-C', str(repo), 'status', '--short']),
            'profile_sha256': {p.name: sha256(p) for p in sorted(ROOT.iterdir())
                               if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.yaml')}})
        argv = docker+['create', '--name', name, '--label', 'owner=zhiyu', '--label', 'task=p800-pr0',
                      '--network', 'none', '--security-opt', 'no-new-privileges',
                      '--group-add', str(os.getgid()),
                      '--read-only', '--pids-limit', '512', '--shm-size', '128m',
                      '--tmpfs', '/tmp:rw,nosuid,size=512m',
                      '--tmpfs', '/root/.cache:rw,nosuid,size=512m',
                      '--device', f'{host_node}:/dev/xpu{container_card}',
                      '--device', '/dev/xpuctrl:/dev/xpuctrl',
                      '--env', f'CUDA_VISIBLE_DEVICES={visibility}',
                      '--env', 'USE_FLAGGEMS=0',
                      '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'XDG_CACHE_HOME=/tmp/cache',
                      '--env', 'TRITON_CACHE_DIR=/tmp/triton',
                      '--mount', f'type=bind,src={repo},dst=/workspace/FlagPerf,readonly',
                      '--mount', f'type=bind,src={result_dir},dst=/results',
                      '--entrypoint', '/bin/bash', manifest['image_id'], profile+'/container_bootstrap.sh',
                      '--allow-candidate', '--inspect-json', '/results/image-inspect.json',
                      '--binding-json', '/results/binding.json', '--output-dir', '/results/probes']
        argv[4:4] = ['--cap-drop', 'ALL']
        if args.profile_route:
            argv[4:4] = ['--env', 'P800_PROFILE_ROUTE=1', '--env', 'XPU_ENABLE_PROFILER_TRACING=1']
        save(result_dir/'launch-command.json', argv)
        cid = checked(argv).strip()
        actual = json.loads(checked(docker+['inspect', cid]))[0]
        devices = actual['HostConfig']['Devices']
        expected = {(host_node, f'/dev/xpu{container_card}'), ('/dev/xpuctrl', '/dev/xpuctrl')}
        if actual['Image'] != manifest['image_id'] or actual['HostConfig']['Privileged']:
            raise RuntimeError('container image or privilege mismatch')
        if {(d['PathOnHost'], d['PathInContainer']) for d in devices} != expected:
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
