#!/usr/bin/env python3
"""Bounded PR0 launcher: single-card API probes or selected-device mapping smoke."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import getpass
from pathlib import Path
import shlex
import stat
import subprocess
import time
import uuid

from verify_runtime import ROOT, sha256, validate_manifest
from device_mapping import normalize_bdf, selected_device, device_identity, validate_device_set


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




def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--card', type=int, help='single xpu-smi index')
    selection.add_argument('--cards', type=int, nargs='+', help='explicit xpu-smi indices in requested rank order')
    parser.add_argument('--mapping-only', action='store_true', help='identity and tiny FP32 on each selected device; no collectives')
    parser.add_argument('--privilege-command', default='sudo -n', help='host privilege prefix, or empty when already authorized')
    parser.add_argument('--reservation-end', required=True, help='ISO time with timezone')
    parser.add_argument('--reservation-reference', required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--container-timeout', type=int, default=360)
    parser.add_argument('--profile-route', action='store_true', help='compare native FP32 kernel launch counters with a control')
    args = parser.parse_args()
    cards = [args.card] if args.card is not None else args.cards
    if any(card < 0 for card in cards) or len(set(cards)) != len(cards):
        raise ValueError('card indices must be distinct and nonnegative')
    if len(cards) > 1 and not args.mapping_only:
        raise ValueError('multiple cards currently require --mapping-only; collectives are not implemented')
    if args.mapping_only and args.profile_route:
        raise ValueError('route profiling is only available for the full single-card probes')
    # Enumerate the selected subset, then match framework UUIDs before any work.
    visibility = ','.join(str(i) for i in range(len(cards)))
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
    repo = Path(checked(['git', '-C', str(ROOT), 'rev-parse', '--show-toplevel']).strip())
    profile = '/workspace/FlagPerf/'+str(ROOT.relative_to(repo))
    privilege = shlex.split(args.privilege_command)
    docker = privilege+['docker']
    name = 'flagperf-p800-pr0-'+uuid.uuid4().hex[:12]
    cid = None
    summary = {'status': 'preflight', 'container_name': name, 'cards': cards,
               'validated': False, 'reservation_end': args.reservation_end}
    lock_fds = []
    try:
        inspected = json.loads(checked(docker+['image', 'inspect', manifest['image_id']]))[0]
        image = {key: inspected.get(key) for key in ('Id', 'RepoDigests', 'Architecture', 'Created')}
        validate_manifest(manifest, image)
        save(result_dir/'image-inspect.json', image)
        preflight = command(['xpu-smi', '-m'])
        save(result_dir/'preflight.json', preflight)
        if preflight['returncode']:
            raise RuntimeError('xpu-smi preflight failed')
        devices = []
        for card in cards:
            selected = selected_device(preflight['stdout'], card)
            if selected['used_memory_mib'] or selected['utilization_percent']:
                raise RuntimeError(f'selected card {card} is occupied')
            query = command(['xpu-smi', '-i', str(card), '-q'])
            save(result_dir/f'card-{card}-query.json', query)
            if query['returncode']:
                raise RuntimeError('selected device identity query failed')
            identity = device_identity(query['stdout'], selected['pci_bdf'])
            node_stat = Path(identity['host_device_node']).stat()
            if not stat.S_ISCHR(node_stat.st_mode) or os.minor(node_stat.st_rdev) != identity['device_minor']:
                raise ValueError('host node minor does not match query')
            devices.append(dict(identity, host_physical_id=card,
                                container_node=identity['host_device_node'],
                                pci_bdf=selected['pci_bdf'], serial=selected['serial']))
        validate_device_set(devices)
        # UUID remains stable when management indices change; lock in stable order.
        for device in sorted(devices, key=lambda d: d['uuid']):
            fd = os.open(f"/tmp/flagperf-p800-{device['uuid']}.lock",
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o666)
            lock_fds.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            users = command(privilege+['fuser', device['host_device_node']])
            save(result_dir/f"card-{device['host_physical_id']}-users.json", users)
            # Fail closed even for monitoring handles; rerun after sampling ends.
            if users['returncode'] != 1 or users['stdout'].strip() or users['stderr'].strip():
                raise RuntimeError('selected device has open handles or occupancy check failed')
        # Recheck identity and occupancy after acquiring every selected-device lock.
        fresh = checked(['xpu-smi', '-m'])
        for device in devices:
            selected = selected_device(fresh, device['host_physical_id'])
            identity = device_identity(checked(['xpu-smi', '-i', str(device['host_physical_id']), '-q']),
                                       selected['pci_bdf'])
            if selected['used_memory_mib'] or selected['utilization_percent']:
                raise RuntimeError('selected device became occupied')
            if selected['pci_bdf'] != device['pci_bdf'] or any(identity[k] != device[k] for k in identity):
                raise RuntimeError('device identity changed during preflight')
        binding = dict(schema_version=2 if args.mapping_only else 1,
                       cuda_visible_devices=visibility, xpu_visible_devices=None,
                       reservation_start=now.isoformat(), reservation_end=end.isoformat(),
                       reservation_reference=args.reservation_reference)
        if args.mapping_only:
            binding['devices'] = devices
        else:
            binding.update(devices[0], logical_device=0)
        save(result_dir/'binding.json', binding)
        save(result_dir/'code-identity.json', {
            'head': checked(['git', '-C', str(repo), 'rev-parse', 'HEAD']).strip(),
            'status': checked(['git', '-C', str(repo), 'status', '--short']),
            'profile_sha256': {p.name: sha256(p) for p in sorted(ROOT.iterdir())
                               if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.yaml')}})
        device_flags = [arg for device in devices for arg in
                        ('--device', device['host_device_node']+':'+device['container_node'])]
        argv = docker+['create', '--cap-drop', 'ALL', *device_flags,
                      '--name', name, '--label', 'owner='+getpass.getuser(), '--label', 'task=p800-pr0',
                      '--network', 'none', '--security-opt', 'no-new-privileges',
                      '--group-add', str(os.getgid()),
                      '--read-only', '--pids-limit', '512', '--shm-size', '128m',
                      '--tmpfs', '/tmp:rw,nosuid,size=512m',
                      '--tmpfs', '/root/.cache:rw,nosuid,size=512m',
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
        if args.mapping_only:
            argv.append('--mapping-only')
        if args.profile_route:
            argv[len(docker)+1:len(docker)+1] = ['--env', 'P800_PROFILE_ROUTE=1', '--env', 'XPU_ENABLE_PROFILER_TRACING=1']
        save(result_dir/'launch-command.json', argv)
        cid = checked(argv).strip()
        actual = json.loads(checked(docker+['inspect', cid]))[0]
        actual_devices = actual['HostConfig']['Devices']
        expected = {(d['host_device_node'], d['container_node']) for d in devices} | {('/dev/xpuctrl', '/dev/xpuctrl')}
        if actual['Image'] != manifest['image_id'] or actual['HostConfig']['Privileged']:
            raise RuntimeError('container image or privilege mismatch')
        if {(d['PathOnHost'], d['PathInContainer']) for d in actual_devices} != expected:
            raise RuntimeError('container mapping mismatch')
        save(result_dir/'container-binding.json', {'id': cid, 'image': actual['Image'], 'devices': actual_devices,
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
                for card in cards:
                    sample = command(['xpu-smi', '-i', str(card), '-q'], timeout=10)
                    sample['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
                    sample['physical_id'] = card
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
        for fd in lock_fds:
            os.close(fd)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get('exit_code') == 0 and summary.get('container_absent') else 1


if __name__ == '__main__':
    raise SystemExit(main())
