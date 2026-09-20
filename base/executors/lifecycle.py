"""Bounded host commands and container ownership/cleanup shared by executors."""
from datetime import datetime, timezone
import json
import re
import shlex
import subprocess
import time


def save(path, value):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


class HostCommands:
    def __init__(self, privilege_command='', deadline=None):
        self.prefix = shlex.split(privilege_command)
        if self.prefix not in ([], ['sudo', '-n']):
            raise ValueError("privilege command must be empty or 'sudo -n'")
        self.deadline = deadline

    def run(self, argv, *, privileged=False, timeout=10, evidence=None):
        argv = [*self.prefix, *argv] if privileged else list(argv)
        started = time.monotonic()
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - started)
        result = {'argv': argv, 'started_at': datetime.now(timezone.utc).isoformat(),
                  'started_monotonic_s': started, 'timeout_s': max(0, timeout)}
        try:
            if timeout <= 0:
                raise subprocess.TimeoutExpired(argv, 0)
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
            result.update(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr, timed_out=False)
        except subprocess.TimeoutExpired as exc:
            def decode(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
            result.update(returncode=124, stdout=decode(exc.stdout), stderr=decode(exc.stderr), timed_out=True)
        except OSError as exc:
            result.update(returncode=127, stdout='', stderr=str(exc), timed_out=False)
        result['finished_monotonic_s'] = time.monotonic()
        save(evidence, result)
        return result

    def checked(self, argv, **kwargs):
        result = self.run(argv, **kwargs)
        if result['returncode']:
            raise RuntimeError(f"command failed ({result['returncode']}): {argv[0]}: {result['stderr']}")
        return result


class ManagedContainer:
    def __init__(self, commands, directory, name, run_id):
        self.commands, self.directory = commands, directory
        self.name, self.run_id = name, run_id
        self.cid = None
        self.create_attempted = False

    def inspect(self, timeout=10):
        result = self.commands.checked(['docker', 'inspect', self.cid or self.name], privileged=True, timeout=timeout)
        records = json.loads(result['stdout'])
        if len(records) != 1:
            raise RuntimeError("container inspect must return one container")
        record = records[0]
        if record.get('Config', {}).get('Labels', {}).get('flagperf.run_id') != self.run_id:
            raise RuntimeError("container ownership differs from this run")
        if not re.fullmatch(r'[0-9a-f]{64}', record.get('Id', '')):
            raise RuntimeError("invalid inspected container ID")
        if self.cid is not None and record['Id'] != self.cid:
            raise RuntimeError("container ID changed")
        self.cid = record['Id']
        return record

    def create(self, arguments, expected):
        self.create_attempted = True
        result = self.commands.checked(['docker', 'create', '--name', self.name,
                                       '--label', 'flagperf.run_id=' + self.run_id, *arguments],
                                      privileged=True, timeout=15, evidence=self.directory / 'container-create.json')
        cid = result['stdout'].strip()
        if not re.fullmatch(r'[0-9a-f]{64}', cid):
            raise RuntimeError("invalid Docker create ID")
        self.cid = cid
        record = self.inspect()
        save(self.directory / 'container-inspect.json', record)
        host = record['HostConfig']
        if record['Image'] != expected['image_id'] or host.get('Privileged'):
            raise RuntimeError("container image/privilege drift")
        actual = {(d['PathOnHost'], d['PathInContainer'], d['CgroupPermissions']) for d in host.get('Devices') or []}
        if actual != set(expected['devices']):
            raise RuntimeError("container device mapping differs from planned selection")
        mounts = {(d['Source'], d['Destination'], d['RW']) for d in record.get('Mounts', []) if d['Type'] == 'bind'}
        if mounts != set(expected['mounts']):
            raise RuntimeError("container mounts differ from planned paths/permissions")
        if host.get('NetworkMode') != 'none' or not host.get('ReadonlyRootfs') or 'ALL' not in (host.get('CapDrop') or []):
            raise RuntimeError("container isolation policy drift")
        if not any(str(s).startswith('no-new-privileges') for s in host.get('SecurityOpt') or []):
            raise RuntimeError("container missing no-new-privileges")
        return record

    def start(self):
        self.commands.checked(['docker', 'start', self.cid], privileged=True,
                              evidence=self.directory / 'container-start.json')

    def wait(self, deadline):
        while time.monotonic() < deadline:
            state = self.inspect(timeout=max(0, deadline - time.monotonic()))['State']
            if not state['Running']:
                if state['Status'] not in ('exited', 'dead'):
                    raise RuntimeError("container was not running or exited")
                return int(state['ExitCode'])
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
        raise TimeoutError("bounded container execution timed out")

    def cleanup(self):
        result = {'container_absent': None, 'status': 'failed', 'commands': []}
        if not self.create_attempted:
            return {'container_absent': True, 'status': 'not-created', 'commands': []}
        try:
            # Recover a create whose CLI timed out after the daemon created it.
            lookup = self.commands.run(['docker', 'ps', '-a', '--no-trunc', '--filter', ('id=' + self.cid) if self.cid else 'name=^/' + self.name + '$',
                                        '--format', '{{.ID}}'], privileged=True, timeout=3)
            result['commands'].append(lookup)
            if lookup['returncode'] == 0 and not lookup['stdout'].strip():
                result.update(container_absent=True, status='passed')
                return result
            record = self.inspect()  # ownership check before every deletion
            logs = self.commands.run(['docker', 'logs', self.cid], privileged=True, timeout=3)
            save(self.directory / 'container-logs.json', logs)
            if record['State']['Running']:
                result['commands'].append(self.commands.run(['docker', 'stop', '-t', '2', self.cid], privileged=True, timeout=5))
            result['commands'].append(self.commands.run(['docker', 'rm', '-f', self.cid], privileged=True, timeout=5))
            remaining = self.commands.run(['docker', 'ps', '-a', '--no-trunc', '--filter', 'id=' + self.cid,
                                           '--format', '{{.ID}}'], privileged=True, timeout=3)
            result['commands'].append(remaining)
            if remaining['returncode'] == 0:
                result['container_absent'] = not remaining['stdout'].strip()
            if result['container_absent'] is True:
                result['status'] = 'passed'
        except (Exception, KeyboardInterrupt) as exc:
            result['error'] = str(exc)
        finally:
            result['container_id'] = self.cid
            save(self.directory / 'cleanup.json', result)
        return result
