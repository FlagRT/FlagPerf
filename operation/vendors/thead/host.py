"""PPU host identity and Base-compatible physical resource leases."""
import copy
import importlib
import os
import json
from pathlib import Path
import socket
import signal
import sys
import time


def base_modules():
    repository = Path(__file__).resolve().parents[3]
    for path in (repository, repository / 'base'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return (importlib.import_module('base.executors.common'),
            importlib.import_module('base.executors.lifecycle'),
            importlib.import_module('base.vendors.thead.provider'))


def check_process_visibility(proc=Path('/proc')):
    mounts = [line.split() for line in (proc / 'mounts').read_text().splitlines()]
    options = [row[3].split(',') for row in mounts if len(row) >= 4 and row[1] == '/proc' and row[2] == 'proc']
    if len(options) != 1 or any(o.startswith('hidepid=') and o not in ('hidepid=0', 'hidepid=off') for o in options[0]):
        raise RuntimeError('PPU host process visibility is incomplete')
    count = 0
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fd = os.open(entry / 'fd', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except FileNotFoundError:
            if not entry.exists():
                continue
            raise RuntimeError('PPU host process handles unavailable: ' + entry.name)
        except PermissionError as exc:
            raise RuntimeError('cannot inspect all host process handles; use an authorized account with full /proc access') from exc
        else:
            os.close(fd)
            count += 1
    return {'method': 'visible /proc inventory and readable fd directories', 'inspected_processes': count}


class HostSession:
    def __init__(self, root, devices, run_id, profile, allow_shared=False):
        self.common, lifecycle, vendor = base_modules()
        self.provider = vendor.TheadProvider()
        self.commands = lifecycle.HostCommands()
        self.root, self.devices, self.run_id, self.profile = Path(root), list(devices), run_id, profile
        self.initial = None
        self.current = None
        self.lease = None
        self.allow_shared = bool(allow_shared)

    def observe(self, label):
        if os.environ.get('THEAD_ALLOW_FOREIGN_HANDLES') == '1':
            raise RuntimeError('PPU Operation does not accept THEAD_ALLOW_FOREIGN_HANDLES=1')
        visibility = check_process_visibility()
        inspect = inspect_shared_host if self.allow_shared else self.provider.inspect_host
        result = inspect(
            {'expected_device_ids': self.profile['expected_device_ids'], 'allow_readonly_smi_handles': True},
            self.devices, self.commands, self.root / 'host' / label)
        if [d['host_physical_id'] for d in result['devices']] != self.devices:
            raise RuntimeError('PPU observation differs from requested physical device order')
        result.update(host=socket.gethostname(), execution_context='host', process_visibility=visibility,
                      sharing_policy='shared-authorized' if self.allow_shared else 'exclusive-required',
                      performance_qualification='partial-shared-host' if self.allow_shared else 'exclusive-host-required')
        self.common.write_json(self.root / 'host' / (label + '-record.json'), result)
        return result

    def prepare_lease(self):
        self.initial = self.observe('prelease')
        self.lease = self.common.DeviceLease([], root=self.common.DEFAULT_LEASE_ROOT,
                                            run_id=self.run_id, kind='operation',
                                            **self.provider.lease_spec(self.initial))
        return self.lease

    def preflight(self, label='preflight'):
        if self.lease is None or not self.lease._streams or self.lease.acquired_at is None:
            raise RuntimeError('PPU preflight requires an acquired physical lease')
        current = self.observe(label)
        self.provider.check_identity(self.initial, current)
        self.current = current
        return current

    def binding(self, device):
        if self.current is None:
            raise RuntimeError('PPU host session is not locked')
        rows = [d for d in self.current['devices'] if d['host_physical_id'] == device]
        if len(rows) != 1:
            raise ValueError('selected PPU device is absent from locked observation')
        return copy.deepcopy(rows[0])


def inspect_shared_host(config, requested, commands, directory):
    """Record real holders without declaring any foreign process read-only."""
    _, _, vendor = base_modules()
    from base.vendors.thead import preflight
    directory.mkdir(parents=True, exist_ok=False)
    raw = commands.checked(['ppu-smi', '-q'], timeout=15, evidence=directory / 'machine.json')
    inventory = preflight.machine_inventory(raw['stdout'])
    if sorted(d['host_physical_id'] for d in inventory) != sorted(config['expected_device_ids']):
        raise RuntimeError('PPU physical inventory differs from configured inventory')
    by_id = {d['host_physical_id']: d for d in inventory}
    devices, holders = [], {}
    for index in requested:
        row = by_id[index]
        raw = commands.checked(['ppu-smi', '-i', str(index), '-q'], timeout=15,
                               evidence=directory / ('query-%d.json' % index))
        info = preflight.query_record(raw['stdout'], row['pci_bdf'])
        if info['driver'] != preflight.runtime.locked_host_observation()['driver']:
            raise RuntimeError('PPU driver differs from profile')
        if any(row[k] != info[k] for k in ('uuid', 'device_minor', 'pci_bdf', 'total_memory_mib')):
            raise RuntimeError('PPU inventory/query identity mismatch')
        devices.append({**info, 'host_physical_id': index, 'container_node': info['host_device_node']})
        node = info['host_device_node']
        owners = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if any(os.readlink(fd) == node for fd in (proc / 'fd').iterdir()):
                    owners.append({'pid': int(proc.name), 'executable': os.readlink(proc / 'exe'),
                                   'classification': 'shared-holder; no read-only claim'})
            except FileNotFoundError:
                continue
        holders[str(index)] = owners
    vendor.validate_device_set(devices)
    preflight.check_nodes(devices)
    result = {'schema_version': 1, 'status': 'passed', 'devices': devices, 'holders': holders,
              'occupancy_qualification': 'shared-authorized; exclusivity not established',
              'selection': {'source': 'physical-device-ids', 'requested_ids': requested}}
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def cleanup_worker(worker, run_id):
    _, lifecycle, _ = base_modules()
    container = lifecycle.ManagedContainer(lifecycle.HostCommands(), worker['control'], worker['name'], run_id)
    # A docker run client can fail after the daemon has created the container.
    # Recover by exact name and check its ownership label before any deletion.
    container.create_attempted = True
    result = container.cleanup()
    # Docker --rm may still be deleting an owned container after stop/rm
    # returns. Confirm disappearance within a bound; never infer success from
    # a removal-in-progress message, and never delete a different container.
    if result.get('container_absent') is False and container.cid:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            observed = container.commands.run(
                ['docker', 'ps', '-a', '--no-trunc', '--filter', 'id=' + container.cid,
                 '--format', '{{.ID}}'], privileged=True, timeout=min(1, remaining))
            result['commands'].append(observed)
            if observed['returncode'] == 0 and not observed['stdout'].strip():
                result.update(container_absent=True, status='passed')
                break
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        lifecycle.save(worker['control'] / 'cleanup.json', result)
    return result

def cleanup_process(proc, timeout=5):
    """Clean the session created by start_new_session, including surviving children."""
    def exists():
        proc.poll()  # Reap the leader before checking the remaining process group.
        try:
            os.killpg(proc.pid, 0)
            return True
        except ProcessLookupError:
            return False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not exists():
            break
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + timeout
        while exists() and time.monotonic() < deadline:
            time.sleep(0.05)
    absent = not exists()
    if absent:
        proc.wait(timeout=timeout)
    return {'process_group': proc.pid, 'process_group_absent': absent,
            'status': 'passed' if absent else 'failed'}
