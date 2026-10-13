# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""P800 host contracts backed by the existing Base provider and flock leases."""
import copy
import importlib
import json
import os
from pathlib import Path
import socket
import signal
import sys
import time


def base_modules():
    repository = Path(__file__).resolve().parents[3]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    return (importlib.import_module('base.executors.common'),
            importlib.import_module('base.executors.lifecycle'),
            importlib.import_module('base.vendors.kunlunxin.provider'))


def check_process_visibility(proc=Path('/proc')):
    """A silent fuser omission must not turn an unreadable host into 'idle'."""
    mounts = [line.split() for line in (proc / 'mounts').read_text().splitlines()]
    options = [row[3].split(',') for row in mounts if len(row) >= 4 and row[1] == '/proc' and row[2] == 'proc']
    if len(options) != 1 or any(o.startswith('hidepid=') and o not in ('hidepid=0', 'hidepid=off') for o in options[0]):
        raise RuntimeError('P800 host process visibility is incomplete: missing/hidden proc mount')
    count = 0
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fd = os.open(entry / 'fd', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except FileNotFoundError:
            if not entry.exists():
                continue  # Process exited during the observation.
            raise RuntimeError(f'P800 host process handles unavailable: {entry.name}')
        except PermissionError as exc:
            raise RuntimeError('P800 host account cannot inspect all process handles; use an authorized account with full /proc access') from exc
        else:
            os.close(fd);count += 1
    return {'method': 'visible /proc inventory and readable fd directories', 'inspected_processes': count}


class HostSession:
    """Observe before locking, then recheck the same identities while locked."""
    def __init__(self, root, devices, run_id, lock, allow_shared=False):
        self.common, lifecycle, vendor = base_modules()
        self.provider = vendor.KunlunxinProvider()
        self.commands = lifecycle.HostCommands()  # Never escalates privileges implicitly.
        self.root, self.devices, self.run_id = Path(root), list(devices), run_id
        self.config = {'expected_device_ids': lock['expected_device_ids'],
                       'allow_readonly_smi_handles': True}
        self.allow_shared = bool(allow_shared)
        self.lock = lock
        self.initial = None
        self.current = None
        self.lease = None

    def observe(self, label):
        # The legacy override is never inherited. Shared Operation runs use a
        # separate explicit opt-in. Shared observations retain all identity
        # checks and record real holders without classifying them as read-only.
        if os.environ.get('P800_ALLOW_FOREIGN_HANDLES') == '1':
            raise RuntimeError('P800 Operation does not accept P800_ALLOW_FOREIGN_HANDLES=1')
        visibility = check_process_visibility()
        inspect = inspect_shared_host if self.allow_shared else self.provider.inspect_host
        host = inspect(self.config, self.devices, self.commands, self.root / 'host' / label)
        _, _, vendor = base_modules()
        vendor.mapping.validate_device_set(host['devices'])
        if [d['host_physical_id'] for d in host['devices']] != self.devices:
            raise RuntimeError('host observation differs from requested physical device order')
        host = {**host, 'host': socket.gethostname(), 'execution_context': 'host', 'process_visibility': visibility,
                'sharing_policy': 'shared-authorized' if self.allow_shared else 'exclusive-required',
                'performance_qualification': 'partial-shared-host' if self.allow_shared else 'exclusive-host-required'}
        self.common.write_json(self.root / 'host' / (label + '-record.json'), host)
        return host

    def prepare_lease(self):
        self.initial = self.observe('prelease')
        self.lease = self.common.DeviceLease([], root=self.common.DEFAULT_LEASE_ROOT,
                                            run_id=self.run_id, kind='operation',
                                            **self.provider.lease_spec(self.initial))
        return self.lease

    def preflight(self, label='preflight'):
        if self.lease is None or self.lease.acquired_at is None or not self.lease._streams:
            raise RuntimeError('P800 host recheck requires an acquired physical lease')
        current = self.observe(label)
        self.provider.check_identity(self.initial, current)
        self.current = current
        return current

    def binding(self, device):
        if self.current is None:
            raise RuntimeError('P800 device mapping requires locked host preflight')
        matches = [d for d in self.current['devices'] if d['host_physical_id'] == device]
        if len(matches) != 1:
            raise ValueError('device is outside the locked P800 selection')
        return copy.deepcopy(matches[0])


def resolve_framework_bindings(devices, observed_uuids):
    """Pure UUID join; the device worker must supply actual observations."""
    _, _, vendor = base_modules()
    return vendor.binding_records(devices, observed_uuids)


def inspect_shared_host(config, requested, commands, directory):
    """Keep identity checks; record occupancy/holders without asserting exclusivity."""
    _, _, vendor = base_modules()
    check = vendor.preflight
    directory.mkdir(parents=True, exist_ok=False)
    raw = commands.checked(['xpu-smi','-m'], evidence=directory/'machine.json')
    rows = check.machine_inventory(raw['stdout'])
    if sorted(r['physical_id'] for r in rows) != sorted(config['expected_device_ids']):
        raise RuntimeError('host physical inventory differs from configured inventory')
    inventory = {r['physical_id']:r for r in rows}
    locked = json.loads((check.runtime.ROOT/'stack.lock.yaml').read_text())['host_observed']
    devices = [];holders = {}
    for index in requested:
        row = inventory[index]
        raw = commands.checked(['xpu-smi','-i',str(index),'-q'], evidence=directory/f'query-{index}.json')
        info = check.query_record(raw['stdout'], row['pci_bdf'])
        if any(info[k] != locked[k] for k in ('driver','xpu_smi_runtime')):
            raise RuntimeError('host driver/runtime differs from locked observation')
        if row['total_memory_mib'] != info['total_memory_mib']:
            raise RuntimeError('machine/query capacity mismatch')
        devices.append({**info, 'host_physical_id':index, 'pci_bdf':row['pci_bdf'],
                        'serial':row['serial'], 'container_node':info['host_device_node']})
        # Full /proc readability was checked before entering this function.
        node = info['host_device_node']
        owners = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit(): continue
            try:
                if any(os.readlink(fd) == node for fd in (proc/'fd').iterdir()):
                    owners.append({'pid':int(proc.name), 'executable':os.readlink(proc/'exe'),
                                   'classification':'shared-holder; no read-only claim'})
            except FileNotFoundError: continue
        holders[str(index)] = owners
    vendor.mapping.validate_device_set(devices);check.check_nodes(devices)
    return {'schema_version':1,'status':'passed','devices':devices,
            'occupancy_qualification':'shared-authorized; exclusivity not established',
            'holders':holders, 'selection':{'source':'physical-device-ids','requested_ids':requested}}


def cleanup_worker(worker, run_id):
    _, lifecycle, _ = base_modules()
    container = lifecycle.ManagedContainer(lifecycle.HostCommands(), worker['control'], worker['name'], run_id)
    # A docker run client can fail after the daemon has created the container.
    # Recover by exact name and check its ownership label before any deletion.
    container.create_attempted = True
    return container.cleanup()


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
