"""Strict P800 host inventory and occupancy checks; imports never query devices."""
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat

from base.vendors.kunlunxin.reuse import mapping, runtime


def machine_inventory(raw):
    rows = []
    for line in raw.splitlines():
        fields = shlex.split(line)
        if not fields:
            continue
        if len(fields) != 32 or not re.fullmatch(r"\d+", fields[1]):
            raise ValueError("unrecognized xpu-smi -m field schema")
        row = mapping.selected_device(line, int(fields[1]))
        if not (0 <= row['used_memory_mib'] <= row['total_memory_mib'] and
                row['total_memory_mib'] > 0 and 0 <= row['utilization_percent'] <= 100):
            raise ValueError("invalid memory or utilization range")
        # Machine help says MB; identity queries explicitly say MiB. Until
        # cross-checked, these are raw units, not a capacity measurement.
        rows.append(row)
    if not rows:
        raise ValueError("empty xpu-smi inventory")
    for field in ('physical_id', 'pci_bdf', 'serial'):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError("duplicate machine identity: " + field)
    return rows


def one(raw, pattern, label):
    values = re.findall(pattern, raw, re.M)
    if len(values) != 1:
        raise ValueError("missing or ambiguous query field: " + label)
    return values[0]


def query_record(raw, bdf):
    identity = mapping.device_identity(raw, bdf)
    identity['driver'] = one(raw, r'^Driver Version[ \t]*:[ \t]*(\S+)[ \t]*$', 'driver')
    identity['xpu_smi_runtime'] = one(raw, r'^XPU-RT Version[ \t]*:[ \t]*(\S+)[ \t]*$', 'runtime')
    memory = one(raw, r"^    Memory Usage\s*\n((?:[ \t]+[^\n]*\n)+?)(?=    \S)", 'Memory Usage')
    values = {}
    for label in ('Total', 'Used', 'Free'):
        value = one(memory, rf"^\s+{label}\s*:\s*(\d+) MiB\s*$", label)
        values[label.lower() + '_memory_mib'] = int(value)
    values['utilization_percent'] = int(one(raw, r"^\s+Xpu\s*:\s*(\d+) %\s*$", 'utilization'))
    if not (0 <= values['used_memory_mib'] <= values['total_memory_mib'] and
            0 <= values['free_memory_mib'] <= values['total_memory_mib'] and
            values['total_memory_mib'] > 0 and 0 <= values['utilization_percent'] <= 100):
        raise ValueError("invalid query telemetry range")
    process_sections = re.findall(r'^    Processes(?:[ \t]*:[ \t]*(None))?[ \t]*$', raw, re.M)
    if len(process_sections) != 1:
        raise ValueError('missing or ambiguous query field: processes')
    values['processes_empty'] = process_sections[0] == 'None'
    if not values['processes_empty'] and not re.search(r'^        Process ID[ \t]*:[ \t]*[0-9]+[ \t]*$', raw, re.M):
        raise ValueError('unrecognized nonempty process table')
    for field, pattern in [('temperature_c', r"^\s+XPU Current Temp\s*:\s*([0-9.]+) C\s*$"),
                           ('power_w', r"^\s+Power Draw\s*:\s*([0-9.]+) W\s*$")]:
        found = re.findall(pattern, raw, re.M)
        if len(found) == 1:
            value = float(found[0])
            if math.isfinite(value) and value >= 0:
                values[field] = value
    return {**identity, **values}


def check_nodes(devices, control='/dev/xpuctrl'):
    for device in devices:
        info = Path(device['host_device_node']).lstat()
        if not stat.S_ISCHR(info.st_mode) or os.minor(info.st_rdev) != device['device_minor']:
            raise RuntimeError("device node type/minor differs from queried identity")
        device['device_major'] = os.major(info.st_rdev)
    if not stat.S_ISCHR(Path(control).lstat().st_mode):
        raise RuntimeError("missing character control device")


def readonly_smi_arguments(arguments):
    return (arguments in (['-m'], ['-q'], []) or
            len(arguments) == 3 and arguments[0] == '-i' and arguments[1].isdigit() and arguments[2] == '-q')


def inspect_handles(device, commands, directory, allow_observers=False):
    node = device['host_device_node']
    for attempt in range(3):
        handles = commands.run(['fuser', node], privileged=True,
                               evidence=directory / f"handles-{device['host_physical_id']}-{attempt}.json")
        if handles['returncode'] == 1 and not handles['stdout'].strip() and not handles['stderr'].strip():
            return []
        pids = handles['stdout'].split()
        if (not allow_observers or handles['returncode'] != 0 or not pids or
                any(not pid.isdigit() for pid in pids) or handles['stderr'].strip() != node + ':'):
            raise RuntimeError('device has open handles or handle check failed')
        observers = []
        vanished = False
        for pid in pids:
            prefix = directory / f'observer-{pid}-{attempt}'
            executable = commands.run(['readlink', '-e', f'/proc/{pid}/exe'], privileged=True,
                                      evidence=prefix.with_suffix('.exe.json'))
            command = commands.run(['cat', f'/proc/{pid}/cmdline'], privileged=True,
                                   evidence=prefix.with_suffix('.argv.json'))
            if executable['returncode'] or command['returncode']:
                existence = commands.run(['test', '-d', f'/proc/{pid}'], privileged=True,
                                         evidence=prefix.with_suffix('.exists.json'))
                if existence['returncode'] == 1 and not existence['stdout'] and not existence['stderr']:
                    observers.append({'pid': int(pid), 'classification': 'exited-before-inspection'})
                    continue
                vanished = True
                break
            arguments = command['stdout'].rstrip('\0').split('\0')
            if (executable['stdout'].strip() != '/usr/local/bin/xpu-smi' or
                    not arguments or not readonly_smi_arguments(arguments[1:])):
                raise RuntimeError('device handle is not a verified read-only xpu-smi observer')
            observers.append({'pid': int(pid), 'executable': executable['stdout'].strip(),
                              'arguments': arguments[1:], 'classification': 'read-only-observer'})
        if not vanished:
            return observers
    raise RuntimeError('device handle owners changed before they could be verified')


def inspect_host(config, requested, commands, directory):
    directory.mkdir(parents=True, exist_ok=False)
    machine = commands.checked(['xpu-smi', '-m'], evidence=directory / 'machine.json')
    inventory = machine_inventory(machine['stdout'])
    if sorted(row['physical_id'] for row in inventory) != sorted(config['expected_device_ids']):
        raise RuntimeError("host physical inventory differs from configured inventory")
    by_id = {row['physical_id']: row for row in inventory}
    devices = []
    for index in requested:
        if index not in by_id:
            raise RuntimeError("selected physical device absent")
        row = by_id[index]
        query = commands.checked(['xpu-smi', '-i', str(index), '-q'], evidence=directory / f'query-{index}.json')
        info = query_record(query['stdout'], row['pci_bdf'])
        locked = json.loads((runtime.ROOT / 'stack.lock.yaml').read_text())['host_observed']
        if any(info[key] != locked[key] for key in ('driver', 'xpu_smi_runtime')):
            raise RuntimeError('host driver/runtime differs from locked observation')
        if row['used_memory_mib'] or row['utilization_percent'] or info['used_memory_mib'] or info['utilization_percent'] or not info['processes_empty']:
            raise RuntimeError(f"selected physical device {index} is occupied")
        if row['total_memory_mib'] != info['total_memory_mib']:
            raise RuntimeError("machine/query memory unit or capacity mismatch")
        devices.append({**info, 'host_physical_id': index, 'pci_bdf': row['pci_bdf'],
                        'serial': row['serial'], 'container_node': info['host_device_node']})
    mapping.validate_device_set(devices)
    check_nodes(devices)
    observers = {str(device['host_physical_id']): inspect_handles(
        device, commands, directory, config.get('allow_readonly_smi_handles', False)) for device in devices}
    result = {'schema_version': 1, 'status': 'passed', 'devices': devices,
              'readonly_observers': observers,
              'selection': {'source': 'physical-device-ids', 'requested_ids': requested},
              'machine_memory_unit': 'numerically-cross-checked-with-query-MiB'}
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def same_identity(first, second):
    fields = ('host_physical_id', 'uuid', 'pci_bdf', 'host_device_node', 'container_node', 'device_minor', 'device_major')
    if [[d[k] for k in fields] for d in first['devices']] != [[d[k] for k in fields] for d in second['devices']]:
        raise RuntimeError("device identity changed after acquiring leases")
