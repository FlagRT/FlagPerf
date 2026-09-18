"""P800 identity mapping, independent of torch, hostname and installed card count."""
import re
import shlex
import uuid


def normalize_bdf(value):
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
    if not match or int(match[1], 16) > 65535:
        raise ValueError('invalid or unsupported PCI domain/BDF')
    return f'{int(match[1], 16):04x}:{match[2].lower()}:{match[3].lower()}.{match[4]}'


def selected_device(raw, card):
    if type(card) is not int or card < 0:
        raise ValueError('physical index must be nonnegative')
    matches = []
    for line in raw.splitlines():
        fields = shlex.split(line)
        if not fields:
            continue
        # xpu-smi machine format for the locked runtime; unknown formats fail.
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
    bdf = re.findall(r'^XPU\s+([0-9a-fA-F:.]+)\s*$', raw, re.M)
    minor = re.findall(r'^\s*Minor Number\s*:\s*(\d+)\s*$', raw, re.M)
    uid = re.findall(r'^\s*XPU UUID\s*:\s*GPU-([0-9a-fA-F-]+)\s*$', raw, re.M)
    if len(bdf) != 1 or len(minor) != 1 or len(uid) != 1 or normalize_bdf(bdf[0]) != expected_bdf:
        raise ValueError('query identity missing, ambiguous or PCI mismatch')
    return {'host_device_node': f'/dev/xpu{int(minor[0])}', 'device_minor': int(minor[0]),
            'uuid': str(uuid.UUID(uid[0]))}


def validate_device_set(devices):
    if not devices:
        raise ValueError('explicit nonempty device selection required')
    for device in devices:
        physical = device.get('host_physical_id')
        if type(physical) is not int or physical < 0:
            raise ValueError('physical index must be nonnegative')
        node = device.get('host_device_node', '')
        match = re.fullmatch(r'/dev/xpu(0|[1-9][0-9]*)', node)
        if not match or device.get('container_node') != node:
            raise ValueError('invalid or renamed device node')
        if type(device.get('device_minor')) is not int or device['device_minor'] != int(match[1]):
            raise ValueError('device minor mismatch')
        try:
            if str(uuid.UUID(device.get('uuid', ''))) != device['uuid']:
                raise ValueError('noncanonical UUID')
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError('canonical device UUID required') from exc
        if normalize_bdf(device.get('pci_bdf', '')) != device['pci_bdf']:
            raise ValueError('canonical PCI BDF required')
    for field in ('host_physical_id', 'host_device_node', 'uuid', 'pci_bdf'):
        if len({d[field] for d in devices}) != len(devices):
            raise ValueError(f'duplicate device identity: {field}')


def resolve_logical_devices(devices, observed_uuids):
    """Return requested rank order with actual logical IDs, never guessed order."""
    validate_device_set(devices)
    observed = [str(value).lower() for value in observed_uuids]
    if len(set(observed)) != len(observed) or set(observed) != {d['uuid'] for d in devices}:
        raise ValueError('framework UUID set differs from selected physical cards')
    return [dict(device, requested_rank=rank, logical_device=observed.index(device['uuid']))
            for rank, device in enumerate(devices)]
