"""Join SMI inventory with the device selected by the framework."""
import json
import os
from pathlib import Path
import re
import stat
import subprocess


def _field(output, name):
    values = re.findall(r'^[ \t]*' + re.escape(name) + r'[ \t]*:[ \t]*(\S[^\n]*)$', output, re.M)
    unique = {value.strip() for value in values}
    return unique.pop() if len(unique) == 1 else None


def _inventory(command, prefix, runtime, node_pattern, model_names):
    try:
        output = subprocess.run([command, '-q'], check=True, capture_output=True, text=True,
                                timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f'cannot read {command} inventory: {error}') from error
    blocks = re.split(r'^' + prefix + r'[ \t]+[0-9a-fA-F:.]+[ \t]*$', output, flags=re.M)
    nodes = {}
    for path in Path('/dev').glob(node_pattern):
        info = path.stat()
        if stat.S_ISCHR(info.st_mode):
            nodes.setdefault(os.minor(info.st_rdev), []).append(str(path))
    devices = []
    for block in (blocks[1:] if len(blocks) > 1 else blocks):
        model = model_names.get(_field(block, 'Product Name'))
        if not model:
            continue
        minor = _field(block, 'Minor Number')
        paths = nodes.get(int(minor), []) if minor and minor.isdigit() else []
        device = dict(model=model, uuid=_field(block, prefix + ' UUID'), pci_bdf=_field(block, 'Bus Id'),
                      driver=_field(blocks[0], 'Driver Version') or _field(block, 'Driver Version'), host_device_node=paths[0] if len(paths)==1 else None)
        if runtime:
            device['xpu_smi_runtime'] = _field(blocks[0], runtime) or _field(block, runtime)
        devices.append(device)
    return dict(source=command, devices=devices)


def hardware_identity(selected, *, command, prefix, env_key, node_pattern, model_names, runtime=None):
    import torch
    logical = torch.cuda.current_device() if selected is None else selected
    props = torch.cuda.get_device_properties(logical)
    observed = str(props.uuid).lower().removeprefix('gpu-')
    supplied = os.environ.get(env_key)
    if supplied:
        path = Path(supplied)
        if path.stat().st_size > 65536:
            raise ValueError('identity document byte budget')
        inventory = json.loads(path.read_bytes())
    else:
        try:
            inventory = _inventory(command, prefix, runtime, node_pattern, model_names)
        except ValueError as error:
            raise ValueError(f'{error}; a sourced host inventory may be supplied with {env_key}') from error
    model = next(iter(model_names.values()))
    if inventory.get('source') != command:
        raise ValueError('sourced ' + model + ' identity required')
    matches = [d for d in inventory['devices'] if d.get('uuid') and
               str(d['uuid']).lower().removeprefix('gpu-') == observed]
    if len(matches) != 1:
        raise ValueError(f'{prefix} physical/framework UUID mismatch or ambiguous device')
    device = matches[0]
    if device.get('model') != model:
        raise ValueError('sourced ' + model + ' model required')
    pci = getattr(props, 'pci_bus_id', None)
    if isinstance(pci, str) and pci.lower().lstrip('0') != str(device.get('pci_bdf')).lower().lstrip('0'):
        raise ValueError('physical/framework PCI address mismatch')
    fields = ['pci_bdf', 'driver', 'host_device_node'] + (['xpu_smi_runtime'] if runtime else [])
    for field in fields:
        if not device.get(field):
            label = 'P800' if runtime else prefix
            raise ValueError(f'missing sourced {label} {field}; check {command} or supply {env_key}')
    return dict(device, logical_device=logical, source=command+' host query joined to torch.cuda UUID',
                compatibility_name=torch.cuda.get_device_name(logical))
