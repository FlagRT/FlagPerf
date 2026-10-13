"""PPU host policy for T-Head ZW810E; day-2 skeleton, container wiring lands day 3."""
import os
import re
import uuid as uuid_module

from base.vendors.protocol import ConfigurationError, DeviceBinding, runtime_root

# ppu-smi prints the PCI address with an eight-digit domain and the driver
# exposes /dev/alixpu_ppuN with minor N+1 (day1 probes A/B evidence). The BDF
# text keeps the printed domain width so monitor sections and preflight
# records compare without reformatting.
_BDF = r'([0-9A-Fa-f]{8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])'


def normalize_bdf(value):
    match = re.fullmatch(_BDF, value or '')
    if not match:
        raise ValueError('invalid PPU PCI domain/bus/device.function: %r' % (value,))
    return match[1] + ':' + match[2].lower() + ':' + match[3].lower() + '.' + match[4]


def canonical_uuid(value):
    try:
        return str(uuid_module.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError('canonical PPU UUID required') from exc


def validate_device_set(devices):
    if not devices:
        raise ValueError('explicit nonempty device selection required')
    for device in devices:
        physical = device.get('host_physical_id')
        if type(physical) is not int or not 0 <= physical <= 15:
            raise ValueError('physical index must be a PPU slot 0..15')
        node = '/dev/alixpu_ppu%d' % physical
        if device.get('host_device_node') != node or device.get('container_node') != node:
            raise ValueError('device node must be the driver-mapped ' + node)
        if device.get('device_minor') != physical + 1:
            raise ValueError('device minor must be the physical slot plus one')
        if canonical_uuid(device.get('uuid', '')) != device['uuid']:
            raise ValueError('canonical device UUID required')
        if normalize_bdf(device.get('pci_bdf', '')) != device['pci_bdf']:
            raise ValueError('canonical PCI BDF required')
    for field in ('host_physical_id', 'host_device_node', 'uuid', 'pci_bdf'):
        if len({d[field] for d in devices}) != len(devices):
            raise ValueError('duplicate device identity: ' + field)


def binding_records(devices, observed):
    """Join container-visible torch UUIDs to host records; never guess order."""
    validate_device_set(devices)
    observed = [canonical_uuid(value) for value in observed]
    if len(set(observed)) != len(observed) or set(observed) != {d['uuid'] for d in devices}:
        raise ValueError('framework UUID set differs from selected physical cards')
    return [DeviceBinding(vendor='thead', host_physical_id=d['host_physical_id'],
                          pci_bdf=d['pci_bdf'], serial_or_uuid=d['uuid'],
                          host_device_node=d['host_device_node'], container_device_node=d['container_node'],
                          framework_local_rank=rank, framework_logical_id=observed.index(d['uuid']),
                          framework_device_name='cuda:' + str(observed.index(d['uuid'])),
                          request_index=rank,
                          resource_key='thead/' + d['uuid']).record()
            for rank, d in enumerate(devices)]


# Day-1 host survey (probes/day1 U8 evidence) found no resident GPU telemetry
# agent: ppu-smi invocations are short-lived and no daemon holds the device
# nodes open. The whitelist therefore starts empty; a name is honored only
# when listed in THEAD_KNOWN_READONLY_MONITORS, and every other holder still
# fails the run with its classification recorded.
KNOWN_READONLY_MONITORS = {}


def authorized_monitors():
    raw = os.environ.get('THEAD_KNOWN_READONLY_MONITORS', '')
    return {name.strip() for name in raw.split(';') if name.strip()}


class TheadProvider:
    name = 'thead'
    display_name = 'T-Head PPU'
    default_runtime_profile = 'torch210_ppu_candidate'
    supports_preflight = True
    supports_bounded_benchmark = True
    fallback_markers = ()
    # Same no-egress intent as the P800 internal bridge: the NCCL socket
    # bootstrap enumerates interfaces and needs a usable one, so containers
    # join a bridge created with (day 3 prerequisite):
    #   docker network create --internal --subnet 172.31.253.0/24 thead-internal
    container_network = 'thead-internal'

    def runtime_root(self, base_dir, profile):
        if profile not in (None, self.default_runtime_profile):
            raise ConfigurationError('unqualified PPU runtime profile')
        return runtime_root(base_dir, self.name, profile or self.default_runtime_profile)

    def validate_config(self, config):
        if type(config.get('allow_readonly_smi_handles', False)) is not bool:
            raise ConfigurationError('allow_readonly_smi_handles must be boolean')
        if config.get('chip') != 'PPU' or config.get('requires_privileged_root') is not False:
            raise ConfigurationError('PPU profile requires chip=PPU and nonprivileged containers')
        if config.get('host_mounts') != [] or config.get('required_devices') != ['/dev/alixpu']:
            raise ConfigurationError('PPU profile has unreviewed mounts or required devices')
        if config.get('runtime_environment') != {}:
            raise ConfigurationError('PPU runtime_environment must be empty; the image is preconfigured')
        if config.get('shm_size') != '16g':
            raise ConfigurationError('PPU bounded probe requires shm_size=16g for NCCL shared memory')

    def container_policy(self, config):
        return {'network': self.container_network, 'ipc_namespace': 'private', 'pid_namespace': 'private',
                'privileged_root': False, 'read_only_root': True, 'cap_drop': ['ALL']}

    def monitor_targets(self, host, selected=None):
        return [{'physical_id': d['host_physical_id'], 'pci_bdf': d['pci_bdf'], 'uuid': d['uuid'],
                 'device_id': 'thead/' + d['uuid']} for d in host['devices']]

    def monitor_policy(self, enabled):
        return {'enabled': enabled, 'vendor': self.name, 'collector': 'ppu-smi single -q full parse',
                'target_interval_s': 1.0, 'required_samples_per_target': 10,
                'target_resource': 'ppu-device', 'automatic_workload_extension': False,
                'metric_fields': [{'key': 'utilization_percent', 'label': 'Device utilization', 'unit': '%'},
                                  {'key': 'used_memory_mib', 'label': 'Allocated device memory', 'unit': 'MiB'}]}

    def create_monitor(self, targets):
        from monitoring.thead_usage import UsageMonitor
        return UsageMonitor(targets)

    def rank_target(self, targets, selected, local_rank, binding):
        matches = [t for t in targets if binding and t['device_id'] == binding.resource_key]
        return matches[0] if len(matches) == 1 else None

    def measurement_identity(self, target, binding):
        return {'vendor': self.name, 'device_id': target['device_id'], 'binding': binding.record()}

    def validate_selection(self, context):
        raise NotImplementedError('thead selection contract lands on day 3')

    def preflight(self, *args, **kwargs):
        raise NotImplementedError('thead host preflight lands on day 3')

    def bindings(self, preflight, selected):
        raise NotImplementedError('thead binding evidence lands on day 3')

    def docker_args(self, config, bindings):
        raise NotImplementedError('thead container arguments land on day 3')

    def bootstrap(self, base_dir, config):
        raise NotImplementedError('thead runtime bootstrap lands on day 3')

    def worker_python(self, config):
        raise NotImplementedError('thead worker entrypoint lands on day 3')

    def container_preflight(self, config, bindings):
        raise NotImplementedError('thead in-container preflight lands on day 3')

    def inspect_host(self, config, requested, commands, directory):
        raise NotImplementedError('thead host inventory lands on day 3')

    def check_identity(self, first, second):
        raise NotImplementedError('thead identity comparison lands on day 3')

    def validate_image(self, manifest, observed):
        raise NotImplementedError('thead image validation lands on day 3')

    def lease_spec(self, host):
        raise NotImplementedError('thead lease contract lands on day 3')

    def probe_spec(self, base_dir, result_dir, config, host, image_id):
        raise NotImplementedError('thead probe specification lands on day 3')

    def resolve_bindings(self, host, context_hash, result):
        raise NotImplementedError('thead binding resolution lands on day 3')
