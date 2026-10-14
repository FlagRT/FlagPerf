"""PPU host policy and explicit two-phase runtime binding contract."""
import os
import re
import uuid as uuid_module
from pathlib import Path

from base.vendors.protocol import ConfigurationError, DeviceBinding, runtime_root
from base.vendors.thead import runtime
# preflight is imported lazily inside inspect_host/check_identity: it imports
# this module's identity helpers, so a top-level import would be circular.

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


# Bounded performance scopes are registered together with their case
# contracts: the six computation precisions and the two transfer directions
# on day 5, memory/communication on day 6. Every unregistered case still
# fails closed in validate_benchmark and inside the container probe.
PERFORMANCE_CASES = ('computation-FP32:PPU', 'computation-FP16:PPU', 'computation-BF16:PPU',
                     'computation-INT8:PPU', 'computation-FP64:PPU', 'computation-TF32:PPU',
                     'interconnect-h2d:PPU', 'interconnect-d2h:PPU')


class TheadProvider:
    name = 'thead'
    display_name = 'T-Head PPU'
    default_runtime_profile = 'torch210_ppu_candidate'
    supports_preflight = True
    supports_bounded_benchmark = True
    fallback_markers = ()
    # Same no-egress intent as the P800 internal bridge: the NCCL socket
    # bootstrap enumerates interfaces and needs a usable one, so containers
    # join a bridge created with (day 3 prerequisite, verified conflict-free):
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

    def validate_selection(self, context):
        context.validate(require_selection=True)
        if context.physical_device_ids is None:
            raise ConfigurationError('PPU requires --physical-device-ids; legacy aliases are undefined')

    def validate_benchmark(self, request, assets):
        if request.case not in PERFORMANCE_CASES:
            raise ConfigurationError('thead case is not registered for bounded performance yet; '
                                     'case contracts land with the day-5 computation and transfer assets')
        if request.allow_privileged_root:
            raise ConfigurationError('PPU performance uses nonprivileged containers')
        if not 30 <= request.context.timeout <= 600:
            raise ConfigurationError('PPU performance timeout must be 30..600 seconds')
        if request.result_dir is not None and request.context.result_root is not None:
            raise ConfigurationError('choose result-dir or result-root, not both')
        requirements = assets.get('requirements', {})
        if requirements.get('process_scope') != {'nnodes': 1, 'nproc_per_node': 1}:
            raise ConfigurationError('thead bounded performance cases are single-rank per node')
        if request.nproc_per_node not in (None, 1):
            raise ConfigurationError('thead bounded performance cases run exactly one local rank')
        from benchmarks.computation_contract import validate_config as validate_computation
        from benchmarks.transfer_contract import CASES as TRANSFER_CASES
        from benchmarks.transfer_contract import validate_config as validate_transfer
        merged = assets.get('merged_config', {})
        if request.case in TRANSFER_CASES:
            validate_transfer(merged, request.case)
        else:
            validate_computation(merged, request.case)

    def code_identity_files(self):
        """Repository files that define this vendor's bounded runtime identity."""
        return ('configs/thead_ppu_torch210.yaml', 'vendors/thead/runtime_bootstrap.sh')

    def benchmark_spec(self, base_dir, result_dir, config, host, image_id):
        return self.probe_spec(base_dir, result_dir, config, host, image_id)

    def inspect_host(self, config, requested, commands, directory):
        from base.vendors.thead import preflight
        return preflight.inspect_host(config, requested, commands, directory)

    def check_identity(self, first, second):
        from base.vendors.thead import preflight
        preflight.same_identity(first, second)

    def validate_image(self, manifest, observed):
        runtime.validate_manifest(manifest, observed)

    def lease_spec(self, host):
        return {'resource_keys': ['thead/' + d['uuid'] for d in host['devices']],
                'compatibility_paths': [Path('/tmp') / ('flagperf-ppu-' + d['uuid'] + '.lock') for d in host['devices']]}

    def container_policy(self, config):
        return {'network': self.container_network, 'ipc_namespace': 'private', 'pid_namespace': 'private',
                'privileged_root': False, 'read_only_root': True, 'cap_drop': ['ALL']}

    def probe_spec(self, base_dir, result_dir, config, host, image_id):
        control = result_dir / 'control'
        output = result_dir / 'artifacts'
        paths = [(str(base_dir), '/workspace/FlagPerf/base', False),
                 (str(control), '/run/flagperf', False),
                 (str(output), '/workspace/FlagPerf/results', True)]
        devices = [(d['host_device_node'], d['container_node'], 'rwm') for d in host['devices']]
        devices.append(('/dev/alixpu', '/dev/alixpu', 'rwm'))
        visibility = ','.join(str(i) for i in range(len(host['devices'])))
        args = ['--network=' + self.container_network, '--ipc=private', '--cap-drop=ALL',
                '--security-opt=no-new-privileges',
                '--read-only', '--pids-limit=512', '--shm-size=' + config['shm_size'],
                # cap-drop ALL removes DAC_OVERRIDE, so container root needs the
                # host user's group to write the 0o770 artifacts mount.
                '--group-add', str(os.getgid()),
                '--tmpfs', '/tmp:rw,nosuid,size=512m',
                '--env', 'CUDA_VISIBLE_DEVICES=' + visibility,
                # Re-exported by runtime_bootstrap.sh after envsetup.sh so no
                # vendor script can silently clobber the selection.
                '--env', 'FLAGPERF_CVD=' + visibility,
                '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONNOUSERSITE=1',
                '--env', 'XDG_CACHE_HOME=/tmp/cache', '--env', 'TRITON_CACHE_DIR=/tmp/triton']
        for source, target, perms in devices:
            args.extend(['--device', source + ':' + target + ':' + perms])
        for source, target, writable in paths:
            args.extend(['--mount', 'type=bind,src=%s,dst=%s%s' % (source, target, '' if writable else ',readonly')])
        args.extend(['--entrypoint', '/bin/bash', image_id,
                     '/workspace/FlagPerf/base/vendors/thead/runtime_bootstrap.sh',
                     '--context', '/run/flagperf/host-context.json', '--output', '/workspace/FlagPerf/results'])
        return args, {'image_id': image_id, 'devices': devices, 'mounts': paths,
                      'network': self.container_network}

    def resolve_bindings(self, host, context_hash, result):
        if result.get('schema_version') != 1 or result.get('context_sha256') != context_hash:
            raise RuntimeError('runtime binding context/hash mismatch')
        expected = binding_records(host['devices'], result['observed_uuids'])
        if result.get('bindings') != expected:
            raise RuntimeError('runtime bindings differ from host UUID join')
        return [DeviceBinding(**record) for record in expected]

    def bindings(self, preflight, selected):
        raise ConfigurationError('PPU bindings require container UUID evidence; use the bounded path')

    def preflight(self, *args, **kwargs):
        raise ConfigurationError('use benchmark preflight for PPU identity qualification')

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
