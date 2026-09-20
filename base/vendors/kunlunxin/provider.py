"""P800 host policy and explicit two-phase runtime binding contract."""
import json
import os
from pathlib import Path

from base.vendors.protocol import ConfigurationError, DeviceBinding, runtime_root
from base.vendors.kunlunxin import preflight
from base.vendors.kunlunxin.reuse import mapping, runtime


class KunlunxinProvider:
    name = 'kunlunxin'
    display_name = 'Kunlunxin P800'
    default_runtime_profile = 'xpytorch_2.9_p800_candidate'
    supports_preflight = True
    fallback_markers = ()  # PR0 did not prove a universal fallback detector.

    def runtime_root(self, base_dir, profile):
        if profile not in (None, self.default_runtime_profile):
            raise ConfigurationError('unqualified P800 runtime profile')
        return runtime_root(base_dir, self.name, profile or self.default_runtime_profile)

    def validate_config(self, config):
        if config.get('chip') != 'P800' or config.get('requires_privileged_root') is not False:
            raise ConfigurationError('P800 profile requires chip=P800 and nonprivileged containers')
        if config.get('host_mounts') != [] or config.get('required_devices') != ['/dev/xpuctrl']:
            raise ConfigurationError('P800 profile has unreviewed mounts or required devices')
        if config.get('runtime_environment') != {'USE_FLAGGEMS': '0'}:
            raise ConfigurationError('P800 runtime_environment must be exactly USE_FLAGGEMS=0')
        if config.get('shm_size') != '128m':
            raise ConfigurationError('P800 bounded probe requires shm_size=512m')

    def validate_selection(self, context):
        context.validate(require_selection=True)
        if context.physical_device_ids is None:
            raise ConfigurationError('P800 requires --physical-device-ids; legacy aliases are undefined')

    def inspect_host(self, config, requested, commands, directory):
        return preflight.inspect_host(config, requested, commands, directory)

    def check_identity(self, first, second):
        preflight.same_identity(first, second)

    def validate_image(self, manifest, observed):
        runtime.validate_manifest(manifest, observed)

    def lease_spec(self, host):
        return {'resource_keys': ['kunlunxin/' + d['uuid'] for d in host['devices']],
                'compatibility_paths': [Path('/tmp') / ('flagperf-p800-' + d['uuid'] + '.lock') for d in host['devices']]}

    def container_policy(self, config):
        return {'network': 'none', 'ipc_namespace': 'private', 'pid_namespace': 'private',
                'privileged_root': False, 'read_only_root': True, 'cap_drop': ['ALL']}

    def probe_spec(self, base_dir, result_dir, config, host, image_id):
        control = result_dir / 'control'
        output = result_dir / 'artifacts'
        paths = [(str(base_dir), '/workspace/FlagPerf/base', False),
                 (str(control), '/run/flagperf', False),
                 (str(output), '/workspace/FlagPerf/results', True)]
        devices = [(d['host_device_node'], d['container_node'], 'rwm') for d in host['devices']]
        devices.append(('/dev/xpuctrl', '/dev/xpuctrl', 'rwm'))
        args = ['--network=none', '--ipc=private', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                '--read-only', '--pids-limit=512', '--shm-size=512m', '--group-add', str(os.getgid()),
                '--tmpfs', '/tmp:rw,nosuid,size=512m', '--tmpfs', '/root/.cache:rw,nosuid,size=512m',
                '--env', 'CUDA_VISIBLE_DEVICES=' + ','.join(str(i) for i in range(len(host['devices']))),
                '--env', 'USE_FLAGGEMS=0', '--env', 'PYTHONDONTWRITEBYTECODE=1',
                '--env', 'PYTHONNOUSERSITE=1', '--env', 'XDG_CACHE_HOME=/tmp/cache',
                '--env', 'TRITON_CACHE_DIR=/tmp/triton']
        for source, target, perms in devices:
            args.extend(['--device', source + ':' + target + ':' + perms])
        for source, target, writable in paths:
            args.extend(['--mount', f'type=bind,src={source},dst={target}' + ('' if writable else ',readonly')])
        args.extend(['--entrypoint', '/bin/bash', image_id,
                     '/workspace/FlagPerf/base/vendors/kunlunxin/runtime_bootstrap.sh',
                     '--context', '/run/flagperf/host-context.json', '--output', '/workspace/FlagPerf/results'])
        return args, {'image_id': image_id, 'devices': devices, 'mounts': paths}

    def resolve_bindings(self, host, context_hash, result):
        if result.get('schema_version') != 1 or result.get('context_sha256') != context_hash:
            raise RuntimeError('runtime binding context/hash mismatch')
        expected = binding_records(host['devices'], result['observed_uuids'])
        if result.get('bindings') != expected:
            raise RuntimeError('runtime bindings differ from host UUID join')
        return [DeviceBinding(**record) for record in expected]

    def bindings(self, preflight, selected):
        raise ConfigurationError('P800 bindings require container UUID evidence; performance driver is not implemented')

    def preflight(self, *args, **kwargs):
        raise ConfigurationError('use benchmark preflight for P800 identity qualification')

    def monitor_targets(self, host, selected=None):
        return [{'physical_id': d['host_physical_id'], 'pci_bdf': d['pci_bdf'], 'uuid': d['uuid'],
                 'device_id': 'kunlunxin/' + d['uuid']} for d in host['devices']]

    def monitor_policy(self, enabled):
        return {'enabled': enabled, 'vendor': self.name, 'collector': 'xpu-smi -m and selected -q',
                'target_interval_s': 1.0, 'required_samples_per_target': 10,
                'target_resource': 'p800-device', 'automatic_workload_extension': False,
                'metric_fields': [{'key': 'utilization_percent', 'label': 'Device utilization', 'unit': '%'},
                                  {'key': 'used_memory_mib', 'label': 'Allocated device memory', 'unit': 'MiB'}]}

    def create_monitor(self, targets):
        from monitoring.kunlunxin_usage import UsageMonitor
        return UsageMonitor(targets)

    def rank_target(self, targets, selected, local_rank, binding):
        matches = [t for t in targets if binding and t['device_id'] == binding.resource_key]
        return matches[0] if len(matches) == 1 else None

    def measurement_identity(self, target, binding):
        return {'vendor': self.name, 'device_id': target['device_id'], 'binding': binding.record()}


def binding_records(devices, observed):
    resolved = mapping.resolve_logical_devices(devices, observed)
    return [DeviceBinding(vendor='kunlunxin', host_physical_id=d['host_physical_id'],
                          pci_bdf=d['pci_bdf'], serial_or_uuid=d['uuid'],
                          host_device_node=d['host_device_node'], container_device_node=d['container_node'],
                          framework_local_rank=d['requested_rank'], framework_logical_id=d['logical_device'],
                          framework_device_name='cuda:' + str(d['logical_device']), request_index=d['requested_rank'],
                          resource_key='kunlunxin/' + d['uuid']).record() for d in resolved]
