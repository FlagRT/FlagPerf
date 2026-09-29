"""P800 host policy and explicit two-phase runtime binding contract."""
import json
import os
from pathlib import Path

from base.vendors.protocol import ConfigurationError, DeviceBinding, runtime_root
from base.vendors.kunlunxin import preflight
from base.vendors.kunlunxin.reuse import mapping, runtime
from benchmarks.day6_contract import COMMUNICATION_CASES


class KunlunxinProvider:
    name = 'kunlunxin'
    display_name = 'Kunlunxin P800'
    default_runtime_profile = 'xpytorch_2.9_p800_candidate'
    supports_preflight = True
    supports_bounded_benchmark = True
    fallback_markers = ()  # PR0 did not prove a universal fallback detector.
    # The BKCL/FlagCX socket bootstrap enumerates IPv4/IPv6 interfaces and does
    # not accept a loopback-only namespace, so `--network none` leaves two-rank
    # collectives without a net card.  This bridge is created with `--internal`
    # (no gateway, no external route), which keeps the no-egress intent while
    # giving the runtime a usable interface.  Prerequisite on the host:
    #   docker network create --internal --subnet 172.31.254.0/24 flagperf-p800-internal
    container_network = 'flagperf-p800-internal'

    def runtime_root(self, base_dir, profile):
        if profile not in (None, self.default_runtime_profile):
            raise ConfigurationError('unqualified P800 runtime profile')
        return runtime_root(base_dir, self.name, profile or self.default_runtime_profile)

    def validate_config(self, config):
        if type(config.get('allow_readonly_smi_handles', False)) is not bool:
            raise ConfigurationError('allow_readonly_smi_handles must be boolean')
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

    def validate_benchmark(self, request, assets):
        from benchmarks.computation_contract import validate_config, CASES
        from benchmarks.transfer_contract import validate_config as validate_transfer_config, CASES as TRANSFER_CASES
        from benchmarks.day6_contract import CASES as DAY6_CASES
        if request.case not in (*CASES, *TRANSFER_CASES, *DAY6_CASES):
            raise ConfigurationError('P800 case is not registered')
        if request.case in DAY6_CASES:
            from benchmarks.day6_contract import validate_config as validate_day6_config
            validate_day6_config(assets['merged_config'], request.case)
            if request.case in COMMUNICATION_CASES:
                nproc = request.nproc_per_node or 2
                max_nproc = 2 if request.case == 'interconnect-P2P_intraserver:P800' else 8
                if not 2 <= nproc <= max_nproc:
                    raise ConfigurationError('P800 communication case requires nproc-per-node in 2..%d' % max_nproc)
            elif (request.nproc_per_node or 1) != 1:
                raise ConfigurationError('P800 day-six memory case requires exactly nproc-per-node=1')
        elif request.nproc_per_node not in (None, 1):
            raise ConfigurationError('P800 computation and transfer cases are single-rank')
        if request.case in TRANSFER_CASES:
            validate_transfer_config(assets['merged_config'], request.case)
        elif request.case not in DAY6_CASES:
            validate_config(assets['merged_config'], request.case)
        if assets['environments']:
            raise ConfigurationError('P800 native case must not inject unqualified environment scripts')
        selected = request.context.selection_request()['requested_ids']
        if request.case in COMMUNICATION_CASES:
            required = request.nproc_per_node or 2
        else:
            required = 1
        if len(selected) != required or not set(selected).issubset(request_assets_inventory(request)):
            raise ConfigurationError(f'P800 performance requires exactly {required} configured physical device(s)')
        if not 30 <= request.context.timeout <= 600:
            raise ConfigurationError('P800 performance timeout must be 30..600 seconds; pass --timeout 300')
        if request.allow_privileged_root:
            raise ConfigurationError('P800 performance uses nonprivileged containers')
        if request.result_dir is not None and request.context.result_root is not None:
            raise ConfigurationError('choose result-dir or result-root, not both')

    def benchmark_spec(self, base_dir, result_dir, config, host, image_id):
        return self.probe_spec(base_dir, result_dir, config, host, image_id)

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
        return {'network': self.container_network, 'ipc_namespace': 'private', 'pid_namespace': 'private',
                'privileged_root': False, 'read_only_root': True, 'cap_drop': ['ALL']}

    def probe_spec(self, base_dir, result_dir, config, host, image_id):
        control = result_dir / 'control'
        output = result_dir / 'artifacts'
        paths = [(str(base_dir), '/workspace/FlagPerf/base', False),
                 (str(control), '/run/flagperf', False),
                 (str(output), '/workspace/FlagPerf/results', True)]
        devices = [(d['host_device_node'], d['container_node'], 'rwm') for d in host['devices']]
        devices.append(('/dev/xpuctrl', '/dev/xpuctrl', 'rwm'))
        args = ['--network=' + self.container_network, '--ipc=private', '--cap-drop=ALL',
                '--security-opt=no-new-privileges',
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


def request_assets_inventory(request):
    from executors.common import load_host_config
    return load_host_config(request.context.config)[1]['expected_device_ids']
