# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit P800 Operation profile; the historical Kunlunxin route stays intact."""
import hashlib
import json
from pathlib import Path

from vendors.compat import Adapter as CompatAdapter

PROFILE = Path(__file__).parent / 'profiles/p800-xpytorch29.json'


class Adapter(CompatAdapter):
    strict_worker_lifecycle = True
    postflight_after_preflight = True

    def __init__(self):
        super().__init__('kunlunxin')
        self.lock = json.loads(PROFILE.read_text())
        self.host_session = None

    def default_image(self):
        return self.lock['image']

    def validate_options(self, args, devices):
        if args.execution == 'docker' and any(d not in self.lock['expected_device_ids'] for d in devices):
            raise ValueError('P800 Docker physical device IDs must be unique and in 0..7')
        if args.allow_privileged_root or args.container_device or args.mount:
            raise ValueError('P800 uses managed device mappings and mounts; privileged/additional mappings are not supported')
        if getattr(args, 'profiling', 'off') != 'off':
            raise ValueError('P800 device profiling is not integrated; use --profiling off')

    def task_metadata(self, args, device):
        return {'worker_device': None if args.execution == 'docker' else device,
                'runtime_config': {'runtime_profile': self.lock['runtime_profile'],
                                   'profile_sha256': hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                                   'execution': args.execution,
                                   'expected_image_id': self.lock['image_id'] if args.execution == 'docker' else None},
                'device_binding': {'status': 'pending-host-and-framework-observation' if args.execution == 'docker'
                                  else 'externally-managed', 'requested_id': device}}

    def runtime_identity(self, image, info):
        if info.get('Id') != self.lock['image_id']:
            raise ValueError('P800 image ID differs from the Operation runtime lock')
        return {'runtime_profile': self.lock['runtime_profile'], 'image': image, 'image_id': info['Id'],
                'validated': False, 'validation': 'image-identity-only; Operation execution not validated',
                'expected_packages': self.lock['packages'], 'packages_observed': False}

    def ensure_execution_ready(self):
        raise RuntimeError('P800 Operation device workers are not integrated yet; use --dry-run for planning')

    def bootstrap(self, device_id):
        self.ensure_execution_ready()

    def environment(self, device, mode='probe', local=False):
        # The legacy R300p environment is not a qualified P800 bootstrap.
        self.ensure_execution_ready()

    def docker_options(self, device, args):
        self.validate_options(args, [device])
        if self.host_session is None:
            raise RuntimeError('P800 device mapping requires a host session')
        binding = self.host_session.binding(device)
        return ['--device', binding['host_device_node'] + ':' + binding['container_node'] + ':rwm',
                '--device', '/dev/xpuctrl:/dev/xpuctrl:rwm', '--ipc=private',
                '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=512']

    def lease_root(self):
        from .host import base_modules
        return base_modules()[0].DEFAULT_LEASE_ROOT

    def local_lease_root(self):
        return Path('/tmp/flagperf-operation-kunlunxin-local-leases')

    def create_lease(self, root, devices, run_id):
        from .host import HostSession
        self.host_session = HostSession(root, devices, run_id, self.lock)
        return self.host_session.prepare_lease()

    def preflight(self, root, devices, label='preflight'):
        if self.host_session is None or self.host_session.root != Path(root) or self.host_session.devices != list(devices):
            raise RuntimeError('P800 preflight does not match its host session')
        return self.host_session.preflight(label)

    def bind_tasks(self, tasks, host):
        for task in tasks:
            binding = self.host_session.binding(task['device_id'])
            task['device_binding'] = {**binding, 'status': 'host-observed; framework-pending',
                                      'framework_logical_id': None}
            task['runtime_config'] = {**task['runtime_config'], 'physical_uuid': binding['uuid'],
                                      'pci_bdf': binding['pci_bdf'], 'device_minor': binding['device_minor']}
            task['worker_device'] = None

    def cleanup_worker(self, worker, run_id):
        from .host import cleanup_worker
        return cleanup_worker(worker, run_id)

    def cleanup_process(self, proc):
        from .host import cleanup_process
        return cleanup_process(proc)

    def kernel_time(self, fn, cfg):
        return None
