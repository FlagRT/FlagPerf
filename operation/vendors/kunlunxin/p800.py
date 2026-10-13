# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit P800 Operation profile; the historical Kunlunxin route stays intact."""
import hashlib
import json
from pathlib import Path

from vendors.compat import Adapter as CompatAdapter

PROFILE = Path(__file__).parent / 'profiles/p800-xpytorch29.json'


class Adapter(CompatAdapter):
    def __init__(self):
        super().__init__('kunlunxin')
        self.lock = json.loads(PROFILE.read_text())

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
        self.ensure_execution_ready()

    def kernel_time(self, fn, cfg):
        return None
