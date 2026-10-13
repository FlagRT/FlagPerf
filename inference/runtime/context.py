"""Explicit device and parallel orchestration, independent of vendor implementations."""
import os
from dataclasses import dataclass
from runtime.adapters import vendor_adapter


@dataclass
class ParallelContext:
    device_type: str
    world_size: int
    mesh: object = None
    backend: str = None
    owner: object = None

    def device_mesh(self):
        if self.mesh is None:
            from torch.distributed.device_mesh import init_device_mesh
            self.mesh = init_device_mesh(self.device_type, (self.world_size,), mesh_dim_names=('tp',))
        return self.mesh


def initialize(cfg):
    adapter = vendor_adapter(cfg)
    device, backend = adapter.initialize(cfg)
    adapter.configure_numerics(cfg, backend)
    if cfg['runtime'].get('parallelism') == 'tp' and 'LOCAL_RANK' in os.environ:
        from runtime.tp import initialize_group
        initialize_group(cfg)
    session = adapter.device_session(backend)
    session.require('synchronize', 'device_info')
    return device, session


def parallel_context(cfg):
    if cfg['runtime'].get('parallelism') != 'tp':
        return None
    from vendors.stack import communication_backend
    return ParallelContext(vendor_adapter(cfg).device_type, len(cfg['runtime']['devices']),
                           backend=communication_backend(cfg))
