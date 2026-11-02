"""Two-rank model/control groups; pinned FlagCX owner survives until process exit."""
from datetime import timedelta
import os

_OWNERS = []


class Parallel:
    def __init__(self, device, backend):
        import torch.distributed as dist
        if int(os.environ.get('WORLD_SIZE', '1')) != 2 or backend not in ('hccl', 'flagcx', 'nccl'):
            raise ValueError('registered TP requires two ranks and hccl, flagcx or nccl')
        self.backend = backend
        self.device = device
        self.control = None
        if backend == 'flagcx':
            previous = os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')
            import flagcx
            if previous is not None:
                os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = previous
        dist.init_process_group(backend, timeout=timedelta(seconds=180))
        if dist.get_backend() != backend:
            raise ValueError('actual process group backend mismatch')
        if backend == 'flagcx':
            _OWNERS.append(dist.distributed_c10d._get_default_group())
        self.control = dist.new_group(backend='gloo', timeout=timedelta(seconds=180))
        from flagprof import current_attachment
        handle = current_attachment()
        if handle:
            handle.state['parallel'] = dict(world_size=2, rank=dist.get_rank(), backend=backend,
                                            model_group=os.environ['FLAGPROF_GROUP_ID']+'/model',
                                            control_group=os.environ['FLAGPROF_GROUP_ID']+'/control')
            if handle.state['effective_config'].get('probes') != 'none':
                from flagprof.probes.communication import CommunicationProbe
                probe = CommunicationProbe(handle.probe_manager, self.control)
                handle.probe_manager.communication = probe
                probe.install()
            handle._save()

    def mesh(self):
        from torch.distributed.device_mesh import init_device_mesh
        mesh=init_device_mesh(self.device.split(':')[0], (2,), mesh_dim_names=('tp',))
        from flagprof import current_attachment
        handle=current_attachment()
        if handle and getattr(handle.probe_manager,'communication',None):
            handle.probe_manager.communication.model_groups.append(mesh.get_group('tp'))
        return mesh

    def barrier(self):
        import torch.distributed as dist
        dist.barrier(group=self.control)

    def _check(self, state):
        import torch.distributed as dist
        values = [None, None]
        dist.all_gather_object(values, state, group=self.control)
        if values[0] != values[1]:
            raise ValueError('TP rank execution state mismatch')

    def check_step(self, context, tokens):
        state = {k:context[k] for k in ('phase','generation_step','mode','active_samples','kv_length_before')}
        self._check(dict(state, tokens=tokens))

    def check_request(self, samples, tokens, stops):
        self._check(dict(samples=samples, tokens=tokens, stop_reasons=stops))

    def close(self):
        import torch.distributed as dist
        from flagprof import current_attachment
        handle = current_attachment()
        if handle and getattr(handle.probe_manager, 'communication', None):
            handle.probe_manager.communication.uninstall()
        if dist.is_initialized():
            if self.control is not None:
                dist.destroy_process_group(self.control)
                self.control = None
            dist.destroy_process_group()
