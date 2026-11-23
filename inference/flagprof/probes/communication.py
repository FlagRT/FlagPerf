# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit collective call identity; host duration never claims device duration."""
from collections import defaultdict
import functools
import os

from .base import Probe, clock

# Wrapped entry points. The functional set covers FSDP2's parameter
# all-gather and gradient reduce-scatter, which bypass torch.distributed.
# The point-to-point set covers Megatron-style pipeline parallelism, whose
# activations cross stages through send/recv and batch_isend_irecv.
DIST_NAMES = ('all_reduce', 'all_gather', 'all_gather_into_tensor', 'all_gather_object',
              'barrier', 'reduce_scatter_tensor', 'broadcast', 'reduce', 'scatter',
              'gather', 'gather_object', 'scatter_object_list', 'all_to_all',
              'all_to_all_single', 'send', 'recv', 'isend', 'irecv', 'batch_isend_irecv')
FUNCTIONAL_NAMES = ('all_reduce', 'all_gather_tensor', 'reduce_scatter_tensor')
# Point-to-point entries carry a peer rank instead of a process group, so they
# get their own role and sequence counter.
P2P_NAMES = frozenset(('send', 'recv', 'isend', 'irecv', 'batch_isend_irecv'))
OP_BY_NAME = {'all_reduce': 'allreduce', 'all_gather': 'allgather',
              'all_gather_into_tensor': 'allgather', 'all_gather_tensor': 'allgather',
              'all_gather_object': 'allgather', 'barrier': 'barrier',
              'reduce_scatter_tensor': 'reducescatter', 'broadcast': 'broadcast',
              'reduce': 'reduce', 'scatter': 'scatter', 'gather': 'gather',
              'gather_object': 'gather', 'scatter_object_list': 'scatter',
              'all_to_all': 'alltoall', 'all_to_all_single': 'alltoall',
              'send': 'send', 'isend': 'send', 'recv': 'recv', 'irecv': 'recv',
              'batch_isend_irecv': 'batch_p2p'}
# Payload and output parameter names per entry. These names drift across torch
# releases (torch>=2.10 renamed reduce_scatter_tensor's tensor/input_tensor
# pair), so the guard test asserts them against the live inspect signatures
# rather than trusting this table blindly.
SPEC = {
    'all_reduce': ('tensor', 'tensor'), 'all_gather': ('tensor', 'tensor_list'),
    'all_gather_into_tensor': ('input_tensor', 'output_tensor'),
    'all_gather_object': ('obj', 'object_list'), 'barrier': (None, None),
    'reduce_scatter_tensor': ('input', 'output'), 'broadcast': ('tensor', 'tensor'),
    'reduce': ('tensor', 'tensor'), 'scatter': ('scatter_list', 'tensor'),
    'gather': ('tensor', 'gather_list'), 'gather_object': ('obj', 'object_gather_list'),
    'scatter_object_list': ('scatter_object_input_list', 'scatter_object_output_list'),
    'all_to_all': ('input_tensor_list', 'output_tensor_list'),
    'all_to_all_single': ('input', 'output'), 'send': ('tensor', None),
    'recv': ('tensor', None), 'isend': ('tensor', None), 'irecv': ('tensor', None),
    'batch_isend_irecv': ('p2p_op_list', None),
}
FUNCTIONAL_SPEC = {'all_reduce': ('self', 'self'), 'all_gather_tensor': ('self', None),
                   'reduce_scatter_tensor': ('self', None)}
# Root rank or remote peer rank field per entry, when one exists.
PEER_FIELDS = {'broadcast': ('src',), 'reduce': ('dst',), 'scatter': ('src',),
               'gather': ('dst',), 'send': ('dst',), 'isend': ('dst',),
               'recv': ('src',), 'irecv': ('src',)}


def payload_items(value):
    """Flatten a payload argument into its tensor items.

    Accepts a single tensor, a list of tensors, or a list of P2POp objects
    (batch_isend_irecv); anything else is object-or-control traffic whose byte
    size stays unmeasured rather than guessed.
    """
    if value is None:
        return []
    import torch
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        if any(hasattr(item, 'tensor') for item in value):
            return [item.tensor for item in value if hasattr(item, 'tensor')]
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


def tensor_bytes(tensors):
    import torch
    return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))


class CommunicationProbe(Probe):
    id = 'communication'
    produces = ('comm',)
    requires = ('torch.distributed',)

    def __init__(self, manager, control):
        super().__init__(manager)
        self.control = control
        self.model_groups = []
        self.native = {}
        self.functional_native = {}
        self.sequences = defaultdict(int)
        self.busy = False

    def install(self):
        import torch.distributed as dist
        self.installed = True
        for name in DIST_NAMES:
            native = getattr(dist, name)
            self.native[name] = native
            setattr(dist, name, self._wrap(name, native))
        import torch.distributed._functional_collectives as functional
        for name in FUNCTIONAL_NAMES:
            native = getattr(functional, name)
            self.functional_native[name] = native
            setattr(functional, name, self._wrap(name, native, functional=True))
        self.manager.state['probes'][self.id] = dict(status='available', descriptor=self.descriptor())

    def _wrap(self, name, native, functional=False):
        @functools.wraps(native)
        def observed(*args, **kwargs):
            if self.busy or not self.active():
                return native(*args, **kwargs)
            import inspect
            import torch
            import torch.distributed as dist
            bound = inspect.signature(native).bind(*args, **kwargs).arguments
            payload_field, output_field = (FUNCTIONAL_SPEC if functional else SPEC)[name]
            group = bound.get('group')
            if functional:
                import torch.distributed._functional_collectives as collectives
                group_name = collectives._resolve_group_name(group, bound.get('tag', ''))
                # Resolve only exact registered process-group names. Do not
                # infer model ownership from ranks or neighboring kernels.
                candidates = [self.control, dist.group.WORLD] + self.model_groups
                group = next((g for g in candidates if g is not None and g.group_name == group_name), group)
            role = ('p2p' if name in P2P_NAMES else
                    'control' if group is self.control else
                    'model' if group is None or group is dist.group.WORLD or any(group is g for g in self.model_groups)
                    else 'unknown')
            group_id = os.environ['FLAGPROF_GROUP_ID'] + '/' + role
            seq = self.sequences[group_id]
            self.sequences[group_id] += 1
            tensors = payload_items(bound.get(payload_field))
            size = tensor_bytes(tensors)
            output_size = tensor_bytes(payload_items(bound.get(output_field))) if output_field else 0
            if functional and name == 'all_gather_tensor' and role != 'unknown':
                output_size = size * dist.get_world_size(group)
            if functional and name == 'reduce_scatter_tensor' and role != 'unknown':
                output_size = size // dist.get_world_size(group)
            peer = next((bound[field] for field in PEER_FIELDS.get(name, ()) if field in bound), None)
            marker = 'flagprof/comm/' + role + '/' + str(seq)
            diagnostic = self.context.diagnostic
            call = None
            if diagnostic:
                call = diagnostic.comm_enter(group_id, seq, name, role,
                    bool(bound.get('async_op', False)))
            start = clock()
            self.busy = True
            error = None
            try:
                with torch.profiler.record_function(marker):
                    return native(*args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                if diagnostic:
                    diagnostic.comm_leave(call, error)
                duration = clock() - start
                self.busy = False
                scope = self.context.current_scope()
                generation = self.context.probe_state().get('generation', {})
                op = OP_BY_NAME[name]
                batch = bound.get('p2p_op_list') if name == 'batch_isend_irecv' else None
                event = dict(t='comm', op=op, group=group_id, group_seq=seq,
                             bytes=size, dur=duration, ts=start,
                             lib=dist.get_backend(group) if role != 'unknown' else 'unknown', phase='enqueue',
                             ext=dict(communication=dict(role=role, marker=marker, host_duration_ns=duration,
                                      api=('torch.distributed._functional_collectives.' if functional else 'torch.distributed.')+name,
                                      logical_input_bytes=size, logical_output_bytes=output_size,
                                      payload_status='tensor' if tensors else 'object_or_control',
                                      peer=peer if isinstance(peer, int) else None,
                                      batch_peers=([p.peer for p in batch] if batch else None),
                                      batch_ops=([str(p.op).rsplit('.', 1)[-1] for p in batch] if batch else None),
                                      runtime_padding_bytes=None, device_wire_bytes=None,
                                      device_link_status='explicit_link_unavailable', error=error,
                                      generation=dict(generation))))
                if tensors:
                    event.update(dtype=str(tensors[0].dtype), count=sum(t.numel() for t in tensors),
                                 dev=str(tensors[0].device))
                if scope:
                    event['corr'] = scope.corr
                self.context.writer.emit(event)
        return observed

    def uninstall(self):
        if not self.installed:
            return
        import torch.distributed as dist
        for name, native in self.native.items():
            setattr(dist, name, native)
        import torch.distributed._functional_collectives as functional
        for name, native in self.functional_native.items():
            setattr(functional, name, native)
        self.installed = False
