# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Explicit collective call identity; host duration never claims device duration."""
from collections import defaultdict
import functools
import os

from .base import Probe, clock


class CommunicationProbe(Probe):
    id = 'communication'
    produces = ('comm',)
    requires = ('torch.distributed',)

    def __init__(self, manager, control):
        super().__init__(manager)
        self.control = control
        self.model_groups = []
        self.native = {}
        self.sequences = defaultdict(int)
        self.busy = False

    def install(self):
        import torch.distributed as dist
        self.installed = True
        for name in ('all_reduce', 'all_gather', 'all_gather_into_tensor', 'all_gather_object', 'barrier'):
            native = getattr(dist, name)
            self.native[name] = native
            setattr(dist, name, self._wrap(name, native))
        self.manager.state['probes'][self.id] = dict(status='available', descriptor=self.descriptor())

    def _wrap(self, name, native):
        @functools.wraps(native)
        def observed(*args, **kwargs):
            if self.busy or not self.active():
                return native(*args, **kwargs)
            import inspect
            import torch
            import torch.distributed as dist
            bound = inspect.signature(native).bind(*args, **kwargs).arguments
            group = bound.get('group')
            role = ('control' if group is self.control else 'model'
                    if group is None or group is dist.group.WORLD or any(group is g for g in self.model_groups)
                    else 'unknown')
            group_id = os.environ['FLAGPROF_GROUP_ID']+'/'+role
            seq = self.sequences[group_id]
            self.sequences[group_id] += 1
            tensor = bound.get('tensor', bound.get('input_tensor'))
            tensors = [tensor] if isinstance(tensor, torch.Tensor) else []
            size = sum(t.numel()*t.element_size() for t in tensors)
            output = bound.get('tensor_list', bound.get('output_tensor', tensor))
            outputs = output if isinstance(output, list) else [output]
            output_size = sum(t.numel()*t.element_size() for t in outputs if isinstance(t, torch.Tensor))
            marker = 'flagprof/comm/'+role+'/'+str(seq)
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
                duration = clock()-start
                self.busy = False
                scope = self.context.current_scope()
                generation = self.context.probe_state().get('generation', {})
                op = {'all_reduce':'allreduce','all_gather':'allgather',
                      'all_gather_into_tensor':'allgather','all_gather_object':'allgather','barrier':'barrier'}[name]
                event = dict(t='comm', op=op, group=group_id, group_seq=seq,
                             bytes=size, dur=duration, ts=start, lib=dist.get_backend(group), phase='enqueue',
                             ext=dict(communication=dict(role=role, marker=marker, host_duration_ns=duration,
                                      api=name, logical_input_bytes=size, logical_output_bytes=output_size,
                                      payload_status='tensor' if tensors else 'object_or_control',
                                      runtime_padding_bytes=None, device_wire_bytes=None,
                                      device_link_status='explicit_link_unavailable', error=error,
                                      generation=dict(generation))))
                if tensors:
                    event.update(dtype=str(tensor.dtype), count=tensor.numel(), dev=str(tensor.device))
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
        self.installed = False
