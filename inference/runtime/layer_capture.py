# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Execution-side module scopes. Timing, memory and trace use separate workers."""
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

from runtime.common import write_json
from runtime.performance import tensor_storage_bytes


def memory_groups(names):
    """No selected ancestor and descendant may reset the same peak counter."""
    groups = defaultdict(list)
    for name in names:
        depth = sum(name.startswith(parent + '.') for parent in names if parent != name)
        groups[depth].append(name)
    return [groups[k] for k in sorted(groups)]


def stream_id(backend):
    stream = backend.current_stream()
    return int(getattr(stream, 'npu_stream', getattr(stream, 'cuda_stream', 0)))


def memory(backend, rank):
    return {'allocated_bytes': int(backend.memory_allocated(rank)),
            'reserved_bytes': int(backend.memory_reserved(rank))}


class Scopes:
    """Inclusive scopes, paired by invocation rather than just module name."""
    def __init__(self, modules, mode, backend=None, rank=0):
        self.modules, self.mode, self.backend, self.rank = modules, mode, backend, rank
        self.handles, self.stack, self.rows = [], [], []
        self.active = False
        self.context = {}
        self.counts = Counter()
        self.events = {}
        self.sequence = 0

    def begin(self, context):
        if self.stack:
            raise RuntimeError('module scope leaked across batches')
        self.context = context
        self.counts.clear()
        self.rows = []
        self.active = True

    def enter(self, module, args, name):
        if not self.active:
            return
        index = self.counts[name]
        self.counts[name] += 1
        row = dict(self.context, module=name, call_index=index, rank=self.rank,
                   sequence=self.sequence, status='entered',
                   parent_sequence=self.stack[-1]['sequence'] if self.stack else None)
        self.sequence += 1
        if self.mode == 'layer_memory':
            if self.stack:
                raise RuntimeError('overlapping memory scopes; peak counters cannot be nested')
            self.backend.synchronize()
            row['before'] = memory(self.backend, self.rank)
            self.backend.reset_peak_memory_stats(self.rank)
        elif self.mode == 'layer_timing':
            key = (name, index)
            if key not in self.events:
                self.events[key] = (self.backend.Event(enable_timing=True),
                                    self.backend.Event(enable_timing=True))
            row['_events'] = self.events[key]
            row['start_stream'] = stream_id(self.backend)
            row['_events'][0].record()
        elif self.mode == 'layer_profile':
            import torch
            row['marker'] = 'flagperf/layer/' + str(row['sequence'])
            row['_marker'] = torch.profiler.record_function(row['marker'])
            row['_marker'].__enter__()
        row['host_start_ns'] = time.perf_counter_ns()
        self.stack.append(row)

    def leave(self, module, args, output, name):
        if not self.active:
            return
        if not self.stack or self.stack[-1]['module'] != name:
            if sys.exc_info()[0] is not None:
                return
            raise RuntimeError('unbalanced module scope: ' + name)
        row = self.stack.pop()
        row['host_end_ns'] = time.perf_counter_ns()
        row['host_ns'] = row['host_end_ns'] - row['host_start_ns']
        failed = sys.exc_info()[0] is not None
        row['status'] = 'failed' if failed else 'completed'
        if self.mode == 'layer_timing':
            row['end_stream'] = stream_id(self.backend)
            row['_events'][1].record()
        elif self.mode == 'layer_memory' and not failed:
            self.backend.synchronize()
            row['after'] = memory(self.backend, self.rank)
            row['peak'] = {'allocated_bytes': int(self.backend.max_memory_allocated(self.rank)),
                           'reserved_bytes': int(self.backend.max_memory_reserved(self.rank))}
            row['peak_increment'] = {k: row['peak'][k] - row['before'][k] for k in row['peak']}
            row['net_change'] = {k: row['after'][k] - row['before'][k] for k in row['peak']}
        elif self.mode == 'layer_profile':
            row.pop('_marker').__exit__(*sys.exc_info())
        self.rows.append(row)

    def finish(self, model_events=None, model_stream=None):
        self.active = False
        if self.stack:
            raise RuntimeError('incomplete module invocation')
        for row in self.rows:
            events = row.pop('_events', None)
            if events:
                if row['start_stream'] != row['end_stream'] or row['start_stream'] != model_stream:
                    row.update(device_ns=None, device_status='stream_mismatch')
                else:
                    row.update(device_ns=round(events[0].elapsed_time(events[1]) * 1e6),
                               device_start_ns=round(model_events[0].elapsed_time(events[0]) * 1e6),
                               device_status='observed')
        return sorted(self.rows, key=lambda row: row['sequence'])

    def __enter__(self):
        try:
            for name, module in self.modules.items():
                self.handles.append(module.register_forward_pre_hook(
                    lambda m, a, name=name: self.enter(m, a, name), prepend=True))
                # Append after Transformers' output hook, including its collective.
                self.handles.append(module.register_forward_hook(
                    lambda m, a, o, name=name: self.leave(m, a, o, name), always_call=True))
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *exc):
        self.active = False
        for handle in self.handles:
            handle.remove()
        for row in reversed(self.stack):
            if '_marker' in row:
                row.pop('_marker').__exit__(*exc)
        self.stack.clear()


class RouteScopes(Scopes):
    """Count real top-level device ATen / registered function calls per scope."""
    def __init__(self, modules):
        super().__init__(modules, 'route')
        self.inventory = defaultdict(Counter)
        self.functions = defaultdict(Counter)
        self.invocations = Counter()

    def enter(self, module, args, name):
        super().enter(module, args, name)
        if self.active:
            self.invocations[name] += 1

    def observe(self, name, kind):
        if not self.active:
            return
        owners = [row['module'] for row in self.stack] or ['__outside_selected_layers__']
        for owner in set(owners):
            (self.functions if kind == 'function' else self.inventory)[owner][name] += 1

    def evidence(self):
        return {name: {'invocations': self.invocations[name],
                       'inventory': {k: {'calls': v} for k, v in self.inventory[name].items()},
                       'actual_function_calls': dict(self.functions[name])}
                for name in dict.fromkeys([*self.modules, *self.inventory, *self.functions])}


def cpu_outputs(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [value.detach().cpu()]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in cpu_outputs(v)]
    raise TypeError('unsupported model output in instrumentation check')


def run_pass(cfg, root, model, forward, data, metadata, backend, repeat, phase,
             identity_key, registered, rank=0, barrier=lambda: None):
    import torch
    from models.qwen3_embedding.model import selected_layers
    modules = selected_layers(model, cfg['performance']['layers'])
    root = Path(root)
    inventory = [{'module': name, 'type': type(module).__name__,
                  'selected_parent': max((n for n in modules if name.startswith(n + '.')),
                                         key=len, default=None),
                  'weight_storage_bytes': tensor_storage_bytes(module)} for name, module in modules.items()]
    write_json(root/'modules.json', inventory)
    references = []
    for inputs, _, _ in data:
        references.append(cpu_outputs(forward(inputs)))
    backend.synchronize()
    result = {'status': 'completed', 'phase': phase, 'rank': rank, 'repeat': repeat,
              'identity_key': identity_key, 'modules': inventory, 'rows': [], 'batches': [],
              'route': {'registered': registered}, 'failure_count': 0,
              'instrumentation_output_equal': True, 'instrumentation_checks': 0}
    rounds = cfg['performance']['measure_rounds'] if phase == 'layer_timing' else cfg['performance']['layer_profile_rounds']
    groups = memory_groups(list(modules)) if phase == 'layer_memory' else [list(modules)]
    result['memory_groups'] = groups if phase == 'layer_memory' else None
    result['expected_batches'] = len(data) * rounds * len(groups)
    result['expected_calls'] = len(modules) * len(data) * rounds
    def checked(output, index):
        actual = cpu_outputs(output)
        expected = references[index]
        if len(actual) != len(expected) or any(not torch.equal(a, b) for a, b in zip(actual, expected)):
            result['instrumentation_output_equal'] = False
            raise RuntimeError('instrumented forward changed the model output')
        result['instrumentation_checks'] += 1
    try:
        with (root/'layers.jsonl').open('w') as stream, (root/'batches.jsonl').open('w') as batch_stream:
            for group_index, names in enumerate(groups):
                with Scopes({n: modules[n] for n in names}, phase, backend, rank) as capture:
                    profiler, comm = nullcontext(), nullcontext()
                    warm = 0
                    if phase == 'layer_profile':
                        if cfg['runtime']['vendor'] == 'ascend':
                            import torch_npu
                            api = torch_npu.profiler
                            activities = [api.ProfilerActivity.CPU, api.ProfilerActivity.NPU]
                            extra = {'experimental_config': api._ExperimentalConfig(profiler_level=api.ProfilerLevel.Level1)}
                        else:
                            api = torch.profiler
                            activities = [api.ProfilerActivity.CPU, api.ProfilerActivity.CUDA]
                            extra = {}
                        warm = len(data)
                        profiler = api.profile(activities=activities,
                            schedule=api.schedule(wait=0, warmup=warm, active=rounds*len(data), repeat=1),
                            on_trace_ready=api.tensorboard_trace_handler(str(root/'profiler')),
                            record_shapes=True, **extra)
                        if cfg['runtime'].get('parallelism') == 'tp':
                            from runtime.communication import Capture
                            comm = Capture(model, root, rank)
                    with comm as communication, profiler as prof:
                        model_events = (backend.Event(enable_timing=True), backend.Event(enable_timing=True)) if phase == 'layer_timing' else None
                        # Prime lazy event allocations outside retained measurements.
                        if model_events:
                            capture.begin({'warmup': True})
                            model_events[0].record()
                            output = forward(data[0][0])
                            model_events[1].record(); backend.synchronize()
                            capture.finish(model_events, stream_id(backend)); del output
                        for step in range(warm + rounds*len(data)):
                            index = step % len(data)
                            cycle = (step-warm)//len(data)
                            active = step >= warm
                            inputs, samples, tokens = data[index]
                            context = dict(repeat=repeat, cycle=cycle, batch_index=index,
                                           samples=samples, tokens=tokens, **metadata[index],
                                           measurement_kind=phase, group_index=group_index)
                            backend.synchronize(); barrier()
                            if active: capture.begin(context)
                            if communication is not None:
                                communication.active = active
                                communication.batch_index, communication.cycle = index, cycle
                            marker = torch.profiler.record_function('flagperf/model_batch') if prof else nullcontext()
                            model_stream = stream_id(backend)
                            if model_events: model_events[0].record()
                            started = time.perf_counter_ns()
                            with marker:
                                output = forward(inputs)
                                if model_events: model_events[1].record()
                                backend.synchronize()
                            ended = time.perf_counter_ns()
                            if active:
                                rows = capture.finish(model_events, model_stream)
                                if any(r['status'] != 'completed' for r in rows):
                                    raise RuntimeError('incomplete layer invocation')
                                batch = dict(context, rank=rank, start_ns=started, end_ns=ended, latency_ns=ended-started)
                                if model_events: batch['device_ns'] = round(model_events[0].elapsed_time(model_events[1])*1e6)
                                result['rows'].extend(rows); result['batches'].append(batch)
                                for row in rows: stream.write(json.dumps(row)+'\n')
                                stream.flush(); batch_stream.write(json.dumps(batch)+'\n'); batch_stream.flush()
                                if cycle == 0: checked(output, index)
                            del output
                            barrier()
                            if prof: prof.step()
                            write_json(root/'stage.json', {'stage': phase, 'group': group_index,
                                       'completed_batches': len(result['batches']), 'expected_batches': result['expected_batches']})
        result['completed_batches'] = len(result['batches'])
        observed = {row['module'] for row in result['rows']}
        if set(modules) != observed:
            raise RuntimeError('selected modules not observed: ' + ', '.join(set(modules)-observed))
        if phase == 'layer_profile' and cfg['runtime'].get('parallelism') == 'tp':
            result['expected_model_collectives'] = len(data)*rounds*2*len(model.layers)
        return result
    except BaseException as error:
        result.update(status='failed', error=str(error), failure_count=1,
                      failure_count_scope='worker failure; layer root cause not inferred',
                      completed_batches=len(result['batches']))
        write_json(root/'layer-failure.json', result)
        raise
