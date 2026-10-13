"""Model-neutral, request-owned execution sessions; no device state survives close."""
from dataclasses import dataclass, field
import time


@dataclass
class ExecutionStep:
    outputs: dict
    context: dict
    token_ids: object = None
    regions: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)


class ExecutionSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        pass

    def cache_snapshot(self):
        return {'status': 'not_applicable', 'storage_bytes': None}


class SingleStepExecution(ExecutionSession):
    def __init__(self, adapter, model, inputs, mode):
        self.adapter, self.model, self.inputs = adapter, model, inputs
        self.done = False
        self.context = {'phase': 'forward', 'generation_step': 0, 'mode': mode}

    def advance(self):
        if self.done:
            raise StopIteration
        self.done = True
        return ExecutionStep(self.adapter.forward(self.model, self.inputs), dict(self.context))


def run_execution(adapter, model, inputs, mode='performance', backend=None, observer=None, inspect_cache=True):
    """Synchronization belongs to the device session. Capture/checks are optional observers."""
    steps, last, tokens = [], None, []
    with adapter.open_execution(model, inputs, mode) as session:
        while not session.done:
            context = dict(session.context)
            if observer:
                observer('before', context, None, session)
            if backend:
                backend.synchronize()
            start = time.perf_counter_ns()
            step = session.advance()
            if backend:
                backend.synchronize()
            end = time.perf_counter_ns()
            steps.append({**step.context, 'start_ns': start, 'end_ns': end, 'latency_ns': end-start})
            if step.token_ids is not None:
                tokens.append(step.token_ids)
            last = step.outputs
            if observer:
                observer('after', step.context, step, session)
        cache = session.cache_snapshot() if inspect_cache else {'status':'not_collected_in_timed_loop'}
        stops = getattr(session, 'stop_reasons', None)
    if last is None:
        raise ValueError('execution produced no steps')
    if tokens:
        import torch
        last = {**last, 'generated_ids': torch.stack(tokens, dim=1)}
    return last, {'steps': steps, 'cache': cache, 'stop_reasons': stops,
                  'generated_tokens': sum(sum(s.get('active_samples', [])) for s in steps)}


def performance_forward(adapter, model, backend):
    def forward(inputs):
        import torch
        with torch.inference_mode():
            output, evidence = run_execution(adapter,model,inputs,'performance',backend,forward.observer,
                                             inspect_cache=False)
        forward.last_execution = evidence
        return adapter.performance_output(output)
    forward.observer = None
    forward.last_execution = None
    return forward


def generation_row(forward, input_tokens):
    if not getattr(forward,'last_execution',None):
        return {}
    evidence=forward.last_execution
    return {'input_tokens':input_tokens,'tokens':evidence['generated_tokens'],
            'generated_tokens':evidence['generated_tokens'], 'generation_steps':evidence['steps']}


def inspect_execution(adapter, model, inputs, backend, rank=0, parallel=False):
    """Untimed cache and per-phase allocator observations; returns no live device state."""
    rows=[]
    before=None
    def observer(when,context,step,session):
        nonlocal before
        from runtime.device_session import increment
        if when=='before':
            backend.synchronize()
            before={'allocated_bytes':backend.memory_allocated(rank),'reserved_bytes':backend.memory_reserved(rank)}
            backend.reset_peak_memory_stats(rank)
            return
        peak={'allocated_bytes':backend.max_memory_allocated(rank),'reserved_bytes':backend.max_memory_reserved(rank)}
        if parallel:
            from runtime.tp import check_execution_step
            check_execution_step(context,step.token_ids.cpu().tolist())
        rows.append({**context,'cache':session.cache_snapshot(), 'allocator_before':before,
                     'allocator_after':{'allocated_bytes':backend.memory_allocated(rank),'reserved_bytes':backend.memory_reserved(rank)},
                     'allocator_peak':peak, 'allocator_peak_increment':increment(peak,before)})
    import torch
    with torch.inference_mode():
        output,_=run_execution(adapter,model,inputs,'performance',backend,observer)
    del output
    return {'status':'measured','scope':'independent untimed generation; allocator occupancy is not added to KV storage',
            'prefill':rows[0], 'final_decode':rows[-1], 'steps':rows}
