"""Single-device resident-input timing; diagnostics run in separate workers."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from runtime.common import write_json
from runtime.device_session import increment


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    return percentile_sorted(ordered, percent)


def percentile_sorted(ordered, percent):
    if not ordered: return None
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def summarize(rows):
    if not rows:
        return None
    seconds = [row['latency_ns'] / 1e9 for row in rows]
    duration = sum(seconds)
    tokens = sum(row['tokens'] for row in rows) if all(row.get('tokens') is not None for row in rows) else None
    result = {'batches':len(rows), 'samples':sum(row['samples'] for row in rows),
            'tokens':tokens, 'measured_seconds':duration,
            'latency_mean_ms':1000 * duration / len(rows),
            'latency_p50_ms':1000 * percentile(seconds,50),
            'latency_p90_ms':1000 * percentile(seconds,90),
            'latency_p99_ms':1000 * percentile(seconds,99),
            'samples_per_second':sum(row['samples'] for row in rows) / duration,
            'tokens_per_second':tokens / duration if tokens is not None else None}
    if all('generation_steps' in r for r in rows):
        steps=[s for r in rows for s in r['generation_steps']]
        prefill=[s['latency_ns'] for s in steps if s['phase']=='prefill']
        decode=[s['latency_ns'] for s in steps if s['phase']=='decode']
        input_tokens=sum(r['input_tokens'] for r in rows)
        generated=sum(r['generated_tokens'] for r in rows)
        decode_tokens=sum(sum(s['active_samples']) for s in steps if s['phase']=='decode')
        result['generation']={'input_tokens':input_tokens,'generated_tokens':generated,
            'generated_tokens_per_second':generated/duration,
            'prefill_mean_ms':sum(prefill)/len(prefill)/1e6,
            'first_token_mean_ms':sum(prefill)/len(prefill)/1e6,
            'input_tokens_per_prefill_second':input_tokens*1e9/sum(prefill),
            'decode_steps':len(decode),'decode_tokens':decode_tokens,
            'decode_mean_ms':sum(decode)/len(decode)/1e6 if decode else None,
            **{f'decode_p{p}_ms':percentile(decode,p)/1e6 if decode else None for p in (50,90,99)},
            'decode_tokens_per_second':decode_tokens*1e9/sum(decode) if decode else None,
            'first_token_scope':'resident-input model prefill through first greedy token and synchronization; not service TTFT'}
    return result


def tensor_storage_bytes(model):
    seen = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def memory_state(backend):
    return {'allocated_bytes':backend.memory_allocated(0),
            'reserved_bytes':backend.memory_reserved(0)}


def run(cfg, root, prepared, include, repeat_index, phase=None):
    import torch
    from runtime.context import initialize
    from runtime.adapters import model_adapter, create
    from runtime.workload import tensors, tensor_tree
    adapter = model_adapter(cfg)
    from runtime.worker import identity
    from runtime.common import read_json, digest, file_hash

    root = Path(root)
    device, backend = initialize(cfg)
    if phase == 'layer_timing': backend.require('events','stream')
    if phase == 'layer_memory': backend.require('memory','peak_memory')
    expected = read_json(prepared/'identity.json')
    current = identity(cfg,backend)
    current['tokenized_sha256'] = file_hash(prepared/'inputs.pt')
    if current != expected:
        write_json(root/'observed-identity.json',current)
        from runtime.common import identity_differences
        raise ValueError('performance worker identity differs: '+', '.join(identity_differences(expected,current)))
    backend.synchronize()
    before_load = memory_state(backend)
    write_json(root/'stage.json', {'stage':'model_load','repeat':repeat_index})
    started = time.perf_counter_ns()
    loaded = adapter.load_model(cfg, device, backend)
    model = loaded.model
    cpu_weight_bytes = loaded.transfer_bytes
    model_cpu_load_seconds = loaded.cpu_load_seconds
    model_transfer_seconds = loaded.transfer_seconds
    model_load_seconds = loaded.total_seconds
    after_load = memory_state(backend)
    weight_bytes = tensor_storage_bytes(model)
    batches = torch.load(prepared/'inputs.pt',map_location='cpu',weights_only=True)
    if not batches:
        raise ValueError('prepared input archive is empty')
    batch_metadata = [adapter.batch_metadata(b) for b in batches]
    data = []
    transfer_ns = 0
    transfer_bytes = 0
    write_json(root/'stage.json', {'stage':'input_transfer','repeat':repeat_index})
    for batch in batches:
        backend.synchronize()
        start = time.perf_counter_ns()
        values = {k:v.to(device) for k,v in batch['inputs'].items()}
        backend.synchronize()
        transfer_ns += time.perf_counter_ns()-start
        transfer_bytes += sum(v.numel()*v.element_size() for v in batch['inputs'].values())
        data.append((values, *adapter.sample_counts(batch)))

    lib, registered = None, []
    if include is not None:
        lib, registered = create('components', 'flaggems').enable_timing(include)

    def forward(values):
        with torch.inference_mode():
            outputs = adapter.forward(model, values)
            return adapter.performance_output(outputs)

    cache_evidence = None
    if adapter.execution_kind != 'single_step':
        from runtime.execution import performance_forward, inspect_execution
        forward = performance_forward(adapter,model,backend)
        cache_evidence = inspect_execution(adapter,model,data[0][0],backend)

    try:
        write_json(root/'stage.json',{'stage':'sanity','repeat':repeat_index})
        for values,_,_ in data:
            with torch.inference_mode():
                adapter.validate_outputs(tensor_tree(adapter.forward(model,values),lambda t:t.detach().cpu()))
            output = forward(values)
            backend.synchronize()
            if not all(bool(torch.isfinite(t).all().item()) for t in tensors(output)):
                raise RuntimeError('nonfinite output in untimed sanity pass')
            del output
        # Output transfer is a separate measurement, after inference has completed.
        output = forward(data[0][0])
        backend.synchronize()
        start = time.perf_counter_ns()
        copied = tensor_tree(adapter.output_transfer(output), lambda t: t.cpu())
        backend.synchronize()
        output_transfer_ns = time.perf_counter_ns()-start
        output_transfer_bytes = sum(t.numel()*t.element_size() for t in tensors(copied))
        del copied,output

        started = time.perf_counter_ns()
        write_json(root/'stage.json',{'stage':'warmup','repeat':repeat_index})
        for _ in range(cfg['performance']['warmup_rounds']):
            for values,_,_ in data:
                output = forward(values)
                backend.synchronize()
                del output
        warmup_seconds = (time.perf_counter_ns()-started)/1e9
        if phase:
            from runtime.layer_capture import run_pass
            return run_pass(cfg,root,model,forward,data,batch_metadata,backend,repeat_index,phase,
                            digest(expected),registered)
        backend.synchronize()
        baseline = memory_state(backend)
        backend.reset_peak_memory_stats(0)
        rows = []
        write_json(root/'stage.json',{'stage':'measurement','repeat':repeat_index,'completed_batches':0})
        with (root/'batches.jsonl').open('w') as stream:
            for cycle in range(cfg['performance']['measure_rounds']):
                for index,(values,samples,tokens) in enumerate(data):
                    backend.synchronize()
                    start = time.perf_counter_ns()
                    output = forward(values)
                    backend.synchronize()
                    end_ns = time.perf_counter_ns()
                    elapsed = end_ns-start
                    del output
                    row = {'cycle':cycle,'batch_index':index,'latency_ns':elapsed,
                           'start_ns':start,'end_ns':end_ns,
                           'samples':samples,'tokens':tokens, **batch_metadata[index]}
                    from runtime.execution import generation_row
                    row.update(generation_row(forward,tokens))
                    rows.append(row)
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
        peak = {'allocated_bytes':backend.max_memory_allocated(0),
                'reserved_bytes':backend.max_memory_reserved(0)}
        end = memory_state(backend)
        write_json(root/'stage.json',{'stage':'completed','repeat':repeat_index,
                                      'completed_batches':len(rows)})
        return {'status':'completed','identity_key':digest(expected),'repeat':repeat_index,
                'path':'on' if include is not None else 'off',
                'route':{'requested':include is not None,'allowed':include,
                         'registered':registered,'counting_in_timed_loop':False},
                'summary':summarize(rows),'batches':rows,
                'failure_count':0,'completed_batches':len(rows),
                'expected_batches':len(data)*cfg['performance']['measure_rounds'],
                'model_load_seconds':model_load_seconds,
                'model_cpu_load_seconds':model_cpu_load_seconds,
                'warmup_seconds':warmup_seconds,
                'transfer':{'model_weights':{'seconds':model_transfer_seconds,
                                             'logical_bytes':cpu_weight_bytes,
                                             'objects':['model parameters','model buffers'],
                                             'logical_link':'host_cpu_to_selected_device' if model_transfer_seconds is not None else None,
                                             'physical_link':None,
                                             'scope':loaded.transfer_scope,
                                             'unmeasured_reason':loaded.unmeasured_reason},
                            'input':{'seconds':transfer_ns/1e9,'logical_bytes':transfer_bytes,
                                     'objects':sorted(batches[0]['inputs']),
                                     'logical_link':'host_cpu_to_selected_device',
                                     'physical_link':None,
                                     'scope':'each prepared batch once before warmup'},
                            adapter.transfer_key:{'seconds':output_transfer_ns/1e9,
                                                'logical_bytes':output_transfer_bytes,
                                                'objects':adapter.transfer_objects,
                                                'logical_link':'selected_device_to_host_cpu',
                                                'physical_link':None,
                                                'scope':adapter.transfer_scope}},
                'memory':{'before_model_load':before_load,'after_model_load':after_load,
                          'weight_storage_bytes':weight_bytes,'measurement_baseline':baseline,
                          'measurement_peak':peak,'measurement_end':end,
                          'peak_increment_bytes':increment(peak,baseline),
                          'measurement':backend.metadata()},
                **({'generation_cache':cache_evidence} if cache_evidence else {})}
    finally:
        if lib is not None:
            lib._destroy()
