# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Single-device resident-input timing; diagnostics run in separate workers."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from runtime.common import write_json


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def summarize(rows):
    if not rows:
        return None
    seconds = [row['latency_ns'] / 1e9 for row in rows]
    duration = sum(seconds)
    return {'batches':len(rows), 'samples':sum(row['samples'] for row in rows),
            'tokens':sum(row['tokens'] for row in rows), 'measured_seconds':duration,
            'latency_mean_ms':1000 * duration / len(rows),
            'latency_p50_ms':1000 * percentile(seconds,50),
            'latency_p90_ms':1000 * percentile(seconds,90),
            'latency_p99_ms':1000 * percentile(seconds,99),
            'samples_per_second':sum(row['samples'] for row in rows) / duration,
            'tokens_per_second':sum(row['tokens'] for row in rows) / duration}


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
    from vendors.device import initialize
    from models.qwen3_embedding.model import load_model_cpu, pool
    from runtime.worker import identity
    from runtime.common import read_json, digest, file_hash

    root = Path(root)
    device, backend = initialize(cfg)
    expected = read_json(prepared/'identity.json')
    current = identity(cfg,backend)
    current['tokenized_sha256'] = file_hash(prepared/'inputs.pt')
    if current != expected:
        write_json(root/'observed-identity.json',current)
        from runtime.common import identity_differences
        raise ValueError('performance worker identity differs: '+', '.join(identity_differences(expected,current)))
    backend.synchronize()
    before_load = memory_state(backend)
    started = time.perf_counter_ns()
    model = load_model_cpu(cfg)
    if any(tensor.device.type != 'cpu' for tensor in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError('model transfer measurement requires CPU-resident parameters and buffers')
    cpu_weight_bytes = tensor_storage_bytes(model)
    transfer_started = time.perf_counter_ns()
    model = model.to(device)
    backend.synchronize()
    load_finished = time.perf_counter_ns()
    model_cpu_load_seconds = (transfer_started-started)/1e9
    model_transfer_seconds = (load_finished-transfer_started)/1e9
    model_load_seconds = (load_finished-started)/1e9
    after_load = memory_state(backend)
    weight_bytes = tensor_storage_bytes(model)
    batches = torch.load(prepared/'inputs.pt',map_location='cpu',weights_only=True)
    if not batches:
        raise ValueError('prepared input archive is empty')
    batch_metadata = [{'input_shape':list(b['inputs']['input_ids'].shape), 'sample_ids':b['ids']} for b in batches]
    data = []
    transfer_ns = 0
    transfer_bytes = 0
    for batch in batches:
        backend.synchronize()
        start = time.perf_counter_ns()
        values = {k:v.to(device) for k,v in batch['inputs'].items()}
        backend.synchronize()
        transfer_ns += time.perf_counter_ns()-start
        transfer_bytes += sum(v.numel()*v.element_size() for v in batch['inputs'].values())
        data.append((values,len(batch['ids']),int(batch['inputs']['attention_mask'].sum().item())))

    lib = None
    registered = []
    if include is not None:
        import flag_gems
        lib = torch.library.Library('aten','IMPL')
        flag_gems.only_enable(include=include,lib=lib,record=False)
        registrar = flag_gems.current_work_registrar
        registered = list(zip(registrar.all_keys,registrar.all_ops))
        if not set(include) <= {name for _,name in registered}:
            raise RuntimeError('FlagGems registration is incomplete; see selected policy')

    def forward(values):
        with torch.inference_mode():
            hidden = model(**values,use_cache=False,return_dict=True).last_hidden_state
            _, embedding = pool(hidden,values['attention_mask'])
        return embedding

    try:
        write_json(root/'stage.json',{'stage':'sanity','repeat':repeat_index})
        for values,_,_ in data:
            embedding = forward(values)
            backend.synchronize()
            if not bool(torch.isfinite(embedding).all().item()):
                raise RuntimeError('nonfinite embedding in untimed sanity pass')
            del embedding
        # Output transfer is a separate measurement, after inference has completed.
        embedding = forward(data[0][0])
        backend.synchronize()
        start = time.perf_counter_ns()
        copied = embedding.cpu()
        backend.synchronize()
        output_transfer_ns = time.perf_counter_ns()-start
        output_transfer_bytes = copied.numel()*copied.element_size()
        del copied,embedding

        started = time.perf_counter_ns()
        write_json(root/'stage.json',{'stage':'warmup','repeat':repeat_index})
        for _ in range(cfg['performance']['warmup_rounds']):
            for values,_,_ in data:
                embedding = forward(values)
                backend.synchronize()
                del embedding
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
                    embedding = forward(values)
                    backend.synchronize()
                    elapsed = time.perf_counter_ns()-start
                    del embedding
                    row = {'cycle':cycle,'batch_index':index,'latency_ns':elapsed,
                           'samples':samples,'tokens':tokens, **batch_metadata[index]}
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
                                             'logical_link':'host_cpu_to_selected_device',
                                             'physical_link':None,
                                             'scope':'one model.to(device) after CPU load; includes allocation and synchronization'},
                            'input':{'seconds':transfer_ns/1e9,'logical_bytes':transfer_bytes,
                                     'objects':sorted(batches[0]['inputs']),
                                     'logical_link':'host_cpu_to_selected_device',
                                     'physical_link':None,
                                     'scope':'each prepared batch once before warmup'},
                            'output_embedding':{'seconds':output_transfer_ns/1e9,
                                                'logical_bytes':output_transfer_bytes,
                                                'objects':['embedding'],
                                                'logical_link':'selected_device_to_host_cpu',
                                                'physical_link':None,
                                                'scope':'first prepared batch, extra forward before warmup; only embedding copy timed'}},
                'memory':{'before_model_load':before_load,'after_model_load':after_load,
                          'weight_storage_bytes':weight_bytes,'measurement_baseline':baseline,
                          'measurement_peak':peak,'measurement_end':end,
                          'peak_increment_bytes':{k:peak[k]-baseline[k] for k in peak}}}
    finally:
        if lib is not None:
            lib._destroy()
