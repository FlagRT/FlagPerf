# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Single-host TP worker and sealed rank aggregation; imports are device-lazy."""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import json
import os
import time
from runtime.common import read_json, write_json, digest, file_hash
from runtime.performance import summarize, tensor_storage_bytes

CONTROL_GROUP = None
# Locked FlagCX Ascend build borrows an NPU stream: retain its owner until worker exit.
MODEL_GROUP = None


def initialize_group(cfg):
    import torch.distributed as dist
    global CONTROL_GROUP, MODEL_GROUP
    if not dist.is_initialized():
        timeout = timedelta(seconds=cfg['runtime']['timeout_seconds'])
        from vendors.stack import communication_backend
        backend = communication_backend(cfg)
        if backend == 'flagcx':
            previous = os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')
            import flagcx
            if previous is not None: os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = previous
        dist.init_process_group(backend,timeout=timeout)
        if dist.get_backend() != backend: raise RuntimeError('unexpected model process-group backend')
        if backend == 'flagcx':
            MODEL_GROUP = dist.distributed_c10d._get_default_group()
        CONTROL_GROUP = dist.new_group(backend='gloo',timeout=timeout)


def close_group():
    """Deregister groups, retain the FlagCX native owner for this bounded process lifetime.

    The pinned Ascend backend destructor calls streamDestroy on a borrowed stream
    member. Eager final-reference release aborts (invalid free). This is the same
    ownership lifetime as the previously validated standalone probe, not a library fix.
    Workers are one-shot; their normal process exit releases device resources.
    """
    import torch.distributed as dist
    global CONTROL_GROUP
    if not dist.is_initialized(): return
    if CONTROL_GROUP is not None:
        dist.destroy_process_group(CONTROL_GROUP)
        CONTROL_GROUP = None
    dist.destroy_process_group()


def global_batches(ranks):
    """One global sample count and host-clock envelope per matching batch."""
    if not ranks: return []
    lists = [r['batches'] for r in ranks]
    if len({len(rows) for rows in lists}) != 1:
        raise ValueError('rank batch counts differ')
    output = []
    for rows in zip(*lists):
        if any(any(r[k] != rows[0][k] for k in ['cycle','batch_index','samples','tokens']) for r in rows):
            raise ValueError('rank workload/order mismatch')
        if any(r['end_ns'] <= r['start_ns'] for r in rows):
            raise ValueError('invalid rank timing interval')
        metadata = {}
        for key in ('input_shape', 'sample_ids'):
            if any(key in r for r in rows):
                if any(r.get(key) != rows[0].get(key) for r in rows) or key not in rows[0]:
                    raise ValueError('rank batch metadata mismatch: '+key)
                metadata[key] = rows[0][key]
        begin, end = min(r['start_ns'] for r in rows), max(r['end_ns'] for r in rows)
        output.append({**{k:rows[0][k] for k in ['cycle','batch_index','samples','tokens']},
                       'start_ns':begin,'end_ns':end,'latency_ns':end-begin,
                       'rank_entry_skew_ns':max(r['start_ns'] for r in rows)-begin, **metadata})
    return output


def aggregate(cfg, destination, action, process, prepared=None):
    root = Path(destination)
    ranks = {}
    missing = []
    for rank in range(len(cfg['runtime']['devices'])):
        p = root/f'rank-{rank}'/'result.json'
        if not p.is_file(): missing.append(str(rank))
        ranks[str(rank)] = read_json(p) if p.is_file() else {'status':'failed','error':'rank exited without result'}
    result = {'status':'failed','ranks':ranks,'world_size':len(ranks),'parallelism':'tp'}
    failures = [(r,v) for r,v in ranks.items() if v['status'] != 'completed']
    if failures or process['exit_code']:
        # Prefer the actual exception over the peer that torchrun terminated.
        failures.sort(key=lambda item: 'traceback' not in item[1])
        rank, value = failures[0] if failures else ('launcher',{'error':'nonzero group exit'})
        result.update(error=f'rank {rank}: {value.get("error",value["status"] if "status" in value else "failed")}',
                      exception_type=value.get('exception_type'),failure_count=len(failures),
                      failure_count_scope='failed_or_missing_rank_results; not failed batches')
        if missing and cfg.get('_preview_trial'):
            result.update(preview_evidence_error=True,preview_evidence_validated=False,
                          missing_ranks=missing)
    else:
        try:
            values = list(ranks.values())
            if len({r['identity_key'] for r in values}) != 1:
                raise ValueError('rank identities differ')
            result.update(status='completed',identity_key=values[0]['identity_key'],failure_count=0)
            if action == 'forward':
                inventory, candidates, calls = {}, {}, {}
                for rank, value in ranks.items():
                    if cfg.get('_preview_trial'):
                        from runtime.preview_evidence import validate_checks
                        validate_checks(root/f'rank-{rank}',value,read_json(Path(prepared)/'forward-inputs.json'))
                        if value['forward_evidence']['mode'] != cfg['_preview_trial']['evidence_mode']:
                            raise ValueError('rank preview evidence mode differs from request')
                    elif not isinstance(value.get('artifacts'), list) or not value['artifacts']:
                        raise ValueError(f'rank {rank} has no sealed forward artifacts')
                    for key,row in value.get('inventory',{}).items():
                        merged = inventory.setdefault(key,{'calls':0,'signatures':[]})
                        merged['calls'] += row['calls']
                        for sig in row['signatures']:
                            if sig not in merged['signatures']: merged['signatures'].append(sig)
                    for row in value.get('candidates',[]):
                        dest = candidates.setdefault(row['function'],dict(row,signatures=[]))
                        for sig in row['signatures']:
                            if sig not in dest['signatures']: dest['signatures'].append(sig)
                    for name,count in value['route'].get('actual_function_calls',{}).items():
                        calls[name] = calls.get(name,0)+count
                route = dict(values[0]['route'],actual_function_calls=calls,
                    observed=all(v['route'].get('observed',False) for v in values),
                    rank_function_calls={r:v['route'].get('actual_function_calls',{}) for r,v in ranks.items()})
                result.update(inventory=inventory,candidates=list(candidates.values()),route=route,
                    numerical_anomalies=[dict(a,rank=r) for r,v in ranks.items() for a in v.get('numerical_anomalies',[])])
                if cfg.get('_preview_trial'):
                    result.update(preview_evidence_required=True,preview_evidence_validated=True,
                                  inventory_collected=all(v.get('inventory_collected',False) for v in values))
            elif action == 'performance':
                rows = global_batches(values)
                (root/'batches.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                result.update(batches=rows,summary=summarize(rows),route=values[0]['route'],
                              completed_batches=len(rows),expected_batches=values[0]['expected_batches'])
                if any(v['completed_batches'] != v['expected_batches'] for v in values):
                    raise ValueError('incomplete rank measurement')
            elif action == 'profile':
                from analysis.communication import analyze
                result['communication'] = analyze(root, len(ranks))
                if result['communication']['status'] != 'completed':
                    result.update(status='partial',error='communication evidence is incomplete; see coverage and issues')
            elif action.startswith('layer_'):
                if any(v.get('completed_batches') != v.get('expected_batches') or
                       not v.get('instrumentation_output_equal') for v in values):
                    raise ValueError('incomplete layer measurement or changed instrumented output')
                if any(v['modules'] != values[0]['modules'] for v in values):
                    # Weight storage may differ across ranks in future uneven sharding.
                    if any([m['module'] for m in v['modules']] != [m['module'] for m in values[0]['modules']] for v in values):
                        raise ValueError('rank module selections differ')
        except Exception as error:
            result.update(status='failed',error=str(error),exception_type=type(error).__name__)
            if cfg.get('_preview_trial'):
                result.update(preview_evidence_error=True,preview_evidence_validated=False)
    write_json(root/'result.json',result)
    return result


def run(cfg, root, prepared, include, repeat_index, profile=False, phase=None):
    import torch
    import torch.distributed as dist
    from vendors.device import initialize
    from models.qwen3_embedding.model import load_model, pool
    from runtime.worker import identity
    root = Path(root)
    device, backend = initialize(cfg)
    rank = dist.get_rank()
    expected = read_json(prepared/'identity.json')
    actual = identity(cfg,backend)
    actual['tokenized_sha256'] = file_hash(prepared/'inputs.pt')
    if actual != expected:
        write_json(root/'observed-identity.json',actual)
        from runtime.common import identity_differences
        raise ValueError('TP worker identity differs: '+', '.join(identity_differences(expected,actual)))
    def stage(name): write_json(root/'stage.json',{'stage':name,'rank':rank,'repeat':repeat_index})
    def memory():
        return {'allocated_bytes':backend.memory_allocated(rank),'reserved_bytes':backend.memory_reserved(rank)}
    def barrier(): dist.barrier(group=CONTROL_GROUP)
    before = memory()
    stage('model_load')
    begin = time.perf_counter_ns()
    model = load_model(cfg,device)
    backend.synchronize()
    load_seconds = (time.perf_counter_ns()-begin)/1e9
    after = memory()
    shards = {name:list(t.shape) for name,t in model.named_parameters() if name.startswith('layers.0.')}
    write_json(root/'tp-plan.json',{'plan':model._tp_plan,'layer0_parameter_shapes':shards,
                                  'rank':rank,'physical_device':cfg['runtime']['devices'][rank]})
    batches = torch.load(prepared/'inputs.pt',map_location='cpu',weights_only=True)
    batch_metadata = [{'input_shape':list(b['inputs']['input_ids'].shape), 'sample_ids':b['ids']} for b in batches]
    data, copy_ns, copy_bytes = [], 0, 0
    for batch in batches:
        backend.synchronize(); begin = time.perf_counter_ns()
        inputs = {k:v.to(device) for k,v in batch['inputs'].items()}
        backend.synchronize(); copy_ns += time.perf_counter_ns()-begin
        copy_bytes += sum(v.numel()*v.element_size() for v in batch['inputs'].values())
        data.append((inputs,len(batch['ids']),int(batch['inputs']['attention_mask'].sum())))
    lib, registered = None, []
    if include is not None:
        import flag_gems
        lib = torch.library.Library('aten','IMPL')
        flag_gems.only_enable(include=include,lib=lib,record=False)
        registrar = flag_gems.current_work_registrar
        registered = list(zip(registrar.all_keys,registrar.all_ops))
        if not set(include) <= {n for _,n in registered}: raise RuntimeError('incomplete FlagGems registration')
    def forward(inputs):
        with torch.inference_mode():
            hidden = model(**inputs,use_cache=False,return_dict=True).last_hidden_state
            return pool(hidden,inputs['attention_mask'])
    try:
        stage('sanity')
        for inputs,_,_ in data:
            pooled, embedding = forward(inputs)
            backend.synchronize()
            if not bool(torch.isfinite(embedding).all().item() and torch.isfinite(pooled).all().item()):
                raise RuntimeError('nonfinite TP output in untimed sanity pass')
            del pooled, embedding
        pooled, embedding = forward(data[0][0]); backend.synchronize()
        begin=time.perf_counter_ns(); copied=embedding.cpu(); backend.synchronize()
        output_ns=time.perf_counter_ns()-begin
        output_bytes=copied.numel()*copied.element_size()
        del pooled,embedding,copied
        stage('warmup'); begin=time.perf_counter_ns()
        for _ in range(cfg['performance']['warmup_rounds']):
            for inputs,_,_ in data:
                pooled,embedding=forward(inputs);backend.synchronize();del pooled,embedding
        warmup_seconds=(time.perf_counter_ns()-begin)/1e9
        if phase:
            from runtime.layer_capture import run_pass
            return run_pass(cfg,root,model,forward,data,batch_metadata,backend,repeat_index,phase,
                            digest(expected),registered,rank,barrier)
        barrier(); baseline=memory(); backend.reset_peak_memory_stats(rank)
        result={'status':'completed','identity_key':digest(expected),'rank':rank,'repeat':repeat_index,
                'path':'on' if include is not None else 'off','model_load_seconds':load_seconds,
                'warmup_seconds':warmup_seconds,'failure_count':0,
                'route':{'requested':include is not None,'allowed':include,'registered':registered,
                         'counting_in_timed_loop':False},
                'transfer':{'model_weights':{'status':'not_separately_measured',
                    'reason':'Transformers TP loader fuses loading, sharding and placement'},
                    'input':{'seconds':copy_ns/1e9,'logical_bytes':copy_bytes,'scope':'all prepared inputs once on this rank'},
                    'output_embedding':{'seconds':output_ns/1e9,'logical_bytes':output_bytes,'scope':'first batch embedding copy on this rank'}},
                'memory':{'before_model_load':before,'after_model_load':after,
                          'weight_storage_bytes':tensor_storage_bytes(model),'measurement_baseline':baseline}}
        if profile:
            import torch_npu
            from runtime.communication import Capture
            count=len(data)*cfg['performance']['communication_profile_rounds']
            # Profiler warmup and active phases each use whole input sweeps.
            stage('communication_profile')
            with Capture(model,root,rank) as capture:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0,warmup=len(data),active=count,repeat=1),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(root/'profiler')),
                    record_shapes=True,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1)) as prof:
                    for step in range(len(data)+count):
                        inputs,_,_=data[step%len(data)]
                        capture.active=step>=len(data)
                        capture.batch_index=step%len(data)
                        capture.cycle=max(0,step//len(data)-1)
                        barrier();backend.synchronize()
                        with torch.profiler.record_function('flagperf/model_batch'):
                            pooled,embedding=forward(inputs);backend.synchronize();del pooled,embedding
                        barrier();prof.step()
            result.update(profile_batches=count,expected_model_collectives=count*2*len(model.layers))
        else:
            stage('measurement');rows=[]
            with (root/'batches.jsonl').open('w') as stream:
                for cycle in range(cfg['performance']['measure_rounds']):
                    for index,(inputs,samples,tokens) in enumerate(data):
                        backend.synchronize();barrier()
                        begin=time.perf_counter_ns()
                        pooled,embedding=forward(inputs);backend.synchronize()
                        end=time.perf_counter_ns()
                        del pooled,embedding
                        barrier()
                        row={'cycle':cycle,'batch_index':index,'samples':samples,'tokens':tokens,
                             'start_ns':begin,'end_ns':end,'latency_ns':end-begin, **batch_metadata[index]}
                        rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            peak={'allocated_bytes':backend.max_memory_allocated(rank),'reserved_bytes':backend.max_memory_reserved(rank)}
            result['memory'].update(measurement_peak=peak,measurement_end=memory(),
                                    peak_increment_bytes={k:peak[k]-baseline[k] for k in peak})
            result.update(batches=rows,summary=summarize(rows),completed_batches=len(rows),
                          expected_batches=len(data)*cfg['performance']['measure_rounds'])
        stage('completed')
        return result
    finally:
        if lib is not None: lib._destroy()
