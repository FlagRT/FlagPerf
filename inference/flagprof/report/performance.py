# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Offline runner statistics. Process repeats are never inferred from filenames."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from flagprof.analyzers.gap import distribution

FORWARD_TIMING_NOTE = '各 rank 分别统计同步前向时延。插桩扰动需要有共同配对标识和测量协议的回执。'
GROUP_TIMING_NOTE = 'rank 分别计时；组时延仅使用同主机时钟和组执行回执，取最早开始至最晚结束，逻辑 token 只计一次。插桩扰动仅使用有 pair_id 与 measurement_protocol 的配对回执。'

def summarize(paths, identity=None, limits=None):
    from flagprof.analyzers.workspace import Limits
    from flagprof.store.analysis_read import ResourceLimit
    limits=limits if isinstance(limits,Limits) else Limits(**(limits or {}));observation_count=0
    groups={};sources=[];notes=[];spikes=[];seen=set();condition=None;group_requests={};paired={};group_windows=[]
    for filename in paths:
        path=Path(filename).resolve()
        if path.stat().st_size>limits.cache_bytes:raise ResourceLimit('runner input exceeds cache_bytes')
        raw=path.read_bytes();fingerprint=hashlib.sha256(raw).hexdigest()
        value=json.loads(raw);receipt={}
        if value.get('schema')=='runner-evidence/1':
            receipt=value.get('execution',{})
            target=(path.parent/value['runner']).resolve()
            if target.stat().st_size>limits.cache_bytes:raise ResourceLimit('runner input exceeds cache_bytes')
            raw=target.read_bytes()
            fingerprint=hashlib.sha256(raw).hexdigest();value=json.loads(raw)
        process_key=tuple(receipt.get(k) for k in ('boot_id','pid','start_ticks'))
        dedup_key=(fingerprint,process_key if all(v is not None for v in process_key) else None)
        if dedup_key in seen:
            notes.append('重复 runner 内容只计一次：'+str(path));continue
        seen.add(dedup_key)
        recorded=value.get('oracle_identity') or {}
        if not recorded:raise ValueError('runner workload identity missing: '+str(path))
        # rank/device-local ordinal may differ, all other recorded workload facts must agree.
        comparable={k:v for k,v in recorded.items() if k not in ('rank','device')}
        if condition is None:condition=comparable
        elif comparable!=condition:raise ValueError('runner inputs contain different workload conditions; report separately')
        if identity:
            for key in ('model_files','inputs_sha256','samples','dtype','attention','pooling','generation','device','software','numerics','flaggems','compiler','comm_backend','parallelism'):
                if key in identity and recorded.get(key)!=identity[key]:
                    raise ValueError('runner/report identity mismatch: '+key)
        independent=(tuple(receipt[k] for k in ('boot_id','pid','start_ticks'))
                     if all(receipt.get(k) is not None for k in ('boot_id','pid','start_ticks')) else None)
        rank=value.get('rank',recorded.get('rank',0));instrumentation=value.get('instrumentation','unknown')
        source=dict(warmup_excluded=(sum(bool(r.get('warmup')) for r in value.get('requests',[])) if value.get('schema')=='generation-runner/1' else sum(r.get('warmup',0) for r in value.get('runs',[]))),source=str(path),sha256=fingerprint,rank=rank,instrumentation=instrumentation,
                    execution=receipt,identity=recorded,load_s=value.get('load_s'),
                    semantics=value.get('semantics'),status=value.get('status','not_recorded'))
        sources.append(source)
        def add(scope,phase,step,shape,latency,repeat,index,numerator=0,token_count=0):
            nonlocal observation_count
            observation_count+=1
            if observation_count*512>limits.cache_bytes:raise ResourceLimit('runner observation statistics exceed cache_bytes')
            key=json.dumps([rank,instrumentation,scope,phase,step,shape],sort_keys=True)
            row=groups.setdefault(key,dict(rank=rank,instrumentation=instrumentation,scope=scope,
                phase=phase,generation_step=step,shape=shape,records=[],invalid_observations=0))
            if not isinstance(latency,(int,float)) or not math.isfinite(latency) or latency<=0:
                row['invalid_observations']+=1;return
            row['records'].append(dict(source=str(path),latency_ns=latency,repeat=repeat,index=index,
                independent=independent,samples=numerator,tokens=token_count))
            if receipt.get('pair_id') and receipt.get('measurement_protocol'):
                pair_key=json.dumps([receipt['pair_id'],receipt['measurement_protocol'],rank,scope,phase,step,shape],sort_keys=True)
                pair=paired.setdefault(pair_key,dict(pair_id=receipt['pair_id'],protocol=receipt['measurement_protocol'],rank=rank,
                    scope=scope,phase=phase,generation_step=step,shape=shape,conditions={}))
                pair['conditions'].setdefault(instrumentation,{}).setdefault(str(path),[]).append(latency)
        if value.get('schema')=='generation-runner/1':
            for index,request in enumerate(value.get('requests',[])):
                if request.get('warmup'):
                    continue
                batch=len(request.get('sample_ids',[]));shape=dict(batch=batch,input_tokens=request.get('input_tokens'))
                repeat=request.get('repeat',0)
                if receipt.get('group_execution') and receipt.get('boot_id') and receipt.get('clock')=='perf_counter_ns':
                    group_key=(receipt['group_execution'],receipt['boot_id'],request.get('request_id'))
                    group_requests.setdefault(group_key,[]).append(dict(rank=rank,request=request,world=recorded.get('parallelism',{}).get('world_size'),
                        source=str(path),instrumentation=instrumentation))
                add('request','request',None,shape,request.get('latency_ns'),repeat,index,batch,request.get('generated_tokens',0))
                steps=request.get('steps',[])
                for step in steps:
                    add('step',step.get('phase','unknown'),step.get('generation_step'),step.get('input_shape',shape),
                        step.get('latency_ns'),repeat,index,0,sum(step.get('active_samples',[])))
                decode=[s.get('latency_ns') for s in steps if s.get('phase')=='decode']
                if decode and all(isinstance(v,(int,float)) and v>0 for v in decode):
                    add('request_decode','decode',None,shape,sum(decode),repeat,index,0,
                        sum(sum(s.get('active_samples',[])) for s in steps if s.get('phase')=='decode'))
        else:
            batches=recorded.get('batches') or [[]]
            for repeat,run in enumerate(value.get('runs',[])):
                batch=batches[repeat%len(batches)]
                count=len(batch) if isinstance(batch,list) else value.get('batch_size')
                for index,latency in enumerate(run.get('latencies_ns',[])):
                    add('forward','measure',None,dict(batch=count,samples=batch),latency,
                        repeat//len(batches),index,count or 0)
    for identity,entries in group_requests.items():
        world=entries[0]['world'];ranks=[e['rank'] for e in entries]
        request=entries[0]['request']
        if not world or sorted(ranks)!=list(range(world)) or len(set(e['instrumentation'] for e in entries))!=1:
            notes.append('组请求缺少唯一的完整 rank 配对：'+str(identity));continue
        if any(any(e['request'].get(k)!=request.get(k) for k in ('sample_ids','tokens','stop_reasons','generated_tokens')) for e in entries):
            notes.append('组请求输出控制不一致，未计算组时延：'+str(identity));continue
        starts=[e['request'].get('start_ns') for e in entries];ends=[e['request'].get('end_ns') for e in entries]
        if not all(isinstance(x,int) for x in starts+ends) or any(b<=a for a,b in zip(starts,ends)):
            notes.append('组请求缺少有效计时区间：'+str(identity));continue
        latency=max(ends)-min(starts);shape=dict(batch=len(request.get('sample_ids',[])),input_tokens=request.get('input_tokens'))
        key=json.dumps(['group',entries[0]['instrumentation'],'group_request','request',None,shape],sort_keys=True)
        group=groups.setdefault(key,dict(rank='group',instrumentation=entries[0]['instrumentation'],scope='group_request',
            phase='request',generation_step=None,shape=shape,records=[],invalid_observations=0))
        group['records'].append(dict(source='group:'+str(identity[0]),latency_ns=latency,repeat=request.get('repeat',0),
            index=request.get('request_id'),independent=(identity[1],identity[0]),samples=len(request.get('sample_ids',[])),tokens=request.get('generated_tokens',0)))
        group_windows.append(dict(group_execution=identity[0],boot_id=identity[1],request_id=identity[2],rank='group',
            start_ns=min(starts),end_ns=max(ends),latency_ns=latency,ranks=ranks,
            rule='same boot and declared perf_counter_ns clock; min(rank start) to max(rank end); logical tokens counted once'))
    perturbations=[]
    for pair in paired.values():
        groups_by_mode=pair.pop('conditions');base=groups_by_mode.get('none',{});candidate=groups_by_mode.get('attachment',{})
        valid=len(base)==len(candidate)==1
        a=next(iter(base.values()),[]);b=next(iter(candidate.values()),[])
        valid=valid and len(a)==len(b) and bool(a)
        am=statistics.mean(a) if a else None;bm=statistics.mean(b) if b else None
        perturbations.append(dict(pair,baseline_mean_ns=am,instrumented_mean_ns=bm,
            overhead_ratio=(bm-am)/am if valid else None,status='ok' if valid else 'unavailable',
            reason=None if valid else 'paired_protocol_requires_one_source_per_mode_and_equal_observation_counts'))
    result=[]
    for group in groups.values():
        records=group.pop('records');values=[r['latency_ns'] for r in records];d=distribution(values)
        independent={}
        for row in records:
            if row['independent'] is not None:
                independent.setdefault(row['independent'],[]).append(row['latency_ns'])
        means=[statistics.mean(v) for v in independent.values()]
        repeats={}
        for record in records:repeats.setdefault((record['source'],record['repeat']),[]).append(record['latency_ns'])
        group['process_repeat_distributions']=[dict(source=key[0],repeat=key[1],**distribution(values)) for key,values in repeats.items()]
        cv=statistics.stdev(means)/statistics.mean(means) if len(means)>=2 else None
        median=d['median'];found=[r for r in records if median and r['latency_ns']>5*median]
        for row in found:
            spikes.append(dict(row,rank=group['rank'],phase=group['phase'],generation_step=group['generation_step'],
                               median_ns=median,screening_active=len(values)>=20))
        elapsed=sum(values);sample_count=sum(r['samples'] for r in records);tokens=sum(r['tokens'] for r in records)
        group.update(observations=len(values),mean_ns=d['mean'],median_ns=median,p90_ns=d['p90'],p95_ns=d['p95'],
            minimum_ns=min(values) if values else None,maximum_ns=max(values) if values else None,
            elapsed_ns=elapsed,sample_count=sample_count,generated_tokens=tokens,
            samples_per_second=sample_count*1e9/elapsed if elapsed and sample_count else None,
            generated_tokens_per_second=tokens*1e9/elapsed if elapsed and tokens else None,
            independent_repeats=len(means),repeat_means_ns=means,repeat_cv=cv,
            cv_screening='unstable' if len(means)>=3 and cv>.10 else 'stable' if len(means)>=3 else 'insufficient_independent_repeats',
            spike_count=len(found),spike_screening='triggered' if len(values)>=20 and found else 'clear' if len(values)>=20 else 'insufficient_observations',
            quantile_rule='nearest_rank',timing_boundary='runner synchronized forward' if group['scope']=='forward' else 'runner recorded '+group['scope'],
            status='partial' if group['invalid_observations'] else 'ok',
            repetitions='same process repeats unless execution receipt identifies a separate process')
        result.append(group)
    notes.extend(['预热不进入时延、吞吐与分布；runner latencies_ns 已由采集端排除预热。',
        'CV 使用独立进程均值，至少三次才筛查 >10%；尖峰超过中位数五倍，至少二十个有效观测才筛查。',
        '未保存进程身份的历史输入只能提供进程内重复分布；文件数量不作为独立重复依据。',
        FORWARD_TIMING_NOTE if result and all(g['scope']=='forward' for g in result) else GROUP_TIMING_NOTE])
    return dict(schema='runner-summary/1',groups=result,spikes=spikes,sources=sources,notes=notes,
                condition=condition,group_windows=group_windows,instrumentation_pairs=perturbations)


def generate(paths, out, *, limits=None):
    from flagprof.offline.worker import output_operation
    return output_operation(out, paths, _generate, paths, limits=limits)


def _generate(paths,out,*,limits=None):
    from .product import write_report
    out=Path(out).resolve()
    if any(out==Path(p).resolve() or out in Path(p).resolve().parents for p in paths):
        raise ValueError('baseline report must not contain source runner inputs')
    from flagprof.offline.worker import new_output
    new_output(out)
    summary=summarize(paths,limits=limits)
    document=dict(schema='runner-result/1',report_context=dict(identity=summary['condition']),results={},runner_summary=summary)
    (out/'result.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
    write_report(out,document,limits=limits)
    return dict(output=str(out),groups=len(summary['groups']))
