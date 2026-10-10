"""Unpaired cross-run statistics from raw measurements, never cumulative-time ratios."""
from collections import defaultdict
from pathlib import Path
import math
import statistics
from runtime.common import read_json, digest
from runtime.performance import percentile, percentile_sorted


def exact_coverage(rows, fields, domains):
    """A bitmap checks the Cartesian coverage without millions of tuple objects."""
    indexes = [{value: index for index, value in enumerate(domain)} for domain in domains]
    count = math.prod(len(domain) for domain in indexes)
    if len(rows) != count: return False
    seen = bytearray((count+7)//8)
    for row in rows:
        position = 0
        for field, values in zip(fields, indexes):
            value = str(row[field]) if field == 'rank' else row[field]
            if value not in values: return False
            position = position*len(values) + values[value]
        bit = 1 << (position % 8)
        if seen[position//8] & bit: return False
        seen[position//8] |= bit
    return True


def stats(rows,field='latency_ns'):
    if not rows:
        raise ValueError('empty performance observations')
    values, repeats = [], defaultdict(list)
    samples, tokens, tokens_complete = 0, 0, True
    for row in rows:
        values.append(row[field])
        repeats[str(row['repeat'])].append(row[field]/1e6)
        samples += row['samples']
        if row.get('tokens') is None: tokens_complete = False
        else: tokens += row['tokens']
    if any(type(x) not in (int,float) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('invalid performance interval')
    total=sum(values)
    means={k:statistics.mean(v) for k,v in repeats.items()}
    ordered = sorted(values)
    return {'observations':len(rows),'mean_ms':statistics.mean(values)/1e6,
            **{f'p{q}_ms':percentile_sorted(ordered,q)/1e6 for q in (50,90,99)},
            'samples_per_second':samples*1e9/total,
            'effective_tokens_per_second':tokens*1e9/total if tokens_complete else None,
            'repeat_means_ms':means,'repeat_mean_ms':statistics.mean(means.values()),
            'repeat_stddev_ms':statistics.stdev(means.values()) if len(means)>1 else None,
            'independent_repeats':len(means)}


def extract(root,result,cfg):
    if result.get('status') != 'completed': raise ValueError('incomplete candidate run')
    out={'schema_version':1,'total':{},'layer':{},'layer_memory':{},'memory':{},'communication':{},'layouts':{},
         'sampling':cfg['performance'],'device':read_json(root/'prepared/identity.json'),
         'input_batches':read_json(root/'prepared/forward-inputs.json')['batches']}
    for side,repeats in result['paths'].items():
        rows=[]; memory=[]; layouts={}
        for label,worker in repeats.items():
            if worker['status'] != 'completed': raise ValueError('incomplete repeat')
            rows.extend(dict(r,repeat=label) for r in worker['batches'])
            for rank,value in worker.get('ranks',{'0':worker}).items():
                memory.append({'repeat':label,'rank':str(rank),'source':'framework_allocator',**value['memory']})
                if value.get('generation_cache'):
                    memory[-1]['generation_cache']=value['generation_cache']
                path=root/side/label/('rank-'+rank)/'tp-plan.json'
                if path.is_file():
                    plan=read_json(path)
                    normalized={k:v for k,v in plan.items() if k not in ('rank','physical_device')}
                    if rank in layouts and layouts[rank] != normalized:
                        raise ValueError('TP layout changed across repeats')
                    layouts[rank]=normalized
        out['total'][side]=rows;out['memory'][side]=memory;out['layouts'][side]=layouts
    for side,repeats in result.get('layer',{}).get('runs',{}).get('layer_timing',{}).items():
        from runtime.artifacts import Store
        out['layer'][side]=Store(root).records(dict(row,repeat=label,rank=str(rank))
            for label,worker in repeats.items()
            for rank,value in worker.get('ranks',{'0':worker}).items() for row in value['rows'])
    out['communication']=result.get('profiles',{})
    for side,repeats in result.get('layer',{}).get('runs',{}).get('layer_memory',{}).items():
        from runtime.artifacts import Store
        out['layer_memory'][side]=Store(root).records(dict(row,repeat=label,rank=str(rank))
            for label,worker in repeats.items()
            for rank,value in worker.get('ranks',{'0':worker}).items() for row in value['rows'])
    return out


def validate_metrics(data,contract):
    if data.get('schema_version') != 1 or not data.get('total'):
        raise ValueError('missing performance measurements')
    sampling=data['sampling']
    repeats=sampling['repeats'];rounds=sampling['measure_rounds']
    generation=contract.get('generation') if contract.get('execution_kind') == 'autoregressive' else None
    maximum=generation['max_new_tokens'] if generation else 1
    for side,rows in data['total'].items():
        stats(rows)
        keys=[(r['repeat'],r['cycle'],r['batch_index']) for r in rows]
        if len(keys) != len(set(keys)): raise ValueError('duplicate total timing record')
        batch_ids=sorted({r['batch_index'] for r in rows})
        if not batch_ids or batch_ids != list(range(len(batch_ids))): raise ValueError('missing input batch')
        repeat_ids=sorted({r['repeat'] for r in rows})
        expected={(f'repeat-{r:02}',c,b) for r in range(repeats) for c in range(rounds) for b in batch_ids}
        if set(keys) != expected: raise ValueError('missing repeat/cycle/batch timing record')
        workload={}
        for row in rows:
            signature=tuple(str(row[k]) for k in ('samples','tokens','input_shape','sample_ids'))
            if row['batch_index'] in workload and workload[row['batch_index']] != signature:
                raise ValueError('workload changed during measurement')
            workload[row['batch_index']]=signature
            if row['samples'] != len(row['sample_ids']) or (row['tokens'] is not None and row['tokens'] <= 0):
                raise ValueError('invalid logical workload counts')
            if generation:
                steps=row.get('generation_steps',[])
                if (row.get('generated_tokens') != row['samples']*maximum or row['tokens'] != row['generated_tokens']
                        or len(steps)!=maximum):
                    raise ValueError('fixed generation workload mismatch')
                for index,step in enumerate(steps):
                    if (step.get('generation_step')!=index or step.get('phase')!=('prefill' if index==0 else 'decode')
                            or step.get('active_samples')!=[True]*row['samples']
                            or step.get('input_shape')!=[row['samples'],row['input_shape'][1] if index==0 else 1]
                            or step.get('kv_length_before')!=(0 if index==0 else row['input_shape'][1]+index-1)
                            or type(step.get('latency_ns')) not in (int,float) or step['latency_ns']<=0):
                        raise ValueError('invalid generation timing step')
                sealed_batch=data['input_batches'][row['batch_index']]
                if row.get('input_tokens')!=sum(sealed_batch['valid_tokens']):
                    raise ValueError('generation input token count mismatch')
        sealed={b['batch_index']:(str(len(b['sample_ids'])),str(len(b['sample_ids'])*maximum if generation else sum(b['valid_tokens']) if 'valid_tokens' in b else b.get('tokens')),
                str(b['input_shape']),str(b['sample_ids'])) for b in data['input_batches']}
        if workload != sealed:
            raise ValueError('timing does not cover sealed input batches')
        expected_ranks={str(i) for i in range(contract['runtime_devices'])}
        memory=data['memory'][side]
        if {(m['repeat'],m['rank']) for m in memory} != {(r,k) for r in repeat_ids for k in expected_ranks}:
            raise ValueError('missing memory rank/repeat')
        if len(memory) != repeats*len(expected_ranks): raise ValueError('duplicate memory record')
        if generation:
            length=data['input_batches'][0]['input_shape'][1]
            for observation in memory:
                cache=observation.get('generation_cache',{})
                if cache.get('status')!='measured':
                    raise ValueError('generation baseline lacks independent cache measurement')
                for stage,index in [('prefill',0),('final_decode',maximum-1)]:
                    value=cache.get(stage,{})
                    storage=value.get('cache',{})
                    if (value.get('generation_step')!=index or storage.get('status')!='measured'
                            or storage.get('sequence_length')!=length+index
                            or type(storage.get('storage_bytes')) is not int or storage['storage_bytes']<=0):
                        raise ValueError('invalid phase KV cache observation')
        layouts=data['layouts'][side]
        if contract['tp_degree'] > 1 and set(layouts) != expected_ranks:
            raise ValueError('missing normalized TP layout')
        layer=data.get('layer',{}).get(side)
        if layer is not None:
            if not layer: raise ValueError('empty performance observations')
            if not exact_coverage(layer, ['repeat','cycle','batch_index','rank','module','call_index'],
                [repeat_ids, range(rounds), batch_ids, sorted(expected_ranks), contract['instrumentation'], range(maximum)]):
                raise ValueError('missing or duplicate layer/rank/batch timing record')
            for row in layer:
                interval = row['device_ns']
                if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
                    raise ValueError('invalid performance interval')
                if generation:
                    index=row['call_index'];batch=data['input_batches'][row['batch_index']]
                    if (row.get('generation_step')!=index or row.get('phase')!=('prefill' if index==0 else 'decode')
                            or row['sample_ids']!=batch['sample_ids'] or row['samples']!=len(batch['sample_ids'])
                            or row['input_shape']!=[row['samples'],batch['input_shape'][1] if index==0 else 1]
                            or row['tokens']!=(sum(batch['valid_tokens']) if index==0 else row['samples'])
                            or row.get('kv_length_before')!=(0 if index==0 else batch['input_shape'][1]+index-1)):
                        raise ValueError('layer generation step workload mismatch')
                    continue
                signature=tuple(str(row[k]) for k in ('samples','tokens','input_shape','sample_ids'))
                if signature != workload[row['batch_index']]: raise ValueError('layer workload mismatch')


def ratios(a,b):
    return {k:(b[k]/a[k] if type(a.get(k)) in (int,float) and a[k] > 0 and
               type(b.get(k)) in (int,float) and math.isfinite(b[k]) else None)
            for k in ('mean_ms','p50_ms','p90_ms','p99_ms','samples_per_second','effective_tokens_per_second')}


def grouped(rows,fields):
    from runtime.artifacts import Records
    if isinstance(rows, Records):
        from runtime.record_index import Index
        return Index(rows, fields).groups()
    groups=defaultdict(list)
    for row in rows:
        key=tuple(str(row[f]) for f in fields)
        groups[key].append(row)
    return groups


def logical_communication(profiles,side):
    groups=defaultdict(lambda:{'calls':0,'logical_payload_bytes':0,'profiled_batches':0})
    for run in profiles.get(side,{}).values():
        for rank,state in run.get('communication',{}).get('ranks',{}).items():
            events=state.get('events',[])
            batches={(e['cycle'],e['batch_index']) for e in events if e.get('module')}
            for category,values in state.get('categories',{}).items():
                row=groups[(str(rank),category)]
                row['calls']+=values['collectives']
                if values.get('logical_payload_bytes') is None:
                    row['logical_payload_bytes']=None
                elif row['logical_payload_bytes'] is not None:
                    row['logical_payload_bytes']+=values['logical_payload_bytes']
                row['profiled_batches']+=len(batches)
    for row in groups.values():
        n=row['profiled_batches']
        row['calls_per_batch']=row['calls']/n if n else None
        row['payload_bytes_per_batch']=row['logical_payload_bytes']/n if n and row['logical_payload_bytes'] is not None else None
    return groups


def compare(reference,candidate,contract):
    validate_metrics(reference,contract);validate_metrics(candidate,contract)
    if set(reference['total']) != {'off'}: raise ValueError('baseline must be native only')
    out={'schema_version':1,'status':'completed','ratio_definition':'candidate / A100',
         'direction':{'latency':'lower is faster','throughput':'higher is faster','memory':'lower uses less memory'},
         'sampling':{'a100':reference['sampling'],'candidate':candidate['sampling']},
         'independent_runs':True,'outliers_removed':False,'paths':{}}
    refwork={r['batch_index']:tuple(str(r[k]) for k in ('samples','tokens','input_shape','sample_ids'))
             for r in reference['total']['off']}
    reference_groups, reference_stats = {}, {}
    def ref_group(name, fields):
        key = (name, tuple(fields))
        if key not in reference_groups:
            reference_groups[key] = grouped(reference[name]['off'], fields)
        return reference_groups[key]
    def ref_stats(rows, field='latency_ns'):
        key = (id(rows), field)
        if key not in reference_stats: reference_stats[key] = stats(rows, field)
        return reference_stats[key]
    for side,rows in candidate['total'].items():
        work={r['batch_index']:tuple(str(r[k]) for k in ('samples','tokens','input_shape','sample_ids')) for r in rows}
        if work != refwork: raise ValueError('candidate/reference batch workload mismatch')
        if candidate['layouts'][side] != reference['layouts']['off']:
            raise ValueError('candidate/reference TP shard layout mismatch')
        a=ref_stats(reference['total']['off']);b=stats(rows)
        value={'total':{'a100':a,'candidate':b,'candidate_over_a100':ratios(a,b)},'shapes':[],'layers':[],'layer_memory':[],'memory':[]}
        if contract.get('generation'):
            value['generation']={}
            def phase_rows(all_rows,phase):
                return [dict(step,repeat=row['repeat'],samples=row['samples'],
                             tokens=row['input_tokens'] if phase=='prefill' else sum(step['active_samples']))
                        for row in all_rows for step in row['generation_steps'] if step['phase']==phase]
            for phase in ('prefill','decode'):
                left=phase_rows(reference['total']['off'],phase);right=phase_rows(rows,phase)
                if not left and not right: continue
                a=stats(left);b=stats(right)
                value['generation'][phase]={'a100':a,'candidate':b,'candidate_over_a100':ratios(a,b)}
            value['generation_cache']=[]
        ag=ref_group('total',['input_shape']);bg=grouped(rows,['input_shape'])
        if ag.keys()!=bg.keys(): raise ValueError('shape coverage mismatch')
        for shape in sorted(ag):
            a=ref_stats(ag[shape]);b=stats(bg[shape])
            value['shapes'].append({'shape':shape[0],'a100':a,'candidate':b,'candidate_over_a100':ratios(a,b)})
        if side in candidate['layer']:
            fields=['rank','module','input_shape']+(['phase','generation_step','kv_length_before'] if contract.get('generation') else [])
            ag=ref_group('layer',fields)
            bg=grouped(candidate['layer'][side],fields)
            if ag.keys()!=bg.keys(): raise ValueError('layer coverage mismatch')
            for key in sorted(ag):
                a=ref_stats(ag[key],'device_ns');b=stats(bg[key],'device_ns')
                value['layers'].append({'rank':key[0],'module':key[1],'shape':key[2],
                    **dict(zip(fields[3:],key[3:])),
                    'a100':a,'candidate':b,'candidate_over_a100':ratios(a,b),
                    'scope':'rank-local inclusive Event window, not global layer latency'})
        for rank in range(contract['runtime_devices']):
            a=[x for x in reference['memory']['off'] if x['rank']==str(rank)]
            b=[x for x in candidate['memory'][side] if x['rank']==str(rank)]
            if contract.get('generation'):
                for stage in ('prefill','final_decode'):
                    for field,keys in [('kv_storage_bytes',('cache','storage_bytes')),
                                       ('allocator_peak_allocated_bytes',('allocator_peak','allocated_bytes')),
                                       ('allocator_peak_reserved_bytes',('allocator_peak','reserved_bytes'))]:
                        def maximum_seen(items):
                            values=[x['generation_cache'][stage].get(keys[0],{}).get(keys[1]) for x in items]
                            return max(values) if values and all(v is not None for v in values) else None
                        av,bv=maximum_seen(a),maximum_seen(b)
                        value['generation_cache'].append({'rank':str(rank),'stage':stage,'field':field,'a100':av,'candidate':bv,
                            'candidate_over_a100':bv/av if av is not None and av>0 and bv is not None else None,
                            'scope':'first sealed input batch, independent untimed generation; maximum rank observation across repeats'})
            for field in ('allocated_bytes','reserved_bytes'):
                av=max((x['measurement_peak'][field] for x in a if x['measurement_peak'][field] is not None),default=None)
                bv=max((x['measurement_peak'][field] for x in b if x['measurement_peak'][field] is not None),default=None)
                value['memory'].append({'rank':str(rank),'source':'framework_allocator','field':field,
                    'a100':av,'candidate':bv,'candidate_over_a100':bv/av if av is not None and av>0 and bv is not None else None,
                    'scope':'maximum observed rank peak across independent repeats'})
        if side in candidate.get('layer_memory',{}):
            fields=['rank','module','input_shape']+(['phase','generation_step','kv_length_before'] if contract.get('generation') else [])
            ag=ref_group('layer_memory',fields)
            bg=grouped(candidate['layer_memory'][side],fields)
            if ag.keys()!=bg.keys(): raise ValueError('layer memory coverage mismatch')
            for key in sorted(ag):
                for field in ('allocated_bytes','reserved_bytes'):
                    av=max(r['peak_increment'][field] for r in ag[key])
                    bv=max(r['peak_increment'][field] for r in bg[key])
                    value['layer_memory'].append({'rank':key[0],'module':key[1],'shape':key[2],
                        **dict(zip(fields[3:],key[3:])),
                        'field':field,'source':'framework_allocator_synchronized_layer_scope',
                        'a100':av,'candidate':bv,'candidate_over_a100':bv/av if av>0 else None,
                        'scope':'maximum layer peak increment; includes temporaries and workspace, not pure activation'})
        value['communication']={'a100':reference['communication'], 'candidate':{side: candidate['communication'][side]} if side in candidate['communication'] else {},
            'time_ratio':None,'time_ratio_status':'not_comparable',
            'reason':'NCCL kernel duration and CANN collective elapsed have different boundaries; logical payload is per rank, not physical wire traffic'}
        ag=logical_communication(reference['communication'],'off');bg=logical_communication(candidate['communication'],side)
        logical=[]
        for key in sorted(set(ag)|set(bg)):
            a=ag.get(key);b=bg.get(key)
            logical.append({'rank':key[0],'category':key[1],'a100':a,'candidate':b,
                'status':'comparable' if a and b else 'missing',
                'candidate_over_a100':{k:b[k]/a[k] if a and b and a.get(k) and b.get(k) is not None else None
                                       for k in ('calls_per_batch','payload_bytes_per_batch')}})
        value['communication']['logical']=logical
        value['communication']['status']='not_applicable' if contract['tp_degree']==1 else 'observed'
        out['paths'][side]=value
    return out


def render(root,result):
    # Keep the exact internal execution configuration, and separately expose a
    # public config that the strict CLI can load without private worker flags.
    config=root/'effective.yaml'
    if config.is_file():
        import yaml
        value=yaml.safe_load(config.read_text())
        if isinstance(value,dict) and any(k.startswith('_') for k in value):
            target=root/'effective-public.yaml'
            temporary=root/'effective-public.yaml.tmp'
            temporary.write_text(yaml.safe_dump({k:v for k,v in value.items() if not k.startswith('_')},
                                               allow_unicode=True,sort_keys=False))
            temporary.replace(target)
    path=root/'performance-comparison.json'
    if not path.is_file(): return
    value=read_json(path)
    lines=['','## A100 离线性能比较','',f"基线标识：`{value['baseline_id']}`。",
           '倍率均为 **被测值 / A100 值**；时延越低越快，吞吐越高越快。跨服务器重复独立，不作配对检验，不设业务达标阈值。',
           '正式总体计时不含逐层插桩、Profiler、加载、预热和输入输出搬运。','',
           '| 路径 | 指标 | A100 | 被测 | 被测/A100 |','|---|---|---:|---:|---:|']
    def fmt(v): return '不适用/不足' if v is None else f'{v:.6g}'
    sampling_lines = []
    for side,p in value['paths'].items():
        for metric,ratio in p['total']['candidate_over_a100'].items():
            lines.append(f"| {side} | {metric} | {fmt(p['total']['a100'][metric])} | {fmt(p['total']['candidate'][metric])} | {fmt(ratio)} |")
        for who in ('a100','candidate'):
            s=p['total'][who]
            sampling_lines += ['',f"{side}/{who}：{s['observations']} 条总体计时，{s['independent_repeats']} 次独立重复；重复均值标准差 {fmt(s['repeat_stddev_ms'])} ms。"]
    lines += sampling_lines
    lines += ['','[完整比较与分 shape/rank/layer 统计](performance-comparison.json) · [A100 原始数值摘录](a100-performance-extract.json) · [被测原始数值](candidate-performance-extract.json)','',
              '逐层为 rank 本地包含式 Event 窗口，不相加为全局层时延。显存按 allocator 来源及 rank 比较，各 rank 峰值之和不是同时发生的系统峰值。NCCL kernel 时间与 CANN 通信时间不计算倍率。']
    lines += ['','| 路径 | shape | A100 均值 ms | 被测均值 ms | 被测/A100 |', '|---|---|---:|---:|---:|']
    for side,p in value['paths'].items():
        for row in p['shapes']:
            lines.append(f"| {side} | {row['shape']} | {fmt(row['a100']['mean_ms'])} | {fmt(row['candidate']['mean_ms'])} | {fmt(row['candidate_over_a100']['mean_ms'])} |")
    lines += ['','| 路径 | rank | allocator 峰值字段 | A100 bytes | 被测 bytes | 被测/A100 |', '|---|---|---|---:|---:|---:|']
    for side,p in value['paths'].items():
        for row in p['memory']:
            lines.append(f"| {side} | {row['rank']} | {row['field']} | {row['a100']} | {row['candidate']} | {fmt(row['candidate_over_a100'])} |")
    if result.get('level')=='layer':
        from runtime.layer_experience import matches
        from runtime.common import write_json
        selected=result.get('report_view',{}).get('selection',{})
        detail=['# A100 逐层比较','', '同 rank、module、shape 的包含式 Event 窗口；倍率为被测/A100。', '',
                '| 路径 | module | rank | shape | A100 ms | 被测 ms | 倍率 |', '|---|---|---|---|---:|---:|---:|']
        view={}
        for side,p in value['paths'].items():
            view[side]=[r for r in p['layers'] if matches(r,selected)]
            for row in view[side]:
                detail.append(f"| {side} | {row['module']} | {row['rank']} | {row['shape']} | {fmt(row['a100']['mean_ms'])} | {fmt(row['candidate']['mean_ms'])} | {fmt(row['candidate_over_a100']['mean_ms'])} |")
        (root/'a100-layer-comparison.md').write_text('\n'.join(detail)+'\n')
        write_json(root/'performance-comparison-view.json',{'selection':selected,'paths':view})
        lines += ['','[按模块、rank、shape 筛选的 A100 逐层比较](a100-layer-comparison.md) · [当前比较视图](performance-comparison-view.json)']
    lines += ['','| 路径 | rank | 通信类别 | A100 每批逻辑 bytes | 被测每批逻辑 bytes | 被测/A100 |',
              '|---|---|---|---:|---:|---:|']
    for side,p in value['paths'].items():
        for row in p['communication'].get('logical',[]):
            lines.append(f"| {side} | {row['rank']} | {row['category']} | {fmt((row['a100'] or {}).get('payload_bytes_per_batch'))} | {fmt((row['candidate'] or {}).get('payload_bytes_per_batch'))} | {fmt(row['candidate_over_a100']['payload_bytes_per_batch'])} |")
    lines += ['','| 平台 | rank | 运行时设备名称 | 显存 bytes |', '|---|---|---|---:|']
    for label,name in [('A100','a100-performance-extract.json'),('被测','candidate-performance-extract.json')]:
        ident=read_json(root/name)['device']
        devices=ident.get('devices',[dict(ident.get('device_properties',{}),rank=0)])
        for device in devices:
            lines.append(f"| {label} | {device['rank']} | {device.get('name','未知')} | {device.get('total_memory','未知')} |")
    lines += ['', '匹配的是运行时计算设备数和 TP 度；物理板卡与芯片映射见设备快照，不据此认定板卡资源相等。']
    if (root/'retention.json').is_file():
        lines += ['','[证据保留清单](retention.json)：精简结果保留数值与报告重绘能力，原始 trace 删除后无法重新解析。']
    if (root/'effective-public.yaml').is_file():
        lines += ['','[可重新加载的公开配置](effective-public.yaml)；原始 effective.yaml 保留内部执行控制字段。迁移后仍需调整模型、基线和策略路径。']
    target=root/'report.md'
    target.write_text(target.read_text()+'\n'+'\n'.join(lines)+'\n')
