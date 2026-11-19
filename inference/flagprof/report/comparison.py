# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Declared condition comparisons with immutable timing and output boundaries."""
import copy
import json
from pathlib import Path
from flagprof.analyzers.gap import comparable,distribution


DECLARABLE=frozenset(('device','software','flaggems','include','compiler','communication','comm_backend','parallelism'))


def conditions(left,right,declare=None):
    def checked(a,b):
        result=comparable(a,b)
        for side,value in (('left',a),('right',b)):
            identity=value.get('workload_identity') or {}
            if identity.get('task')=='generate':
                if not identity.get('generation'):result['reasons'].append(side+':output_semantics_missing')
            elif not identity.get('pooling'):result['reasons'].append(side+':output_semantics_missing')
        if result['reasons']:result['status']='not_comparable'
        return result
    original=checked(left,right)
    names=(declare or {}).get('differences',[])
    if not isinstance(names,list) or any(not isinstance(n,str) or n.split('.')[0] not in DECLARABLE for n in names):
        raise ValueError('only device, software, components, backend and parallelism differences may be declared')
    if not names:return original
    a,b=copy.deepcopy(left),copy.deepcopy(right)
    details=[]
    def visit(path,x,y):
        if x==y:return
        if isinstance(x,dict) and isinstance(y,dict):
            for key in sorted(set(x)|set(y)):visit(path+'.'+key if path else key,x.get(key),y.get(key))
        else:details.append(dict(field='workload_identity.'+path,left=x,right=y,
            declared=any(path==name or path.startswith(name+'.') for name in names)))
    visit('',left.get('workload_identity') or {},right.get('workload_identity') or {})
    for name in names:
        x=a.get('workload_identity') or {};y=b.get('workload_identity') or {}
        parts=name.split('.')
        for part in parts[:-1]:
            x=x.get(part,{});y=y.get(part,{})
            if not isinstance(x,dict) or not isinstance(y,dict):break
        else:
            if parts[-1] in x and parts[-1] in y:y[parts[-1]]=copy.deepcopy(x[parts[-1]])
    check=checked(a,b)
    check['differences']=details+[d for d in original['differences'] if d['field'].startswith('window_identity.')]
    check['declaration']=names
    return check


def change(a,b,allowed):
    valid=allowed and a is not None and b is not None
    return dict(baseline=a,candidate=b,delta=b-a if valid else None,
                relative_change=(b-a)/a if valid and a else None,
                reason=None if valid and a else 'zero_baseline' if valid else 'not_comparable_or_missing')


def generate(reports, out, *, declare=None, limits=None):
    from flagprof.offline.worker import output_operation
    return output_operation(out, reports, _generate, reports, declare=declare, limits=limits)


def _generate(reports,out,*,declare=None,limits=None):
    from .model import opened_model,read_document
    from . import write_json
    from .product import write_report
    out=Path(out).resolve()
    if any(out==Path(p).resolve() or Path(p).resolve() in out.parents or out in Path(p).resolve().parents for p in reports):
        raise ValueError('comparison output must be separate from source reports')
    out.mkdir(parents=True,exist_ok=True)
    with opened_model(limits=limits) as model:
        sources=[]
        for report in reports:
            path=Path(report).resolve();value=read_document(path/'result.json',model)
            panel=value.get('results',{}).get('perf.device',{})
            coverage=panel.get('coverage',{});gap=value.get('results',{}).get('perf.gap',{})
            context={k:coverage.get(k) for k in ('window_identity','workload_identity')}
            runner=value.get('runner_summary')
            if not panel and runner:
                identity=dict(runner.get('condition') or {})
                identity['device']={str(s['rank']):s.get('identity',{}).get('device') for s in runner.get('sources',[])}
                groups=runner.get('groups',[]);source_rows=runner.get('sources',[])
                context=dict(workload_identity=identity,window_identity=dict(window_kind='runner_synchronized',
                    warmup=[s.get('warmup_excluded') for s in source_rows],
                    forwards=[g.get('observations') for g in groups],repeats=len(source_rows),
                    profiler=sorted(set(s.get('instrumentation','unknown') for s in source_rows)),
                    dispatch='none' if all(s.get('instrumentation')=='none' for s in source_rows) else 'not_recorded',
                    boundary_rule='runner synchronized recorded forward or request; excludes load and warmup'))
            sources.append(dict(report=str(path),context=context,
                windows=gap.get('data',()),data=coverage.get('headline') or panel.get('data',()),
                runner=value.get('runner_summary')))
        pairs=[];window_pairs=[];metrics=[]
        for i,left in enumerate(sources):
            for right in sources[i+1:]:
                checked=conditions(left['context'],right['context'],declare)
                pairs.append(dict(left=left['report'],right=right['report'],**checked))
                # A window class represents repeated observations; window IDs are not aligned between runs.
                kinds=sorted(set(r['kind'] for r in left['windows'])|set(r['kind'] for r in right['windows']))
                for kind in kinds:
                    a=[r for r in left['windows'] if r['kind']==kind];b=[r for r in right['windows'] if r['kind']==kind]
                    ac=dict(left['context'],window_identity=a[0].get('window_identity')) if a else {}
                    bc=dict(right['context'],window_identity=b[0].get('window_identity')) if b else {}
                    check=conditions(ac,bc,declare)
                    # Every member must have the same definition and complete coverage.
                    if any(conditions(dict(ac,window_identity=r.get('window_identity')),ac)['status']!='comparable' for r in a) or any(conditions(dict(bc,window_identity=r.get('window_identity')),bc)['status']!='comparable' for r in b):
                        check=dict(check,status='not_comparable',reasons=check['reasons']+['mixed_window_definitions'])
                    window_pairs.append(dict(left=left['report'],right=right['report'],kind=kind,**check))
                    if i!=0:continue
                    for metric in ('window_ns','all_activity_union_ns','blank_ns','compute_union_ns','comm_union_ns'):
                        av=[r[metric] for r in a if r.get(metric) is not None];bv=[r[metric] for r in b if r.get(metric) is not None]
                        for stat in ('mean','median','p95'):
                            metrics.append(dict(source=right['report'],scope=kind,metric=metric+'.'+stat,unit='ns',
                                baseline_observations=len(av),candidate_observations=len(bv),comparison_status=check['status'],
                                **change(distribution(av)[stat],distribution(bv)[stat],check['status']=='comparable')))
                if i==0:
                    if left.get('runner') and right.get('runner'):
                        def key(row):return json.dumps([row.get(k) for k in ('rank','instrumentation','scope','shape','phase','generation_step')],sort_keys=True)
                        left_groups={key(r):r for r in left['runner']['groups']}
                        for row in right['runner']['groups']:
                            baseline=left_groups.get(key(row),{})
                            for metric in ('mean_ns','median_ns','p95_ns','samples_per_second','generated_tokens_per_second'):
                                allowed=checked['status']=='comparable' and bool(baseline)
                                metrics.append(dict(source=right['report'],rank=row.get('rank'),scope=row.get('scope'),
                                    phase=row.get('phase'),generation_step=row.get('generation_step'),shape=row.get('shape'),
                                    metric=metric,unit='ns' if metric.endswith('_ns') else 'samples/s' if metric=='samples_per_second' else 'tokens/s',
                                    comparison_status='comparable' if allowed else 'not_comparable',
                                    **change(baseline.get(metric),row.get(metric),allowed)))
                    # Same-category sums are descriptive workload changes, never speedups.
                    for category in sorted(set(r.get('category','unknown') for s in (left,right) for r in s['data'])):
                        def total(source):
                            values=[r['device_work_ns'] for r in source['data'] if r.get('category')==category and r.get('device_work_ns') is not None]
                            return sum(values) if values else None
                        metrics.append(dict(source=right['report'],scope='source_window',metric=category+'.device_work_ns',unit='ns',
                            comparison_status=checked['status'],aggregation='sum_including_overlap',
                            **change(total(left),total(right),checked['status']=='comparable')))
        for index,source in enumerate(sources):
            from .product import redraw
            local=out/('source-'+str(index))
            if not local.exists():redraw(source['report'],local,limits=limits)
            source['local_report']='source-'+str(index)+'/report.html'
            source['comparison_status']='not_comparable' if any(p['status']=='not_comparable' for p in pairs if source['report'] in (p['left'],p['right'])) else 'comparable'
        result=dict(schema='device-comparison/1',declaration=declare or {},sources=sources,comparisons=pairs,
                    window_comparisons=window_pairs,metric_comparisons=metrics,
                    note='第一份报告为基准。变化因素完整保留；差值描述观测变化，不自动归因于设备或组件。工作量时长和不作为加速比。')
        with (out/'result.json').open('w') as stream:write_json(stream,result);stream.write('\n')
        write_report(out,result,limits=limits)
        # Caller-compatible materialized summaries; large input tables remain on disk.
        return dict(result,sources=[dict(s,data=list(s['data']),windows=list(s['windows'])) for s in sources])
