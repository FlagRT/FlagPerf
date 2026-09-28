"""Recompute Day 7 single/two-rank groups; never modify run evidence."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUT=ROOT/'base/result/p800-day7-20260924'
sys.path.insert(0,str(ROOT/'base'))
from qualification import summarize
from executors.bounded_benchmark import validate_metric

def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(values):
    mean=statistics.mean(values); deviation=statistics.stdev(values)
    cv=deviation/mean*100
    return dict(status='passed' if cv<=5 else 'unstable',repetitions=len(values),median=statistics.median(values),
                min=min(values),max=max(values),mean=mean,sample_standard_deviation=deviation,ddof=1,cv_percent=cv,max_cv_percent=5)

def validated_ab(root, monitor_on):
    summary=read(root/'summary.json'); context=read(root/'control/host-context.json')
    assets=read(root/'control/case-assets.json'); result=read(root/'benchmark-result.json')
    code=read(root/'code-identity.json')
    assert summary['status']=='passed' and summary['cleanup_status']=='passed' and summary['lease_released']
    assert context['case_assets_sha256']==digest(root/'control/case-assets.json')
    ranks=result['expected_ranks']; metrics=[]
    assert len(result['metrics'])==len(ranks) and not result['missing_ranks']
    for rank in ranks:
        metric=read(root/('artifacts/metric-rank-'+str(rank)+'.json'))
        correctness=read(root/('artifacts/correctness-rank-'+str(rank)+'.json'))
        validate_metric(metric,correctness,context,digest(root/'control/host-context.json'),summary['device_bindings'][rank],assets['merged_config'])
        assert metric in result['metrics'] and metric['elapsed_seconds']>=15
        if monitor_on:
            monitor=read(root/'benchmark-monitor/summary.json')
            assert summary['monitoring_status']=='passed'
            assert monitor['primary_sample_counts_by_target'].get(metric['binding']['resource_key'],0)>=10
        metrics.append(metric)
    identity=(metrics[0]['case_assets_sha256'],tuple(m['binding']['resource_key'] for m in metrics),
              summary['runtime']['image_id'],json.dumps(code['source_sha256'],sort_keys=True))
    return min(m['value'] for m in metrics),identity,dict(path=str(root.relative_to(OUT)),
        summary_sha256=digest(root/'summary.json'),rank_values=[m['value'] for m in metrics])

def comm_group(paths):
    rows=[]; identities=set(); ids=set()
    for root in paths:
        s=read(root/'summary.json'); result=read(root/'benchmark-result.json')
        monitor=read(root/'benchmark-monitor/summary.json')
        for key in ('status','correctness_status','measurement_status','monitoring_status','cleanup_status','postflight_status'):
            assert s.get(key)=='passed',(root,key,s.get(key))
        assert s.get('lease_released') is True
        context=read(root/'control/host-context.json'); assets=read(root/'control/case-assets.json')
        assert context['case_assets_sha256']==digest(root/'control/case-assets.json')
        metrics=[]
        assert len(result['metrics'])==2
        for rank in (0,1):
            metric=read(root/('artifacts/metric-rank-'+str(rank)+'.json'))
            correctness=read(root/('artifacts/correctness-rank-'+str(rank)+'.json'))
            validate_metric(metric,correctness,context,digest(root/'control/host-context.json'),s['device_bindings'][rank],assets['merged_config'])
            assert metric in result['metrics'] and metric['elapsed_seconds']>=15
            resource=metric['binding']['resource_key']
            assert monitor['primary_sample_counts_by_target'].get(resource,0)>=10
            metrics.append(metric)
        code=read(root/'code-identity.json')
        identities.add((tuple(m['binding']['resource_key'] for m in metrics),s['runtime']['image_id'],
                        metrics[0]['case_assets_sha256'],json.dumps(code['source_sha256'],sort_keys=True)))
        ids.add(s['run_id'])
        rows.append(dict(directory=str(root),run_id=s['run_id'],rank_values=[m['value'] for m in metrics],
                         value=min(m['value'] for m in metrics),elapsed=[m['elapsed_seconds'] for m in metrics],summary_sha256=digest(root/'summary.json')))
    assert len(paths)==5 and len(ids)==5 and len(identities)==1
    return dict(schema_version=1,**stats([r['value'] for r in rows]),metric=metrics[0]['metric'],unit='GB/s',runs=rows,
                aggregation='minimum of the two validated rank bandwidths per run; rank values retained; never summed')

def main():
    q=OUT/'qualification'; q.mkdir(exist_ok=True)
    results=[]
    names=[g['group'] for g in read(OUT/'qualification-plan.json')['groups']]
    names += ['comm-MPI-m64','comm-P2P-m64']
    for name in names:
        paths=[OUT/'runs'/(name+'-q'+str(i)) for i in range(1,6)]
        if not all((p/'summary.json').exists() and read(p/'summary.json').get('status')!='running' for p in paths):
            results.append(dict(group=name,status='incomplete' if '--final' in sys.argv else 'pending',
                completed_runs=sum((p/'summary.json').exists() and read(p/'summary.json').get('status')!='running' for p in paths))); continue
        try:
            data=comm_group(paths) if name.startswith('comm-') else summarize(paths)
            if name=='memory-bandwidth':
                data['scope']='single physical UUID; native device copy, 4 GiB payload, read+write traffic, frozen configuration'
            (q/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
            results.append(dict(group=name,**{k:data[k] for k in ('status','median','unit','cv_percent','repetitions')}))
        except Exception as e:
            results.append(dict(group=name,status='not-qualified',reason=str(e)))
    ab=[]
    for group in ('FP32','FP16','h2d-pinned','comm-MPI','comm-P2P'):
        pairs=[]; identities=set(); excluded=[]
        for i in range(1,4):
            values={}; evidence={}
            for mode in ('off','on'):
                root=OUT/'runs'/(group+'-ab-'+mode+'-'+str(i))
                if not (root/'artifacts/metric-rank-0.json').exists() or not (root/'summary.json').exists(): continue
                try:
                    value,identity,detail=validated_ab(root,mode=='on')
                    identities.add(identity);values[mode]=value;evidence[mode]=detail
                except Exception as e:
                    excluded.append(dict(path=str(root.relative_to(OUT)),reason=repr(e)))
            if len(values)==2: pairs.append(dict(pair=i,**values,evidence=evidence,on_minus_off_percent=(values['on']/values['off']-1)*100))
        ab.append(dict(group=group,pairs=pairs,excluded=excluded,identity_matches=len(identities)==1,
            status='complete' if len(pairs)==3 and len(identities)==1 else 'incomplete',
            median_on_minus_off_percent=statistics.median(p['on_minus_off_percent'] for p in pairs) if len(pairs)==3 else None,
            interpretation='observed paired difference, shared-host noise included; not a causal overhead estimate'))
    (q/'overview.json').write_text(json.dumps(results,indent=2)+'\n')
    (q/'monitor-ab.json').write_text(json.dumps(ab,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    print('AB',[(p['group'],len(p['pairs']),p['median_on_minus_off_percent']) for p in ab])

if __name__=='__main__': main()
