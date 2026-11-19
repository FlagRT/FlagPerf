# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Three independent pair reports with exact call-key/sample joins."""
import json
from pathlib import Path
from .comparison import change
from .model import opened_model,read_document


def generate(ref,baseline,cmp,out=None,*,declare=None,limits=None,**options):
    from . import generate_oracle,write_json
    from .product import write_report
    out=Path(out).resolve() if out else Path(ref).resolve().parent/(Path(ref).name+'-three-way')
    for source in (ref,baseline,cmp):
        path=Path(source).resolve();path=path.parent if path.is_file() else path
        if out==path or path in out.parents or out in path.parents:raise ValueError('three-way output must be outside inputs')
    from flagprof.offline.worker import new_output
    new_output(out)
    pairs=(('reference-baseline',ref,baseline,{}),('reference-candidate',ref,cmp,options),('baseline-candidate',baseline,cmp,{}))
    for label,a,b,linked in pairs:generate_oracle(a,b,out/label,declare=declare,limits=limits,**linked)
    with opened_model(limits=limits) as model:
        db=model.db;db.execute('create table pair_samples(pair text,rank integer,key text,payload text,primary key(pair,rank,key))')
        statuses=[]
        for label,_,_,_ in pairs:
            roots=sorted((out/label).glob('rank-*')) or [out/label]
            for root in roots:
                rank=int(root.name.split('-')[1]) if root.name.startswith('rank-') else -1
                document=read_document(root/'result.json',model)
                identity=document.get('results',{}).get('oracle.identity_diff',{})
                statuses.append(dict(pair=label,rank=rank,status=identity.get('status')))
                path=root/'exports/oracle.samples.jsonl'
                if path.is_file():
                    with path.open() as stream:
                        for line in stream:
                            if len(line.encode())>model.limits.record_bytes:raise ValueError('three-way sample exceeds record budget')
                            row=json.loads(line)
                            db.execute('insert into pair_samples values(?,?,?,?)',(label,rank,row['call_key'],line))
        db.commit()
        for rank,key in db.execute('select distinct rank,key from pair_samples order by rank,key'):
            values={p:json.loads(r) for p,r in db.execute('select pair,payload from pair_samples where rank=? and key=?',(rank,key))}
            first=next(iter(values.values()))
            for metric in ('max_abs','mae','mse','relative_l2','cosine_similarity','normalized_rmse'):
                a=values.get('reference-baseline',{});b=values.get('reference-candidate',{});c=values.get('baseline-candidate',{})
                same=bool(a and b and c and a.get('level')==b.get('level')==c.get('level') and a.get('valid_elements')==b.get('valid_elements')==c.get('valid_elements') and
                    all(r.get('valid_elements')==r.get('total_elements') for r in (a,b,c)))
                delta=change(a.get(metric),b.get(metric),same)
                model.add('three_way',dict(rank=rank if rank>=0 else None,call_key=key,scope=first['scope'],boundary=first['boundary'],
                    sample=first['sample'],metric=metric,level=a.get('level'),reference_to_baseline=a.get(metric),
                    reference_to_candidate=b.get(metric),baseline_to_candidate=c.get(metric),delta=delta['delta'],
                    relative_change=delta['relative_change'],reason=delta['reason'] if same else 'missing_pair_or_different_evidence_population'))
        result=dict(schema='oracle-three-way/1',declaration=declare or {},comparisons=model.rows('three_way'),pair_statuses=statuses,
                    reports={label:label+'/report.html' for label,_,_,_ in pairs},
                    note='同一参考下逐样本比较；余弦越接近 1 越相似，其余误差指标越小越接近。数值变化不自动证明组件因果。')
        with (out/'result.json').open('w') as stream:write_json(stream,result);stream.write('\n')
        write_report(out,result,limits=limits)
    return dict(output=str(out),schema='oracle-three-way/1',pairs=statuses)
