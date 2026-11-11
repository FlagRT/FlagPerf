# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Pair corresponding rank Stores, preserving one logical sample count."""
from html import escape
import json
from pathlib import Path
import tempfile

from .group import validate_manifest, check_ranks, replicated_boundary
from flagprof.store import StoreView
from flagprof.analyzers.workspace import Workspace
from flagprof.analyzers.oracle import identity


def write_index(result,out):
    out=Path(out)
    status='完整' if result['status']=='ok' else '部分'
    with (out/'report.html').open('w') as page:
        page.write('<!doctype html><meta charset="utf-8"><title>TP2 精度对照</title>'
            '<style>body{font:16px sans-serif;max-width:1000px;margin:32px auto;padding:0 20px}'
            'table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:10px}</style>'
            '<h1>TP2 精度对照</h1><p>状态：'+status+'；逻辑输入样本：'+str(result['samples'])+
            '；每侧 rank 数：'+str(result['world_size'])+'。</p><p>参考后端：'+escape(result['ref_backend'])+
            '；比较后端：'+escape(result['cmp_backend'])+'。</p>'
            '<p>复制的模型与 block 边界已核对跨 rank 一致。两侧内部算子按相同 rank 配对，'
            '逻辑输入样本只计一次。打开各 rank 报告查看身份、首差异、逐层及算子结果。</p>'
            '<table><tr><th>配对</th><th>详细报告</th></tr>')
        for row in result['ranks']:
            rank=str(row['rank'])
            page.write('<tr><td>rank '+rank+' → rank '+rank+'</td><td><a href="'+row['report']+'">查看精度对照</a></td></tr>')
        page.write('</table>')
        if any(r['panels'].get('oracle.op_sensitivity')=='partial' for r in result['ranks']):
            page.write('<p>部分算子的输入来源或调用配置不完整；数值配对继续保留，introduced 判断仅使用完整来源。请在对应 rank 面板核对原因。</p>')


def inspect_group(path, *, work=None):
    path = Path(path).resolve()
    group = validate_manifest(path)
    if group['status'] != 'complete':
        raise ValueError('oracle requires complete TP groups')
    states, runs = [], []
    for entry in group['ranks']:
        if entry.get('status') != 'complete' or not entry.get('run'):
            raise ValueError('oracle requires complete rank Stores')
        run = path.parent/entry['run']
        with StoreView(run) as view, tempfile.TemporaryDirectory(dir=work) as tmp:
            ws = Workspace(view,tmp)
            try:
                ws.ingest()
                if ws.partial():raise ValueError('oracle rank Store is partial')
                observed = identity(ws)
                if (observed.get('parallelism',{}).get('world_size') != 2 or
                    observed.get('comm_backend') != group['backend'] or
                    observed.get('rank',entry['rank']) != entry['rank']):
                    raise ValueError('oracle rank topology/backend mismatch')
                observed.pop('device',None);observed.pop('rank',None)
                tensors={};calls={};keys=set();sequences={}
                for _,_,_,event in ws.events(('tensor',)):
                    if event['call_key'] in keys:raise ValueError('duplicate rank tensor call_key')
                    keys.add(event['call_key'])
                    if replicated_boundary(event):
                        tensors[event['call_key']]={k:event.get(k) for k in ('shape','dtype','digest','mask')}
                if not tensors:raise ValueError('oracle rank accuracy tensors missing')
                for _,_,_,event in ws.events(('comm',)):
                    key=event['group'],event['group_seq']
                    if key in calls or event.get('rank') != entry['rank']:
                        raise ValueError('duplicate or misattributed rank collective')
                    if key[1]!=sequences.get(key[0],0):raise ValueError('rank collective sequence missing or out of order')
                    sequences[key[0]]=key[1]+1
                    calls[key]=event
                    role=event.get('ext',{}).get('communication',{}).get('role')
                    if role=='model' and event['lib']!=group['backend']:
                        raise ValueError('actual collective backend mismatch')
                states.append(dict(identity=observed,trajectories={},tensors=tensors,calls=calls))
                runs.append(run)
            finally:ws.close()
    check_ranks(states,group['group_id'])
    return group,runs,states[0]['identity']


def generate(ref,cmp,out,*,declare=None,work=None,limits=None,**options):
    from . import generate_oracle
    if any(options.values()):
        raise ValueError('group oracle uses rank Stores directly; per-rank replay/performance options require Store inputs')
    if work:Path(work).mkdir(parents=True,exist_ok=True)
    a,ar,ai=inspect_group(ref,work=work);b,br,bi=inspect_group(cmp,work=work)
    out=Path(out).resolve()
    for source in (Path(ref).resolve().parent,Path(cmp).resolve().parent):
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError('group oracle output must be outside both deliveries')
    out.mkdir(parents=True,exist_ok=False)
    ranks=[]
    for rank,(left,right) in enumerate(zip(ar,br)):
        result=generate_oracle(left,right,out/('rank-'+str(rank)),declare=declare,work=work,limits=limits)
        ranks.append(dict(rank=rank,report='rank-'+str(rank)+'/report.html',panels=result['panels']))
    status='ok' if all(all(v in ('ok','unavailable') for v in r['panels'].values()) and
                       r['panels'].get('oracle.identity_diff')=='ok' for r in ranks) else 'partial'
    result=dict(schema='oracle-group/1',status=status,samples=len(ai.get('samples',[])),world_size=2,
                ref_backend=a['backend'],cmp_backend=b['backend'],ranks=ranks,
                sample_count_scope='logical input samples, counted once; operator shards compare within the same rank')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    write_index(result,out)
    return dict(output=str(out),status=status,ranks=ranks)
