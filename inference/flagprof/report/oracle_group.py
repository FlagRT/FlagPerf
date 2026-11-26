# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Pair corresponding rank Stores, preserving one logical sample count."""
import json
from pathlib import Path
import tempfile

from .group import validate_manifest, check_ranks, replicated_boundary
from flagprof.store import StoreView
from flagprof.analyzers.workspace import Workspace
from flagprof.analyzers.oracle import identity


def inspect_group(path):
    path = Path(path).resolve()
    group = validate_manifest(path)
    if group['status'] != 'complete':
        raise ValueError('oracle requires complete TP groups')
    states, runs = [], []
    for entry in group['ranks']:
        if entry.get('status') != 'complete' or not entry.get('run'):
            raise ValueError('oracle requires complete rank Stores')
        run = path.parent/entry['run']
        with StoreView(run) as view, tempfile.TemporaryDirectory() as tmp:
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


def generate(ref,cmp,out,*,declare=None,limits=None,**options):
    from . import generate_oracle
    def rank_inputs(value):
        if not value:return {}
        path=Path(value).resolve()
        if not path.is_file():raise ValueError('group linked input requires group.json or a rank manifest')
        data=json.loads(path.read_text())
        if data.get('schema')=='flagprof-group/1':
            group=validate_manifest(path)
            mapping={str(r['rank']):r.get('run') for r in group['ranks']}
        else:mapping=data.get('ranks',{})
        if set(mapping)!={'0','1'} or any(not isinstance(v,str) for v in mapping.values()):
            raise ValueError('linked rank manifest requires exactly ranks 0 and 1')
        return {int(k):(path.parent/v).resolve() for k,v in mapping.items()}
    mapped={k:rank_inputs(v) for k,v in options.items()}
    a,ar,ai=inspect_group(ref);b,br,bi=inspect_group(cmp)
    out=Path(out).resolve()
    for source in (Path(ref).resolve().parent,Path(cmp).resolve().parent):
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError('group oracle output must be outside both deliveries')
    from flagprof.offline.worker import new_output
    new_output(out)
    ranks=[]
    for rank,(left,right) in enumerate(zip(ar,br)):
        result=generate_oracle(left,right,out/('rank-'+str(rank)),declare=declare,limits=limits,
            **{k:v.get(rank) for k,v in mapped.items()})
        ranks.append(dict(rank=rank, report='rank-'+str(rank)+'/report.html', panels=result['panels'],
                          comparison_status=result['comparison_status'], blocking_fields=result['blocking_fields'],
                          paired_tensors=result['paired_tensors'], paired_boundaries=result['paired_boundaries']))
    status='ok' if all(all(v in ('ok','unavailable') for v in r['panels'].values()) and
                       r['panels'].get('oracle.identity_diff')=='ok' for r in ranks) else 'partial'
    result=dict(schema='oracle-group/1',declaration=declare or {},status=status,samples=len(ai.get('samples',[])),world_size=2,
                ref_backend=a['backend'],cmp_backend=b['backend'],ranks=ranks,
                sample_count_scope='logical input samples, counted once; operator shards compare within the same rank')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    from .product import write_report
    write_report(out,result,limits=limits)
    return dict(output=str(out),status=status,ranks=ranks)
