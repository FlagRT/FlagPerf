# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Exact per-side accuracy/performance links; layouts stay in the frozen sid."""
import json
from pathlib import Path
from .workspace import Workspace
from .oracle import identity
from .core import perf_op
from .device import device
from flagprof.store import StoreView

FIELDS=('model_files','oracle_id','inputs_sha256','samples','batches','dtype','attention','pooling',
        'device','software','numerics','flaggems','include')


def link(ref,cmp,runs,directory,limits):
    totals=dict(requested=any(runs.values()),joined=0,unmatched=0,route_agree=0,route_mismatch=0,route_unknown=0,sides={})
    for side,accuracy in (('ref',ref),('cmp',cmp)):
        if not runs.get(side):continue
        with StoreView(runs[side]) as view:
            perf=Workspace(view,Path(directory)/('perf-'+side),limits)
            try:
                perf.ingest();perf_op(perf,{});device(perf,{})
                wanted,actual=identity(accuracy),identity(perf)
                mismatches=[dict(field=k,accuracy=wanted.get(k),performance=actual.get(k)) for k in FIELDS
                    if k not in wanted or k not in actual or wanted[k]!=actual[k]]
                totals['sides'][side]=dict(run=str(view.run),identity_mismatches=mismatches,joined=0,unmatched=0)
                perf.db.execute("create index perf_sid on rows(json_extract(payload,'$.sid')) where panel='perf.op'")
                for row in ref.table('oracle.op_sensitivity'):
                    sid=row.get('sid_'+side)
                    reasons=['identity_mismatch'] if mismatches else []
                    if perf.partial():reasons.append('performance_read_partial')
                    matches=[] if reasons else [json.loads(p[0]) for p in perf.db.execute(
                        "select payload from rows where panel='perf.op' and json_extract(payload,'$.sid')=? order by key",(sid,))]
                    if not matches and not reasons:reasons.append('exact_signature_missing')
                    decisions=sorted({p.get('route_decision') or 'unknown' for p in matches})
                    rule=row['expected_route'] if side=='cmp' else {}
                    expected='fallback' if rule.get('status')=='not_registered' else 'excluded' if rule.get('device_unsupported') or rule.get('policy_excluded') else 'hit' if rule.get('registered') else None
                    agreement=None if not matches or expected is None or 'unknown' in decisions else decisions==[expected]
                    totals['joined' if matches else 'unmatched']+=1
                    totals['sides'][side]['joined' if matches else 'unmatched']+=1
                    totals['route_unknown' if agreement is None else 'route_agree' if agreement else 'route_mismatch']+=1
                    ref.row('oracle.perf_links',side+':'+row['call_key'],dict(side=side,call_key=row['call_key'],sid=sid,
                        matched_rows=len(matches),groups=matches,route_decisions=decisions,expected_decision=expected,
                        route_agrees=agreement,reasons=reasons,identity_mismatches=mismatches))
            finally:perf.close()
    return totals
