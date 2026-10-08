"""Read-only Base evidence re-audit; outputs only to a new audit directory."""
import collections, hashlib, json, pathlib, re, statistics, subprocess, sys
R = pathlib.Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
O = R / 'base/result/p800-base-doc-audit-20261008'
O.mkdir(exist_ok=True)
sys.path[:0] = [str(R / 'base'), str(R)]
from qualification import summarize
from executors.bounded_benchmark import validate_metric
def save(name, obj):
    (O/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
rows, groups, errors, hashes = [], collections.defaultdict(list), [], {}
for label in ['p800-base-final-20261008', 'p800-idle-accept-20261008']:
    for p in sorted((R/'base/result'/label/'base-runs').iterdir()):
        if not (p/'summary.json').exists(): continue
        s=json.loads((p/'summary.json').read_text()); b=json.loads((p/'benchmark-result.json').read_text())
        ctxraw=(p/'control/host-context.json').read_bytes(); ctx=json.loads(ctxraw)
        cfg=json.loads((p/'control/case-assets.json').read_text())['merged_config']
        metrics=b.get('metrics', [])
        for m in metrics:
            try:
                rank=m['rank']; raw=json.loads((p/f'artifacts/metric-rank-{rank}.json').read_text())
                assert m==raw
                c=json.loads((p/f'artifacts/correctness-rank-{rank}.json').read_text())
                validate_metric(m,c,ctx,hashlib.sha256(ctxraw).hexdigest(),s['device_bindings'][rank],cfg)
            except Exception as e: errors.append({'run':label+'/'+p.name,'error':str(e)})
        row={'root':label,'run':p.name,'case':s.get('case'),'status':s.get('status'),
             'layers':{k:s.get(k) for k in ['execution_status','measurement_status','correctness_status','measurement_evidence_status','monitoring_status','cleanup_status','postflight_status','lease_released']},
             'health_degraded':any(any(d.get('uncorrectable_ecc_counts',[])) for d in ctx['host']['devices']),
             'mode':cfg.get('MODE'),'ranks':len(metrics),'minimum_rank_value':min(m['value'] for m in metrics),
             'unit':metrics[0]['unit'],'minimum_window_s':min(m['elapsed_seconds'] for m in metrics),
             'iterations':cfg.get('ITERS'),'warmup':cfg.get('WARMUP')}
        rows.append(row)
        match=re.match(r'(.+)-q[1-5]$',p.name)
        if match: groups[(label,match[1])].append(p)
        for f in p.rglob('*'):
            if f.is_file(): hashes[str(f.relative_to(R))]=sha(f)
save('runs.json',rows);save('source-sha256.json',hashes)
group_rows=[]
for (label,name),paths in sorted(groups.items()):
    values=[next(x['minimum_rank_value'] for x in rows if x['root']==label and x['run']==p.name) for p in paths]
    g={'root':label,'group':name,'runs':[p.name for p in paths],'n':len(paths),'median':statistics.median(values),
       'cv_percent':statistics.stdev(values)/statistics.mean(values)*100 if len(values)>1 else None}
    try:
        q=summarize(paths);g['strict_status']=q['status'];g['qualification_status']=q['qualification_status']
    except Exception as e:
        g['strict_status']='rejected';g['reason']=str(e)
    group_rows.append(g)
save('groups.json',group_rows)
evidence=R/'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence'
index_rows=[]
for p in sorted(evidence.rglob('sha256-index.json')):
    obj=json.loads(p.read_text());bad=[];checked=0
    if isinstance(obj,dict) and all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in obj.values()):
        for rel,digest in obj.items():
            f=p.parent/rel
            if not f.is_file() or sha(f)!=digest: bad.append(rel)
            checked+=1
        index_rows.append({'index':str(p.relative_to(R)),'checked':checked,'mismatches':bad})
    else:index_rows.append({'index':str(p.relative_to(R)),'schema':'not-flat-digest-map','keys':list(obj)[:8] if isinstance(obj,dict) else type(obj).__name__})
save('archive-index-audit.json',index_rows)
report={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        'run_count':len(rows),'measurement_counts':dict(collections.Counter(x['layers']['measurement_status'] for x in rows)),
        'run_status_counts':dict(collections.Counter(x['status'] for x in rows)),
        'health_degraded_count':sum(x['health_degraded'] for x in rows),'raw_metric_errors':errors,
        'group_status_counts':dict(collections.Counter(x['strict_status'] for x in group_rows)),
        'source_file_count':len(hashes),'archive_indices':len(index_rows),
        'archive_mismatches':sum(len(x.get('mismatches',[])) for x in index_rows)}
save('audit-summary.json',report);print(json.dumps(report,indent=2))
for g in group_rows: print(g['root'],g['group'],g['n'],g['strict_status'],g.get('reason',''),flush=True)
