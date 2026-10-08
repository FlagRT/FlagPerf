import collections,hashlib,json,pathlib,re,shutil,subprocess,sys,tempfile
R=pathlib.Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf');O=R/'base/result/p800-base-doc-audit-20261008'
sys.path[:0]=[str(R/'base'),str(R)]
from generate_benchmark_report import generate_benchmark_report
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x): (O/n).write_text(json.dumps(x,indent=2)+'\n')
indices=[]
for p in (R/'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence').rglob('sha256-index.json'):
    if 'run-records' in p.parts: continue  # metadata subsets retain full-source indices; see Toolkit README
    obj=json.loads(p.read_text());entries=obj.get('files',obj)
    if isinstance(entries,dict): entries=[dict(path=k,sha256=v) for k,v in entries.items()]
    bad=[]
    for e in entries:
        f=p.parent/e['path'];actual=sha(f) if f.is_file() else None
        if actual!=e['sha256']:bad.append(dict(path=e['path'],expected=e['sha256'],actual=actual))
    indices.append(dict(index=str(p.relative_to(R)),checked=len(entries),mismatches=bad))
save('archive-index-audit.json',indices)
print('INDICES',len(indices),'ENTRIES',sum(x['checked'] for x in indices),'MISMATCHES',sum(len(x['mismatches']) for x in indices),flush=True)
report_rows=[]
tmp=pathlib.Path(tempfile.mkdtemp(prefix='p800-doc-report-'))
for root in ['p800-base-final-20261008','p800-idle-accept-20261008']:
 for source in sorted((R/'base/result'/root/'base-runs').iterdir()):
    if not (source/'summary.json').exists():continue
    copy=tmp/root/source.name;shutil.copytree(source,copy)
    before_summary=sha(copy/'summary.json')
    def reports(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and (f.suffix=='.svg' or f.name in ['report.md','report_monitor.md'])}
    try:
        old=reports(copy);generate_benchmark_report(copy);first=reports(copy);generate_benchmark_report(copy);second=reports(copy)
        report_rows.append(dict(run=root+'/'+source.name,deterministic=first==second,summary_unchanged=sha(copy/'summary.json')==before_summary,stored_report_matches=old==first,drift=[k for k in set(old)|set(first) if old.get(k)!=first.get(k)]))
    except Exception as e:report_rows.append(dict(run=root+'/'+source.name,error=str(e)))
save('report-verification.json',report_rows)
source_hashes=json.loads((O/'source-sha256.json').read_text());changed=[p for p,h in source_hashes.items() if sha(R/p)!=h]
save('source-unchanged.json',dict(checked=len(source_hashes),changed=changed))
print('REPORTS',len(report_rows),'NONDETERMINISTIC',sum(not x.get('deterministic') for x in report_rows),'STORED_DRIFT',sum(not x.get('stored_report_matches') for x in report_rows),'SOURCE_CHANGED',changed,flush=True)
matrix=[]
for d in sorted((R/'base/benchmarks').iterdir()):
 if not (d/'kunlunxin/P800/runtime_requirements.json').exists():continue
 case=d.name;comm=case in ['interconnect-MPI_intraserver','interconnect-P2P_intraserver']
 args=['python3','-B','base/run.py','benchmark','run','--case',case+':P800','--config','base/configs/kunlunxin_p800_xpytorch29.yaml','--physical-device-ids','0,5' if comm else '5','--nproc-per-node','2' if comm else '1','--timeout','300','--allow-candidate-runtime','--dry-run']
 if case=='main_memory-capacity':args+=['--allow-high-risk-case']
 p=subprocess.run(args,cwd=R,text=True,capture_output=True,timeout=30)
 try:plan=json.loads(p.stdout)
 except Exception:plan={}
 matrix.append(dict(case=case,returncode=p.returncode,applicability=plan.get('applicability'),error=p.stderr))
save('dry-run-matrix.json',matrix)
print('MATRIX',json.dumps(matrix),flush=True)
