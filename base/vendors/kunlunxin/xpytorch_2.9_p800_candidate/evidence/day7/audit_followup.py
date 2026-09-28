from pathlib import Path
import hashlib, json, subprocess, sys, tempfile, shutil
ROOT=Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUT=ROOT/'base/result/p800-day7-20260924'
sys.path.insert(0,str(ROOT/'base'))
from generate_benchmark_report import generate_benchmark_report
matrix=json.loads((OUT/'regression/planning-matrix.json').read_text())
for item in matrix:
    if item['vendor']=='ascend' and item['case'] in ('interconnect-MPI_intraserver','interconnect-P2P_intraserver'):
        command=['python3','-B','base/run.py','benchmark','run','--case',item['case'],
           '--config','base/configs/ascend910_cann9_p2p_candidate.yaml','--device-ids','14,15',
           '--nproc-per-node','2','--timeout','300','--allow-candidate-runtime','--dry-run']
        if item['case']=='interconnect-P2P_intraserver':
            command += ['--case-config','base/benchmarks/interconnect-P2P_intraserver/ascend/case_config.smoke.yaml']
        p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        stem=OUT/'regression'/('dry-ascend-communication-'+item['case'])
        stem.with_suffix('.stdout.log').write_text(p.stdout)
        stem.with_suffix('.stderr.log').write_text(p.stderr)
        stem.with_suffix('.command.json').write_text(json.dumps(dict(command=command,exit_code=p.returncode),indent=2))
        plan=json.loads(p.stdout) if not p.returncode else {}
        item.update(exit_code=p.returncode,applicability=plan.get('applicability'),runtime_profile='torch_fl_2.10_flagcx',
                    previous_attempt='operator profile correctly rejected; original outputs retained')
(OUT/'regression/planning-matrix-final.json').write_text(json.dumps(matrix,indent=2))
print('MATRIX',[(v,sum(x['exit_code']==0 for x in matrix if x['vendor']==v)) for v in ('ascend','kunlunxin')])
skip_parent=OUT/'early-skip-fp64-a02'
command=['python3','-B','base/run.py','benchmark','run','--case','computation-FP64:P800',
    '--config','base/configs/kunlunxin_p800_xpytorch29.yaml','--physical-device-ids','5',
    '--nproc-per-node','1','--timeout','300','--allow-candidate-runtime','--result-root',str(skip_parent)]
if not skip_parent.exists():
    p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (OUT/'regression/early-skip-a02.log').write_text(p.stdout+p.stderr)
    (OUT/'regression/early-skip-a02-command.json').write_text(json.dumps(dict(command=command,exit_code=p.returncode),indent=2))
skip=next(skip_parent.glob('benchmark-*'))
report=json.loads((OUT/'regression/report-regeneration.json').read_text())
assert json.loads((skip/'summary.json').read_text())['status']=='skipped'
assert not (skip/'container-create.json').exists() and not (skip/'lease.json').exists()
with tempfile.TemporaryDirectory() as t:
    copy=Path(t)/'skip'; shutil.copytree(skip,copy)
    before=(copy/'summary.json').read_bytes()
    generate_benchmark_report(copy)
    hashes={n:hashlib.sha256((copy/n).read_bytes()).hexdigest() for n in ('report.md','report_monitor.md')}
    generate_benchmark_report(copy)
    report.append(dict(status='skipped',source=str(skip.relative_to(ROOT)),summary_unchanged=before==(copy/'summary.json').read_bytes(),
       deterministic=hashes=={n:hashlib.sha256((copy/n).read_bytes()).hexdigest() for n in hashes},hashes=hashes))
(OUT/'regression/report-regeneration-final.json').write_text(json.dumps(report,indent=2))
gate_tests=[]
base=['python3','-B','base/run.py','benchmark','run','--config','base/configs/kunlunxin_p800_xpytorch29.yaml',
      '--case','main_memory-capacity:P800','--physical-device-ids','5','--timeout','300','--allow-candidate-runtime']
for name,extra in [('capacity-no-opt-in',[]),('eight-rank-rejected',['--case','interconnect-MPI_intraserver:P800','--physical-device-ids','0,1,2,3,4,5,6,7','--nproc-per-node','8','--dry-run'])]:
    cmd=base+extra+['--result-dir',str(OUT/'runs'/name)]
    p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    gate_tests.append(dict(name=name,command=cmd,exit_code=p.returncode,output=p.stdout+p.stderr))
(OUT/'regression/rejection-gates.json').write_text(json.dumps(gate_tests,indent=2))
paths=['base/vendors/kunlunxin','base/executors/benchmark.py','base/executors/bounded_benchmark.py','base/benchmark_worker.py',
       'base/benchmarks/drivers/kunlunxin.py','base/benchmarks/drivers/day6.py','base/benchmarks/drivers/utils.py']
p=subprocess.run(['rg','-n',r'\.cuda\(|\.to\(local_rank|/dev/davinci|ASCEND_RT_VISIBLE_DEVICES|HCCL|fallback|expected_ranks|nproc_per_node','--glob','*.py',
                  '--glob','!**/evidence/**','--glob','!**/xpytorch_2.9_p800_candidate/test*',*paths],cwd=ROOT,capture_output=True,text=True)
(OUT/'regression/static-routing-scan.log').write_text(p.stdout+p.stderr)
print('REPORTS',[(r['status'],r.get('deterministic'),r.get('summary_unchanged')) for r in report])
print('REJECTION',[(r['name'],r['exit_code']) for r in gate_tests])
