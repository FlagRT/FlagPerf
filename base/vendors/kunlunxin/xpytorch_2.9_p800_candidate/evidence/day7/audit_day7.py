"""CPU-only regression, complete planning matrix, and read-only evidence audit."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUT = ROOT / 'base/result/p800-day7-20260924'
PROFILE = ROOT / 'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate'
os.chdir(ROOT)
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
def execute(name, command):
    folder = OUT / 'regression'
    folder.mkdir(exist_ok=True)
    p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    (folder / (name + '.stdout.log')).write_text(p.stdout)
    (folder / (name + '.stderr.log')).write_text(p.stderr)
    write(folder / (name + '.command.json'), dict(command=command, exit_code=p.returncode))
    print(name, p.returncode, p.stderr[-150:], flush=True)
    return p
venv = str(ROOT / 'base/result/p800-pr1-20260920-162724/cpu-test-venv/bin/python')
suite = {}
for name, cmd in [
    ('base-final', [venv,'-B','-m','unittest','discover','-s','base/tests','-p','test_*.py','-v']),
    ('pr0', ['python3','-S','-B','-m','unittest','discover','-s',str(PROFILE),'-p','test_*.py','-v']),
    ('toolkit', ['python3','-B','-m','unittest','discover','-s','base/toolkits/_common/ascend/A3/tests','-v']),
    ('day6', [venv,'-B','-m','unittest','base.tests.test_p800_day6','-v']),
    ('bootstrap', ['bash','-n',str(PROFILE/'container_bootstrap.sh')]),
]:
    p=execute(name,cmd); suite[name]=dict(exit_code=p.returncode, tail=p.stderr[-600:])
write(OUT/'regression/suites.json',suite)
cases = sorted(p.parent.parent.parent.name for p in (ROOT/'base/benchmarks').glob('*/kunlunxin/P800/runtime_requirements.json'))
assert len(cases)==15, cases
matrix=[]
for vendor in ('kunlunxin','ascend'):
    for case in cases:
        pair=case in ('interconnect-MPI_intraserver','interconnect-P2P_intraserver')
        cmd=['python3','-B','base/run.py','benchmark','run','--case',case+(':P800' if vendor=='kunlunxin' else ''),
             '--config','base/configs/'+('kunlunxin_p800_xpytorch29.yaml' if vendor=='kunlunxin' else 'ascend910_cann9_local.yaml'),
             '--nproc-per-node','2' if pair else '1','--timeout','300','--allow-high-risk-case','--allow-candidate-runtime','--dry-run']
        cmd += ['--physical-device-ids','5,6' if pair else '5'] if vendor=='kunlunxin' else ['--device-ids','14,15' if pair else '14']
        p=execute('dry-'+vendor+'-'+case,cmd)
        try: plan=json.loads(p.stdout)
        except json.JSONDecodeError: plan={}
        matrix.append(dict(vendor=vendor,case=case,exit_code=p.returncode,applicability=plan.get('applicability'),
                           entrypoint=plan.get('case_assets',{}).get('entrypoint')))
write(OUT/'regression/planning-matrix.json',matrix)
sys.path.insert(0,str(ROOT/'base'))
from generate_benchmark_report import generate_benchmark_report
reports=[]
chosen={}
for day in ('day6','day5','day4','day2'):
    for f in sorted((PROFILE/'evidence'/day).rglob('summary.json')):
        if f.parent.name.endswith('monitor') or not (f.parent/'resolved-plan.json').exists(): continue
        try: s=json.loads(f.read_text())
        except (ValueError,OSError): continue
        status=s.get('status')
        if s.get('kind')=='benchmark' and status in ('passed','partial','failed','skipped') and status not in chosen:
            chosen[status]=f.parent
for status,root in chosen.items():
    before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    import tempfile, shutil
    with tempfile.TemporaryDirectory(prefix='p800-day7-report-') as tmp:
        copy=Path(tmp)/root.name
        shutil.copytree(root,copy)
        original=(copy/'summary.json').read_bytes()
        try:
            generate_benchmark_report(copy)
            first={n:hashlib.sha256((copy/n).read_bytes()).hexdigest() for n in ('report.md','report_monitor.md')}
            generate_benchmark_report(copy)
            second={n:hashlib.sha256((copy/n).read_bytes()).hexdigest() for n in first}
            unchanged=(copy/'summary.json').read_bytes()==original
            reports.append(dict(status=status,source=str(root.relative_to(ROOT)),deterministic=first==second,summary_unchanged=unchanged,hashes=second))
        except Exception as e:
            reports.append(dict(status=status,source=str(root.relative_to(ROOT)),error=repr(e)))
    after={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    assert before==after, 'historical source changed'
write(OUT/'regression/report-regeneration.json',reports)
print('REPORTS',json.dumps(reports),flush=True)
print('AUDIT-DONE',flush=True)
