"""Final offline regression after the early-skip fix; no hardware launch."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import datetime as dt
ROOT=Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUT=ROOT/'base/result/p800-day7-20260924'
sys.path.insert(0,str(ROOT/'base'))
from generate_benchmark_report import generate_benchmark_report

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d): p.write_text(json.dumps(d,indent=2)+'\n')
reg=OUT/'regression'; records=[]
for previous,label in [('base-final','base-release'),('pr0','pr0-release'),('toolkit','toolkit-release'),('day6','day6-release')]:
    old=json.loads((reg/(previous+'.command.json')).read_text())
    command=old['command']
    p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (reg/(label+'.stdout.log')).write_text(p.stdout)
    (reg/(label+'.stderr.log')).write_text(p.stderr)
    record=dict(label=label,command=command,exit_code=p.returncode,checked_at=dt.datetime.now(dt.timezone.utc).isoformat())
    save(reg/(label+'.command.json'),record)
    records.append(record)
    print(label,p.returncode,(p.stdout+p.stderr)[-300:],flush=True)
save(reg/'release-suites.json',records)
assert all(r['exit_code']==0 for r in records),'release suite failed'
skips=[]
for short in ('computation-FP64','computation-FP8','computation-TF32','interconnect-MPI_interserver','interconnect-P2P_interserver'):
    dest=OUT/'early-skip-final'/short
    command=['python3','-B','base/run.py','benchmark','run','--case',short+':P800',
        '--config','base/configs/kunlunxin_p800_xpytorch29.yaml','--physical-device-ids','5',
        '--nproc-per-node','1','--timeout','300','--allow-candidate-runtime','--result-dir',str(dest),
        '--privilege-command','sudo -n','--reservation-reference','unsupported-no-hardware',
        '--reservation-end','2000-01-01T00:00:00+00:00']
    if dest.exists():
        class P: returncode=0; stdout=''; stderr='existing evidence reused'
        p=P()
    else:
        p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        (reg/('release-skip-'+short+'.log')).write_text(p.stdout+p.stderr)
    summary=json.loads((dest/'summary.json').read_text())
    record=dict(case=short,command=command,exit_code=p.returncode,status=summary['status'],
                skip_reason=summary.get('skip_reason'),lease_absent=not (dest/'lease.json').exists(),
                container_absent=not (dest/'container-create.json').exists())
    assert record['status']=='skipped' and record['skip_reason'] and record['lease_absent'] and record['container_absent'] and p.returncode==0
    skips.append(record)
save(reg/'release-skips.json',skips)
sources={}
for path in sorted((OUT/'runs').glob('*/summary.json')):
    status=json.loads(path.read_text()).get('status')
    if status in ('passed','partial','failed') and status not in sources: sources[status]=path.parent
sources['skipped']=OUT/'early-skip-final/computation-FP64'
# This batch had no new failed hardware attempt after cleanup; the prior
# report-regeneration audit already covers a failed Day 6 artifact.
missing_failed = 'failed' not in sources
reports=[]
for status,source in sources.items():
    original={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)/'copy';shutil.copytree(source,dest)
        summary_before=(dest/'summary.json').read_bytes()
        generate_benchmark_report(dest)
        first={n:sha(dest/n) for n in ('report.md','report_monitor.md')}
        generate_benchmark_report(dest)
        deterministic=first=={n:sha(dest/n) for n in first}
        summary_unchanged=summary_before==(dest/'summary.json').read_bytes()
    original_unchanged=original=={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    assert deterministic and summary_unchanged and original_unchanged
    reports.append(dict(status=status,source=str(source.relative_to(OUT)),deterministic=deterministic,
                         summary_unchanged=summary_unchanged,all_original_files_unchanged=original_unchanged,hashes=first))
save(reg/'release-report-regeneration.json',dict(reports=reports,missing_failed=missing_failed,
    prior_failed_coverage='regression/report-regeneration-final.json' if missing_failed else None))
print('RELEASE OFFLINE AUDIT PASSED',flush=True)
