"""Build a scoped Day 7 review from immutable run outcomes, after all jobs."""
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUT=ROOT/'base/result/p800-day7-20260924'
sys.path.insert(0,str(OUT))
from collect_day7 import validated_ab

def read(p): return json.loads(p.read_text())
def save(p,d): p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
q=read(OUT/'qualification/overview.json'); ab=read(OUT/'qualification/monitor-ab.json')
assert all(x['status']!='pending' for x in q)
resource=read(OUT/'final-resource-check.json')
assert resource['all_started_runs_terminal']
suites=read(OUT/'regression/release-suites.json')
assert all(x['exit_code']==0 for x in suites)
for suite in suites:
 text=(OUT/'regression'/(suite['label']+'.stdout.log')).read_text()+(OUT/'regression'/(suite['label']+'.stderr.log')).read_text()
 counts=re.findall(r'Ran (\d+) tests?',text);suite['tests']=int(counts[-1]) if counts else None
states=[]
for p in sorted((OUT/'runs').glob('*/summary.json')):
 s=read(p);assert s['status']!='running'
 states.append(dict(name=p.parent.name,**{k:s.get(k) for k in ('status','failure_stage','error','correctness_status','measurement_status','monitoring_status','cleanup_status','postflight_status','lease_released')}))
curves=[]
for short in ('MPI','P2P'):
 for m in (1,4,16,64):
  root=OUT/'runs'/('comm-'+short+'-m'+str(m)+'-q1')
  record=dict(case=short,message_mib=m*4,run=str(root.relative_to(OUT)))
  try:
   value,identity,detail=validated_ab(root,True)
   metrics=read(root/'benchmark-result.json')['metrics']
   record.update(status='passed',gb_s=value,rank_values=detail['rank_values'],elapsed_seconds=[x['elapsed_seconds'] for x in metrics])
  except Exception as e:record.update(status='not-qualified',reason=repr(e))
  curves.append(record)
save(OUT/'qualification/communication-curves.json',curves)
capacity_root=OUT/'runs/capacity'
capacity=dict(summary=read(capacity_root/'summary.json'),metric=read(capacity_root/'artifacts/metric-rank-0.json'),
              correctness=read(capacity_root/'artifacts/correctness-rank-0.json'))
timeouts=[]
for short in ('MPI','P2P'):
 path=OUT/'runs'/('comm-'+short+'-timeout')/'summary.json'
 if path.exists():
  s=read(path);timeouts.append(dict(case=short,**{k:s.get(k) for k in ('status','error','failure_stage','cleanup_status','postflight_status','lease_released')}))
historical=read(OUT/'historical-evidence-audit.json')
history=ROOT/'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day6'
index=read(history/'sha256-index.json')
bad=[name for name,value in index.items() if sha(history/name)!=value]
assert not bad and len(index)==historical['index_entries']
assert all(sha(ROOT/x['path'])==x['summary_sha256'] for x in historical['formal_two_card_references'])
end=dict(index_entries=len(index),index_mismatches=bad,formal_summaries_unchanged=True,checked_at=dt.datetime.now(dt.timezone.utc).isoformat())
save(OUT/'historical-end-check.json',end)
source=read(OUT/'measurement-source-index.json')
assert all(sha(OUT/'measurement-source'/p)==h for p,h in source['sha256'].items())
code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
final=dict(schema_version=1,date='2026-09-24',timezone='Asia/Shanghai',status='scoped-delivery; full-week acceptance incomplete',
 measurement_code_commit=source['code_commit'],release_code_commit=code_commit,
 qualification=q,monitor_ab=ab,communication_curves=curves,timeouts=timeouts,
 capacity={k:capacity['summary'].get(k) for k in ('status','measurement_status','monitoring_status','correctness_status','cleanup_status','lease_released')},
 capacity_metric=capacity['metric'],offline_suites=suites,run_status_counts=dict(Counter(x['status'] for x in states)),
 eight_card=dict(status='not-implemented-and-not-qualified',reasons=['cards 1/2 health exclusions','no healthy idle whole-host window','provider, runtime requirements, contract and reference remain exactly two ranks']),
 candidate_validated=False,resource_check='final-resource-check.json',historical_unchanged=end)
save(OUT/'acceptance-summary.json',final)
save(OUT/'run-inventory.json',states)
lines=['# P800 Day 7 review — 2026-09-24','',
'## Verdict','',
'**Scoped delivery, not full-week or eight-card acceptance.** Runtime remains candidate / validated:false. Per-case outcome is listed below; unstable groups are retained as unstable. CPU and static Ascend coverage is not Ascend hardware acceptance.','',
'## Scope and provenance','',
'- User-authorized Day 7 execution uses healthy idle card 5 for single-card work and a freshly preflighted 5+6 pair for two-rank work. No foreign-handle override, tenant termination, driver/firmware change or eight-card run.',
'- Historical Day 6 formal 4+7 and 3+4 runs were already completed by the user. Their 5811 indexed files and 16 curve summaries were rechecked without modification. Day 7 results do not replace those records.',
'- Measurement code: '+source['code_commit']+'. Release code: '+code_commit+'. Source/config/image/UUID identity is recorded per run; measurement-source/ retains the frozen measurement snapshot.',
'- Eight-card work remains blocked by health/resources and the existing exactly-two-rank implementation. Both implementation and hardware qualification are still required.',
'- Authorization and time budget: authorization.json. Raw files: base/result/p800-day7-20260924/. Portable evidence is a labelled sanitized projection, with original and archive SHA256 in provenance.json.','',
'## Fixed defects','',
'1. synchronize() and memory_info() used rank 0 instead of the current verified rank. c1d900d8 tracks the active rank after set_device succeeds. Reordered logical-device mappings are covered by regression. Fresh communication runs use the corrected code.',
'2. Unsupported cases with bounded CLI arguments and --result-dir incorrectly failed before producing skip evidence. The release fix routes supported bounded providers to the existing skip renderer before Docker, lease or host inspection. Five real skip cases and three regression methods cover reasons, no external calls, immutable existing output, and conflicting output paths.',
'3. A stale driver test fixture retained _bindings across cases; fixtures now isolate that state. System Python without torch is not used as the full regression environment.',
'', '## Five-run qualification','',
'Computation uses 8192 cubed. Transfers use 512 MiB and mode-specific frozen iterations. Device memory bandwidth uses 4 GiB and read+write traffic. Every admitted sample has correctness, binding, synchronized timing >=15 seconds, monitoring and successful cleanup. CV is sample standard deviation / mean, ddof=1, limit 5%. Two-rank repetitions use the lower validated rank bandwidth, never a sum.','',
'| Group | Outcome | Median | Unit | CV % | Runs |','|---|---|---:|---|---:|---:|']
for x in q:lines.append('| '+x['group']+' | '+x['status']+' | '+(format(x['median'],'.6f') if 'median' in x else '—')+' | '+x.get('unit','—')+' | '+(format(x['cv_percent'],'.4f') if 'cv_percent' in x else '—')+' | '+str(x.get('repetitions',x.get('completed_runs','—')))+' |')
lines += ['', 'No slow samples were removed and no configuration/UUID groups were pooled. A run can pass its correctness/lifecycle checks while the five-run group is unstable. See qualification/*.json for min/max/mean/std and every run path.', '', '## Two-card curve and timeout regression','',
'| Case | MiB | Outcome | Lower-rank GB/s | Per-rank windows (s) |','|---|---:|---|---:|---|']
for x in curves:lines.append('| '+x['case']+' | '+str(x['message_mib'])+' | '+x['status']+' | '+(format(x['gb_s'],'.6f') if 'gb_s' in x else '—')+' | '+', '.join(format(t,'.3f') for t in x.get('elapsed_seconds',[]))+' |')
lines += ['', 'At world size 2, AllReduce busbw equals algbw (2*(N-1)/N = 1). P2P is one-way rank 0 to rank 1. The first three curve sizes are one qualified run each, not five-run stability claims; the 256 MiB point has a separate five-run group.',
'The XPULink bandwidth discrepancy remains open. New card-pair results do not establish a causal XPULink-vs-PCIe comparison. Ring counts alone do not prove an interconnect maximum.', '', 'Timeout evidence:', '']
for t in timeouts:lines.append('- '+t['case']+': '+json.dumps(t,ensure_ascii=False))
lines += ['', '## Capacity and monitoring','',
'Capacity search is not a bandwidth qualification. Full bounded search, held allocations, OOM classification and release evidence are retained. Search duration can produce partial monitoring; that state is not rewritten as passed.',
'', 'Capacity outcome: '+json.dumps(final['capacity'])+'.',
'Held allocation: '+str(capacity['metric'].get('held_mib'))+' MiB; search window '+str(capacity['metric'].get('elapsed_seconds'))+' s. See runs/capacity/artifacts/.', '',
'Alternating monitor off/on: three pairs per configured case; positive numbers mean monitored measurement was faster. These are observed differences including host variation, not causal overhead estimates.', '',
'| Group | Complete pairs | Same identity | Median (on/off - 1), % | Outcome |','|---|---:|---|---:|---|']
for x in ab:lines.append('| '+x['group']+' | '+str(len(x['pairs']))+' | '+str(x['identity_matches'])+' | '+(format(x['median_on_minus_off_percent'],'.4f') if x['median_on_minus_off_percent'] is not None else '—')+' | '+x['status']+' |')
lines += ['', '## Offline, reports and rejection gates','']
for s in suites:lines.append('- '+s['label']+': '+str(s['tests'])+' tests, exit '+str(s['exit_code'])+'.')
lines += ['- P800 15/15 dry-runs and Ascend 15/15 dry-runs complete: ten applicable, five skipped each. Ascend two-rank profile and explicit P2P smoke config are required.',
'- Public distribution remains 6/7; the existing Ascend private host-path check fails. Retained in regression/public-distribution.log; no unrelated Ascend configuration is changed.',
'- Passed/partial/failed/skipped reports were regenerated twice in temporary copies. Output hashes deterministic; original files and summary bytes unchanged. See release-report-regeneration.json.',
'- All five unsupported P800 cases accept --result-dir plus bounded arguments and expired reservation, then skip before any external execution; see release-skips.json. Capacity without opt-in and eight ranks are rejected before hardware.',
'- The initial overlong standalone preflight timeout, wrong Ascend planning profile and pre-fix early-skip error remain in their original logs. They are not hardware failures.', '',
'## Cleanup and delivery boundary','',
"Final resource check: final-resource-check.json. It independently inspects the task container IDs, reacquires both lock protocols and checks selected cards under lock. Shared internal network is retained. All attempt statuses are listed in run-inventory.json.",
'', 'Observed states: '+json.dumps(final['run_status_counts'])+'.',
'', 'Remaining work: eight-rank implementation and healthy whole-host qualification; any unstable qualification groups listed above; vendor investigation of XPULink throughput and BF16 arithmetic/dispatch limits; known 4 GiB pinned-nonblocking runtime error. Toolkit feature expansion and interserver tests remain outside this Base delivery.',
'', 'Support/runbook: ../../../../../docs/p800-day7.md (from this evidence directory), or base/docs/p800-day7.md from the repository root. No push or publication was performed.','']
(OUT/'review.md').write_text('\n'.join(lines))
profile=ROOT/'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/qualification-record.json'
record=read(profile)
record['day7_acceptance']={k:v for k,v in final.items() if k not in ('capacity_metric','monitor_ab','offline_suites','historical_unchanged')}
record['day7_acceptance'].update(review='evidence/day7/review.md',verification='evidence/day7/verification.json',
 cpu_regression={x['label']:dict(tests=x['tests'],exit_code=x['exit_code']) for x in suites},
 historical_day6_formal_reruns='Already completed by user on 4+7 and 3+4; current Day6 index 5811 verified; supersedes the older pending note without rewriting historical evidence')
save(profile,record)
print(json.dumps(dict(status=final['status'],qualification=[(x['group'],x['status']) for x in q],tests=[(x['label'],x['tests']) for x in suites],runs=final['run_status_counts']),indent=2))
