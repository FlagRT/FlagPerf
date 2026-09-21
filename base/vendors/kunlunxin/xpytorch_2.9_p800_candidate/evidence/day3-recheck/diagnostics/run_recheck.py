from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys
import time

REPO = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
sys.path.insert(0, str(REPO/'base'))
from executors.common import DeviceLease
from base.vendors.kunlunxin.provider import KunlunxinProvider

ROOT = Path(__file__).resolve().parent
reservation = ROOT/'reservation.json'
if not reservation.exists():
    reservation.write_text(json.dumps(dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        end=(datetime.now(timezone.utc)+timedelta(minutes=45)).isoformat(),
        cards=[1,2,6,7], reference='User explicitly requested retesting idle physical cards 1,2,6,7 before next-day performance work; sequential bounded probes and cleanup drills'), indent=2)+'\n')
window = json.loads(reservation.read_text())


def run(card, mode, label, monitor='on'):
    for retry in range(1,4):
        name=f'{label}-card{card}-attempt{retry:02}'
        root=ROOT/name
        command=['python3','-B','base/run.py','benchmark','preflight',
                 '--config','base/configs/kunlunxin_p800_xpytorch29.yaml',
                 '--physical-device-ids',str(card),'--probe-mode',mode,
                 '--allow-candidate-runtime','--monitor',monitor,'--timeout','120',
                 '--privilege-command','sudo -n','--reservation-end',window['end'],
                 '--reservation-reference',window['reference'],'--result-dir',str(root)]
        (ROOT/(name+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
        (ROOT/(name+'-reservation.json')).write_text(json.dumps(dict(window,physical_id=card),indent=2)+'\n')
        print('START',name,flush=True)
        with (ROOT/(name+'.stdout')).open('wb') as out,(ROOT/(name+'.stderr')).open('wb') as err:
            process=subprocess.run(command,cwd=REPO,stdout=out,stderr=err)
        (ROOT/(name+'.exit')).write_text(str(process.returncode)+'\n')
        if not (root/'summary.json').exists():
            raise RuntimeError('no summary: '+name)
        summary=json.loads((root/'summary.json').read_text())
        cleanup=json.loads((root/'cleanup.json').read_text())
        print('END',name,summary['status'],summary.get('error'),summary.get('postflight_status'),flush=True)
        if cleanup['container_absent'] is not True or summary.get('recovery_required'):
            raise RuntimeError('cleanup unknown: stop entire recheck')
        if cleanup['status']!='not-created':
            if summary.get('postflight_status')!='passed':
                raise RuntimeError('launched workload postflight failed: stop for review')
            host=json.loads((root/'control/host-context.json').read_text())['host']
            with DeviceLease([],run_id='recheck-acceptance',kind='lock-reacquisition',**KunlunxinProvider().lease_spec(host)) as lease:
                review=dict(container_absent=True,postflight_passed=True,lock_reacquired=True,lease=lease.record())
            (root/'independent-review.json').write_text(json.dumps(review,indent=2)+'\n')
            return name, summary
        # A transient handle check is not permission to bypass occupancy.
        # Each retry repeats the full gate with fresh evidence and no container.
        if 'open handles' not in summary.get('error',''):
            return name, summary
        time.sleep(2)
    return name, summary


records=[]
for card in [1,2,6,7]:
    name,summary=run(card,'identity','normal')
    records.append(dict(card=card,normal_attempt=name,status=summary['status']))
    (ROOT/'matrix.json').write_text(json.dumps(records,indent=2)+'\n')
for row in records:
    if row['status']=='passed':
        name,summary=run(row['card'],'timeout-check','timeout',monitor='off')
        root=ROOT/name
        probe=json.loads((root/'artifacts/probe.json').read_text()) if (root/'artifacts/probe.json').exists() else {}
        row.update(timeout_attempt=name,timeout_drill_passed=summary.get('error_type')=='TimeoutError' and probe.get('status')=='passed' and summary.get('postflight_status')=='passed')
        (ROOT/'matrix.json').write_text(json.dumps(records,indent=2)+'\n')
print('MATRIX_FINISHED',json.dumps(records),flush=True)
