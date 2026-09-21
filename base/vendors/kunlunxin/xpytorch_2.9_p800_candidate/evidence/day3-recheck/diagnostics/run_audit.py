from pathlib import Path
import datetime, hashlib, json, re, subprocess, sys, time

REPO = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
ROOT = REPO/'base/result/p800-recheck-20260920-1952'
sys.path.insert(0, str(REPO/'base'))
from executors.common import DeviceLease
from executors.lifecycle import HostCommands
from base.vendors.kunlunxin.provider import KunlunxinProvider

provider = KunlunxinProvider()
config = json.loads((REPO/'base/configs/kunlunxin_p800_xpytorch29.yaml').read_text())
records = []
hosts = {}
for root in sorted(ROOT.glob('*attempt*')):
    if not root.is_dir() or not (root/'summary.json').exists():
        continue
    summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] != 'running'
    cleanup = json.loads((root/'cleanup.json').read_text())
    card = json.loads((ROOT/(root.name+'-reservation.json')).read_text())['physical_id']
    row = dict(attempt=root.name, card=card, summary=summary, container_created=cleanup['status']!='not-created')
    assert cleanup['container_absent'] is True
    if row['container_created']:
        cid = cleanup['container_id']
        result = subprocess.run(['sudo','-n','docker','ps','-a','--no-trunc','--filter','id='+cid,'--format','{{.ID}}'], text=True,capture_output=True,timeout=10)
        row['absence_check'] = dict(argv=result.args, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        assert result.returncode == 0 and not result.stdout.strip()
        host = json.loads((root/'control/host-context.json').read_text())['host']
        hosts[card] = host
        with DeviceLease([],run_id='final-recheck-audit',kind='lock-reacquisition',**provider.lease_spec(host)) as lease:
            row['locks_reacquired'] = True
        binding = json.loads((root/'artifacts/runtime-bindings.json').read_text())
        assert binding['context_sha256'] == hashlib.sha256((root/'control/host-context.json').read_bytes()).hexdigest()
        assert binding['run_id'] == summary['run_id']
        assert binding['bindings'][0]['host_physical_id'] == card
        row['binding_context_verified'] = True
        row['binding'] = binding['bindings'][0]
    probe = root/'artifacts/probe.json'
    row['probe_passed'] = probe.exists() and json.loads(probe.read_text()).get('status') == 'passed'
    monitor = root/'monitor/summary.json'
    if monitor.exists():
        row['valid_samples'] = json.loads(monitor.read_text())['sample_counts_by_target']
    review = root/'independent-review.json'
    if review.exists(): row['independent_review'] = json.loads(review.read_text())
    records.append(row)

final_idle = []
for card in [1,2,6,7]:
    host = hosts[card]
    row = dict(card=card,checks=[])
    with DeviceLease([],run_id='final-recheck-idle',kind='lock-reacquisition',**provider.lease_spec(host)):
        row['locks_reacquired'] = True
        for n in range(1,7):
            try:
                checked = provider.inspect_host(config,[card],HostCommands('sudo -n'),ROOT/f'final-idle-card{card}-{n}')
                provider.check_identity(host,checked)
                row['checks'].append(dict(attempt=n,idle=True))
                row['idle_verified'] = True
                break
            except Exception as e:
                row['checks'].append(dict(attempt=n,idle=False,error=str(e)))
                time.sleep(2)
    final_idle.append(row)

c=subprocess.run(['sudo','-n','dmesg','--ctime'],text=True,capture_output=True,timeout=10)
lines=[line for line in c.stdout.splitlines() if re.search(r'(0000:16:00|0000:1c:00|0000:b6:00|0000:bb:00)',line,re.I)]
(ROOT/'final-selected-kernel-after.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=c.returncode,selection='exact selected PCI BDF mentions only; not proof of absence of other errors',lines=lines,stderr=c.stderr),indent=2)+'\n')
audit=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt_count=len(records),attempts=records,final_idle=final_idle,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scope='bounded single-card preflight; no performance qualification')
(ROOT/'acceptance-recheck.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(attempt_count=len(records),launched=sum(r['container_created'] for r in records),final_idle=final_idle),indent=2))
