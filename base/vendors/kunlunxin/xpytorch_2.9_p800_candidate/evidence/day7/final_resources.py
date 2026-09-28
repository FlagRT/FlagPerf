"""Final resource checks: inspect only this task's CIDs and selected cards."""
import json
from pathlib import Path
import subprocess
import sys
import datetime as dt
ROOT=Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUT=ROOT/'base/result/p800-day7-20260924'
sys.path.insert(0,str(ROOT/'base'))
from executors.common import DeviceLease
from executors.lifecycle import HostCommands
from base.vendors.kunlunxin.provider import KunlunxinProvider
record={'checked_at':dt.datetime.now(dt.timezone.utc).isoformat()}
ids=set(); devices={}; states=[]
for p in sorted((OUT/'runs').glob('*/summary.json')):
 s=json.loads(p.read_text())
 states.append({'run':p.parent.name,**{k:s.get(k) for k in ('status','failure_stage','cleanup_status','postflight_status','lease_released')}})
 if s.get('container_id'): ids.add(s['container_id'])
 c=p.parent/'control/host-context.json'
 if c.exists():
  for d in json.loads(c.read_text()).get('host',{}).get('devices',[]): devices[d['uuid']]=d
p=subprocess.run(['sudo','-n','docker','ps','-a','--no-trunc','--format','{{.ID}}'],capture_output=True,text=True,check=True)
record['owned_container_ids']=sorted(ids)
record['owned_containers_remaining']=sorted(ids.intersection(p.stdout.split()))
record['all_started_runs_terminal']=all(s['status']!='running' for s in states)
record['run_states']=states
record['unfinished_command_records']=[p.name for p in (OUT/'commands').glob('*.json') if 'exit_code' not in json.loads(p.read_text())]
record['cleanup_failures']=[s['run'] for s in states if s['cleanup_status'] not in (None,'passed') or s['lease_released'] is False]
provider=KunlunxinProvider()
host={'devices':list(devices.values())}
lease=DeviceLease([],run_id='p800-day7-final-reacquire',kind='acceptance',**provider.lease_spec(host))
try:
 with lease:
  record['dual_protocol_locks_reacquired']=True
  record['lease_record']=lease.record()
  config=json.loads((ROOT/'base/configs/kunlunxin_p800_xpytorch29.yaml').read_text())
  try:
   observed=provider.inspect_host(config,sorted(d['host_physical_id'] for d in devices.values()),HostCommands('sudo -n'),OUT/'final-host-check')
   record['selected_cards_idle_under_lock']=True
   record['selected_devices']=observed['devices']
  except Exception as e:
   record['selected_cards_idle_under_lock']=False
   record['host_check_error']=str(e)
 record['final_locks_released']=True
except Exception as e:
 record['dual_protocol_locks_reacquired']=False;record['lock_error']=str(e)
p=subprocess.run(['sudo','-n','docker','network','inspect','flagperf-p800-internal'],capture_output=True,text=True)
if p.returncode==0:
 n=json.loads(p.stdout)[0]; record['network']={k:n.get(k) for k in ('Name','Id','Driver','Internal')}
else: record['network_error']=p.stderr
record['foreign_override_absent']='P800_ALLOW_FOREIGN_HANDLES' not in (ROOT/'base/vendors/kunlunxin/preflight.py').read_text()
record['final_check_passed']=all([not record['owned_containers_remaining'],record['all_started_runs_terminal'],
 not record['unfinished_command_records'],not record['cleanup_failures'],record.get('dual_protocol_locks_reacquired'),
 record.get('final_locks_released'),record.get('selected_cards_idle_under_lock'),record['foreign_override_absent'],
 record.get('network',{}).get('Internal') is True])
(OUT/'final-resource-check.json').write_text(json.dumps(record,indent=2)+'\n')
print({k:v for k,v in record.items() if k not in ('run_states','selected_devices','owned_container_ids','lease_record')})
