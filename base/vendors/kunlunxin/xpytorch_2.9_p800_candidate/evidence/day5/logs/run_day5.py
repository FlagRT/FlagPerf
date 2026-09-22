from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUTPUT = ROOT / 'base/result/p800-pr4-pr5-20260922-0935'
mode, name, card = sys.argv[1:4]
window = OUTPUT / 'authorization-any-idle-card.json'
if not window.exists():
    now = datetime.now(timezone.utc)
    window.write_text(json.dumps({'reference': 'user-20260922-complete-day5-and-continue-existing-draft',
        'authorized_scope': 'Day 5 single-card computation and transfer on any currently idle card; verified read-only xpu-smi observers allowed by user',
        'created_at': now.isoformat(), 'reservation_end': (now + timedelta(hours=8)).isoformat(),
        'interpretation': 'Bounded execution budget under user task authorization; not a reservation with other users'}, indent=2) + '\n')
authorization = json.loads(window.read_text())
if card not in ('0', '3', '4', '5', '6', '7'):
    raise RuntimeError('outside current selected scope')
common = ['--config', 'base/configs/kunlunxin_p800_xpytorch29.yaml', '--physical-device-ids', card,
          '--allow-candidate-runtime', '--privilege-command', 'sudo -n', '--reservation-end', authorization['reservation_end'],
          '--reservation-reference', authorization['reference'], '--result-dir', str(OUTPUT / name)]
if mode == 'preflight' or mode.startswith('capability-'):
    command = ['python3', '-B', 'base/run.py', 'benchmark', 'preflight', *common, '--monitor', 'on', '--timeout', '120']
    if mode != 'preflight':
        command += ['--probe-mode', mode]
else:
    case = sys.argv[4]
    monitor = 'on' if mode in ('qualification', 'measured') else 'off'
    command = ['python3', '-B', 'base/run.py', 'benchmark', 'run', '--case', case + ':P800',
               '--nproc-per-node', '1', *common, '--monitor', monitor, '--timeout', '60' if mode == 'timeout' else '300']
    if len(sys.argv) > 5:
        command += ['--case-config', sys.argv[5]]
    elif mode not in ('qualification', 'measured'):
        command += ['--case-config', 'base/benchmarks/' + case + '/kunlunxin/P800/case_config.' + mode + '.yaml']
record = {'command': command, 'started_at': datetime.now(timezone.utc).isoformat()}
path = OUTPUT / (name + '-command.json')
with path.open('x') as stream:
    json.dump(record, stream, indent=2)
subprocess.run(['sudo', '-n', '-v'], check=True)
with (OUTPUT / (name + '-cli.log')).open('x') as stream:
    completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
record.update(exit_code=completed.returncode, finished_at=datetime.now(timezone.utc).isoformat())
path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
summary = OUTPUT / name / 'summary.json'
if summary.exists():
    data = json.loads(summary.read_text())
    print(json.dumps({key: data.get(key) for key in ('status', 'failure_stage', 'error', 'postflight_status', 'postflight_error', 'cleanup_status', 'lease_released')}), flush=True)
else:
    print((OUTPUT / (name + '-cli.log')).read_text(), flush=True)
raise SystemExit(completed.returncode)
