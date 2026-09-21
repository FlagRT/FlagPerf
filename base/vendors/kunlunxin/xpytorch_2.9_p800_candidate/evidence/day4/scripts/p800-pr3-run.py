from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUTPUT = ROOT / 'base/result/p800-pr3-20260921-1011'
case_root = 'base/benchmarks/computation-FP32/kunlunxin/P800/'
mode, name = sys.argv[1:3]
card = sys.argv[3] if len(sys.argv) > 3 else '6'
window = OUTPUT / 'authorization.json'
if not window.exists():
    now = datetime.now(timezone.utc)
    window.write_text(json.dumps({'authorized_scope': 'User instructed completing all Day 4 tasks; idle single card 6 or 7, sequential FP32 qualification and bounded cleanup drills',
        'reference': 'user-20260920-start-all-day4-tasks', 'created_at': now.isoformat(),
        'reservation_end': (now + timedelta(minutes=90)).isoformat(),
        'interpretation': 'Bounded execution budget under user authorization, not a claim of reservation with other users'}, indent=2) + '\n')
authorization = json.loads(window.read_text())
common = ['--config', 'base/configs/kunlunxin_p800_xpytorch29.yaml', '--physical-device-ids', card,
          '--allow-candidate-runtime', '--privilege-command', 'sudo -n', '--reservation-end', authorization['reservation_end'],
          '--reservation-reference', authorization['reference'], '--result-dir', str(OUTPUT / name)]
if mode == 'preflight':
    command = ['python3', '-B', 'base/run.py', 'benchmark', 'preflight', *common, '--monitor', 'on', '--timeout', '120']
else:
    monitor = 'on' if mode == 'qualification' else 'off'
    command = ['python3', '-B', 'base/run.py', 'benchmark', 'run', '--case', 'computation-FP32:P800',
               '--nproc-per-node', '1', *common, '--monitor', monitor, '--timeout', '120' if mode == 'timeout' else '300']
    if mode != 'qualification':
        command += ['--case-config', case_root + 'case_config.' + mode + '.yaml']
record = {'command': command, 'started_at': datetime.now(timezone.utc).isoformat()}
path = OUTPUT / (name + '-command.json')
if path.exists():
    raise RuntimeError('attempt command record already exists')
path.write_text(json.dumps(record, indent=2) + '\n')
subprocess.run(['sudo', '-n', '-v'], check=True)
with (OUTPUT / (name + '-cli.log')).open('x') as stream:
    completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
record.update(exit_code=completed.returncode, finished_at=datetime.now(timezone.utc).isoformat())
path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
print((OUTPUT / (name + '-cli.log')).read_text(), flush=True)
raise SystemExit(completed.returncode)
