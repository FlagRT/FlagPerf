import ast
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUTPUT = ROOT / 'base/result/p800-pr3-20260921-1011'
PYTHON = str(ROOT / 'base/result/p800-pr1-20260920-162724/cpu-test-venv/bin/python')


def run(name, command):
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (OUTPUT / (name + '.stdout.log')).write_text(completed.stdout)
    (OUTPUT / (name + '.stderr.log')).write_text(completed.stderr)
    record = {'name': name, 'command': command, 'exit_code': completed.returncode}
    matches = re.findall(r'Ran (\d+) tests?', completed.stderr)
    if matches:
        record['test_count'] = int(matches[-1])
    if '-dry-' in name and completed.returncode == 0:
        record['applicability'] = json.loads(completed.stdout).get('applicability')
    (OUTPUT / (name + '.command.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return record


prefix = sys.argv[1]
records = []
for name, command in [('base', [PYTHON, '-B', '-m', 'unittest', 'discover', '-s', 'base/tests', '-p', 'test_*.py', '-v']),
                      ('pr0', ['python3', '-S', '-B', '-m', 'unittest', 'discover', '-s', 'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate', '-p', 'test_*.py', '-v']),
                      ('toolkit', ['python3', '-B', '-m', 'unittest', 'discover', '-s', 'base/toolkits/_common/ascend/A3/tests', '-v']),
                      ('bootstrap', ['bash', '-n', 'base/vendors/kunlunxin/runtime_bootstrap.sh']),
                      ('whitespace', ['git', 'diff', '--check'])]:
    records.append(run(prefix + '-' + name, command))
history = json.loads((ROOT / 'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day2/logs/final-dry-runs.json').read_text())
for item in history['runs']:
    record = run(prefix + '-' + item['name'], item['command'])
    record['expected_exit_code'] = item['exit_code']
    record['matches_baseline'] = record['exit_code'] == item['exit_code'] and ('applicability' not in item or record.get('applicability') == item['applicability'])
    records.append(record)
for mode in ('on', 'off'):
    records.append(run(prefix + '-dry-p800-' + mode, ['python3', '-B', 'base/run.py', 'benchmark', 'run',
        '--config', 'base/configs/kunlunxin_p800_xpytorch29.yaml', '--case', 'computation-FP32:P800',
        '--physical-device-ids', '6', '--nproc-per-node', '1', '--timeout', '300', '--monitor', mode, '--dry-run']))
(OUTPUT / (prefix + '-verification.json')).write_text(json.dumps(records, indent=2) + '\n')
raise SystemExit(1 if any(item['exit_code'] != item.get('expected_exit_code', 0) or item.get('matches_baseline') is False for item in records) else 0)
