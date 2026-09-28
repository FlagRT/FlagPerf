import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUT = ROOT / 'base/result/p800-day7-20260924'
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
authfile = OUT / 'authorization.json'
if not authfile.exists():
    now = dt.datetime.now(dt.timezone.utc)
    authfile.write_text(json.dumps(dict(reference='user-20260924-execute-day7',
        scope='Day7 on idle healthy cards, single-card qualification and capacity, two-card regression; no occupied-card override or eight-card run',
        reservation_end=(now + dt.timedelta(hours=8)).isoformat(),
        interpretation='Execution budget under user instruction, not an external reservation'), indent=2))
auth = json.loads(authfile.read_text())

def run(name, case=None, config=None, monitor='on', cards='5', high=False, timeout=None):
    dest = OUT / 'runs' / name
    if dest.exists():
        raise RuntimeError('refuse overwriting run ' + name)
    command = ['python3', '-B', 'base/run.py', 'benchmark', 'run' if case else 'preflight',
        '--config', 'base/configs/kunlunxin_p800_xpytorch29.yaml', '--physical-device-ids', cards,
        '--allow-candidate-runtime', '--privilege-command', 'sudo -n',
        '--reservation-reference', auth['reference'], '--reservation-end', auth['reservation_end'],
        '--timeout', str(timeout or (600 if case else 120)), '--monitor', monitor, '--result-dir', str(dest)]
    if case:
        command += ['--case', case + ':P800', '--nproc-per-node', str(len(cards.split(',')))]
    if config:
        command += ['--case-config', str(config)]
    if high:
        command += ['--allow-high-risk-case']
    records = OUT / 'commands'
    records.mkdir(exist_ok=True)
    record = dict(command=command, started_at=dt.datetime.now(dt.timezone.utc).isoformat())
    (records / (name + '.json')).write_text(json.dumps(record, indent=2))
    with (records / (name + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    (records / (name + '.json')).write_text(json.dumps(record, indent=2))
    summary = json.loads((dest / 'summary.json').read_text()) if (dest / 'summary.json').exists() else {}
    print(name, result.returncode, {k:summary.get(k) for k in ('status','failure_stage','error','cleanup_status','lease_released')}, flush=True)
    if case and (summary.get('cleanup_status') != 'passed' or not summary.get('lease_released')):
        raise RuntimeError('stop on incomplete cleanup: ' + name)
    return result.returncode

def batch():
    if run('preflight-card5-a02'):
        raise RuntimeError('card5 preflight failed')
    jobs = []
    for dtype in ('FP32','FP16','BF16','INT8'):
        case = 'computation-' + dtype
        jobs.append((dtype, case, None, False))
    for direction in ('h2d','d2h'):
        for variant in ('pageable-blocking','pageable-nonblocking','pinned-blocking','pinned-nonblocking'):
            case = 'interconnect-' + direction
            config = ROOT / 'base/benchmarks' / case / 'kunlunxin/P800' / ('case_config.measured-' + variant + '.yaml')
            jobs.append((direction + '-' + variant, case, config, False))
    jobs.append(('memory-bandwidth', 'main_memory-bandwidth', None, False))
    for name, case, config, high in jobs:
        for repeat in range(1, 6):
            if run(name + '-q' + str(repeat), case, config, high=high):
                break
    run('capacity', 'main_memory-capacity', high=True)
    for name, case, config in [('FP32','computation-FP32',None), ('FP16','computation-FP16',None),
            ('h2d-pinned','interconnect-h2d',ROOT/'base/benchmarks/interconnect-h2d/kunlunxin/P800/case_config.measured-pinned-blocking.yaml')]:
        for repeat in range(1,4):
            for monitor in ('off','on'):
                run(name + '-ab-' + monitor + '-' + str(repeat), case, config, monitor=monitor)
    print('DAY7-BATCH-COMPLETE', flush=True)

if __name__ == '__main__':
    batch()
