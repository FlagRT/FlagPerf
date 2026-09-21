from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUTPUT = ROOT / 'base/result/p800-pr3-20260921-1011'
sys.path.insert(0, str(ROOT / 'base'))
sys.path.insert(0, str(ROOT))
from executors.common import DeviceLease, load_host_config
from executors.lifecycle import HostCommands
from base.vendors.kunlunxin.provider import KunlunxinProvider
from executors.preflight import render_report
from generate_benchmark_report import generate_benchmark_report
from qualification import summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_review():
    prior = json.loads((ROOT / 'base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day2/logs/final-dry-runs.json').read_text())
    current = json.loads((OUTPUT / 'attempt04-verification.json').read_text())
    rows = []
    for expected in prior['runs']:
        actual = next(row for row in current if row['name'] == 'attempt04-' + expected['name'])
        matches = expected['exit_code'] == actual['exit_code'] and ('applicability' not in expected or expected['applicability'] == actual.get('applicability'))
        rows.append({'name': expected['name'], 'matches_baseline': matches, 'exit_code': actual['exit_code'],
                     'applicability': actual.get('applicability')})
    result = {'status': 'passed' if all(row['matches_baseline'] for row in rows) else 'failed', 'rows': rows,
        'comparison_correction': 'Historical selector entries omitted applicability; compare only fields actually recorded in baseline. attempt04-verification.json retained unchanged.'}
    (OUTPUT / 'static-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def hardware_review():
    commands = HostCommands('sudo -n')
    provider = KunlunxinProvider()
    _, config = load_host_config(ROOT / 'base/configs/kunlunxin_p800_xpytorch29.yaml')
    checks = []
    for root in sorted(OUTPUT.iterdir()):
        if not (root / 'summary.json').is_file():
            continue
        summary = json.loads((root / 'summary.json').read_text())
        before = (root / 'summary.json').read_bytes()
        report_hashes = {path.name: digest(path) for path in root.glob('report*.md')}
        if summary['kind'] == 'benchmark-preflight':
            render_report(root)
        else:
            generate_benchmark_report(root)
        row = {'attempt': root.name, 'experiment_status': summary['status'], 'summary_unchanged': before == (root / 'summary.json').read_bytes(),
               'report_deterministic': report_hashes == {path.name: digest(path) for path in root.glob('report*.md')}}
        inspect_path = root / 'container-inspect.json'
        if inspect_path.exists():
            cid = json.loads(inspect_path.read_text())['Id']
            lookup = commands.checked(['docker', 'ps', '-a', '--no-trunc', '--filter', 'id=' + cid, '--format', '{{.ID}}'], privileged=True)
            row.update(container_id=cid, container_absent=not lookup['stdout'].strip())
        metric_path = root / 'artifacts/metric-rank-0.json'
        if metric_path.exists():
            metric = json.loads(metric_path.read_text())
            row.update(tflops=metric['value'], elapsed_seconds=metric['elapsed_seconds'])
            monitor_path = root / 'benchmark-monitor/summary.json'
            monitor = json.loads(monitor_path.read_text())
            row['valid_samples'] = monitor.get('primary_sample_counts_by_target', {}).get(metric['binding']['resource_key'], 0)
        if (root / 'code-identity.json').exists():
            identity = json.loads((root / 'code-identity.json').read_text())
            row.update(code_commit=identity['git_head'], worktree_clean=not identity['git_status'].strip())
        checks.append(row)
    host = provider.inspect_host(config, [6], commands, OUTPUT / 'acceptance-host-preflight')
    with DeviceLease([], run_id='pr3-independent-acceptance', kind='review', **provider.lease_spec(host)) as lease:
        locked = provider.inspect_host(config, [6], commands, OUTPUT / 'acceptance-locked-preflight')
        provider.check_identity(host, locked)
        lock_record = lease.record()
    qualification = summarize([OUTPUT / f'qualification-attempt{index:02d}-card6' for index in range(1, 6)])
    (OUTPUT / 'qualification-statistics.json').write_text(json.dumps(qualification, indent=2) + '\n')
    timeout_root = OUTPUT / 'timeout-attempt01-card6'
    timeout = json.loads((timeout_root / 'summary.json').read_text())
    probe = json.loads((timeout_root / 'artifacts/probe.json').read_text())
    correct = json.loads((timeout_root / 'artifacts/correctness-rank-0.json').read_text())
    timeout_ok = timeout['status'] == 'failed' and timeout.get('error_type') == 'TimeoutError' and probe['status'] == 'passed' \
        and timeout['cleanup_status'] == timeout['postflight_status'] == 'passed' and timeout['lease_released'] is True \
        and all(record['passed'] for record in correct['small_cases']) and correct['shape_before']['passed']
    result = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'checks': checks,
              'lease_reacquired': True, 'lease': lock_record, 'final_idle_status': locked['status'],
              'controlled_timeout_cleanup': 'passed' if timeout_ok else 'failed',
              'qualification_status': qualification['status'],
              'status': 'passed' if timeout_ok and qualification['status'] == 'passed' and all(row['summary_unchanged'] and row['report_deterministic']
                        and row.get('container_absent', True) for row in checks) else 'failed'}
    (OUTPUT / 'acceptance-review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    print(json.dumps(qualification, indent=2), flush=True)
    if result['status'] != 'passed':
        raise RuntimeError('independent acceptance failed')


if sys.argv[1] == 'static':
    static_review()
else:
    hardware_review()
