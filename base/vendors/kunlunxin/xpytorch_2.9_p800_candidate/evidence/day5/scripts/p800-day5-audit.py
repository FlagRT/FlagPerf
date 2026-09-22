from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/kzhang519/Zhiyu/runtime-team/FlagPerf')
OUTPUT = ROOT / 'base/result/p800-pr4-pr5-20260922-0935'
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
        kind = summary.get('kind')
        regenerate = None
        if kind == 'benchmark-preflight':
            regenerate = render_report
        elif kind == 'benchmark' or (root / 'benchmark-result.json').exists():
            regenerate = generate_benchmark_report
        if regenerate is not None:
            regenerate(root)
        regenerated = {path.name: digest(path) for path in root.glob('report*.md')}
        if regenerate is not None:
            regenerate(root)
        row = {'attempt': root.name, 'experiment_status': summary['status'], 'kind': kind,
               'summary_unchanged': before == (root / 'summary.json').read_bytes(),
               'report_deterministic': regenerated == {path.name: digest(path) for path in root.glob('report*.md')},
               'report_normalized_on_first_regen': regenerated != report_hashes}
        inspect_path = root / 'container-inspect.json'
        if inspect_path.exists():
            cid = json.loads(inspect_path.read_text())['Id']
            lookup = commands.checked(['docker', 'ps', '-a', '--no-trunc', '--filter', 'id=' + cid, '--format', '{{.ID}}'], privileged=True)
            row.update(container_id=cid, container_absent=not lookup['stdout'].strip())
        # A rejection at host-preflight creates no container and acquires no lease;
        # its summary records lease_released=false with cleanup_status=not-created.
        # Preflight-only summaries (no benchmark run) carry no lease fields at all.
        row['lease_released'] = summary.get('lease_released')
        if kind == 'benchmark':
            row['lease_check_ok'] = summary.get('lease_released') is True or (
                summary.get('lease_released') is False and summary.get('cleanup_status') == 'not-created'
                and summary.get('failure_stage') == 'host-preflight')
        else:
            row['lease_check_ok'] = True
        metric_path = root / 'artifacts/metric-rank-0.json'
        if metric_path.exists():
            metric = json.loads(metric_path.read_text())
            row.update(value=metric['value'], unit=metric['unit'], metric=metric['metric'], elapsed_seconds=metric['elapsed_seconds'])
            monitor_path = root / 'benchmark-monitor/summary.json'
            if monitor_path.exists():
                monitor = json.loads(monitor_path.read_text())
                row['valid_samples'] = monitor.get('primary_sample_counts_by_target', {}).get(metric['binding']['resource_key'], 0)
        checks.append(row)
    host = provider.inspect_host(config, [5], commands, OUTPUT / 'acceptance-host-preflight-attempt03')
    with DeviceLease([], run_id='pr4-pr5-independent-acceptance', kind='review', **provider.lease_spec(host)) as lease:
        locked = provider.inspect_host(config, [5], commands, OUTPUT / 'acceptance-locked-preflight-attempt03')
        provider.check_identity(host, locked)
        lock_record = lease.record()

    groups = {
        'computation-FP16': [f'qualification-FP16-q0{i}' for i in range(1, 6)],
        'computation-BF16': [f'qualification-BF16-q0{i}' for i in range(1, 6)],
        'computation-INT8-2048': [f'qualification-INT8-q2{i}' for i in range(1, 6)],
        'computation-INT8-4096-warmup10': [f'qualification-INT8-q0{i}' for i in range(1, 6)],
        'computation-INT8-4096-warmup100': [f'qualification-INT8-q1{i}' for i in range(1, 6)],
    }
    for direction in ('h2d', 'd2h'):
        for variant in ('pageable-blocking', 'pageable-nonblocking', 'pinned-blocking', 'pinned-nonblocking'):
            groups[f'interconnect-{direction}-{variant}'] = [f'qualification-{direction}-{variant}-q0{i}' for i in range(1, 6)]
    statistics = {}
    stats_dir = OUTPUT / 'qualification-statistics'
    stats_dir.mkdir(exist_ok=True)
    for name, directories in groups.items():
        record = summarize([OUTPUT / item for item in directories])
        (stats_dir / f'{name}.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        statistics[name] = {'status': record['status'], 'median': record['median'],
                            'cv_percent': record['cv_percent'], 'min': record['min'], 'max': record['max'],
                            'unit': record['unit']}

    timeout_rows = []
    for name in ('timeout-FP16-a01', 'timeout-h2d-a01'):
        root = OUTPUT / name
        summary = json.loads((root / 'summary.json').read_text())
        probe = json.loads((root / 'artifacts/probe.json').read_text()) if (root / 'artifacts/probe.json').exists() else None
        timeout_rows.append({'run': name, 'status': summary['status'], 'error': summary.get('error'),
                             'error_type': summary.get('error_type'), 'failure_stage': summary.get('failure_stage'),
                             'cleanup_status': summary.get('cleanup_status'), 'postflight_status': summary.get('postflight_status'),
                             'lease_released': summary.get('lease_released'), 'probe_status': (probe or {}).get('status')})

    accepted = ('qualification-FP16', 'qualification-BF16', 'qualification-INT8-2048',
                'qualification-h2d-pageable-blocking', 'qualification-h2d-pageable-nonblocking',
                'qualification-d2h-pageable-blocking', 'qualification-d2h-pageable-nonblocking')
    passed_groups = [name for name, row in statistics.items() if row['status'] == 'passed']
    unstable_groups = [name for name, row in statistics.items() if row['status'] == 'unstable']
    integrity_ok = all(row['summary_unchanged'] and row['report_deterministic'] and row.get('container_absent', True)
                       and row.get('lease_check_ok', True) for row in checks)
    timeouts_ok = all(row['status'] == 'failed' and row['error_type'] == 'TimeoutError'
                      and row['cleanup_status'] == 'passed' and row['postflight_status'] == 'passed'
                      and row['lease_released'] is True for row in timeout_rows)
    result = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
              'checks': checks, 'lease_reacquired': True, 'lease': lock_record,
              'final_idle_status': locked['status'], 'qualification_statistics': statistics,
              'passed_groups': passed_groups, 'unstable_groups': unstable_groups,
              'controlled_timeout_cleanup': 'passed' if timeouts_ok else 'failed', 'timeout_runs': timeout_rows,
              'integrity_ok': integrity_ok,
              'status': 'passed' if integrity_ok and timeouts_ok else 'failed'}
    (OUTPUT / 'acceptance-review.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'integrity_ok', 'controlled_timeout_cleanup',
        'final_idle_status', 'passed_groups', 'unstable_groups')}, indent=2))
    if result['status'] != 'passed':
        raise RuntimeError('independent acceptance failed')


if __name__ == '__main__':
    hardware_review()
