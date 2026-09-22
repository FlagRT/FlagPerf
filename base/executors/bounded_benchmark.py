"""Performance hooks for the shared bounded device/container lifecycle."""
from argparse import Namespace
import hashlib
import json
import math
from pathlib import Path
import shutil

from benchmarks.case_assets import portable_assets
from benchmarks.computation_contract import contract, validate_config
from benchmarks.transfer_contract import CASES as TRANSFER_CASES, contract as transfer_contract
from executors.benchmark import parse_benchmark_results, request_assets, snapshot_case_configuration
from executors.common import run_timestamp, write_json
from executors.lifecycle import HostCommands
from executors import preflight
from generate_benchmark_report import generate_benchmark_report


def read(path):
    return json.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_metric(metric, correctness, context, context_hash, binding, config):
    case = context.get('case', 'computation-FP32:P800')
    if case in TRANSFER_CASES:
        from benchmarks.transfer_contract import validate_metric as validate_transfer_metric
        return validate_transfer_metric(metric, correctness, context, context_hash, binding, config)
    spec = contract(case)
    validate_config(config, case)
    for record in (metric, correctness):
        check(record.get('schema_version') == 1 and record.get('status') == 'passed', 'rank artifact failed')
        check(record.get('run_id') == context['run_id'] and record.get('context_sha256') == context_hash,
              'rank artifact context mismatch')
        check(record.get('rank') == 0 and record.get('binding') == binding, 'rank artifact binding mismatch')
    check(metric.get('world_size') == 1 and metric.get('metric') == spec['metric'] and metric.get('unit') == spec['unit'],
          'unexpected rank metric contract')
    check(metric.get('case_assets_sha256') == context['case_assets_sha256'], 'rank case identity mismatch')
    check(metric.get('shape') == [config['M'], config['N'], config['K']] and metric.get('iterations') == config['ITERS']
          and metric.get('warmup') == config['WARMUP'] and metric.get('mode') == config['MODE'], 'rank config drift')
    check(metric.get('implementation') == config['IMPLEMENTATION'], 'rank implementation drift')
    check(correctness.get('seed') == config['SEED'] and correctness.get('dtype') == spec['dtype']
          and correctness.get('reference_dtype') == spec['reference_dtype'], 'correctness config drift')
    checks = [*correctness.get('small_cases', []), correctness.get('shape_before', {}), correctness.get('shape_after', {})]
    check(len(checks) == spec['small_cases'] + 2 and all(record.get('passed') is True and record.get('finite') is True
          and record.get('atol') == config['ATOL'] and record.get('rtol') == config['RTOL'] for record in checks),
          'missing or failed numerical checks')
    if spec['precision'] != 'FP32':
        check(metric.get('output_dtype') == spec['output_dtype'], 'computation output dtype mismatch')
        check(metric.get('dtype') == spec['dtype'] and metric.get('operator') == correctness.get('operator') == spec['operator'],
              'computation dtype/operator mismatch')
        check(metric.get('reference_inputs') == correctness.get('reference_inputs') ==
              spec['reference_inputs'], 'incorrect low-precision reference inputs')
        expected_shapes = spec['shapes']
        check([record.get('shape') for record in correctness['small_cases']] == expected_shapes, 'missing precision diagnostic shapes')
    for name in ('shape_before', 'shape_after'):
        check(correctness[name].get('full_output_finite') is True and correctness[name].get('reduction_size') == config['N'],
              'large shape validation incomplete')
    for key in ('elapsed_seconds', 'value'):
        check(type(metric.get(key)) in (float, int) and math.isfinite(metric[key]) and metric[key] > 0, 'invalid rank ' + key)
    started, finished = metric.get('started_monotonic_ns'), metric.get('finished_monotonic_ns')
    check(type(started) is int and type(finished) is int and finished > started, 'invalid rank clock interval')
    check(math.isclose((finished - started) / 1e9, metric['elapsed_seconds'], rel_tol=1e-12), 'rank elapsed interval differs')
    operations = 2 * config['M'] * config['N'] * config['K'] * config['ITERS']
    check(metric.get('operations') == operations and math.isclose(metric['value'], operations / metric['elapsed_seconds'] / 1e12,
          rel_tol=1e-12), 'rank metric cannot be recomputed')


def validate_stdout(text, raw, case):
    parsed = parse_benchmark_results(text, 1)
    metrics = parsed['metrics']
    spec = transfer_contract(case) if case in TRANSFER_CASES else contract(case)
    expected = {spec['unit']: raw['value']}
    if case in TRANSFER_CASES:
        expected['GiB/s'] = raw['value_gib_s']
    check(parsed['status'] == 'passed' and len(metrics) == len(expected), 'stdout metric count or status differs')
    check({metric['unit'] for metric in metrics} == set(expected), 'stdout units differ')
    for metric in metrics:
        check(metric['rank'] == 0 and metric['metric'] == spec['metric'] and
              math.isclose(metric['value'], expected[metric['unit']], rel_tol=1e-12),
              'stdout rank metric differs from durable evidence')


class BenchmarkWorkload:
    def __init__(self, request, provider):
        self.request, self.provider = request, provider
        self.assets = request_assets(request)
        self.config = self.assets['merged_config']

    def initial_summary(self, static):
        return {'schema_version': 3, 'kind': 'benchmark', 'case': self.request.case,
                'vendor_display_name': self.provider.display_name, 'static_plan': static,
                'execution_status': 'running', 'measurement_status': 'not_started',
                'qualification': {'mode': self.config['MODE'], 'repetitions_required': 5,
                                  'min_measurement_seconds': 15, 'max_cv_percent': 5,
                                  'scope': 'single-card native XPYTORCH ' + self.request.case + '; candidate runtime'},
                'measurement_evidence_status': 'not_started'}

    def prepare(self, root):
        snapshot_case_configuration(self.request, root / 'control')
        contract = portable_assets(self.assets)
        write_json(root / 'control/case-assets.json', contract)
        write_json(root / 'case-assets.json', contract)
        write_json(root / 'benchmark-result.json', {'schema_version': 1, 'status': 'failed', 'metrics': [],
                                                   'expected_ranks': [0], 'observed_ranks': [], 'missing_ranks': [0]})

    def context_record(self, root):
        return {'kind': 'benchmark', 'case': self.request.case, 'probe_mode': 'performance',
                'nproc_per_node': 1, 'master_port': self.request.master_port,
                'case_assets_sha256': hashlib.sha256((root / 'control/case-assets.json').read_bytes()).hexdigest()}

    def collect(self, root, context, context_hash, summary):
        output = root / 'artifacts'
        check(sorted(path.name for path in output.glob('metric-rank-*.json')) == ['metric-rank-0.json'], 'missing/extra rank metrics')
        check(sorted(path.name for path in output.glob('correctness-rank-*.json')) == ['correctness-rank-0.json'], 'missing/extra correctness ranks')
        metric, correctness = read(output / 'metric-rank-0.json'), read(output / 'correctness-rank-0.json')
        validate_metric(metric, correctness, context, context_hash, summary['device_bindings'][0], self.config)
        summary.update(execution_status='passed', measurement_status='passed', correctness_status='passed',
                       runtime={'image': summary['static_plan']['image'], 'image_id': context['image_identity']['Id'],
                                'lock': summary['static_plan']['runtime_lock']},
                       host_preflight='host-preflight/summary.json', host_postflight='host-postflight/summary.json')
        result = {'schema_version': 1, 'status': 'passed', 'case': self.request.case, 'expected_ranks': [0],
                  'observed_ranks': [0], 'missing_ranks': [], 'metrics': [metric],
                  'correctness': 'artifacts/correctness-rank-0.json'}
        write_json(root / 'benchmark-result.json', result)
        try:
            events = sorted((output / 'benchmark-events').glob('measurement-rank-*.json'))
            check(len(events) == 1 and events[0].name == 'measurement-rank-0.json', 'missing/extra measurement events')
            event = read(events[0])
            check(event.get('kind') == 'measurement-window' and event.get('case') == self.request.case
                  and (event.get('rank'), event.get('local_rank'), event.get('world_size')) == (0, 0, 1), 'measurement event identity mismatch')
            check(event.get('pid') == metric['worker_pid'], 'event worker PID mismatch')
            check(event['started_monotonic_ns'] <= metric['started_monotonic_ns'] < metric['finished_monotonic_ns'] <= event['finished_monotonic_ns'],
                  'measurement event does not enclose synchronized timer')
            summary['measurement_evidence_status'] = 'passed'
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            summary.update(status='partial', measurement_evidence_status='partial', measurement_evidence_error=str(exc))
        if self.config['MODE'] == 'qualification' and metric['elapsed_seconds'] < 15:
            summary.update(status='partial', qualification_error='measurement window shorter than 15 seconds')

    def monitor_windows(self, root, monitor, summary):
        path = root / 'artifacts/metric-rank-0.json'
        if not path.is_file() or summary.get('measurement_evidence_status') != 'passed':
            return []
        metric = read(path)
        return [{'role': 'measurement', 'rank': 0, 'device_id': metric['binding']['resource_key'],
                 'started_monotonic_ns': metric['started_monotonic_ns'], 'finished_monotonic_ns': metric['finished_monotonic_ns'],
                 'started_offset_s': metric['started_monotonic_ns'] / 1e9 - monitor.origin_monotonic_s,
                 'finished_offset_s': metric['finished_monotonic_ns'] / 1e9 - monitor.origin_monotonic_s}]

    def finalize(self, root, summary):
        try:
            logs = read(root / 'container-logs.json') if (root / 'container-logs.json').is_file() else {}
            text = logs.get('stdout', '') + logs.get('stderr', '')
            (root / 'runner.log').write_text(text)
            if summary.get('measurement_status') == 'passed':
                raw = read(root / 'benchmark-result.json')['metrics'][0]
                check(logs.get('returncode') == 0, 'container log retrieval failed')
                validate_stdout(text, raw, self.request.case)
            if self.request.monitor == 'off':
                write_json(root / 'benchmark-monitor/summary.json', {'schema_version': 2, 'vendor': self.provider.name,
                           'status': 'not-run', 'policy': self.provider.monitor_policy(False), 'workload_windows': []})
            else:
                path = root / 'benchmark-monitor/summary.json'
                if path.is_file():
                    monitor = read(path)
                    monitor['policy'] = self.provider.monitor_policy(True)
                    write_json(path, monitor)
            probe = root / 'artifacts/probe.json'
            if probe.is_file():
                shutil.copyfile(probe, root / 'container-preflight.json')
        except Exception as exc:
            summary.update(status='failed', measurement_status='failed', finalization_error=str(exc))
        if summary['status'] == 'failed':
            summary['execution_status'] = 'failed'
            if summary.get('measurement_status') != 'passed':
                summary['measurement_status'] = 'failed'
        if self.config['MODE'] == 'qualification' and self.request.monitor != 'on' and summary['status'] == 'passed':
            summary.update(status='partial', qualification_error='qualification requires measurement telemetry')

    def render_report(self, root):
        return generate_benchmark_report(root)


def execute(request, static, config, provider):
    HostCommands(request.privilege_command)
    root = request.result_dir or request.context.resolved_result_root(config) / ('benchmark-' + run_timestamp())
    args = Namespace(dry_run=False, allow_candidate_runtime=request.allow_candidate_runtime,
                     timeout=request.context.timeout, reservation_end=request.reservation_end,
                     reservation_reference=request.reservation_reference, result_dir=root,
                     privilege_command=request.privilege_command, monitor=request.monitor, probe_mode='performance')
    return preflight.execute(args, prepared=(static, config, provider), workload=BenchmarkWorkload(request, provider))
