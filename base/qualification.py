"""Recompute qualification statistics from complete, identical FP32 runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(directories):
    from executors.bounded_benchmark import validate_metric
    if len(directories) != 5 or len({path.resolve() for path in directories}) != 5:
        raise ValueError('qualification requires exactly five distinct result directories')
    records, identities, run_ids = [], set(), set()
    for root in directories:
        summary = json.loads((root / 'summary.json').read_text())
        result = json.loads((root / 'benchmark-result.json').read_text())
        monitor = json.loads((root / 'benchmark-monitor/summary.json').read_text())
        if any(summary.get(key) != 'passed' for key in ('status', 'execution_status', 'measurement_status',
                'correctness_status', 'measurement_evidence_status', 'monitoring_status', 'postflight_status', 'cleanup_status')):
            raise ValueError('incomplete qualification evidence: ' + str(root))
        if summary.get('lease_released') is not True or summary['qualification']['mode'] != 'qualification':
            raise ValueError('unreleased lease or non-qualification run')
        if result.get('status') != 'passed' or len(result.get('metrics', [])) != 1:
            raise ValueError('incomplete rank metrics')
        metric = result['metrics'][0]
        context_raw = (root / 'control/host-context.json').read_bytes()
        context = json.loads(context_raw)
        contract = json.loads((root / 'control/case-assets.json').read_text())
        if hashlib.sha256((root / 'control/case-assets.json').read_bytes()).hexdigest() != context['case_assets_sha256']:
            raise ValueError('case contract hash differs from host context')
        raw_metric = json.loads((root / 'artifacts/metric-rank-0.json').read_text())
        correctness = json.loads((root / 'artifacts/correctness-rank-0.json').read_text())
        if metric != raw_metric:
            raise ValueError('semantic metric differs from raw rank evidence')
        validate_metric(metric, correctness, context, hashlib.sha256(context_raw).hexdigest(),
                        summary['device_bindings'][0], contract['merged_config'])
        if metric.get('rank') != 0 or metric['elapsed_seconds'] < 15 or metric['mode'] != 'qualification':
            raise ValueError('wrong rank, mode, or short measurement window')
        value = metric['value']
        if not math.isfinite(value) or value <= 0:
            raise ValueError('nonpositive or nonfinite metric')
        resource = metric['binding']['resource_key']
        if monitor['status'] != 'passed' or monitor['primary_sample_counts_by_target'].get(resource, 0) < 10:
            raise ValueError('measurement telemetry incomplete')
        code = json.loads((root / 'code-identity.json').read_text())
        identities.add((resource, summary['runtime']['image_id'], metric['case_assets_sha256'],
                        json.dumps(code['source_sha256'], sort_keys=True)))
        run_ids.add(summary['run_id'])
        records.append({'directory': str(root), 'run_id': summary['run_id'], 'tflops': value,
                        'elapsed_seconds': metric['elapsed_seconds'],
                        'summary_sha256': hashlib.sha256((root / 'summary.json').read_bytes()).hexdigest()})
    if len(identities) != 1 or len(run_ids) != 5:
        raise ValueError('qualification runs differ in UUID, image, config, code, or repeat run IDs')
    values = [record['tflops'] for record in records]
    mean, deviation = statistics.mean(values), statistics.stdev(values)
    cv = deviation / mean * 100
    return {'schema_version': 1, 'status': 'passed' if cv <= 5 else 'unstable', 'runs': records,
            'repetitions': 5, 'unit': 'TFLOPS', 'median': statistics.median(values), 'min': min(values),
            'max': max(values), 'mean': mean, 'sample_standard_deviation': deviation, 'ddof': 1,
            'cv_percent': cv, 'max_cv_percent': 5,
            'scope': 'one physical UUID, locked M1/native FP32/config; no other dtype or communication qualification'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs=5, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directories)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
