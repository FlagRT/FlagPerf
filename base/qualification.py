"""Recompute qualification statistics from complete, identical computation runs.

Formal frozen measurement configs carry MODE=qualification for computation cases
and MODE=measured for transfer cases; both are accepted here because every
substantive gate (five identical runs, >=15s windows, >=10 monitor samples,
identical UUID/image/config/code identity, CV<=5%) is enforced independently."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(directories, allow_shared=False):
    from executors.bounded_benchmark import validate_metric
    if len(directories) != 5 or len({path.resolve() for path in directories}) != 5:
        raise ValueError('qualification requires exactly five distinct result directories')
    records, identities, run_ids = [], set(), set()
    shared_seen = degraded_seen = False
    for root in directories:
        summary = json.loads((root / 'summary.json').read_text())
        result = json.loads((root / 'benchmark-result.json').read_text())
        monitor = json.loads((root / 'benchmark-monitor/summary.json').read_text())
        if any(summary.get(key) != 'passed' for key in ('status', 'execution_status', 'measurement_status',
                'correctness_status', 'measurement_evidence_status', 'monitoring_status', 'postflight_status', 'cleanup_status')):
            raise ValueError('incomplete qualification evidence: ' + str(root))
        if summary.get('lease_released') is not True or summary['qualification']['mode'] not in ('qualification', 'measured'):
            raise ValueError('unreleased lease or non-qualification run')
        if result.get('status') != 'passed' or not result.get('metrics'):
            raise ValueError('incomplete rank metrics')
        context_raw = (root / 'control/host-context.json').read_bytes()
        context = json.loads(context_raw)
        shared = bool(context['host'].get('foreign_occupancy_observed')) or any(d.get('foreign_occupancy_observed') for d in context['host']['devices'])
        degraded = any(any(d.get('uncorrectable_ecc_counts', [])) for d in context['host']['devices'])
        shared_seen |= shared
        degraded_seen |= degraded
        if (shared or degraded) and not allow_shared:
            raise ValueError('shared or degraded-health run is not isolated performance qualification')
        contract = json.loads((root / 'control/case-assets.json').read_text())
        if hashlib.sha256((root / 'control/case-assets.json').read_bytes()).hexdigest() != context['case_assets_sha256']:
            raise ValueError('case contract hash differs from host context')
        ranks = context.get('nproc_per_node', 1)
        expected = list(range(ranks))
        metrics = result['metrics']
        if (type(ranks) is not int or not 1 <= ranks <= 8 or [m.get('rank') for m in metrics] != expected
                or result.get('expected_ranks') != expected or result.get('observed_ranks') != expected
                or result.get('missing_ranks')):
            raise ValueError('missing, duplicate or reordered rank evidence')
        resources = []
        for rank, metric in enumerate(metrics):
            raw_metric = json.loads((root / f'artifacts/metric-rank-{rank}.json').read_text())
            correctness = json.loads((root / f'artifacts/correctness-rank-{rank}.json').read_text())
            if metric != raw_metric:
                raise ValueError('semantic metric differs from raw rank evidence')
            validate_metric(metric, correctness, context, hashlib.sha256(context_raw).hexdigest(),
                            summary['device_bindings'][rank], contract['merged_config'])
            if metric['elapsed_seconds'] < 15 or metric['mode'] not in ('qualification', 'measured'):
                raise ValueError('wrong mode or short measurement window')
            if not math.isfinite(metric['value']) or metric['value'] <= 0:
                raise ValueError('nonpositive or nonfinite metric')
            resource = metric['binding']['resource_key']
            resources.append(resource)
            if monitor['status'] != 'passed' or monitor['primary_sample_counts_by_target'].get(resource, 0) < 10:
                raise ValueError('measurement telemetry incomplete')
        if len(set(resources)) != ranks or len({(m['metric'],m['unit']) for m in metrics}) != 1:
            raise ValueError('duplicate resources or inconsistent rank metric units')
        value = min(m['value'] for m in metrics)
        code = json.loads((root / 'code-identity.json').read_text())
        identities.add((tuple(resources), summary['runtime']['image_id'], metric['case_assets_sha256'], metric['metric'], metric['unit'],
                        json.dumps(code['source_sha256'], sort_keys=True)))
        run_ids.add(summary['run_id'])
        records.append({'directory': str(root), 'run_id': summary['run_id'], 'value': value,
                        {'TFLOPS': 'tflops', 'TOPS': 'tops', 'GB/s': 'gb_s'}[metric['unit']]: value,
                        'elapsed_seconds': max(m['elapsed_seconds'] for m in metrics),
                        'rank_values': [m['value'] for m in metrics],
                        'rank_elapsed_seconds': [m['elapsed_seconds'] for m in metrics],
                        'summary_sha256': hashlib.sha256((root / 'summary.json').read_bytes()).hexdigest()})
    if len(identities) != 1 or len(run_ids) != 5:
        raise ValueError('qualification runs differ in UUID, image, config, code, or repeat run IDs')
    values = [record['value'] for record in records]
    mean, deviation = statistics.mean(values), statistics.stdev(values)
    cv = deviation / mean * 100
    return {'schema_version': 1, 'status': 'exploratory' if shared_seen or degraded_seen else 'passed' if cv <= 5 else 'unstable', 'runs': records,
            'stability_status': 'passed' if cv <= 5 else 'unstable',
            'qualification_status': 'passed' if cv <= 5 and not shared_seen and not degraded_seen else 'not-qualified',
            'resource_scope': 'shared' if shared_seen else 'selected-card-idle-at-preflight',
            'health_status': 'degraded' if degraded_seen else 'no-recorded-uncorrectable-ecc',
            'rank_count': ranks, 'aggregation': 'minimum validated rank bandwidth per run; never summed',
            'repetitions': 5, 'metric': metric['metric'], 'unit': metric['unit'], 'median': statistics.median(values), 'min': min(values),
            'max': max(values), 'mean': mean, 'sample_standard_deviation': deviation, 'ddof': 1,
            'cv_percent': cv, 'max_cv_percent': 5,
            'scope': 'fixed physical UUID set, locked image/native ' + metric['metric'] + '/config; limited to this case and rank count'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs=5, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-shared', action='store_true', help='audit shared/degraded runs as exploratory; never qualifies them')
    args = parser.parse_args()
    result = summarize(args.directories, allow_shared=args.allow_shared)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
