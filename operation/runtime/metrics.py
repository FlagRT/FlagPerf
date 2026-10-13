# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Portable performance definitions, repeat statistics and within-run comparisons."""
import hashlib
import json
import math
import random
import statistics

PROTOCOL = 'operation-performance-v1'


def repeat_statistics(samples):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError('timing samples must be finite and positive')
    n = len(samples)
    sd = statistics.stdev(samples) if n > 1 else None
    result = {'protocol': PROTOCOL, 'rounds': n, 'min_us': min(samples), 'max_us': max(samples),
              'median_us': statistics.median(samples), 'stddev_us': sd,
              'cv_percent': 100 * sd / statistics.mean(samples) if sd is not None else None,
              'scope': 'within-process batch means; not single-call tail latency or cross-process confidence',
              'median_ci95_us': None, 'ci_reason': 'requires at least 10 rounds'}
    if n >= 10:
        rng = random.Random(2026)
        boot = sorted(statistics.median(rng.choices(samples, k=n)) for _ in range(2000))
        result.update(median_ci95_us=[boot[49], boot[1949]], ci_reason=None,
                      ci_method='percentile bootstrap; 2000 resamples; seed 2026')
    return result


def logical_traffic(inputs, outputs, backward=False, parameters=()):
    """Each tensor argument read once, each materialized output written once.

    Caller supplies tensor metadata, keeping this module independent of Torch.
    Aliased output views create no write; in-place outputs still write their data.
    """
    if backward:
        return {'status': 'not-supported', 'reason': 'backward logical traffic is not modeled'}
    reads = sum(t['bytes'] for t in [*inputs, *parameters])
    writes = sum(t['bytes'] for t in outputs if not t.get('view', False))
    return {'status': 'estimated', 'read_bytes': reads, 'write_bytes': writes,
            'total_bytes': reads + writes, 'inputs': inputs, 'outputs': outputs,
            'parameters': list(parameters), 'protocol': 'logical-io-v1',
            'boundary': 'one semantic read per input/parameter, materialized outputs written once; broadcast inputs counted before expansion; excludes internal accesses and caches; not measured HBM traffic'}


def pair_key(task):
    keys = ('vendor', 'device_id', 'case', 'dtype', 'seed', 'case_config', 'spectflops', 'performance_protocol')
    return hashlib.sha256(json.dumps({k: task.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()[:24]


def comparisons(summary):
    # Aggregate views must never manufacture a pair from different source runs.
    if summary.get('source_runs') or summary.get('oplib_selection') != 'both':
        return []
    from runtime.reporting import valid_performance
    groups = {}
    for task in summary.get('tasks', []):
        if task.get('pair_id'):
            groups.setdefault(task['pair_id'], []).append(task)
    rows = []
    for pair, tasks in groups.items():
        row = {'pair_id': pair, 'status': 'not-comparable', 'reason': None,
               'scope': 'same-run --oplib both only', 'speedup': None, 'latency_reduction_percent': None,
               'directories': [t.get('directory') for t in tasks]}
        by_lib = {t['oplib']: t for t in tasks}
        if any(summary.get(k) for k in ('error', 'cleanup_error', 'postflight_error')):
            row['reason'] = 'run has unresolved execution, cleanup or device health errors'
        elif len(tasks) != 2 or set(by_lib) != {'nativetorch', 'flaggems'}:
            row['reason'] = 'missing or duplicate path'
        elif not all(valid_performance(t) for t in tasks):
            row['reason'] = 'both paths must pass correctness, routing and measurement gates'
        else:
            a, b = by_lib['nativetorch'], by_lib['flaggems']
            ma, mb = a['measurement'], b['measurement']
            keys = ('run_id', 'actual_image_id', 'vendor', 'device_id', 'case', 'dtype', 'seed', 'case_config', 'performance_protocol')
            mkeys = ('input_sha256', 'case_sha256', 'packages', 'vendor_runtime', 'measurement_protocol', 'timing_boundary')
            if (not a.get('run_id') or a['run_id'] != summary.get('run_id')
                    or not ma.get('input_sha256') or not ma.get('case_sha256')
                    or any(a.get(k) != b.get(k) for k in keys)
                    or any(ma.get(k) != mb.get(k) for k in mkeys)):
                row['reason'] = 'runtime, saved input/module state, source or measurement contract mismatch'
            elif any(not isinstance(m.get('median_us'), (int, float)) or not math.isfinite(m['median_us']) or m['median_us'] <= 0 for m in (ma, mb)):
                row['reason'] = 'invalid timing'
            else:
                row.update(status='comparable', speedup=ma['median_us']/mb['median_us'],
                           latency_reduction_percent=100*(1-mb['median_us']/ma['median_us']))
        rows.append(row)
    return rows


def memory_window(adapter, call, warmup):
    for _ in range(warmup):
        out = call()
        del out
    adapter.synchronize()
    baseline = adapter.memory_stats()
    adapter.reset_peak_memory_stats()
    reset = adapter.memory_stats()
    out = call()  # Keep target outputs alive while observing the peak.
    adapter.synchronize()
    peak = adapter.memory_stats()
    del out
    result = {'status': 'measured', 'protocol': 'allocator-window-v1',
              'scope': 'worker allocator only; inputs/prewarmed gradient cache in baseline; target outputs, gradients and allocator-visible workspace included; not whole-device HBM'}
    result['raw_baseline'] = baseline
    result['raw_reset'] = reset
    result['raw_peak'] = peak
    result['normalization'] = 'absolute peak = pre-reset baseline + (reported peak - post-reset current); records offsets for allocators resetting current counters too'
    result['missing'] = []
    for kind in ('allocated', 'reserved'):
        base, high = baseline.get(f'{kind}_bytes'), peak.get(f'peak_{kind}_bytes')
        origin = reset.get(f'{kind}_bytes')
        if any(not isinstance(x, int) or x < 0 for x in (base, high, origin)) or high < origin:
            result['status'] = 'partial'
            result['missing'].append(f'{kind}: invalid allocator baseline/peak; inspect raw statistics')
            continue
        result[f'baseline_{kind}_bytes'] = base
        result[f'reset_offset_{kind}_bytes'] = base - origin
        result[f'peak_{kind}_bytes'] = base + high - origin
        result[f'incremental_{kind}_bytes'] = high - origin
    return result
