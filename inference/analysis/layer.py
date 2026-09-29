# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Offline layer accounting. No accelerator imports."""
from collections import defaultdict
import math
from runtime.performance import percentile


def ranks(state):
    return state.get('ranks', {'0': state})


def key(row):
    return (row['repeat'], row['cycle'], row['batch_index'], row['rank'], row['module'], row['call_index'])


def workload(row):
    return tuple(str(row.get(k)) for k in ('samples', 'tokens', 'input_shape', 'sample_ids'))


def statistics(rows, field):
    good = [r for r in rows if isinstance(r.get(field), (int, float)) and math.isfinite(r[field]) and r[field] > 0]
    times = [r[field] for r in good]
    if not times: return None
    total = sum(times)
    return {'calls': len(times), 'time_ns': total, 'mean_ms': total/len(times)/1e6,
            'min_ms': min(times)/1e6, 'max_ms': max(times)/1e6,
            **{f'p{p}_ms': percentile(times, p)/1e6 for p in (50, 90, 99)},
            'calls_per_second': len(times)*1e9/total,
            'samples_per_second': sum(r['samples'] for r in good)*1e9/total,
            'tokens_per_second': sum(r['tokens'] for r in good)*1e9/total}


def summarize_layer(layer):
    records, groups, issues = {}, defaultdict(lambda: defaultdict(list)), []
    for side, repeats in layer['runs'].get('layer_timing', {}).items():
        records[side] = {}
        for label, state in repeats.items():
            if state.get('status') != 'completed':
                issues.append(f'{side}/{label}: timing worker incomplete')
                continue
            for rank, value in ranks(state).items():
                expected = value.get('expected_batches')
                if len(value.get('batches', [])) != expected:
                    issues.append(f'{side}/{label}/{rank}: batch count mismatch')
                modules = {m['module'] for m in value.get('modules', [])}
                batches = {(r['repeat'], r['cycle'], r['batch_index']): r for r in value.get('batches', [])}
                seen = defaultdict(set)
                for row in value.get('rows', []):
                    k = key(row)
                    if k in records[side]:
                        issues.append('duplicate layer invocation: '+str(k)); continue
                    if row.get('device_ns') is None or row.get('device_ns', 0) <= 0:
                        issues.append('invalid device interval: '+str(k))
                    records[side][k] = row
                    bkey = (row['repeat'], row['cycle'], row['batch_index'])
                    seen[bkey].add(row['module'])
                    if bkey not in batches or workload(row) != workload(batches[bkey]):
                        issues.append('layer/batch workload mismatch: '+str(k))
                    groups[(row['module'], str(row['rank']), str(row['input_shape']))][side].append(row)
                if set(seen) != set(batches) or any(v != modules for v in seen.values()):
                    issues.append(f'{side}/{label}/{rank}: selected-module coverage incomplete')
    paired = set(records) == {'off', 'on'} and not issues
    if paired:
        paired = records['off'].keys() == records['on'].keys() and all(
            workload(row) == workload(records['on'][k]) for k, row in records['off'].items())
        if not paired: issues.append('off/on layer workload or invocation mismatch')
    output = []
    for (module, rank, shape), sides in groups.items():
        row = {'module': module, 'rank': rank, 'shape': shape, 'paths': {}}
        for side in ('off', 'on'):
            if side not in sides: continue
            values = sides[side]
            row['paths'][side] = {'device': statistics(values, 'device_ns'), 'host': statistics(values, 'host_ns'),
                                 'observed_calls': len(values)}
        if paired and all(row['paths'].get(s, {}).get('device') for s in ('off', 'on')):
            row['on_over_off_device_time'] = row['paths']['on']['device']['time_ns']/row['paths']['off']['device']['time_ns']
        output.append(row)
    imbalance = []
    for side, entries in records.items():
        by_call = defaultdict(list)
        for row in entries.values():
            if row.get('device_ns', 0) and row['device_ns'] > 0:
                by_call[(row['repeat'], row['cycle'], row['batch_index'], row['module'], row['call_index'])].append(row)
        expected_ranks = layer.get('world_size', 1)
        if expected_ranks > 1:
            for k, values in by_call.items():
                if len({r['rank'] for r in values}) != expected_ranks: continue
                times = [r['device_ns'] for r in values]
                imbalance.append({'side': side, 'repeat': k[0], 'cycle': k[1], 'batch_index': k[2],
                    'module': k[3], 'call_index': k[4], 'max_rank_window_ns': max(times),
                    'min_rank_window_ns': min(times), 'spread_ns': max(times)-min(times),
                    'scope': 'rank-local stream windows; not a global layer latency'})
    return {'status': 'completed' if output and not issues else 'partial', 'paired': paired,
            'groups': output, 'issues': issues, 'rank_imbalance': imbalance}


def overhead(result, by_shape=False):
    rows = []
    for side, repeats in result['layer']['runs'].get('layer_timing', {}).items():
        for label, state in repeats.items():
            baseline = result['paths'].get(side, {}).get(label, {})
            for rank, value in ranks(state).items():
                reference = ranks(baseline).get(rank, {})
                left = {(r['cycle'], r['batch_index']): r for r in reference.get('batches', [])}
                right = {(r['cycle'], r['batch_index']): r for r in value.get('batches', [])}
                valid = (state.get('status') == 'completed' and baseline.get('status') == 'completed'
                         and reference.get('status') == 'completed' and value.get('status') == 'completed'
                         and len(left) == len(reference.get('batches', [])) and len(right) == len(value.get('batches', []))
                         and all(isinstance(r.get('latency_ns'), (int, float)) and math.isfinite(r['latency_ns']) and r['latency_ns'] > 0
                                 for r in [*left.values(), *right.values()]))
                if not valid or not left or left.keys() != right.keys() or any(workload(r) != workload(right[k]) for k, r in left.items()):
                    rows.append({'side': side, 'repeat': label, 'rank': rank, 'status': 'unpaired'})
                    continue
                shapes = sorted({str(r.get('input_shape')) for r in left.values()}) if by_shape else [None]
                for shape in shapes:
                    keys = [k for k, r in left.items() if shape is None or str(r.get('input_shape')) == shape]
                    a, b = sum(left[k]['latency_ns'] for k in keys), sum(right[k]['latency_ns'] for k in keys)
                    rows.append({'side': side, 'repeat': label, 'rank': rank, 'status': 'paired' if a > 0 else 'invalid',
                                 **({'shape': shape} if by_shape else {}), 'batches': len(keys),
                                 'baseline_ns': a, 'instrumented_ns': b, 'instrumented_over_baseline': b/a if a > 0 else None})
    return rows
