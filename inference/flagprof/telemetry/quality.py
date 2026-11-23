"""Offline time support and statistics. Acquisition does not call this module."""
import math


def union(intervals):
    result = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(right, result[-1][1]))
        else:
            result.append((left, right))
    return result


def subtract(interval, covered):
    left, right = interval
    for a, b in covered:
        if b <= left:
            continue
        if a >= right:
            break
        if a > left:
            yield left, min(a, right)
        left = max(left, b)
        if left >= right:
            return
    if left < right:
        yield left, right


def intersects(interval, windows):
    a, b = interval
    return any(a < d and b > c for c, d in windows)


def supports(samples, descriptor, interval_ns, conflicts=()):
    """Ordered points: ts/value, optional sample_ts/dur/interval_ns/miss.

    The source window describes resolution. Instant estimates use bounded
    left hold, and a missing observation terminates the preceding estimate.
    """
    points = sorted(samples, key=lambda p: p.get('sample_ts', p['ts']))
    for index, point in enumerate(points):
        timestamp = point.get('sample_ts', point['ts'])
        period = point.get('interval_ns') or interval_ns
        value = point.get('value')
        reason = point.get('miss')
        source_window = descriptor.get('window_ns')
        width = source_window if source_window is not None else period
        source = (timestamp-width, timestamp)
        if descriptor.get('profiler_conflict') and intersects(source, conflicts):
            reason = 'profiler_conflict'
        if value is None or reason or not math.isfinite(value):
            yield dict(point, effective_ts=timestamp, value=None, miss=reason or 'invalid', support=None)
            continue
        if descriptor.get('kind', 'instant') == 'window':
            support = source
            # Explicit missing reads delimit support, even for overlapping
            # firmware windows. We do not invent values across those gaps.
            prior = points[index-1] if index else None
            if prior is not None and (prior.get('value') is None or prior.get('miss')):
                boundary = prior.get('sample_ts', prior['ts'])
                support = (max(support[0], boundary), support[1])
        else:
            following = points[index+1].get('sample_ts', points[index+1]['ts']) if index+1 < len(points) else timestamp+period
            support = (timestamp, min(following, timestamp+period))
        yield dict(point, effective_ts=timestamp, support=support,
                   source_support=source, approximate_window=source_window is None)


def statistics(samples, descriptor, window, interval_ns, conflicts=()):
    begin, end = window
    if end <= begin:
        raise ValueError('positive statistics window required')
    candidates = list(supports(samples, descriptor, interval_ns, conflicts))
    selected = []
    missing = {}
    for point in candidates:
        if begin <= point['effective_ts'] < end and point.get('miss'):
            reason = point['miss']
            missing[reason] = missing.get(reason, 0)+1
        if point['support'] is None:
            continue
        a, b = point['support']
        if descriptor.get('kind') == 'window':
            sa, sb = point['source_support']
            if sa < begin or sb > end:
                continue
        a, b = max(begin, a), min(end, b)
        if a < b:
            selected.append((point, a, b))
    # Later readings own overlapping support. No moment has two weights.
    covered, weighted = [], []
    for point, a, b in reversed(selected):
        pieces = list(subtract((a, b), covered))
        weight = sum(y-x for x, y in pieces)
        if weight:
            weighted.append((point['value'], weight))
            covered = union(covered+pieces)
    effective = sum(w for _, w in weighted)
    minimum = (descriptor.get('window_ns') or (interval_ns if descriptor.get('kind') == 'window' else 0))+interval_ns
    result = dict(status='available', samples=len(weighted), coverage=effective/(end-begin),
                  covered_ns=effective, duration_ns=end-begin, miss=missing,
                  mean=None, maximum=None, p95=None, start_value=None, jump=None,
                  approximate_window=descriptor.get('window_ns') is None,
                  _weighted=weighted, _intervals=covered)
    if len(weighted) < 2 or end-begin < minimum or not effective:
        result['status'] = 'insufficient_resolution'
        return result
    result.update(mean=sum(v*w for v, w in weighted)/effective,
                  maximum=max(v for v, _ in weighted),
                  maximum_kind='window_mean_maximum' if descriptor.get('kind') == 'window' else 'sample_maximum',
                  start_value=selected[0][0]['value'])
    result['jump'] = result['maximum']-result['start_value']
    if len(weighted) >= 20:
        result['p95'] = quantile(weighted, .95)
    return result


def quantile(weighted, fraction):
    limit = sum(w for _, w in weighted)*fraction
    total = 0
    for value, weight in sorted(weighted):
        total += weight
        if total >= limit:
            return value
    return None


def aggregate(rows):
    """Combine disjoint instances using their measured weights, never p95s."""
    valid = [row for row in rows if row['status'] == 'available']
    weighted = [item for row in valid for item in row['_weighted']]
    duration = sum(row['duration_ns'] for row in rows)
    covered = sum(row['covered_ns'] for row in valid)
    samples = sum(row['samples'] for row in valid)
    return dict(status='available' if weighted else 'insufficient_resolution',
                instances=len(rows), samples=samples, duration_ns=duration,
                covered_ns=covered, coverage=covered/duration if duration else 0,
                mean=sum(v*w for v, w in weighted)/covered if covered else None,
                maximum=max((v for v, _ in weighted), default=None),
                p95=quantile(weighted, .95) if samples >= 20 else None)


def energy(points, window, *, cumulative=False, max_gap_ns=None):
    """Integrate observed adjacent pairs, clip/interpolate boundaries, keep gaps."""
    begin, end = window
    if end <= begin:
        raise ValueError('positive energy window required')
    points = sorted(points, key=lambda p: p['ts'])
    joules = 0.0
    covered = []
    resets = 0
    for first, second in zip(points, points[1:]):
        a, b = first['ts'], second['ts']
        x, y = first.get('value'), second.get('value')
        if (x is None or y is None or first.get('miss') or second.get('miss') or
                not math.isfinite(x) or not math.isfinite(y) or b <= a or
                (max_gap_ns and b-a > max_gap_ns)):
            continue
        if cumulative and y < x:
            resets += 1
            continue
        left, right = max(a, begin), min(b, end)
        if left >= right:
            continue
        if cumulative:
            joules += (y-x)*(right-left)/(b-a)
        else:
            lv = x+(y-x)*(left-a)/(b-a)
            rv = x+(y-x)*(right-a)/(b-a)
            joules += (lv+rv)*.5*(right-left)/1e9
        covered.append((left, right))
    duration = sum(b-a for a, b in union(covered))
    return dict(joules=joules if duration else None, coverage=duration/(end-begin),
                method='energy_counter_difference' if cumulative else 'power_trapezoid_estimate', resets=resets)


def residual(framework, device_points, *, baseline, tolerance, interval_ns, preexisting=False):
    if preexisting:
        return dict(status='unavailable', reason='preexisting_usage')
    if not framework.get('observed', False) or framework.get('value') is None:
        return dict(status='unavailable', reason='framework_unobserved')
    valid = [p for p in device_points if p.get('value') is not None and not p.get('miss')]
    if not valid or baseline is None:
        return dict(status='unavailable', reason='baseline_or_device_unavailable')
    point = min(valid, key=lambda p: abs(p.get('sample_ts',p['ts'])-framework['ts']))
    difference = abs(point.get('sample_ts',point['ts'])-framework['ts'])
    if difference > (point.get('interval_ns') or interval_ns)/2:
        return dict(status='unavailable', reason='unpaired')
    return dict(status='available', bytes=point['value']-baseline-framework['value'],
                delta_ns=difference, baseline=baseline, baseline_tolerance=tolerance,
                evidence='time_candidate', label='扣除基线后的分配器外显存估算')


def violation(points, window):
    """A positive cumulative delta proves an occurrence only inside its bracket.

    Unlike energy interpolation, a partly overlapping bracket cannot locate a
    thermal/power restriction in the requested interval.
    """
    begin,end=window
    total=0;covered=[];resets=0
    for first,second in zip(points,points[1:]):
        a=first.get('sample_ts',first['ts']);b=second.get('sample_ts',second['ts'])
        x=first.get('value');y=second.get('value')
        if not begin<=a<b<=end or x is None or y is None or first.get('miss') or second.get('miss'):
            continue
        if y<x:resets+=1;continue
        total+=y-x;covered.append((a,b))
    return dict(restricted_ns=total if covered else None,observed_window=union(covered),resets=resets,
                evidence='mechanism' if total>0 else 'time_candidate',
                conclusion='区间内发生限制' if total>0 else '未观察到区间累计增加' if covered else '区间内累计观测不足')


def energy_per_token(value, tokens, members, selected, preexisting):
    value.update(generated_tokens=tokens,includes_other_chips=bool(set(members)-set(selected)),
                 preexisting_usage=preexisting,joules_per_token=None)
    if value['includes_other_chips']:
        value['token_reason']='power_domain_includes_other_chips'
    elif not tokens:
        value['token_reason']='token_denominator_unavailable'
    elif value.get('joules') is None or value.get('coverage',0)<1-1e-9:
        value['token_reason']='incomplete_energy_coverage'
    else:
        value['joules_per_token']=value['joules']/tokens
    return value


def low_frequency(points, nominal, interval_ns):
    """At least two consecutive readings below 95% of observed maximum."""
    maxima=[p for p in supports(nominal,{},interval_ns) if p['support'] is not None]
    result=[];run=[];cursor=0
    def finish():
        if len(run)>=2:
            result.append(dict(window=[run[0][0],run[-1][1]],samples=len(run),
                frequency_min_hz=min(p[2] for p in run),frequency_max_hz=max(p[2] for p in run),
                evidence='time_candidate'))
        run.clear()
    for point in supports(points,{},interval_ns):
        if point['support'] is None:
            finish();continue
        a,b=point['support']
        while cursor<len(maxima) and maxima[cursor]['support'][1]<=a:cursor+=1
        maximum=maxima[cursor] if cursor<len(maxima) else None
        if maximum is None or not maximum['support'][0]<=a<maximum['support'][1] or point['value']>=.95*maximum['value']:
            finish();continue
        b=min(b,maximum['support'][1])
        if run and run[-1][1]!=a:finish()
        run.append((a,b,point['value']))
    finish()
    return result
