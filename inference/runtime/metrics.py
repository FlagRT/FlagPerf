"""CPU FP64 descriptive metrics. No tolerance and no accuracy pass/fail decision."""
import numpy as np
from runtime.tensor_io import concatenate


def interpretation(a, b, valid, *, pairs=None, delta=None):
    """Add scale/weight context without changing legacy metrics or verdicts."""
    x, y = (a[valid], b[valid]) if pairs is None else pairs
    floor = 1e-12
    near = np.abs(x) <= floor
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        if delta is None:
            delta = np.abs(y-x)
        scale = float(np.max(np.abs(x))) if x.size else None
        error_scale = float(np.max(delta)) if delta.size else None
        reference_rms = scale * float(np.sqrt(np.mean((x/scale)**2))) if scale else scale
        error_rms = error_scale * float(np.sqrt(np.mean((delta/error_scale)**2))) if error_scale else error_scale
        normalized = error_rms/reference_rms if reference_rms else None
        quantiles = np.quantile(delta, [0.5, 0.9, 0.99], method='linear') if delta.size else [None]*3
    def finite(v):
        return float(v) if v is not None and np.isfinite(v) else None
    reason = ('no_finite_pairs' if not x.size else 'zero_reference_rms' if reference_rms == 0
              else 'overflow' if finite(normalized) is None else None)
    return {'interpretation_version': 1, 'reference_rms': finite(reference_rms),
            'rmse': finite(error_rms), 'normalized_rmse': finite(normalized),
            'normalized_rmse_undefined_reason': reason,
            'absolute_error_quantiles': {f'p{q}': finite(v) for q, v in zip((50, 90, 99), quantiles)},
            'near_zero_reference': {'absolute_cutoff': floor, 'elements': int(near.sum()),
                'finite_pairs': int(x.size), 'fraction': float(near.mean()) if x.size else None,
                'max_abs_error': finite(np.max(delta[near])) if near.any() else None},
            'interpretation_scope': 'all_elements' if valid.all() else 'finite_pairs_only'}


def metrics(reference, candidate):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        return {'status':'shape_mismatch', 'off_shape':list(a.shape), 'on_shape':list(b.shape)}
    valid = np.isfinite(a) & np.isfinite(b)
    special = {}
    for label, value in [('off',a),('on',b)]:
        special[label] = {'nan':int(np.isnan(value).sum()), 'positive_inf':int(np.isposinf(value).sum()),
                          'negative_inf':int(np.isneginf(value).sum())}
    x, y = a[valid], b[valid]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        delta = np.abs(y-x)
        relative = delta / np.maximum(np.abs(x), 1e-12)
        values = {'mse':float(np.mean(delta**2)) if x.size else None,
                  'mae':float(np.mean(delta)) if x.size else None,
                  'max_abs':float(np.max(delta)) if x.size else None,
                  'relative_mean':float(np.mean(relative)) if x.size else None,
                  'relative_max':float(np.max(relative)) if x.size else None}
        norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        reason = None
        if not valid.all(): reason = 'nonfinite_elements'
        elif not a.size: reason = 'empty_tensor'
        elif norm_a == 0 or norm_b == 0: reason = 'zero_norm'
        elif not np.isfinite(norm_a*norm_b): reason = 'norm_overflow'
        cosine = float(np.dot(a.ravel()/norm_a, b.ravel()/norm_b)) if reason is None else None
    overflow = [k for k,v in values.items() if v is not None and not np.isfinite(v)]
    for k in overflow: values[k] = None
    return dict(status='finite' if valid.all() else 'nonfinite', **values,
                cosine_similarity=cosine, cosine_undefined_reason=reason,
                valid_elements=int(valid.sum()), total_elements=int(a.size),
                statistics_scope='all_elements' if valid.all() else 'finite_pairs_only',
                reference_zero_elements=int((a == 0).sum()), special=special,
                zero_norm={'off':norm_a == 0,'on':norm_b == 0}, overflow_metrics=overflow,
                **interpretation(a, b, valid, pairs=(x, y), delta=delta))


def compare_records(off, on, levels, worst_count, boundary_levels=None):
    """Records contain sample IDs, boundary names and already unpadded arrays."""
    if set(off) != set(on):
        raise ValueError('sample ID sets differ; refusing partial pairing')
    rows, aggregate = [], {}
    for sample in off:
        if set(off[sample]) != set(on[sample]):
            raise ValueError(f'boundary sets differ for {sample}')
        for boundary, a in off[sample].items():
            if boundary_levels is None:
                from runtime.workload import legacy_specs
                level = legacy_specs([boundary])[boundary]['level']
            else:
                level = boundary_levels[boundary]
            if level not in levels:
                continue
            b = on[sample][boundary]
            result = metrics(a,b)
            rows.append({'sample_id':sample,'boundary':boundary,**result})
            if result['status'] != 'shape_mismatch':
                aggregate.setdefault(boundary, [[],[]])[0].append(np.asarray(a).ravel())
                aggregate[boundary][1].append(np.asarray(b).ravel())
    summaries = {name:metrics(concatenate(a),concatenate(b)) for name,(a,b) in aggregate.items()}
    for name, values in summaries.items():
        samples = [r for r in rows if r['boundary']==name]
        cosines = [r['cosine_similarity'] for r in samples if r.get('cosine_similarity') is not None]
        values['sample_cosine_mean'] = float(np.mean(cosines)) if cosines else None
        values['sample_cosine_min'] = min(cosines) if cosines else None
        values['cosine_scope'] = 'flattened_all_samples; sample_cosine_* are separate sample statistics'
        macro = {}
        for metric in ('mse', 'mae', 'normalized_rmse'):
            defined = [r[metric] for r in samples if r.get(metric) is not None]
            macro[metric] = {'value': float(sum(v/len(defined) for v in defined)) if defined else None,
                             'defined_samples': len(defined), 'total_samples': len(samples)}
        values['sample_equal_weight'] = macro
        total = values['valid_elements']
        values['sample_element_weights'] = [
            {'sample_id': r['sample_id'], 'valid_elements': r.get('valid_elements', 0),
             'total_elements': r.get('total_elements'),
             'weight': r.get('valid_elements', 0)/total if total else None} for r in samples]
    worst = {}
    for name in summaries:
        worst[name] = {}
        for metric in ['mse','mae','max_abs','relative_mean','relative_max','cosine_similarity']:
            ranked = [r for r in rows if r['boundary']==name and r.get(metric) is not None]
            ranked.sort(key=lambda r:r[metric],reverse=metric!='cosine_similarity')
            worst[name][metric] = [{'sample_id':r['sample_id'],'value':r[metric]} for r in ranked[:worst_count]]
    from runtime.metric_quality import classify
    return classify({'reference':'FlagGems off (not an absolute oracle)', 'thresholds':None,
            'interpretation_version':1,
            'interpretation_definitions': {
                'normalized_rmse':'sqrt(mean((candidate-reference)^2))/sqrt(mean(reference^2)); zero reference is undefined',
                'absolute_error_quantiles':'linear interpolation of finite-pair absolute errors',
                'near_zero_reference':'abs(reference)<=1e-12 on finite pairs; denominator-floor context, not an accuracy tolerance',
                'sample_equal_weight':'arithmetic mean of defined per-sample metrics; defined/total counts retained',
                'sample_element_weights':'finite-pair elements per sample / total finite-pair elements; padding already removed'},
            'relative_denominator_floor':1e-12, 'aggregate':summaries,'samples':rows,'worst':worst,
            'anomalies':[]})
