# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CPU FP64 descriptive metrics. No tolerance and no accuracy pass/fail decision."""
import numpy as np


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
                zero_norm={'off':norm_a == 0,'on':norm_b == 0}, overflow_metrics=overflow)


def compare_records(off, on, levels, worst_count):
    """Records contain sample IDs, boundary names and already unpadded arrays."""
    if set(off) != set(on):
        raise ValueError('sample ID sets differ; refusing partial pairing')
    rows, aggregate = [], {}
    for sample in off:
        if set(off[sample]) != set(on[sample]):
            raise ValueError(f'boundary sets differ for {sample}')
        for boundary, a in off[sample].items():
            if ('model' if boundary in ['pooled','embedding'] else 'layer') not in levels:
                continue
            b = on[sample][boundary]
            result = metrics(a,b)
            rows.append({'sample_id':sample,'boundary':boundary,**result})
            if result['status'] != 'shape_mismatch':
                aggregate.setdefault(boundary, [[],[]])[0].append(np.asarray(a).ravel())
                aggregate[boundary][1].append(np.asarray(b).ravel())
    summaries = {name:metrics(np.concatenate(a),np.concatenate(b)) for name,(a,b) in aggregate.items()}
    for name, values in summaries.items():
        cosines = [r['cosine_similarity'] for r in rows if r['boundary']==name and r.get('cosine_similarity') is not None]
        values['sample_cosine_mean'] = float(np.mean(cosines)) if cosines else None
        values['sample_cosine_min'] = min(cosines) if cosines else None
        values['cosine_scope'] = 'flattened_all_samples; sample_cosine_* are separate sample statistics'
    worst = {}
    for name in summaries:
        worst[name] = {}
        for metric in ['mse','mae','max_abs','relative_mean','relative_max','cosine_similarity']:
            ranked = [r for r in rows if r['boundary']==name and r.get(metric) is not None]
            ranked.sort(key=lambda r:r[metric],reverse=metric!='cosine_similarity')
            worst[name][metric] = [{'sample_id':r['sample_id'],'value':r[metric]} for r in ranked[:worst_count]]
    return {'reference':'FlagGems off (not an absolute oracle)', 'thresholds':None,
            'relative_denominator_floor':1e-12, 'aggregate':summaries,'samples':rows,'worst':worst,
            'anomalies':[r for r in rows if r['status']!='finite' or r.get('cosine_undefined_reason') or r.get('overflow_metrics')]}
