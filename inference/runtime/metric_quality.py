"""Classify descriptive metric limitations separately from invalid numerics."""

CLASSIFICATION_VERSION = 2
from runtime.artifacts import Records


def classify(diff):
    """Reclassify complete saved rows without changing a numerical measurement."""
    if not isinstance(diff.get('samples'), (list, Records)):
        return diff  # Legacy counts alone cannot establish a new classification.
    anomalies, undefined = [], []
    for row in diff['samples']:
        cosine = row.get('cosine_undefined_reason')
        normalized = row.get('normalized_rmse_undefined_reason')
        invalid = (row.get('status') != 'finite' or
                   cosine not in (None, 'zero_norm') or row.get('overflow_metrics') or
                   normalized == 'overflow')
        if invalid:
            anomalies.append(row)
        reasons = {}
        if cosine == 'zero_norm' and row.get('status') == 'finite':
            reasons['cosine_similarity'] = cosine
        if normalized == 'zero_reference_rms' and row.get('status') == 'finite':
            reasons['normalized_rmse'] = normalized
        if reasons:
            undefined.append({k: row.get(k) for k in
                              ('sample_id', 'boundary', 'zero_norm', 'max_abs', 'mse', 'mae')} |
                             {'metrics': reasons, 'exact_zero_match': bool(
                                 row.get('total_elements', 0) > 0 and
                                 row.get('reference_zero_elements') == row.get('total_elements') and
                                 row.get('max_abs') == 0)})
    diff.update(classification_version=CLASSIFICATION_VERSION,
                anomalies=anomalies, undefined_metrics=undefined)
    return diff


def description(diff):
    if diff.get('classification_version') != CLASSIFICATION_VERSION:
        return f"历史异常记录数：{len(diff.get('anomalies', []))}；分类尚未复核。"
    rows = diff.get('undefined_metrics', [])
    samples = {r.get('sample_id') for r in rows}
    return (f"数值异常记录数：{len(diff.get('anomalies', []))}；指标未定义记录数：{len(rows)}"
            f"（{len(samples)} 条逻辑样本）。零范数的余弦、零参考 RMS 的归一化 RMSE 保留未定义；"
            "结合绝对误差判断两侧差异，不设精度达标阈值。")
