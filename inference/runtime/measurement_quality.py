"""Descriptive repeat variation and call spikes, independent of execution status."""
from collections import Counter, defaultdict
import math
from statistics import mean, median, stdev

DEFAULTS = {'repeat_cv_threshold': 0.10, 'spike_ratio_threshold': 5.0}
VERSION = 1
FIELDS = ('scope', 'side', 'rank', 'module', 'shape', 'phase', 'generation_step', 'kv_length_before')


def settings(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError('performance.variability must be a mapping')
    if set(value or {}) - set(DEFAULTS):
        raise ValueError('unknown performance.variability setting')
    result = dict(DEFAULTS, **(value or {}))
    for key, floor in [('repeat_cv_threshold', 0), ('spike_ratio_threshold', 1)]:
        v = result[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= floor:
            raise ValueError(f'performance.variability.{key} must be finite and greater than {floor}')
    return result


class Collector:
    """Retain small timing tuples, not complete profiler or worker documents."""
    def __init__(self, thresholds=None):
        self.thresholds = settings(thresholds)
        self.groups = defaultdict(list)
        self.evidence_issues = []

    def add(self, row, *, scope, side, repeat, rank='all', source=None, line=None):
        shape = str(row.get('input_shape', row.get('shape')))
        group = (scope, side, str(rank), row.get('module') if scope == 'layer' else None,
                 shape if scope != 'total' else 'all', row.get('phase') if scope == 'layer' else None,
                 row.get('generation_step') if scope == 'layer' else None,
                 row.get('kv_length_before') if scope == 'layer' else None)
        location = (row.get('cycle'), row.get('batch_index'), row.get('call_index'),
                    row.get('sequence'), source, line)
        self.groups[group].append((str(repeat), row.get('device_ns' if scope == 'layer' else 'latency_ns'), location))

    def finish(self):
        groups = []
        for key, raw in self.groups.items():
            valid, issues, seen = [], [], set()
            for repeat, ns, location in raw:
                identity = (repeat, *location[:3])
                reason = ('duplicate_invocation' if identity in seen else
                          'invalid_interval' if type(ns) not in (int, float) or not math.isfinite(ns) or ns <= 0 else None)
                seen.add(identity)
                if reason:
                    issues.append({'reason': reason, 'repeat': repeat, 'location': list(location)})
                else:
                    valid.append((repeat, ns / 1e6, location))
            repeats = defaultdict(list)
            for repeat, ms, _ in valid: repeats[repeat].append(ms)
            means = {k: mean(v) for k, v in repeats.items()}
            cv = stdev(means.values()) / mean(means.values()) if len(means) >= 2 else None
            values = [v for _, v, _ in valid]
            mid = median(values) if values else None
            maximum = max(values) if values else None
            spikes = [(r, v, loc) for r, v, loc in valid if v > self.thresholds['spike_ratio_threshold'] * mid]
            triggers = []
            if len(means) >= 3 and cv > self.thresholds['repeat_cv_threshold']:
                triggers.append('repeat_variation')
            if len(values) >= 20 and spikes:
                triggers.append('call_spikes')
            enough = len(means) >= 3 and len(values) >= 20
            status = ('invalid_evidence' if issues else 'needs_review' if triggers else
                      'no_trigger' if enough else 'insufficient_evidence')
            groups.append(dict(zip(FIELDS, key)) | {
                'observations': len(raw), 'valid_observations': len(valid),
                'independent_repeats': len(means), 'repeat_means_ms': means,
                'repeat_cv': cv, 'repeat_check_evaluable': len(means) >= 3,
                'tail_check_evaluable': len(values) >= 20,
                'median_ms': mid, 'max_ms': maximum,
                'max_over_median': maximum / mid if mid else None,
                'spike_count': len(spikes), 'spike_fraction': len(spikes) / len(values) if values else None,
                'spike_excess_ms': sum(v - mid for _, v, _ in spikes),
                'spikes': [{'repeat': r, 'duration_ms': v, 'cycle': loc[0], 'batch_index': loc[1],
                            'call_index': loc[2], 'sequence': loc[3], 'source': loc[4], 'line': loc[5]}
                           for r, v, loc in spikes],
                'issues': issues, 'triggers': triggers, 'status': status})
        counts = dict(Counter(g['status'] for g in groups))
        status = ('invalid_evidence' if self.evidence_issues or counts.get('invalid_evidence') else
                  'needs_review' if counts.get('needs_review') else
                  'insufficient_evidence' if not groups or counts.get('insufficient_evidence') else 'no_trigger')
        return {'schema_version': VERSION, 'rules_version': VERSION, 'thresholds': self.thresholds,
                'minimum_repeats': 3, 'minimum_tail_calls': 20, 'status': status,
                'group_count': len(groups), 'counts': counts, 'evidence_issues': self.evidence_issues,
                'scope': 'full captured workload; sides/ranks/phases/steps/KV lengths remain separate',
                'definitions': {'repeat_cv': 'sample standard deviation of independent repeat means / their mean',
                                'spike': 'individual call > spike_ratio_threshold * group median',
                                'verdict': 'diagnostic screening only; no significance or business performance verdict',
                                'location': 'source JSONL and line when available, otherwise repeat/cycle/batch/call'},
                'groups': groups}


def analyze(result, thresholds=None):
    collector = Collector(thresholds)
    for side, repeats in result.get('paths', {}).items():
        for label, state in repeats.items():
            if state.get('status') != 'completed':
                collector.evidence_issues.append(f'{side}/{label}: incomplete total worker')
                continue
            for row in state.get('batches', []):
                for scope in ('total', 'shape'):
                    collector.add(row, scope=scope, side=side, repeat=label)
    for side, repeats in result.get('layer', {}).get('runs', {}).get('layer_timing', {}).items():
        for label, state in repeats.items():
            for rank, worker in state.get('ranks', {'0': state}).items():
                if state.get('status') != 'completed' or worker.get('status') != 'completed':
                    collector.evidence_issues.append(f'{side}/{label}/{rank}: incomplete layer worker')
                    continue
                for row in worker.get('rows', []):
                    collector.add(row, scope='layer', side=side, repeat=label, rank=rank)
    return collector.finish()


def prepare(root, result):
    from runtime.common import write_json
    import yaml
    previous = result.get('measurement_quality', {})
    effective = root / 'effective.yaml'
    cfg = yaml.safe_load(effective.read_text()) if effective.is_file() else {}
    thresholds = previous.get('thresholds', (cfg or {}).get('performance', {}).get('variability'))
    evidence = analyze(result, thresholds)
    evidence['threshold_source'] = 'source_result' if previous.get('thresholds') else 'effective_config' if thresholds else 'versioned_defaults'
    write_json(root/'measurement-quality.json', evidence)
    result['measurement_quality'] = {k: v for k, v in evidence.items() if k != 'groups'}
    result['measurement_quality']['evidence'] = 'measurement-quality.json'
    return evidence


def report_lines(evidence):
    counts = evidence['counts']; thresholds = evidence['thresholds']
    lines = ['## 测量质量检查', '',
             f"检查 {evidence['group_count']} 个分组/路径；需复查 {counts.get('needs_review', 0)}，"
             f"证据问题 {counts.get('invalid_evidence', 0)}，样本不足 {counts.get('insufficient_evidence', 0)}。", '',
             f"重复均值 CV > {thresholds['repeat_cv_threshold']:.1%}，或单次耗时 > 中位数的 "
             f"{thresholds['spike_ratio_threshold']:g} 倍时标记；分别至少需要 3 次独立重复、20 次有效调用。"
             '样本不足时稳定性标为待复核；诊断保留执行状态、原始均值和性能倍率，由人工决定是否补跑。', '',
             '| 范围 / 路径 / rank / 层 / shape / 阶段 / step / KV | 状态 | 重复 CV | 中位数 ms | 最大 ms | 尖峰数 | 超额 ms |',
             '|---|---|---:|---:|---:|---:|---:|']
    selected = [g for g in evidence['groups'] if g['status'] in ('needs_review', 'invalid_evidence')]
    def fmt(v): return '未定义' if v is None else f'{v:.4g}'
    for g in sorted(selected, key=lambda g: g['spike_excess_ms'], reverse=True)[:10]:
        label = ' / '.join(str(g[k]) for k in FIELDS).replace('|', '\\|')
        lines.append(f"| {label} | {g['status']} | {fmt(g['repeat_cv'])} | {fmt(g['median_ms'])} | "
                     f"{fmt(g['max_ms'])} | {g['spike_count']} | {fmt(g['spike_excess_ms'])} |")
    return lines + ['', '按尖峰超额耗时展示前 10 项；完整阈值、分组及原始调用定位见 '
                    '[测量质量证据](measurement-quality.json)。异常现场保留用于复查。', '']
