"""Bounded P800 copy semantics and durable evidence validation."""
import math

from base.vendors.protocol import ConfigurationError

CASES = ('interconnect-h2d:P800', 'interconnect-d2h:P800')


def contract(case):
    if case not in CASES:
        raise ConfigurationError('unknown transfer case')
    direction = case.split(':')[0].split('-')[1]
    return {'direction': direction, 'metric': 'transfer-bandwidth' if direction == 'h2d' else 'd2h-bandwidth',
            'unit': 'GB/s'}


def validate_config(config, case):
    spec = contract(case)
    for key, lower, upper in [('PAYLOAD_BYTES', 4096, 256 * 2**20), ('ITERS', 1, 200000),
                              ('WARMUP', 1, 100), ('SEED', 0, 2**31 - 1)]:
        if type(config.get(key)) is not int or not lower <= config[key] <= upper:
            raise ConfigurationError('invalid bounded transfer field: ' + key)
    if config['PAYLOAD_BYTES'] % 4 or config.get('DIRECTION') != spec['direction']:
        raise ConfigurationError('payload alignment or direction mismatch')
    if any(type(config.get(key)) is not bool for key in ('PIN_MEMORY', 'NON_BLOCKING', 'REUSE_DESTINATION')):
        raise ConfigurationError('copy mode requires explicit booleans')
    if config['REUSE_DESTINATION'] is not True or config.get('DTYPE') != 'float32':
        raise ConfigurationError('copy requires preallocated float32 destination')
    if config.get('DIST_BACKEND') != 'gloo' or config.get('IMPLEMENTATION') != 'native-xpytorch':
        raise ConfigurationError('copy requires native XPYTORCH and CPU Gloo')
    if config.get('MODE') not in ('smoke', 'calibration', 'measured', 'qualification', 'diagnostic'):
        raise ConfigurationError('invalid transfer mode')
    fault = config.get('FAULT_MODE', 'none')
    if fault not in ('none', 'cpu-wait', 'error') or (fault != 'none' and config['MODE'] != 'diagnostic'):
        raise ConfigurationError('fault injection is diagnostic-only')
    return config


def bandwidth(payload, iterations, elapsed):
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('copy time must be finite and positive')
    total = payload * iterations
    return total / elapsed / 1e9, total / elapsed / 2**30


def validate_metric(metric, correctness, context, context_hash, binding, config):
    validate_config(config, context['case'])
    spec = contract(context['case'])
    def require(condition, message):
        if not condition:
            raise RuntimeError(message)
    for record in (metric, correctness):
        require(record.get('schema_version') == 1 and record.get('status') == 'passed', 'copy artifact failed')
        require(record.get('run_id') == context['run_id'] and record.get('context_sha256') == context_hash
                and record.get('binding') == binding and record.get('rank') == 0, 'copy identity mismatch')
        for field, expected in [('direction', config['DIRECTION']), ('host_memory', 'pinned' if config['PIN_MEMORY'] else 'pageable'),
                                ('non_blocking', config['NON_BLOCKING']), ('dtype', 'float32'), ('api', 'Tensor.copy_'),
                                ('payload_bytes', config['PAYLOAD_BYTES'])]:
            require(record.get(field) == expected, 'copy semantic drift: ' + field)
    require(metric.get('world_size') == 1 and metric.get('metric') == spec['metric'] and metric.get('unit') == 'GB/s', 'copy metric contract mismatch')
    require(metric.get('case_assets_sha256') == context['case_assets_sha256'], 'copy assets drift')
    require(metric.get('iterations') == config['ITERS'] and metric.get('warmup') == config['WARMUP']
            and metric.get('mode') == config['MODE'] and metric.get('implementation') == config['IMPLEMENTATION'], 'copy config drift')
    started, finished = metric.get('started_monotonic_ns'), metric.get('finished_monotonic_ns')
    require(type(started) is int and type(finished) is int and finished > started, 'copy clock interval invalid')
    require(math.isclose((finished - started) / 1e9, metric['elapsed_seconds'], rel_tol=1e-12), 'copy time mismatch')
    gb, gib = bandwidth(config['PAYLOAD_BYTES'], config['ITERS'], metric['elapsed_seconds'])
    require(metric.get('total_bytes') == config['PAYLOAD_BYTES'] * config['ITERS'], 'copy byte count mismatch')
    require(all(type(metric.get(key)) in (int, float) and math.isfinite(metric[key]) and
                math.isclose(metric[key], expected, rel_tol=1e-12) for key, expected in [('value', gb), ('value_gib_s', gib)]), 'copy units mismatch')
    checks = correctness.get('checks', [])
    require([record.get('phase') for record in checks] == ['sentinel', 'changed-input', 'before', 'after'], 'missing copy checks')
    require(all(record.get('passed') is True and record.get('mismatch_count') == 0 and
                record.get('elements') == config['PAYLOAD_BYTES'] // 4 for record in checks), 'copy content failed')
    require(correctness.get('destination_reused') is True and correctness.get('source_reused') is True, 'copy buffer replaced')
    require(correctness.get('host_is_pinned') is config['PIN_MEMORY'], 'host allocation differs from request')
    for key in ('source', 'destination'):
        placement = correctness.get(key, {})
        expected_device = binding['framework_device_name'] if (key == 'destination') == (config['DIRECTION'] == 'h2d') else 'cpu'
        require(placement.get('device') == expected_device and placement.get('dtype') == 'float32'
                and placement.get('contiguous') is True and placement.get('numel') == config['PAYLOAD_BYTES'] // 4, 'copy placement mismatch')
    budget = correctness.get('memory_budget', {})
    for prefix in ('host', 'device'):
        require(type(budget.get(prefix + '_bytes')) is int and type(budget.get(prefix + '_available')) is int
                and 0 < budget[prefix + '_bytes'] <= budget[prefix + '_available'] // 10, 'copy memory budget exceeded')
