"""Bounded P800 Day 6 contracts, formulas and artifact validation."""
import math
import re

from base.vendors.protocol import ConfigurationError


MEMORY_CASES = ('main_memory-bandwidth:P800', 'main_memory-capacity:P800')
COMMUNICATION_CASES = ('interconnect-P2P_intraserver:P800', 'interconnect-MPI_intraserver:P800')
CASES = MEMORY_CASES + COMMUNICATION_CASES
INTERSERVER_CASES = ('interconnect-P2P_interserver:P800', 'interconnect-MPI_interserver:P800')

METRIC_NAMES = {
    'main_memory-bandwidth:P800': 'device-memory-bandwidth',
    'main_memory-capacity:P800': 'main_memory-capacity',
    'interconnect-MPI_intraserver:P800': 'allreduce-algbw',
    'interconnect-P2P_intraserver:P800': 'p2p-one-way-bandwidth',
}


def _int(c, k, lo, hi):
    if type(c.get(k)) is not int or not lo <= c[k] <= hi:
        raise ConfigurationError('invalid bounded day-six field: ' + k)


def validate_config(c, case):
    if case == 'main_memory-bandwidth:P800':
        _int(c, 'Melements', 1, 1024); _int(c, 'WARMUP', 1, 100); _int(c, 'ITERS', 1, 200000)
        if c.get('DTYPE') != 'float32' or c.get('DIST_BACKEND') != 'gloo':
            raise ConfigurationError('invalid memory bandwidth contract')
    elif case == 'main_memory-capacity:P800':
        _int(c, 'INITSIZE', 1, 98304); _int(c, 'MIN_MIB', 1, 1024)
        if c.get('DTYPE') != 'float32' or c.get('DIST_BACKEND') != 'gloo' \
                or c.get('BOUND_REQUEST_BY_FREE_MEMORY') is not True or c.get('POST_TEST_WAIT_SECONDS') != 0:
            raise ConfigurationError('invalid memory capacity contract')
    elif case in COMMUNICATION_CASES:
        _int(c, 'Melements', 1, 256); _int(c, 'WARMUP', 1, 100); _int(c, 'ITERS', 1, 200000)
        if c.get('DTYPE') != 'float32' or c.get('DIST_BACKEND') != 'cpu:gloo,cuda:flagcx':
            raise ConfigurationError('invalid FlagCX contract')
    else:
        raise ConfigurationError('unknown P800 Day 6 case')
    if c.get('IMPLEMENTATION') != 'native-xpytorch':
        raise ConfigurationError('native XPYTORCH required')
    return c


def expected_ranks(case, world_size=None):
    if case in COMMUNICATION_CASES:
        return world_size or 2
    return 1


def payload_bytes(melements):
    """Melements counts 2^20 float32 elements, so 1024 means a 4 GiB payload."""
    return melements * (1 << 20) * 4


def bandwidth(total_bytes, elapsed):
    if elapsed <= 0 or not math.isfinite(elapsed):
        raise ValueError('invalid elapsed time')
    return total_bytes / elapsed / 1e9, total_bytes / elapsed / 2 ** 30


def memory_bandwidth(payload, iterations, elapsed):
    """One clone/copy moves the payload once out of the source and once into
    the destination, so timed traffic is 2 x payload per iteration."""
    return bandwidth(payload * 2 * iterations, elapsed)


def allreduce_bandwidth(payload, iterations, elapsed, world_size):
    """Whole-window traffic over the window time.

    Every timed iteration moves one payload per rank, so the algorithmic
    bandwidth is (payload x iterations) / elapsed, matching the repository
    reference (`datasize = ITERS * message / elapsed`) and the NCCL
    convention; busbw applies 2 x (n-1) / n and nothing else.
    """
    algbw_gb, algbw_gib = bandwidth(payload * iterations, elapsed)
    factor = 2 * (world_size - 1) / world_size
    return algbw_gb, algbw_gib, algbw_gb * factor, algbw_gib * factor


def p2p_one_way_bandwidth(payload, iterations, elapsed):
    """Single-direction transfer; never doubled and never summed over ranks."""
    return bandwidth(payload * iterations, elapsed)


def allreduce_timed_coefficient(rank, world_size):
    """Exact zero sum for every supported world size, including odd sizes."""
    if type(world_size) is not int or not 2 <= world_size <= 8 or type(rank) is not int or rank not in range(world_size):
        raise ValueError('invalid allreduce rank/world size')
    return world_size - 1 if rank == 0 else -1


def capacity_release_verified(free_before_mib, free_after_bytes):
    """Preserve the 95% recovery gate while comparing the same byte unit."""
    return (type(free_before_mib) is int and free_before_mib > 0
            and type(free_after_bytes) is int and free_after_bytes >= 0
            and free_after_bytes * 100 >= free_before_mib * (1 << 20) * 95)


def _common_metric_checks(metric, correctness, context, context_hash, binding, config, case, world_size):
    for record in (metric, correctness):
        if not (record.get('schema_version') == 1 and record.get('status') == 'passed'):
            raise RuntimeError('rank artifact failed: ' + case)
        if record.get('run_id') != context['run_id'] or record.get('context_sha256') != context_hash:
            raise RuntimeError('rank artifact context mismatch')
    rank = metric.get('rank')
    if rank not in range(world_size) or correctness.get('rank') != rank:
        raise RuntimeError('rank artifact rank mismatch')
    if metric.get('binding') != binding or correctness.get('binding') != binding:
        raise RuntimeError('rank artifact binding mismatch')
    if metric.get('world_size') != world_size:
        raise RuntimeError('rank artifact world size mismatch')
    if metric.get('case_assets_sha256') != context['case_assets_sha256']:
        raise RuntimeError('rank case identity mismatch')
    if metric.get('mode') != config['MODE'] or metric.get('implementation') != config['IMPLEMENTATION']:
        raise RuntimeError('rank config drift')
    for field, key in (('ITERS', 'iterations'), ('WARMUP', 'warmup')):
        if field in config and metric.get(key) != config[field]:
            raise RuntimeError('rank config drift: ' + field)
    started, finished = metric.get('started_monotonic_ns'), metric.get('finished_monotonic_ns')
    if type(started) is not int or type(finished) is not int or finished <= started:
        raise RuntimeError('invalid rank clock interval')
    elapsed = metric.get('elapsed_seconds')
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError('invalid rank elapsed seconds')
    if not math.isclose((finished - started) / 1e9, elapsed, rel_tol=1e-12):
        raise RuntimeError('rank elapsed interval differs')
    return rank, elapsed


def validate_metric(metric, correctness, context, context_hash, binding, config, case):
    """Recompute every published number from the artifact's own raw bytes."""
    world_size = (context.get('nproc_per_node') or expected_ranks(case)) if case in COMMUNICATION_CASES else 1
    rank, elapsed = _common_metric_checks(metric, correctness, context, context_hash, binding, config, case, world_size)
    name = METRIC_NAMES[case]
    unit = 'GB' if case == 'main_memory-capacity:P800' else 'GB/s'
    if metric.get('metric') != name or metric.get('unit') != unit:
        raise RuntimeError('unexpected rank metric contract')
    value = metric.get('value')
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise RuntimeError('invalid rank metric value')

    # A top-level "passed" cannot substitute for the actual content checks.
    checks = correctness.get('checks', [])
    if case != 'main_memory-capacity:P800':
        phases = (['sentinel', 'clone-equivalence', 'changed-input', 'after'] if case == 'main_memory-bandwidth:P800'
                  else ['cold', 'second', 'post-loop'] if case == 'interconnect-MPI_intraserver:P800'
                  else ['sent-verified-repeats'] if rank == 0
                  else [f'recv-{i}' for i in range(config['WARMUP'])] + ['post-loop'])
        if [c.get('phase') for c in checks] != phases or any(c.get('passed') is not True for c in checks):
            raise RuntimeError('missing or failed content correctness phases')
        if case == 'interconnect-P2P_intraserver:P800' and rank == 0:
            if checks[0].get('repeats') != config['WARMUP'] or checks[0].get('sequence_range') != [0, config['WARMUP'] - 1]:
                raise RuntimeError('P2P sender sequence coverage mismatch')
        elif any(c.get('elements') != payload_bytes(config['Melements']) // 4 or c.get('mismatch_count') != 0
                 or c.get('first_mismatch') is not None for c in checks):
            raise RuntimeError('content correctness coverage/mismatch evidence invalid')
        if case == 'main_memory-bandwidth:P800' and (correctness.get('source_reused') is not True
                or correctness.get('destination_reused') is not True):
            raise RuntimeError('preallocated copy buffers were not reused')
        if case == 'interconnect-MPI_intraserver:P800' and world_size > 2 and correctness.get('timed_input_coefficient') != allreduce_timed_coefficient(rank, world_size):
            raise RuntimeError('multi-rank timed inputs do not use verified zero-sum coefficients')

    if case == 'main_memory-bandwidth:P800':
        payload = payload_bytes(config['Melements'])
        expected_gb, expected_gib = memory_bandwidth(payload, config['ITERS'], elapsed)
        if metric.get('total_bytes') != 2 * payload * config['ITERS']:
            raise RuntimeError('memory bandwidth total bytes differ from 2 x payload x iterations')
        if not math.isclose(value, expected_gb, rel_tol=1e-9) or \
                not math.isclose(metric.get('value_gib_s', 0), expected_gib, rel_tol=1e-9):
            raise RuntimeError('memory bandwidth cannot be recomputed from bytes and elapsed')
    elif case == 'main_memory-capacity:P800':
        held = metric.get('held_mib')
        if type(held) is not int or held < config['MIN_MIB']:
            raise RuntimeError('capacity held MiB missing or below the search granularity')
        if not math.isclose(value, held * (1 << 20) / 1e9, rel_tol=1e-9) or \
                not math.isclose(metric.get('value_gib_s', 0), held / 1024, rel_tol=1e-9):
            raise RuntimeError('capacity value cannot be recomputed from held MiB')
        if correctness.get('released') is not True or correctness.get('release_verified') is not True:
            raise RuntimeError('capacity tensors were not verifiably released')
        after = correctness.get('free_after_bytes', correctness.get('free_after_mib', -1) * (1 << 20))
        if not capacity_release_verified(correctness.get('free_before_mib'), after):
            raise RuntimeError('capacity release bytes do not meet the recovery gate')
        held_trail = [r for r in correctness.get('trail', []) if r.get('stage') == 'held']
        if not held_trail or sum(r.get('request_mib', 0) for r in held_trail) != held or correctness.get('held_mib') != held:
            raise RuntimeError('capacity result differs from held allocation trail')
    elif case == 'interconnect-MPI_intraserver:P800':
        payload = payload_bytes(config['Melements'])
        algbw, algbw_gib, busbw, busbw_gib = allreduce_bandwidth(payload, config['ITERS'], elapsed, world_size)
        if metric.get('message_bytes') != payload:
            raise RuntimeError('allreduce message bytes differ from config')
        if metric.get('total_bytes') != payload * config['ITERS']:
            raise RuntimeError('allreduce total bytes differ from message x iterations')
        if not math.isclose(value, algbw, rel_tol=1e-9) or \
                not math.isclose(metric.get('value_gib_s', 0), algbw_gib, rel_tol=1e-9) or \
                not math.isclose(metric.get('busbw_gb_s', 0), busbw, rel_tol=1e-9) or \
                not math.isclose(metric.get('busbw_gib_s', 0), busbw_gib, rel_tol=1e-9):
            raise RuntimeError('allreduce algbw/busbw cannot be recomputed')
        if world_size == 2 and not math.isclose(metric['busbw_gb_s'], value, rel_tol=1e-12):
            raise RuntimeError('two-rank busbw must equal algbw without an extra multiplier')
    else:
        payload = payload_bytes(config['Melements'])
        expected_gb, expected_gib = p2p_one_way_bandwidth(payload, config['ITERS'], elapsed)
        if metric.get('total_bytes') != payload * config['ITERS']:
            raise RuntimeError('p2p total bytes differ from payload x iterations')
        if not math.isclose(value, expected_gb, rel_tol=1e-9) or \
                not math.isclose(metric.get('value_gib_s', 0), expected_gib, rel_tol=1e-9):
            raise RuntimeError('p2p one-way bandwidth cannot be recomputed')
        if metric.get('direction') != 'rank0-to-rank1':
            raise RuntimeError('p2p direction missing or unexpected')
    return metric


STDOUT_UNITS = {
    'main_memory-bandwidth:P800': [('device-memory-bandwidth', 'GB/s'), ('device-memory-bandwidth', 'GiB/s')],
    'main_memory-capacity:P800': [('main_memory-capacity', 'GB'), ('main_memory-capacity', 'GiB')],
    'interconnect-MPI_intraserver:P800': [('allreduce-algbw', 'GB/s'), ('allreduce-busbw', 'GB/s')],
    'interconnect-P2P_intraserver:P800': [('p2p-one-way-bandwidth', 'GB/s'), ('p2p-one-way-bandwidth', 'GiB/s')],
}

STDOUT_VALUE_FIELDS = {
    'GB/s': 'value', 'GiB/s': 'value_gib_s', 'GB': 'value', 'GiB': 'value_gib_s',
    'busbw': 'busbw_gb_s',
}


STDOUT_LINE = re.compile(r"\[FlagPerf Result\]Rank (\d+)'s ([^=\n]+)=([0-9.]+)([A-Za-z/]+)")


def parse_stdout_lines(text):
    """Scan result lines anywhere in the text.

    Every rank writes to the same container stdout, so a lost line boundary
    between two concurrent writes must not hide a metric from the cross-check.
    """
    return [{'rank': int(match.group(1)), 'metric': match.group(2),
             'value': float(match.group(3)), 'unit': match.group(4)}
            for match in STDOUT_LINE.finditer(text)]


def expected_stdout_pairs(case, metric):
    if case == 'interconnect-MPI_intraserver:P800':
        return [(METRIC_NAMES[case], 'GB/s', metric['value']), ('allreduce-busbw', 'GB/s', metric['busbw_gb_s'])]
    unit_value = {'GB/s': metric['value'], 'GiB/s': metric['value_gib_s'], 'GB': metric['value'], 'GiB': metric['value_gib_s']}
    return [(METRIC_NAMES[case], unit, unit_value[unit]) for _, unit in STDOUT_UNITS[case]]
