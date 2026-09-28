"""P800 Day 6 memory and two-rank communication drivers.

Every function follows the verified-copy pattern: identity first, content
correctness before and after the timed window, synchronized wall-clock timing,
durable per-rank artifacts, and no reset/retry that could hide a failure.
"""
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from benchmarks.day6_contract import (
    allreduce_bandwidth,
    expected_ranks,
    memory_bandwidth,
    payload_bytes,
    p2p_one_way_bandwidth,
    validate_config,
)
from .events import benchmark_measurement_finish, benchmark_measurement_start
from .fp32 import save


def _require_scope(case, rank, world_size, local_rank):
    validate_scope = expected_ranks(case, world_size)
    if world_size != validate_scope or rank != local_rank or rank not in range(world_size):
        raise RuntimeError('day-six case process scope differs from the container binding')


def _identity(driver, config, case, rank, world_size, local_rank):
    _require_scope(case, rank, world_size, local_rank)
    driver.set_device(local_rank)
    target = driver.device(local_rank)
    identity = driver.evidence()
    # evidence() reports the first binding; every rank must record its own.
    identity['binding'] = driver.binding(local_rank)
    output = Path(os.environ['FLAGPERF_BENCHMARK_OUTPUT'])
    return target, identity, output


def _context():
    return json.loads(Path(os.environ['FLAGPERF_HOST_CONTEXT']).read_text())


def _fault_point(config):
    fault = config.get('FAULT_MODE', 'none')
    if fault == 'error':
        raise RuntimeError('controlled day-six error after correctness')
    if fault == 'cpu-wait':
        time.sleep(_context()['timeout'] + 30)
        raise RuntimeError('watchdog failed to stop day-six cpu wait')


def _checksum(tensor):
    observed = tensor.detach().to(torch.int64)
    return {'sum': int(observed.sum()), 'first': int(observed[0]), 'last': int(observed[-1]),
            'finite': bool(torch.isfinite(tensor).all())}


def _check_content(actual, reference, phase):
    observed = actual.detach().cpu()
    mismatch = observed != reference
    count = int(mismatch.sum())
    first = int(mismatch.nonzero()[0]) if count else None
    return {'phase': phase, 'passed': count == 0, 'mismatch_count': count,
            'first_mismatch': first, 'elements': reference.numel()}


def _placement(tensor):
    return {'device': str(tensor.device), 'dtype': str(tensor.dtype).removeprefix('torch.'),
            'numel': tensor.numel(), 'contiguous': tensor.is_contiguous(), 'data_ptr': tensor.data_ptr()}


def _num(value):
    """Format one published number with enough significant digits to round-trip.

    A 1 MiB message reports about 0.0006 GB/s, and %.6f keeps only one
    significant digit there, which breaks the stdout cross-check against the
    durable artifact.
    """
    return f'{value:.9g}'


def _emit(line):
    """Write one result line atomically.

    Two ranks share the container stdout; a single write below PIPE_BUF keeps
    the line intact so the framework parser never sees merged lines.
    """
    os.write(1, (line + '\n').encode())


def _metric_common(identity, case, config, rank, world_size, metric, unit, value):
    return {'schema_version': 1, **identity, 'case': case, 'rank': rank, 'world_size': world_size,
            'status': 'passed', 'metric': metric, 'unit': unit, 'value': value, 'mode': config['MODE'],
            'iterations': config.get('ITERS'), 'warmup': config.get('WARMUP'), 'dtype': config['DTYPE'],
            'implementation': config['IMPLEMENTATION'], 'backend': config['DIST_BACKEND'],
            'case_assets_sha256': _context()['case_assets_sha256']}


def run_memory_bandwidth(driver, config, rank, world_size, local_rank):
    """Device-to-device copy bandwidth: one read plus one write per iteration."""
    case = 'main_memory-bandwidth:P800'
    validate_config(config, case)
    target, identity, output = _identity(driver, config, case, rank, world_size, local_rank)
    payload = payload_bytes(config['Melements'])
    record = {'schema_version': 1, **identity, 'case': case, 'rank': rank, 'status': 'failed',
              'checks': [], 'semantics': {'api': 'Tensor.copy_ on a preallocated destination',
                                          'clone_equivalence': 'verified outside the timed window',
                                          'payload_bytes': payload, 'bytes_per_iteration': 2 * payload}}
    try:
        free_bytes, total_bytes = driver.memory_info()
        record['memory_budget'] = {'device_bytes': payload * 2, 'device_available': int(free_bytes),
                                   'device_total': int(total_bytes), 'scope': 'source plus destination'}
        if payload * 2 > free_bytes // 2:
            raise RuntimeError('copy pair exceeds half of available device memory')
        base = torch.arange(payload // 4, dtype=torch.float32).remainder_(251).add_(config['SEED'] % 17)
        alternate = base + 1
        source = base.to(target)
        destination = torch.empty_like(source)
        record.update(source=_placement(source), destination=_placement(destination))
        source_pointer, destination_pointer = source.data_ptr(), destination.data_ptr()
        for phase, expected in [('sentinel', base), ('clone-equivalence', base), ('changed-input', alternate)]:
            destination.fill_(-999)
            if phase == 'clone-equivalence':
                cloned = source.clone()
                driver.synchronize()
                checked = _check_content(cloned, expected, phase)
                del cloned
            else:
                if phase == 'changed-input':
                    source.copy_(alternate.to(target))
                driver.synchronize()
                destination.copy_(source)
                driver.synchronize()
                checked = _check_content(destination, expected, phase)
            record['checks'].append(checked)
            if not checked['passed']:
                raise RuntimeError('memory copy content mismatch: ' + phase)
        if phase == 'changed-input':
            source.copy_(base.to(target))
            driver.synchronize()
        _fault_point(config)
        for _ in range(config['WARMUP']):
            destination.copy_(source)
        driver.synchronize()
        token = benchmark_measurement_start()
        started = time.perf_counter_ns()
        for _ in range(config['ITERS']):
            destination.copy_(source)
        driver.synchronize()
        finished = time.perf_counter_ns()
        benchmark_measurement_finish(token)
        elapsed = (finished - started) / 1e9
        gb, gib = memory_bandwidth(payload, config['ITERS'], elapsed)
        record['checks'].append(_check_content(destination, base, 'after'))
        record.update(source_reused=source.data_ptr() == source_pointer,
                      destination_reused=destination.data_ptr() == destination_pointer)
        if not record['checks'][-1]['passed'] or not record['source_reused'] or not record['destination_reused']:
            raise RuntimeError('post-measurement content or buffer reuse failed')
        record['status'] = 'passed'
        metric = _metric_common(identity, case, config, rank, world_size, 'device-memory-bandwidth', 'GB/s', gb)
        metric.update(value_gib_s=gib, elapsed_seconds=elapsed,
                      started_monotonic_ns=started, finished_monotonic_ns=finished,
                      total_bytes=2 * payload * config['ITERS'],
                      timer='perf_counter_ns with full target synchronization',
                      scope='device copy bandwidth; 2 x payload per iteration; allocation and checks outside the window',
                      qualification_duration_passed=elapsed >= 15)
        save(output / 'metric-rank-0.json', metric)
        _emit(f"[FlagPerf Result]Rank {rank}'s device-memory-bandwidth={_num(gb)}GB/s")
        _emit(f"[FlagPerf Result]Rank {rank}'s device-memory-bandwidth={_num(gib)}GiB/s")
        return gb, gib
    except Exception as exc:
        record.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        save(output / 'correctness-rank-0.json', record)


def run_memory_capacity(driver, config, rank, world_size, local_rank):
    """Report only allocations actually held; release them before exiting."""
    case = 'main_memory-capacity:P800'
    validate_config(config, case)
    target, identity, output = _identity(driver, config, case, rank, world_size, local_rank)
    minimum = config['MIN_MIB']
    record = {'schema_version': 1, **identity, 'case': case, 'rank': rank, 'status': 'failed',
              'trail': [], 'oom_errors': [], 'semantics': {
                  'result': 'sum of device MiB actually held at the search peak',
                  'bound_request_by_free_memory': bool(config['BOUND_REQUEST_BY_FREE_MEMORY']),
                  'granularity_mib': minimum}}
    held = []
    total_mib = 0
    token = benchmark_measurement_start()
    started = time.perf_counter_ns()
    try:
        free_before, _total = driver.memory_info()
        record['free_before_mib'] = int(free_before // (1 << 20))
        request = int(config['INITSIZE'])
        allocations = 0
        while request >= minimum and allocations < 4096:
            free_bytes, _t = driver.memory_info()
            free_mib = int(free_bytes // (1 << 20))
            bounded = request
            if config['BOUND_REQUEST_BY_FREE_MEMORY']:
                bounded = min(request, max(minimum, free_mib // 2))
            if bounded < minimum:
                record['trail'].append({'stage': 'stop', 'request_mib': request,
                                        'free_hint_mib': free_mib, 'reason': 'bounded request below granularity'})
                break
            try:
                tensor = torch.empty((bounded * (1 << 20)) // 4, dtype=torch.float32, device=target)
                tensor.fill_(1.0)
                driver.synchronize()
                held.append(tensor)
                total_mib += bounded
                allocations += 1
                record['trail'].append({'stage': 'held', 'request_mib': bounded, 'held_total_mib': total_mib,
                                        'free_hint_mib': free_mib})
                request = bounded * 2
            except Exception as exc:
                if not driver.is_out_of_memory(exc):
                    raise
                record['oom_errors'].append({'request_mib': bounded, 'error': str(exc)[:2000],
                                             'error_type': type(exc).__name__})
                request = bounded // 2
        if allocations >= 4096:
            raise RuntimeError('capacity search exceeded the bounded allocation count')
        driver.synchronize()
        free_peak, _t = driver.memory_info()
        record.update(held_mib=total_mib, free_at_peak_mib=int(free_peak // (1 << 20)))
        _fault_point(config)
        finished = time.perf_counter_ns()
        elapsed = (finished - started) / 1e9
        benchmark_measurement_finish(token)
        value_gb = total_mib * (1 << 20) / 1e9
        metric = _metric_common(identity, case, config, rank, world_size, 'main_memory-capacity', 'GB', value_gb)
        metric.update(value_gib_s=total_mib / 1024, held_mib=total_mib, allocation_count=allocations,
                      elapsed_seconds=elapsed, started_monotonic_ns=started, finished_monotonic_ns=finished,
                      scope='successfully held device allocation only; free-memory hints bounded requests',
                      timer='perf_counter_ns enclosing the whole bounded search; monitor window evidence')
        save(output / 'metric-rank-0.json', metric)
        _emit(f"[FlagPerf Result]Rank {rank}'s main_memory-capacity={_num(value_gb)}GB")
        _emit(f"[FlagPerf Result]Rank {rank}'s main_memory-capacity={total_mib / 1024:.6f}GiB")
        record['status'] = 'passed'
        return total_mib
    except Exception as exc:
        record.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        held.clear()
        for _ in range(3):
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        try:
            driver.synchronize()
            free_after, _t = driver.memory_info()
            record.update(released=True, release_verified=free_after >= record.get('free_before_mib', 0) * 95 // 100,
                          free_after_mib=int(free_after // (1 << 20)))
        except Exception as exc:
            record.update(released=False, release_error=str(exc), error_type=type(exc).__name__)
        save(output / 'correctness-rank-0.json', record)


def run_allreduce(driver, config, rank, world_size, local_rank):
    """Two-rank nonzero SUM over FlagCX; algbw=S/t, busbw=algbw*2*(n-1)/n."""
    case = 'interconnect-MPI_intraserver:P800'
    validate_config(config, case)
    target, identity, output = _identity(driver, config, case, rank, world_size, local_rank)
    payload = payload_bytes(config['Melements'])
    record = {'schema_version': 1, **identity, 'case': case, 'rank': rank, 'status': 'failed',
              'checks': [], 'semantics': {'collective': 'all_reduce SUM', 'message_bytes': payload,
                                          'reference': 'sum of deterministic rank inputs computed on CPU for the configured world size',
                                          'timed_loop': 'sign-paired inputs (+base and -base) keep repeated SUM at exactly zero; wire traffic is unchanged'}}
    try:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('process group must be initialized before allreduce')
        base = torch.arange(payload // 4, dtype=torch.float32).remainder_(251).add_(config['SEED'] % 17)
        mine = base + rank
        reference = base * world_size + (world_size * (world_size - 1) // 2)
        device_tensor = mine.to(target)
        record.update(input=_placement(device_tensor), input_checksum=_checksum(device_tensor))
        for phase in ('cold', 'second'):
            device_tensor.copy_(mine.to(target))
            dist.all_reduce(device_tensor, op=dist.ReduceOp.SUM)
            driver.synchronize()
            checked = _check_content(device_tensor, reference, phase)
            record['checks'].append(checked)
            record['output_checksum_' + phase] = _checksum(device_tensor)
            if not checked['passed']:
                raise RuntimeError('allreduce content mismatch: ' + phase)
        _fault_point(config)
        # Sign-paired inputs: rank0 holds +base and rank1 holds -base, so every
        # repeated SUM collapses to exactly +0.0 and the buffer can never grow
        # toward float32 overflow no matter how long the timed loop runs.  The
        # collective still moves the identical payload over the interconnect.
        device_tensor.copy_((base * (1.0 if rank == 0 else -1.0)).to(target))
        driver.synchronize()
        record['timed_input_checksum'] = _checksum(device_tensor)
        zero = torch.zeros_like(base)
        for _ in range(config['WARMUP']):
            dist.all_reduce(device_tensor, op=dist.ReduceOp.SUM)
        dist.barrier()
        driver.synchronize()
        token = benchmark_measurement_start()
        started = time.perf_counter_ns()
        for _ in range(config['ITERS']):
            dist.all_reduce(device_tensor, op=dist.ReduceOp.SUM)
        dist.barrier()
        driver.synchronize()
        finished = time.perf_counter_ns()
        benchmark_measurement_finish(token)
        elapsed = (finished - started) / 1e9
        algbw, algbw_gib, busbw, busbw_gib = allreduce_bandwidth(payload, config['ITERS'], elapsed, world_size)
        checked = _check_content(device_tensor, zero, 'post-loop')
        record['checks'].append({'phase': 'post-loop', **checked,
                                 'note': 'sign-paired inputs keep the repeated SUM at exactly zero'})
        if not checked['passed']:
            raise RuntimeError('allreduce timed loop drifted from the exact zero state')
        record['status'] = 'passed'
        metric = _metric_common(identity, case, config, rank, world_size, 'allreduce-algbw', 'GB/s', algbw)
        metric.update(value_gib_s=algbw_gib, busbw_gb_s=busbw, busbw_gib_s=busbw_gib,
                      elapsed_seconds=elapsed, started_monotonic_ns=started, finished_monotonic_ns=finished,
                      message_bytes=payload, total_bytes=payload * config['ITERS'],
                      timer='perf_counter_ns; collective plus barrier and device synchronization inside the window',
                      scope='algbw=message x iterations / window; busbw=algbw*2*(world_size-1)/world_size; no additional multiplier',
                      qualification_duration_passed=elapsed >= 15)
        save(output / f'metric-rank-{rank}.json', metric)
        _emit(f"[FlagPerf Result]Rank {rank}'s allreduce-algbw={_num(algbw)}GB/s")
        _emit(f"[FlagPerf Result]Rank {rank}'s allreduce-busbw={_num(busbw)}GB/s")
        return algbw, busbw
    except Exception as exc:
        record.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        save(output / f'correctness-rank-{rank}.json', record)


def run_p2p(driver, config, rank, world_size, local_rank):
    """One-way rank0->rank1 send/recv; bandwidth never doubled."""
    case = 'interconnect-P2P_intraserver:P800'
    validate_config(config, case)
    target, identity, output = _identity(driver, config, case, rank, world_size, local_rank)
    payload = payload_bytes(config['Melements'])
    record = {'schema_version': 1, **identity, 'case': case, 'rank': rank, 'status': 'failed',
              'checks': [], 'semantics': {'direction': 'rank0-to-rank1', 'payload_bytes': payload,
                                          'sentinel': '-999 destination prefill'}}
    try:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('process group must be initialized before p2p')
        base = torch.arange(payload // 4, dtype=torch.float32).remainder_(251).add_(config['SEED'] % 17)
        expected_checksum = _checksum(base)
        if rank == 0:
            buffer = base.clone().to(target)
            record['buffer'] = _placement(buffer)
            for sequence in range(config['WARMUP']):
                buffer[0] = float(sequence)
                driver.synchronize()
                dist.send(buffer, dst=1)
            record['checks'].append({'phase': 'sent-verified-repeats', 'passed': True,
                                     'repeats': config['WARMUP'], 'sequence_range': [0, config['WARMUP'] - 1]})
        else:
            buffer = torch.full_like(base, -999.0).to(target)
            record['buffer'] = _placement(buffer)
            for sequence in range(config['WARMUP']):
                dist.recv(buffer, src=0)
                driver.synchronize()
                expected = base.clone()
                expected[0] = float(sequence)
                checked = _check_content(buffer, expected, f'recv-{sequence}')
                record['checks'].append(checked)
                if not checked['passed']:
                    raise RuntimeError('p2p receive content mismatch at sequence ' + str(sequence))
        _fault_point(config)
        dist.barrier()
        driver.synchronize()
        token = benchmark_measurement_start()
        started = time.perf_counter_ns()
        if rank == 0:
            for sequence in range(config['ITERS']):
                buffer[0] = float(sequence)
                dist.send(buffer, dst=1)
        else:
            for _ in range(config['ITERS']):
                dist.recv(buffer, src=0)
        dist.barrier()
        driver.synchronize()
        finished = time.perf_counter_ns()
        benchmark_measurement_finish(token)
        elapsed = (finished - started) / 1e9
        gb, gib = p2p_one_way_bandwidth(payload, config['ITERS'], elapsed)
        if rank == 1:
            expected = base.clone()
            expected[0] = float(config['ITERS'] - 1)
            final = _check_content(buffer, expected, 'post-loop')
            record['checks'].append(final)
            if not final['passed']:
                raise RuntimeError('p2p final receive content mismatch')
        record['status'] = 'passed'
        metric = _metric_common(identity, case, config, rank, world_size, 'p2p-one-way-bandwidth', 'GB/s', gb)
        metric.update(value_gib_s=gib, elapsed_seconds=elapsed, started_monotonic_ns=started,
                      finished_monotonic_ns=finished, total_bytes=payload * config['ITERS'],
                      direction='rank0-to-rank1',
                      timer='perf_counter_ns; sends/recvs plus barrier and device synchronization inside the window',
                      scope='one-way payload x iterations / elapsed; never doubled; per-rank elapsed reported independently',
                      qualification_duration_passed=elapsed >= 15)
        save(output / f'metric-rank-{rank}.json', metric)
        _emit(f"[FlagPerf Result]Rank {rank}'s p2p-one-way-bandwidth={_num(gb)}GB/s")
        _emit(f"[FlagPerf Result]Rank {rank}'s p2p-one-way-bandwidth={_num(gib)}GiB/s")
        return gb, gib
    except Exception as exc:
        record.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        save(output / f'correctness-rank-{rank}.json', record)
