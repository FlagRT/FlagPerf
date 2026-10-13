"""Verified copies with preallocated buffers and synchronized wall-clock timing."""
import json
import os
from pathlib import Path
import time

import torch

from benchmarks.transfer_contract import bandwidth, contract, validate_config
from .events import benchmark_measurement_start, benchmark_measurement_finish
from .fp32 import save


def host_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('host memory availability missing')


def placement(tensor):
    return {'device': str(tensor.device), 'dtype': str(tensor.dtype).removeprefix('torch.'),
            'numel': tensor.numel(), 'contiguous': tensor.is_contiguous(), 'data_ptr': tensor.data_ptr()}


def check_content(actual, reference, phase):
    observed = actual.detach().cpu()
    mismatch = observed != reference
    count = int(mismatch.sum())
    first = int(mismatch.nonzero()[0]) if count else None
    return {'phase': phase, 'passed': count == 0, 'mismatch_count': count,
            'first_mismatch': first, 'elements': reference.numel()}


def copy_loop(destination, source, iterations, non_blocking):
    for iteration in range(iterations):
        destination.copy_(source, non_blocking=non_blocking)


def run_verified_transfer(driver, config, rank, world_size, local_rank, direction):
    case = 'interconnect-' + direction + ':P800'
    validate_config(config, case)
    if (rank, world_size, local_rank) != (0, 1, 0):
        raise RuntimeError('copy requires exactly rank zero')
    target = driver.device(local_rank)
    identity = driver.evidence()
    output = Path(os.environ['FLAGPERF_BENCHMARK_OUTPUT'])
    payload = config['PAYLOAD_BYTES']
    semantics = {'direction': direction, 'host_memory': 'pinned' if config['PIN_MEMORY'] else 'pageable',
                 'non_blocking': config['NON_BLOCKING'], 'dtype': 'float32', 'api': 'Tensor.copy_', 'payload_bytes': payload}
    record = {'schema_version': 1, **identity, **semantics, 'rank': rank, 'status': 'failed', 'checks': []}
    try:
        budget = {'host_bytes': payload * 12, 'device_bytes': payload * 2,
                  'host_available': host_available(), 'device_available': int(driver.memory_info()[0]),
                  'pinned_bytes': payload if config['PIN_MEMORY'] else 0,
                  'scope': 'conservative simultaneous buffer and validation temporary upper bounds'}
        record['memory_budget'] = budget
        if any(budget[prefix + '_bytes'] > budget[prefix + '_available'] // 10 for prefix in ('host', 'device')):
            raise RuntimeError('copy exceeds ten percent available memory budget')
        reference = torch.arange(payload // 4, dtype=torch.float32).remainder_(251).add_(config['SEED'] % 17)
        alternate = reference + 1
        host = torch.empty_like(reference, pin_memory=config['PIN_MEMORY'])
        if host.is_pinned() != config['PIN_MEMORY']:
            raise RuntimeError('host pinned allocation differs from request')
        accelerator = torch.empty_like(reference, device=target)
        source, destination = (host, accelerator) if direction == 'h2d' else (accelerator, host)
        record.update(source=placement(source), destination=placement(destination), host_is_pinned=host.is_pinned())
        expected_source = torch.device('cpu') if direction == 'h2d' else target
        expected_destination = target if direction == 'h2d' else torch.device('cpu')
        if source.device != expected_source or destination.device != expected_destination:
            raise RuntimeError('copy device placement drift')
        if any(tensor.dtype != torch.float32 or not tensor.is_contiguous() for tensor in (source, destination)):
            raise RuntimeError('copy dtype/layout drift')
        source_pointer, destination_pointer = source.data_ptr(), destination.data_ptr()
        for phase, expected in [('sentinel', reference), ('changed-input', alternate), ('before', reference)]:
            source.copy_(expected)
            destination.fill_(-999)
            driver.synchronize()
            destination.copy_(source, non_blocking=config['NON_BLOCKING'])
            driver.synchronize()
            checked = check_content(destination, expected, phase)
            record['checks'].append(checked)
            if not checked['passed']:
                raise RuntimeError('copy content mismatch: ' + phase)
        fault = config.get('FAULT_MODE', 'none')
        if fault == 'error':
            raise RuntimeError('controlled copy error after correctness')
        if fault == 'cpu-wait':
            context = json.loads(Path(os.environ['FLAGPERF_HOST_CONTEXT']).read_text())
            time.sleep(context['timeout'] + 30)
            raise RuntimeError('watchdog failed to stop copy CPU wait')
        copy_loop(destination, source, config['WARMUP'], config['NON_BLOCKING'])
        driver.synchronize()
        token = benchmark_measurement_start()
        started = time.perf_counter_ns()
        copy_loop(destination, source, config['ITERS'], config['NON_BLOCKING'])
        driver.synchronize()
        finished = time.perf_counter_ns()
        benchmark_measurement_finish(token)
        elapsed = (finished - started) / 1e9
        gb, gib = bandwidth(payload, config['ITERS'], elapsed)
        record['checks'].append(check_content(destination, reference, 'after'))
        record.update(destination_reused=destination.data_ptr() == destination_pointer,
                      source_reused=source.data_ptr() == source_pointer)
        if not record['checks'][-1]['passed'] or not record['destination_reused'] or not record['source_reused']:
            raise RuntimeError('post-measurement copy or buffer reuse failed')
        record['status'] = 'passed'
        metric = {'schema_version': 1, **identity, **semantics, 'rank': rank, 'world_size': world_size,
                  'status': 'passed', 'metric': contract(case)['metric'], 'unit': 'GB/s', 'value': gb, 'value_gib_s': gib,
                  'mode': config['MODE'], 'iterations': config['ITERS'], 'warmup': config['WARMUP'],
                  'elapsed_seconds': elapsed, 'total_bytes': payload * config['ITERS'],
                  'started_monotonic_ns': started, 'finished_monotonic_ns': finished,
                  'timer': 'perf_counter_ns with full target synchronization', 'implementation': config['IMPLEMENTATION'],
                  'allocation_semantics': 'preallocated source/destination; timed loop only Tensor.copy_; runtime staging may occur',
                  'scope': 'effective single-direction API bandwidth; non_blocking is a request, not proof of overlap or DMA',
                  'qualification_duration_passed': elapsed >= 15,
                  'case_assets_sha256': json.loads(Path(os.environ['FLAGPERF_HOST_CONTEXT']).read_text())['case_assets_sha256']}
        save(output / 'metric-rank-0.json', metric)
        return gb, gib
    except Exception as exc:
        record.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        save(output / 'correctness-rank-0.json', record)
