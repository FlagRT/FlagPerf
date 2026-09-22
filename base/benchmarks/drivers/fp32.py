"""Verified FP32 measurement, with all validation outside the timed loop."""
import json
import math
import os
from pathlib import Path
import time

import torch

from benchmarks.computation_contract import contract, validate_config
from .correctness import compare, input_pair, sampled_reference
from .events import benchmark_measurement_finish, benchmark_measurement_start


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def require_placement(tensors, device, dtype=torch.float32):
    if any(tensor.device != device or tensor.dtype != dtype for tensor in tensors):
        raise RuntimeError('computation tensor dtype/device differs from the verified target')


def run_verified_fp32(driver, config, rank, world_size, local_rank):
    return run_verified_computation(driver, config, rank, world_size, local_rank, 'FP32')


def run_verified_computation(driver, config, rank, world_size, local_rank, precision):
    case = 'computation-' + precision + ':P800'
    validate_config(config, case)
    spec = contract(case)
    dtype = getattr(torch, spec['dtype'])
    output_dtype = getattr(torch, spec['output_dtype'])
    reference_dtype = getattr(torch, spec['reference_dtype'])
    multiply = torch._int_mm if precision == 'INT8' else torch.mm

    def inputs(rows, inner, columns):
        if precision == 'INT8':
            generator = torch.Generator().manual_seed(config['SEED'])
            return (torch.randint(-8, 9, (rows, inner), generator=generator, dtype=dtype),
                    torch.randint(-8, 9, (inner, columns), generator=generator, dtype=dtype))
        return input_pair(rows, inner, columns, config['SEED'])
    if (rank, world_size, local_rank) != (0, 1, 0):
        raise RuntimeError('computation candidate contract requires exactly rank zero')
    output_dir = Path(os.environ['FLAGPERF_BENCHMARK_OUTPUT'])
    identity = driver.evidence()
    target = driver.device(local_rank)
    driver.seed(config['SEED'])
    correctness = {'schema_version': 1, **identity, 'rank': rank, 'dtype': spec['dtype'],
                   'reference_dtype': spec['reference_dtype'], 'seed': config['SEED'], 'small_cases': [], 'status': 'failed'}
    if precision != 'FP32':
        correctness.update(operator=spec['operator'], reference_inputs=spec['reference_inputs'],
                           accumulation_scope='numerical observations only; internal accumulator dtype not certified')
    correctness_path = output_dir / 'correctness-rank-0.json'
    try:
        shapes = spec['shapes'][:spec['small_cases']]
        for rows, inner, columns in shapes:
            left, right = inputs(rows, inner, columns)
            if (rows, inner, columns) == (16, 16, 16):
                left = torch.eye(16, dtype=torch.float32)
            if (rows, inner, columns) == (2, 1024, 2):
                left.fill_(1 / 32)
                right[1::2] = -right[::2]
                right[-1] += 0.125
            if precision == 'INT8' and inner == 1024:
                left[:, ::2], left[:, 1::2] = 127, -128
                right[::2], right[1::2] = -128, 127
            left, right = left.to(dtype), right.to(dtype)
            device_left, device_right = left.to(target), right.to(target)
            actual = multiply(device_left, device_right)
            require_placement((device_left, device_right), target, dtype)
            require_placement((actual,), target, output_dtype)
            driver.synchronize()
            check = compare(actual, left.to(reference_dtype) @ right.to(reference_dtype), atol=config['ATOL'], rtol=config['RTOL'])
            correctness['small_cases'].append({'shape': [rows, inner, columns], 'scope': 'full output', **check})
            if not check['passed']:
                raise RuntimeError('small computation CPU reference failed')
        del device_left, device_right, actual
        left, right = inputs(config['M'], config['N'], config['K'])
        left, right = left.to(dtype), right.to(dtype)
        device_left, device_right = left.to(target), right.to(target)
        actual = multiply(device_left, device_right)
        require_placement((device_left, device_right), target, dtype)
        require_placement((actual,), target, output_dtype)
        driver.synchronize()
        correctness['shape_before'] = sampled_reference(left, right, actual, atol=config['ATOL'], rtol=config['RTOL'], reference_dtype=reference_dtype)
        if not correctness['shape_before']['passed']:
            raise RuntimeError('measurement shape CPU reference failed')
        save(correctness_path, correctness)
        if config.get('FAULT_MODE') == 'error':
            raise RuntimeError('controlled error after correctness, before measurement')
        if config.get('FAULT_MODE') == 'cpu-wait':
            context = json.loads(Path(os.environ['FLAGPERF_HOST_CONTEXT']).read_text())
            time.sleep(context['timeout'] + 30)
            raise RuntimeError('watchdog failed to stop controlled CPU wait')
        for iteration in range(config['WARMUP']):
            actual = multiply(device_left, device_right)
        driver.synchronize()
        token = benchmark_measurement_start()
        started_ns = time.perf_counter_ns()
        for iteration in range(config['ITERS']):
            actual = multiply(device_left, device_right)
        driver.synchronize()
        finished_ns = time.perf_counter_ns()
        benchmark_measurement_finish(token)
        elapsed = (finished_ns - started_ns) / 1e9
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError('invalid synchronized measurement time')
        require_placement((device_left, device_right), target, dtype)
        require_placement((actual,), target, output_dtype)
        correctness['shape_after'] = sampled_reference(left, right, actual, atol=config['ATOL'], rtol=config['RTOL'], reference_dtype=reference_dtype)
        if not correctness['shape_after']['passed']:
            raise RuntimeError('post-measurement computation reference failed')
        correctness['status'] = 'passed'
        operations = 2 * config['M'] * config['N'] * config['K'] * config['ITERS']
        value = operations / elapsed / 1e12
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError('invalid computation throughput')
        metric = {'schema_version': 1, **identity, 'rank': rank, 'world_size': world_size,
                  'status': 'passed', 'metric': spec['metric'], 'value': value, 'unit': spec['unit'],
                  'mode': config['MODE'], 'shape': [config['M'], config['N'], config['K']],
                  'iterations': config['ITERS'], 'warmup': config['WARMUP'], 'elapsed_seconds': elapsed,
                  'operations': operations, 'started_monotonic_ns': started_ns, 'finished_monotonic_ns': finished_ns,
                  'timer': 'perf_counter_ns with full target synchronization', 'implementation': config['IMPLEMENTATION'],
                  'allocation_semantics': spec['operator'] + ' returned output; allocator cost may be included',
                  'qualification_duration_passed': elapsed >= 15,
                  'case_assets_sha256': json.loads(Path(os.environ['FLAGPERF_HOST_CONTEXT']).read_text())['case_assets_sha256']}
        if precision != 'FP32':
            metric.update(dtype=spec['dtype'], output_dtype=spec['output_dtype'], operator=spec['operator'],
                          reference_inputs=correctness['reference_inputs'], accumulation_scope=correctness['accumulation_scope'])
        save(output_dir / 'metric-rank-0.json', metric)
        return value
    except Exception as exc:
        correctness.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        save(correctness_path, correctness)
