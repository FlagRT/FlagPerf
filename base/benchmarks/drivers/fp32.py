"""Verified FP32 measurement, with all validation outside the timed loop."""
import json
import math
import os
from pathlib import Path
import time

import torch

from benchmarks.fp32_contract import validate_config
from .correctness import compare, input_pair, sampled_reference
from .events import benchmark_measurement_finish, benchmark_measurement_start


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def require_placement(tensors, device):
    if any(tensor.device != device or tensor.dtype != torch.float32 for tensor in tensors):
        raise RuntimeError('FP32 tensor dtype/device differs from the verified target')


def run_verified_fp32(driver, config, rank, world_size, local_rank):
    validate_config(config)
    if (rank, world_size, local_rank) != (0, 1, 0):
        raise RuntimeError('FP32 candidate contract requires exactly rank zero')
    output_dir = Path(os.environ['FLAGPERF_BENCHMARK_OUTPUT'])
    identity = driver.evidence()
    target = driver.device(local_rank)
    driver.seed(config['SEED'])
    correctness = {'schema_version': 1, **identity, 'rank': rank, 'dtype': 'float32',
                   'reference_dtype': 'float64', 'seed': config['SEED'], 'small_cases': [], 'status': 'failed'}
    correctness_path = output_dir / 'correctness-rank-0.json'
    try:
        for rows, inner, columns in [(32, 32, 32), (17, 29, 11), (16, 16, 16)]:
            left, right = input_pair(rows, inner, columns, config['SEED'])
            if (rows, inner, columns) == (16, 16, 16):
                left = torch.eye(16, dtype=torch.float32)
            device_left, device_right = left.to(target), right.to(target)
            actual = torch.mm(device_left, device_right)
            require_placement((device_left, device_right, actual), target)
            driver.synchronize()
            check = compare(actual, left.double() @ right.double(), atol=config['ATOL'], rtol=config['RTOL'])
            correctness['small_cases'].append({'shape': [rows, inner, columns], 'scope': 'full output', **check})
            if not check['passed']:
                raise RuntimeError('small FP32 CPU reference failed')
        del device_left, device_right, actual
        left, right = input_pair(config['M'], config['N'], config['K'], config['SEED'])
        device_left, device_right = left.to(target), right.to(target)
        actual = torch.mm(device_left, device_right)
        require_placement((device_left, device_right, actual), target)
        driver.synchronize()
        correctness['shape_before'] = sampled_reference(left, right, actual, atol=config['ATOL'], rtol=config['RTOL'])
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
            actual = torch.mm(device_left, device_right)
        driver.synchronize()
        token = benchmark_measurement_start()
        started_ns = time.perf_counter_ns()
        for iteration in range(config['ITERS']):
            actual = torch.mm(device_left, device_right)
        driver.synchronize()
        finished_ns = time.perf_counter_ns()
        benchmark_measurement_finish(token)
        elapsed = (finished_ns - started_ns) / 1e9
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError('invalid synchronized measurement time')
        require_placement((device_left, device_right, actual), target)
        correctness['shape_after'] = sampled_reference(left, right, actual, atol=config['ATOL'], rtol=config['RTOL'])
        if not correctness['shape_after']['passed']:
            raise RuntimeError('post-measurement FP32 reference failed')
        correctness['status'] = 'passed'
        operations = 2 * config['M'] * config['N'] * config['K'] * config['ITERS']
        value = operations / elapsed / 1e12
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError('invalid FP32 throughput')
        metric = {'schema_version': 1, **identity, 'rank': rank, 'world_size': world_size,
                  'status': 'passed', 'metric': 'computation-FP32', 'value': value, 'unit': 'TFLOPS',
                  'mode': config['MODE'], 'shape': [config['M'], config['N'], config['K']],
                  'iterations': config['ITERS'], 'warmup': config['WARMUP'], 'elapsed_seconds': elapsed,
                  'operations': operations, 'started_monotonic_ns': started_ns, 'finished_monotonic_ns': finished_ns,
                  'timer': 'perf_counter_ns with full target synchronization', 'implementation': config['IMPLEMENTATION'],
                  'allocation_semantics': 'torch.mm returned output; allocator cost may be included',
                  'qualification_duration_passed': elapsed >= 15,
                  'case_assets_sha256': json.loads(Path(os.environ['FLAGPERF_HOST_CONTEXT']).read_text())['case_assets_sha256']}
        save(output_dir / 'metric-rank-0.json', metric)
        return value
    except Exception as exc:
        correctness.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        save(correctness_path, correctness)
