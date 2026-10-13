"""Verified INT8 measurement on the device int8 kernel; validation outside the timed loop.

The locked XPU stack exposes no device int8 path through ATen: torch.mm rejects int8
and torch._int_mm runs on the host CPU (its process CPU time equals the thread count
times wall time, so the earlier "device" number was a host measurement). The case
therefore uses the vendor kernel xtorch_ops.gemm_I8_I8_bf16_nt, a block-quantized int8
GEMM with int32 accumulation and bfloat16 output:

    out[m, n] = scale_a * scale_b * sum_k A[m, k] * B[n, k]

The kernel takes the int8 quantization maxima (dequant = q * max / 127) and divides by
127 internally, so passing 127 * SCALE_A and 127 * SCALE_B reproduces the case contract
out = (A @ B) * SCALE_A * SCALE_B cast to OUTPUT_DTYPE.
"""
import json
import math
import os
from pathlib import Path
import time

import torch

try:
    import xtorch_ops
except ImportError:  # CPU verification environments deliberately lack the vendor package.
    xtorch_ops = None

from benchmarks.computation_contract import contract, validate_config
from .correctness import compare, sampled_reference
from .events import benchmark_measurement_finish, benchmark_measurement_start

OUTPUT_DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def resolve_output_dtype(name):
    try:
        return OUTPUT_DTYPES[str(name).lower()]
    except KeyError as exc:
        raise RuntimeError('OUTPUT_DTYPE must be one of: ' + ', '.join(sorted(OUTPUT_DTYPES))) from exc


def quantized_inputs(rows, inner, columns, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    left = torch.randint(-8, 9, (rows, inner), generator=generator, dtype=torch.int8)
    right = torch.randint(-8, 9, (inner, columns), generator=generator, dtype=torch.int8)
    return left, right


def run_verified_int8(driver, config, rank, world_size, local_rank):
    spec = contract('computation-INT8:' + config.get('VENDOR_CHIP', 'P800'))
    validate_config(config, 'computation-INT8:P800')
    if (rank, world_size, local_rank) != (0, 1, 0):
        raise RuntimeError('computation candidate contract requires exactly rank zero')
    output_dir = Path(os.environ['FLAGPERF_BENCHMARK_OUTPUT'])
    identity = driver.evidence()
    target = driver.device(local_rank)
    if xtorch_ops is None:
        raise RuntimeError('device int8 kernel is unavailable in this environment')
    kernel = getattr(xtorch_ops, 'gemm_I8_I8_bf16_nt')
    output_dtype = resolve_output_dtype(config.get('OUTPUT_DTYPE', 'bfloat16'))
    scale_a, scale_b = float(config.get('SCALE_A', 1.0)), float(config.get('SCALE_B', 1.0))
    driver.seed(config['SEED'])
    correctness = {'schema_version': 1, **identity, 'rank': rank, 'dtype': spec['dtype'],
                   'reference_dtype': spec['reference_dtype'], 'seed': config['SEED'], 'small_cases': [], 'status': 'failed',
                   'operator': spec['operator'], 'output_dtype': spec['output_dtype'],
                   'reference_inputs': spec['reference_inputs'], 'scale_a': scale_a, 'scale_b': scale_b,
                   'accumulation_scope': 'int32 accumulation inside the device kernel; scales applied by the kernel'}

    def reference_of(left, right):
        return (left.double() @ right.double()) * (scale_a * scale_b)

    def prepare(left_int8, right_int8, out):
        """Everything the timed loop must not do: transport-free layout prep and scales."""
        blocks = (left_int8.shape[1] + 127) // 128
        right_nt = right_int8.t().contiguous()
        a_scale = torch.full((left_int8.shape[0], blocks), 127.0 * scale_a, dtype=torch.float32, device=target)
        # nt layout: the right operand is [columns, inner] after the transpose, so its
        # per-block quantization maxima are indexed by output column, not by reduction size.
        b_scale = torch.full((right_int8.shape[1], blocks), 127.0 * scale_b, dtype=torch.float32, device=target)
        return (left_int8, a_scale), (right_nt, b_scale), out

    correctness_path = output_dir / 'correctness-rank-0.json'
    try:
        for rows, inner, columns in spec['shapes'][:spec['small_cases']]:
            left, right = quantized_inputs(rows, inner, columns, config['SEED'])
            if (rows, inner, columns) == (16, 16, 16):
                left = torch.eye(16, dtype=torch.int8)
                right = torch.eye(16, dtype=torch.int8)
            if inner == 1024:
                left[:, ::2], left[:, 1::2] = 127, -128
                right[::2], right[1::2] = -128, 127
            device_left, device_right = left.to(target), right.to(target)
            out = torch.zeros(rows, columns, dtype=output_dtype, device=target)
            lhs, rhs, _ = prepare(device_left, device_right, out)
            kernel(lhs, rhs, out)
            driver.synchronize()
            if device_left.dtype != torch.int8 or out.dtype != output_dtype or out.device != target:
                raise RuntimeError('int8 kernel placement or dtype drift')
            check = compare(out, reference_of(left, right), atol=config['ATOL'], rtol=config['RTOL'])
            correctness['small_cases'].append({'shape': [rows, inner, columns], 'scope': 'full output', **check})
            if not check['passed']:
                raise RuntimeError('small computation CPU reference failed')
        del device_left, device_right, out, lhs, rhs

        left, right = quantized_inputs(config['M'], config['K'], config['N'], config['SEED'])
        device_left, device_right = left.to(target), right.to(target)
        out = torch.zeros(config['M'], config['N'], dtype=output_dtype, device=target)
        lhs, rhs, out = prepare(device_left, device_right, out)
        kernel(lhs, rhs, out)
        driver.synchronize()
        correctness['shape_before'] = sampled_reference(
            left.double() * scale_a, right.double() * scale_b, out, atol=config['ATOL'], rtol=config['RTOL'])
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
            kernel(lhs, rhs, out)
        driver.synchronize()
        token = benchmark_measurement_start()
        started_ns = time.perf_counter_ns()
        for iteration in range(config['ITERS']):
            kernel(lhs, rhs, out)
        driver.synchronize()
        finished_ns = time.perf_counter_ns()
        benchmark_measurement_finish(token)
        elapsed = (finished_ns - started_ns) / 1e9
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError('invalid synchronized measurement time')
        if out.dtype != output_dtype or out.device != target:
            raise RuntimeError('post-measurement int8 kernel placement drift')
        correctness['shape_after'] = sampled_reference(
            left.double() * scale_a, right.double() * scale_b, out, atol=config['ATOL'], rtol=config['RTOL'])
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
                  'allocation_semantics': 'preallocated bf16 output reused across iterations; no allocation in the timed loop',
                  'qualification_duration_passed': elapsed >= 15,
                  'dtype': spec['dtype'], 'output_dtype': spec['output_dtype'], 'operator': spec['operator'],
                  'reference_inputs': correctness['reference_inputs'], 'accumulation_scope': correctness['accumulation_scope'],
                  'scale_a': scale_a, 'scale_b': scale_b,
                  'case_assets_sha256': json.loads(Path(os.environ['FLAGPERF_HOST_CONTEXT']).read_text())['case_assets_sha256']}
        save(output_dir / 'metric-rank-0.json', metric)
        return value
    except Exception as exc:
        correctness.update(status='failed', error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        save(correctness_path, correctness)
