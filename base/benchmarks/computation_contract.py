"""Explicit, CPU-only contracts for candidate P800/PPU computation cases."""
import math

from base.vendors.protocol import ConfigurationError
from base.benchmarks.fp32_contract import validate_config as validate_fp32


CONTRACTS = {
    'FP32': {'dtype': 'float32', 'unit': 'TFLOPS', 'atol': 1e-4, 'rtol': 1e-4, 'small_cases': 3},
    'FP16': {'dtype': 'float16', 'unit': 'TFLOPS', 'atol': 1e-3, 'rtol': 1e-3, 'small_cases': 5},
    'BF16': {'dtype': 'bfloat16', 'unit': 'TFLOPS', 'atol': 8e-3, 'rtol': 8e-3, 'small_cases': 5},
    'INT8': {'dtype': 'int8', 'unit': 'TOPS', 'atol': 0.02, 'rtol': 8e-3, 'small_cases': 5},
    # FP64/TF32 joined on day 5 from the day-4 capability probe: PPU torch.mm
    # keeps float64 exactly (the 2**-40 probe passed) and honors the TF32
    # flag. TF32 inherits the BF16 error budget so a bf16x3 emulation inside
    # the CUDA compatibility layer would still pass.
    'FP64': {'dtype': 'float64', 'unit': 'TFLOPS', 'atol': 1e-12, 'rtol': 1e-12, 'small_cases': 5},
    'TF32': {'dtype': 'float32', 'unit': 'TFLOPS', 'atol': 8e-3, 'rtol': 8e-3, 'small_cases': 5},
}
# Case selectors are chip-scoped. A chip registers here together with its
# vendor provider PERFORMANCE_CASES and per-chip case assets; the
# per-precision contract itself stays chip-independent.
CHIPS = ('P800', 'PPU')
CASES = tuple('computation-' + precision + ':' + chip for precision in CONTRACTS for chip in CHIPS)


def contract(case):
    if case not in CASES:
        raise ConfigurationError('computation case has no verified implementation contract: ' + str(case))
    selector, chip = case.rsplit(':', 1)
    precision = selector.split('-')[1]
    integer = precision == 'INT8'
    # The P800 INT8 kernel is the vendor xtorch op with a bf16 output; the
    # day-4 PPU probe fixed torch._int_mm as the real device kernel (torch.mm
    # raises NotImplementedError for int8 on PPU, and the P800 _int_mm runs
    # on the host CPU, so the two chips must not share the operator).
    device_int8 = integer and chip == 'PPU'
    # torch._int_mm refuses M <= 16 on the device ("self.size(0) needs to be
    # greater than 16"), so the PPU INT8 small cases run at M = 32; K and N
    # keep their 8-multiple alignment. The 8192-cube measurement shape and
    # the P800 small cases are unchanged.
    int8_shapes = [[32, 32, 32], [32, 32, 48], [32, 16, 16], [32, 2048, 16], [32, 1024, 32]]
    spec = {'precision': precision, 'metric': 'computation-' + precision,
            'operator': 'torch._int_mm' if device_int8 else 'xtorch_ops.gemm_I8_I8_bf16_nt' if integer else 'torch.mm',
            'reference_dtype': 'float64',
            'output_dtype': 'int32' if device_int8 else 'bfloat16' if integer else CONTRACTS[precision]['dtype'],
            'reference_inputs': 'signed int8 inputs; CPU int64 accumulation reference; device int32 accumulation '
                                'and int32 output via torch._int_mm' if device_int8 else
                                'signed int8 inputs; CPU float64 accumulation of the quantized operands times SCALE_A*SCALE_B; '
                                'int32 accumulation and bf16 output inside the device kernel' if integer else
                                'CPU inputs quantized to target dtype before FP64 reference',
            'shapes': int8_shapes if device_int8 else
                      [[32, 32, 32], [16, 32, 48], [16, 16, 16], [16, 2048, 16], [16, 1024, 32]] if integer else
                      [[32, 32, 32], [17, 29, 11], [16, 16, 16], [4, 2048, 4], [2, 1024, 2]], **CONTRACTS[precision]}
    if device_int8:
        spec.update(atol=0.0, rtol=0.0)
    return spec


def validate_config(config, case):
    spec = contract(case)
    if spec['precision'] == 'FP32':
        return validate_fp32(config)
    for key, lower, upper in [('M', 1, 8192), ('N', 1, 8192), ('K', 1, 8192),
                              ('WARMUP', 1, 100), ('ITERS', 1, 200000), ('SEED', 0, 2**31 - 1)]:
        if type(config.get(key)) is not int or not lower <= config[key] <= upper:
            raise ConfigurationError('invalid bounded computation field: ' + key)
    if config.get('DTYPE') != spec['dtype'] or config.get('OPERATOR') != spec['operator']:
        raise ConfigurationError('case dtype/operator mismatch')
    for key in ('ATOL', 'RTOL'):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value != spec[key.lower()]:
            raise ConfigurationError('low-precision tolerance must match the frozen error budget')
    if config.get('DIST_BACKEND') != 'gloo' or config.get('IMPLEMENTATION') != 'native-xpytorch':
        raise ConfigurationError('computation requires native XPYTORCH and CPU Gloo')
    if config.get('MODE') not in ('smoke', 'calibration', 'qualification', 'diagnostic'):
        raise ConfigurationError('invalid computation mode')
    if config.get('FAULT_MODE', 'none') not in ('none', 'cpu-wait', 'error'):
        raise ConfigurationError('unknown diagnostic fault')
    if config.get('FAULT_MODE', 'none') != 'none' and config['MODE'] != 'diagnostic':
        raise ConfigurationError('fault injection is diagnostic-only')
    return config
