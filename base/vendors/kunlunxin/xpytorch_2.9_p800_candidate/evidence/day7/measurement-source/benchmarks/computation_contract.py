"""Explicit, CPU-only contracts for candidate P800 computation cases."""
import math

from base.vendors.protocol import ConfigurationError
from base.benchmarks.fp32_contract import validate_config as validate_fp32


CONTRACTS = {
    'FP32': {'dtype': 'float32', 'unit': 'TFLOPS', 'atol': 1e-4, 'rtol': 1e-4, 'small_cases': 3},
    'FP16': {'dtype': 'float16', 'unit': 'TFLOPS', 'atol': 1e-3, 'rtol': 1e-3, 'small_cases': 5},
    'BF16': {'dtype': 'bfloat16', 'unit': 'TFLOPS', 'atol': 8e-3, 'rtol': 8e-3, 'small_cases': 5},
    'INT8': {'dtype': 'int8', 'unit': 'TOPS', 'atol': 0.02, 'rtol': 8e-3, 'small_cases': 5},
}
CASES = tuple('computation-' + precision + ':P800' for precision in CONTRACTS)


def contract(case):
    if case not in CASES:
        raise ConfigurationError('P800 computation case has no verified implementation contract: ' + str(case))
    precision = case.split(':')[0].split('-')[1]
    integer = precision == 'INT8'
    return {'precision': precision, 'metric': 'computation-' + precision,
            'operator': 'xtorch_ops.gemm_I8_I8_bf16_nt' if integer else 'torch.mm',
            'reference_dtype': 'float64' if integer else 'float64',
            'output_dtype': 'bfloat16' if integer else CONTRACTS[precision]['dtype'],
            'reference_inputs': 'signed int8 inputs; CPU float64 accumulation of the quantized operands times SCALE_A*SCALE_B; '
                                'int32 accumulation and bf16 output inside the device kernel' if integer else
                                'CPU inputs quantized to target dtype before FP64 reference',
            'shapes': [[32, 32, 32], [16, 32, 48], [16, 16, 16], [16, 2048, 16], [16, 1024, 32]] if integer else
                      [[32, 32, 32], [17, 29, 11], [16, 16, 16], [4, 2048, 4], [2, 1024, 2]], **CONTRACTS[precision]}


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
