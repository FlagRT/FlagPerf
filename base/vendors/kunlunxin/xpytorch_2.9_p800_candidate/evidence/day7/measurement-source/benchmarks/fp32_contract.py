"""CPU-only bounds for the candidate single-rank FP32 contract."""
import math

from base.vendors.protocol import ConfigurationError


def validate_config(config):
    for key in ('M', 'N', 'K'):
        # 8192 allows the Ascend-parity shape; the day-four qualification used 4096 and
        # keeps its own snapshotted contract inside each result directory.
        if type(config.get(key)) is not int or not 1 <= config[key] <= 8192:
            raise ConfigurationError('FP32 shape must be integer dimensions in 1..8192')
    for key, lower, upper in [('WARMUP', 1, 100), ('ITERS', 1, 20000), ('SEED', 0, 2**31 - 1)]:
        if type(config.get(key)) is not int or not lower <= config[key] <= upper:
            raise ConfigurationError('invalid bounded FP32 field: ' + key)
    if config.get('DIST_BACKEND') != 'gloo' or config.get('IMPLEMENTATION') != 'native-xpytorch':
        raise ConfigurationError('FP32 requires native XPYTORCH and CPU Gloo')
    if config.get('MODE') not in ('smoke', 'calibration', 'qualification', 'diagnostic'):
        raise ConfigurationError('invalid FP32 mode')
    for key in ('ATOL', 'RTOL'):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1e-4:
            raise ConfigurationError('FP32 tolerance must be finite and no larger than 1e-4')
    if config.get('FAULT_MODE', 'none') not in ('none', 'cpu-wait', 'error'):
        raise ConfigurationError('unknown diagnostic fault')
    if config.get('FAULT_MODE', 'none') != 'none' and config['MODE'] != 'diagnostic':
        raise ConfigurationError('fault injection is diagnostic-only')
    return config
