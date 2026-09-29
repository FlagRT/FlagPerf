# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Case discovery and bounded workload planning. No Torch, YAML or device imports."""
from pathlib import Path
import ast
import math

ROOT = Path(__file__).resolve().parents[1]
FLOATS = ('FP32', 'FP16', 'BF16')
INTEGERS = ('INT32', 'INT16', 'BOOL')


def names():
    return sorted(p.parent.name for p in (ROOT / 'benchmarks').glob('*/main.py'))


def dtypes(case):
    if case.startswith('bitwise_'):
        return INTEGERS
    if case == 'all':
        return ('INT64',)
    return FLOATS


def defaults(case):
    # Repository configs contain only scalar numbers. Host planning needs no YAML dependency.
    values = {}
    for line in (ROOT / 'benchmarks' / case / 'case_config.yaml').read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            key, value = line.split(':', 1)
            values[key] = ast.literal_eval(value.strip())
    return values


def workload(case, profile='daily', size=None, warmup=None, iters=None, rounds=None):
    cfg = defaults(case)
    smoke = profile == 'smoke'
    # Retain each case's shape equations, including its existing multipliers.
    caps = {'ELEMENT_UNIT': 128 if smoke else 1024, 'Melements': 1, 'M': 128 if smoke else 512, 'N': 128 if smoke else 512,
            'K': 128 if smoke else 512, 'BS': 2, 'bs': 2 if smoke else 8,
            'channel': 8 if smoke else 32, 'hiddensize': 32 if smoke else 128,
            'elements': 128 if smoke else 1024}
    if '1024' in (ROOT / 'benchmarks' / case / 'main.py').read_text():
        cfg['ELEMENT_UNIT'] = caps['ELEMENT_UNIT']
    if case in ('amax', 'argmax', 'triu', 'outer'):
        caps.update(M=2 if smoke else 8, N=2 if smoke else 8)
    if case in ('isinf', 'isnan', 'log_softmax'):
        caps['M'] = 1
    for key in cfg:
        if key in caps:
            cfg[key] = min(cfg[key], caps[key])
    cfg.update(WARMUP=2 if smoke else 5, ITERS=5 if smoke else 20,
               KERNELWARMUP=5, KERNELITERS=20, rounds=1 if smoke else 10)
    if size:
        for field in size.split(','):
            key, value = field.split('=', 1)
            if key not in caps or key not in cfg:
                raise ValueError(f'{case}: unsupported size key {key!r}; use its config dimension names')
            number = int(value)
            if number <= 0:
                raise ValueError('size dimensions must be positive')
            cfg[key] = number
    for key, value in [('WARMUP', warmup), ('ITERS', iters), ('rounds', rounds)]:
        if value is not None:
            if value <= 0:
                raise ValueError(f'{key} must be positive')
            cfg[key] = value
    return cfg


def expand(cases=None, dtype=None, oplib='nativetorch', **kwargs):
    selected = list(dict.fromkeys(cases or names()))
    unknown = set(selected) - set(names())
    if unknown:
        raise ValueError(f'unknown cases: {sorted(unknown)}')
    tasks = []
    for case in selected:
        for dt in dict.fromkeys(dtype or [dtypes(case)[0]]):
            for lib in (('nativetorch', 'flaggems') if oplib == 'both' else (oplib,)):
                for size in kwargs.get('sizes') or [None]:
                    tasks.append({'case': case, 'dtype': dt, 'oplib': lib,
                                  'applicable': dt in dtypes(case),
                                  'case_config': workload(case, size=size, **{k:v for k,v in kwargs.items() if k != 'sizes'})})
    return tasks
