# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Execution configuration shared by package runners and their attach targets."""
from contextlib import contextmanager
import json
from pathlib import Path


def select_compiler(handle, *, flagtree=None, vendor_compiler=None, outputs):
    requested = dict(flagtree=flagtree, vendor_compiler=vendor_compiler)
    if handle:
        for key, value in requested.items():
            if key == 'vendor_compiler' and value is not None:
                value = str(Path(value).resolve())
            if value is not None and value != handle.state['effective_config'].get(key):
                raise ValueError('runner compiler configuration conflicts with attachment')
    else:
        from flagprof.components.compiler import select
        select(requested, cache_root=Path(outputs).parent/'component-cache')


@contextmanager
def execution(*, device, outputs, numerics='default', flaggems=None,
              flaggems_include=None, flagtree=None, vendor_compiler=None):
    from flagprof import current_attachment
    from flagprof.probes.config import resolve
    handle = current_attachment()
    select_compiler(handle, flagtree=flagtree, vendor_compiler=vendor_compiler, outputs=outputs)
    component = None
    if handle:
        effective = handle.state['effective_config']
        if flaggems is not None and flaggems != effective['flaggems']:
            raise ValueError('runner component configuration conflicts with attachment')
        if flaggems_include is not None:
            include = json.loads(Path(flaggems_include).read_text())
            if include != effective['flaggems_include']:
                raise ValueError('runner include conflicts with attachment')
    else:
        _, effective = resolve('coverage', dict(flaggems=flaggems or 'off',
                               flaggems_include=flaggems_include, probes='none'))
    from .fixed import numerics as configure_numerics, software
    numeric_settings = configure_numerics(device, numerics)
    import torch
    if device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(device)
        sync = torch.npu.synchronize
    elif device.startswith('cuda'):
        torch.cuda.set_device(device)
        sync = torch.cuda.synchronize
    else:
        sync = lambda: None
    try:
        if not handle and effective['flaggems'] == 'on':
            from flagprof.components.flaggems import FlagGems
            component = FlagGems(effective['flaggems_include']).enable()
        from flagprof.components.compiler import identity as compiler_identity
        identity = dict(device=device, software=software(), numerics=numeric_settings,
                        flaggems=effective['flaggems'], include=effective['flaggems_include'],
                        compiler=handle.state['compiler_identity'] if handle else compiler_identity())
        yield dict(identity=identity, sync=sync, effective_config=effective,
                   component_identity=component.identity if component else
                       handle.state.get('flaggems_identity') if handle else None,
                   component_owner='attachment' if handle else 'runner')
    finally:
        if component:
            component.close()


def options(args):
    return {key: getattr(args, key, default) for key, default in
            [('numerics', 'default'), ('flaggems', None), ('flaggems_include', None),
             ('flagtree', None), ('vendor_compiler', None)]}


def batch_identity(batches):
    rows = []
    for samples, inputs in batches:
        row = dict(samples=samples, inputs={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                                           for key, value in inputs.items()})
        if 'attention_mask' in inputs:
            row['valid_tokens'] = inputs['attention_mask'].sum(1).tolist()
        rows.append(row)
    return rows


def validate_counts(batch_size, warmup, forwards, repeats):
    if any(type(v) is not int or v < 1 for v in (forwards, repeats)) or warmup < 0:
        raise ValueError('positive forwards/repeats and nonnegative warmup required')
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError('batch-size must be positive')
