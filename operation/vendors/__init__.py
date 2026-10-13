# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Lazy vendor registry: device dependencies are imported only inside workers."""
import importlib
import importlib.util

AVAILABLE = ('ascend', 'nvidia', 'cambricon', 'kunlunxin', 'iluvatar', 'metax')


def get_vendor(name):
    if name not in AVAILABLE:
        raise ValueError(f'unsupported vendor: {name}; available: {", ".join(AVAILABLE)}')
    module = f'vendors.{name}.adapter'
    if importlib.util.find_spec(module) is not None:
        return importlib.import_module(module).Adapter()
    if name not in ('nvidia','cambricon','kunlunxin','iluvatar','metax'):
        raise ValueError(f'vendor adapter is missing: {module}')
    from vendors.compat import Adapter
    return Adapter(name)


def legacy_reporter(schema_version):
    readers = {1: 'legacy.report_v1'}
    if schema_version not in readers:
        raise ValueError(f'unknown legacy schema: {schema_version}')
    return importlib.import_module(readers[schema_version]).report


def report_metadata(name):
    """Optional offline vocabulary, independent of runtime adapter imports."""
    modules = {'ascend': 'vendors.ascend.report_metrics'}
    module = modules.get(name)
    return importlib.import_module(module) if module else None
