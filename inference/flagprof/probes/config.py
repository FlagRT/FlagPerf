# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""One serializable configuration for API, bootstrap and final targets."""
import hashlib
import json
from pathlib import Path

PRESETS = ('perf', 'modules', 'coverage', 'perf-lite', 'accuracy')
KEYS = {'flaggems', 'flaggems_include', 'dispatch', 'layers', 'probes',
        'each_start', 'each_count', 'fallback_calls', 'summary_seconds',
        'tensor_blob','tensor_blob_bytes','tensor_ops_in','tensor_boundaries'}


def resolve(preset, config=None):
    if preset not in PRESETS:
        raise ValueError('unsupported preset: ' + preset)
    config = dict(config or {})
    options = {k: config.pop(k) for k in list(config) if k in KEYS}
    result = dict(flaggems='off', flaggems_include=None, dispatch='off',
                  layers='selected', probes='default', each_start=1,
                  each_count=1, fallback_calls=20000, summary_seconds=1.0)
    if preset == 'coverage':
        result.update(dispatch='each-window', each_start=0)
    if preset == 'accuracy':
        result.update(layers='all',tensor_blob='none',tensor_blob_bytes=1<<30,tensor_ops_in=None,tensor_boundaries=None)
    result.update(options)
    for key, allowed in [('flaggems', ('off', 'on')), ('probes', ('default', 'none')),
                         ('dispatch', ('off', 'summary', 'each-window')),
                         ('layers', ('selected', 'all'))]:
        if result[key] not in allowed:
            raise ValueError('invalid ' + key)
    for key in ('each_start', 'each_count', 'fallback_calls'):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError('invalid ' + key)
    if not 0 < float(result['summary_seconds']) < float('inf'):
        raise ValueError('invalid summary_seconds')
    if result.get('tensor_blob') not in ('none','all',None) and not isinstance(result.get('tensor_blob'),list):
        raise ValueError('tensor_blob requires none, all or a prefix list')
    if preset != 'accuracy' and any(k.startswith('tensor_') for k in options):
        raise ValueError('tensor options require accuracy preset')
    if preset=='accuracy' and (result['dispatch']!='off' or result['layers']!='all'):
        raise ValueError('accuracy requires dispatch off and layers all')
    if 'tensor_boundaries' in result:
        boundaries = result['tensor_boundaries']
        # Canonical values keep CLI/API aliases identical to existing captures.
        if boundaries == 'model':
            boundaries = []
        elif boundaries == 'layer':
            boundaries = None
        if boundaries is not None and (not isinstance(boundaries, list) or
                                       not all(isinstance(path, str) for path in boundaries)):
            raise ValueError('tensor_boundaries requires model, layer, or a list of module paths')
        result['tensor_boundaries'] = boundaries
    if 'tensor_blob_bytes' in result and (type(result['tensor_blob_bytes']) is not int or result['tensor_blob_bytes']<0):
        raise ValueError('tensor_blob_bytes must be a nonnegative integer')
    include = result['flaggems_include']
    if isinstance(include, (str, Path)):
        include = json.loads(Path(include).read_text())
    if include is not None and (not isinstance(include, list) or
                               not all(isinstance(s, str) for s in include)):
        raise ValueError('flaggems include must be an array of function names')
    if result['flaggems'] == 'on' and not include:
        raise ValueError('FlagGems on requires a nonempty include list')
    result['flaggems_include'] = include
    result['include_sha256'] = hashlib.sha256(json.dumps(include, separators=(',', ':')).encode()).hexdigest()
    return config, result
