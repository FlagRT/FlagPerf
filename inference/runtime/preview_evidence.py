# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Small, sealed forward checks and crash-readable preview stage observations."""
from contextlib import contextmanager
from pathlib import Path
import math
import re
import time

from runtime.common import read_json, write_json, file_hash


class Stages:
    def __init__(self, root, append=False):
        self.root = Path(root)
        self.rows = read_json(self.root/'stages.json') if append and (self.root/'stages.json').is_file() else []

    @contextmanager
    def measure(self, name):
        started = time.monotonic()
        row = {'stage': name, 'started': started, 'status': 'running'}
        self.rows.append(row)
        write_json(self.root/'stages.json', self.rows)
        try:
            yield
        except BaseException:
            row['status'] = 'failed'
            raise
        else:
            row['status'] = 'completed'
        finally:
            row['seconds'] = time.monotonic() - started
            write_json(self.root/'stages.json', self.rows)


def input_batches(batches):
    return [{'batch_index': i, 'sample_ids': b['ids'],
             'input_shape': list(b['inputs']['input_ids'].shape),
             'valid_tokens': b['inputs']['attention_mask'].sum(dim=1).tolist()}
            for i, b in enumerate(batches)]


def seal_checks(root, mode, batches):
    path = Path(root)/'forward-checks.json'
    write_json(path, {'schema_version': 1, 'mode': mode, 'complete': True, 'batches': batches})
    return {'mode': mode, 'manifest': path.name, 'sha256': file_hash(path)}


def validate_checks(root, result, expected):
    """A missing tensor file is allowed only with complete, verified compact evidence."""
    descriptor = result.get('forward_evidence', {})
    path = (Path(root)/descriptor.get('manifest', '')).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('missing or external preview forward checks')
    if file_hash(path) != descriptor.get('sha256'):
        raise ValueError('preview forward checks digest mismatch')
    value = read_json(path)
    if (value.get('schema_version') != 1 or value.get('complete') is not True or
            value.get('mode') not in ['lightweight', 'full'] or value['mode'] != descriptor.get('mode')):
        raise ValueError('incomplete preview forward checks')
    actual = value.get('batches', [])
    if len(actual) != len(expected['batches']):
        raise ValueError('preview batch count mismatch')
    anomalies = set()
    for batch, reference in zip(actual, expected['batches']):
        if any(batch.get(k) != reference[k] for k in ['batch_index', 'sample_ids', 'input_shape']):
            raise ValueError('preview batch/sample/input shape mismatch')
        wanted = {(sample, boundary) for sample in reference['sample_ids'] for boundary in expected['boundaries']}
        seen = set()
        lengths = dict(zip(reference['sample_ids'], reference['valid_tokens']))
        for check in batch.get('checks', []):
            key = (check.get('sample_id'), check.get('boundary'))
            if key not in wanted or key in seen:
                raise ValueError('unexpected or duplicate preview boundary check')
            seen.add(key)
            shape = ([expected['hidden_size']] if key[1] in ['pooled', 'embedding']
                     else [lengths[key[0]], expected['hidden_size']])
            count = check.get('nonfinite_count')
            if (check.get('shape') != shape or check.get('checked_elements') != math.prod(shape)
                    or not isinstance(check.get('dtype'), str) or not check['dtype'].startswith('torch.')
                    or type(count) is not int or not 0 <= count <= math.prod(shape)):
                raise ValueError('invalid preview boundary check')
            if count:
                anomalies.add(key)
        if seen != wanted:
            raise ValueError('missing preview boundary checks')
    reported = {(a.get('sample_id'), a.get('boundary')) for a in result.get('numerical_anomalies', [])}
    if reported != anomalies:
        raise ValueError('preview anomaly summary differs from sealed checks')
    if value['mode'] == 'full':
        artifacts = result.get('artifacts', [])
        if len(artifacts) != len(actual):
            raise ValueError('missing full preview tensor artifacts')
        for name in artifacts:
            p = (Path(root)/name).resolve()
            if not p.is_relative_to(Path(root).resolve()) or not p.is_file():
                raise ValueError('missing full preview tensor artifact')
    return value


def diagnostic_summary(root, result, parallel=False):
    directories = sorted(Path(root).glob('rank-*')) if parallel else [Path(root)]
    observations = {}
    for folder in directories:
        stages = read_json(folder/'stages.json') if (folder/'stages.json').is_file() else []
        observations[folder.name if parallel else 'single'] = {
            'stages': stages, 'last_stage': stages[-1]['stage'] if stages else 'bootstrap',
            'tensor_bytes': sum(p.stat().st_size for p in folder.glob('batch-*.pt')),
            'check_bytes': (folder/'forward-checks.json').stat().st_size if (folder/'forward-checks.json').exists() else 0}
        if (folder/'components.json').is_file():
            observations[folder.name if parallel else 'single']['cache_groups'] = read_json(folder/'components.json').get('cache_groups')
    return observations


def resource_failure(value):
    """Environment faults are not cumulative operator qualification failures."""
    rows = [value, *value.get('ranks', {}).values()]
    for row in rows:
        if row.get('exit_code') in [-9, -15, 137, 143] or row.get('errno') in [12, 16, 19]:
            return True
        if row.get('exception_type') in ['OutOfMemoryError', 'MemoryError']:
            return True
        message = str(row.get('error', '')).lower()
        if re.search(r'\boom\b',message) or any(word in message for word in ['out of memory', 'ebusy', 'resource_busy',
                'resource busy', 'device busy', 'device unavailable', 'device not available',
                'device lost', 'no such device']):
            return True
    return False
