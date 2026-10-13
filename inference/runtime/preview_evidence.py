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


def input_batches(batches, adapter=None):
    if adapter is None:
        from runtime.adapters import create
        adapter = create('models', 'qwen3_embedding_0.6b')
    return [adapter.batch_metadata(batch, i) for i, batch in enumerate(batches)]


def seal_checks(root, mode, batches):
    path = Path(root)/'forward-checks.json'
    write_json(path, {'schema_version': 1, 'mode': mode, 'complete': True, 'batches': batches})
    return {'mode': mode, 'manifest': path.name, 'sha256': file_hash(path)}


def validate_checks(root, result, expected, *, module_shapes=False):
    """A missing tensor file is allowed only with complete, verified compact evidence."""
    if expected.get('execution_kind') == 'autoregressive':
        from runtime.step_capture import validate_checks as validate_steps
        return validate_steps(root,result,expected)
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
        lengths = dict(zip(reference['sample_ids'], reference.get('valid_tokens', [None]*len(reference['sample_ids']))))
        for check in batch.get('checks', []):
            key = (check.get('sample_id'), check.get('boundary'))
            if key not in wanted or key in seen:
                raise ValueError('unexpected or duplicate preview boundary check')
            seen.add(key)
            specs = expected.get('boundary_specs')
            if specs:
                spec = specs[key[1]]
                shape = spec.get('sample_shape')
            else:
                from runtime.workload import legacy_specs
                spec = legacy_specs([key[1]])[key[1]]
                shape = ([expected['hidden_size']] if spec['level'] == 'model'
                         else ['valid_tokens', None if module_shapes else expected['hidden_size']])
            observed = check.get('shape')
            if not isinstance(observed, list) or any(type(n) is not int or n < 1 for n in observed):
                raise ValueError('invalid forward module shape')
            if shape is None:
                shape = observed
            else:
                shape = [lengths[key[0]] if n == 'valid_tokens' else n for n in shape]
                if len(shape) != len(observed):
                    raise ValueError('invalid preview boundary shape')
                shape = [actual if wanted is None else wanted for wanted, actual in zip(shape, observed)]
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


def validate_forward_checks(cfg, root, result, prepared, probe=False):
    """Formal forward coverage; preview keeps its stricter complete-block contract."""
    expected = dict(read_json(Path(prepared)/'forward-inputs.json'))
    from runtime.adapters import model_adapter, create
    if expected.get('model_adapter') or cfg.get('model'):
        adapter = model_adapter({'model': {'name': expected.get('model_adapter') or cfg['model']['name']}})
    elif expected.get('schema_version') == 1:
        adapter = create('models', 'qwen3_embedding_0.6b')
    else:
        raise ValueError('missing workload adapter metadata')
    names = ['all'] if probe else cfg['accuracy']['layers']
    specs = adapter.prepared_specs(expected, names)
    layers = not (probe and cfg.get('_layer_audit')) and (probe or 'layer' in cfg['accuracy']['levels'])
    selected = [s for s in specs if s['level'] == 'model' or layers]
    expected['boundaries'] = [s['name'] for s in selected]
    expected['boundary_specs'] = {s['name']: s for s in selected}
    if expected.get('execution_kind')=='autoregressive':
        expected['execution_modes']=['performance'] if cfg.get('_layer_audit') else ['accuracy','performance'] if probe else ['accuracy']
    try:
        return validate_checks(root, result, expected, module_shapes=True)
    except ValueError as error:
        raise ValueError(str(error).replace('preview', 'formal forward')) from error


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
