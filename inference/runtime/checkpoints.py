"""Dependency checkpoints for current-source offline comparison nodes."""
from pathlib import Path
from runtime.common import read_json, write_json, digest, file_hash
from runtime.artifacts import Records, pack


def verify_records(value):
    if isinstance(value, Records):
        for row in value: verify_records(row)
    elif isinstance(value, dict):
        for child in value.values(): verify_records(child)
    elif isinstance(value, list):
        for child in value: verify_records(child)


def capture_key(root, result, cfg, sources):
    root = Path(root)
    identity = read_json(root/'prepared/identity.json') if (root/'prepared/identity.json').is_file() else {}
    def semantic(section):
        return {key:value for key,value in section.items() if key != 'path' and not (key == 'options' and not value)}
    for group in ('model', 'inputs'):
        if group in identity and group in cfg and semantic(identity[group]) != semantic(cfg[group]):
            raise ValueError('saved configuration differs from captured '+group)
    inputs = {name: file_hash(root/name) for name in ('prepared/identity.json', 'prepared/inputs.pt', 'prepared/workload.json',
              'prepared/samples.json', 'prepared/forward-inputs.json', 'comparison-context.json') if (root/name).is_file()}
    captures = {'paths': result.get('paths'), 'layer_runs': result.get('layer', {}).get('runs'),
                'execution_status': result.get('execution_status', result.get('status'))}
    for side, state in result.get('paths', {}).items():
        if result.get('command') != 'accuracy': continue
        for rank, worker in state.get('ranks', {'single': state}).items():
            directory = root/side if rank == 'single' else root/side/f'rank-{rank}'
            for metadata in ('result.json', 'identity.json', 'execution.json'):
                path = directory/metadata
                if path.is_file(): inputs[path.relative_to(root).as_posix()] = file_hash(path)
            for name in worker.get('artifacts', []):
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts: raise ValueError('unsafe capture artifact')
                inputs[(directory/relative).relative_to(root).as_posix()] = file_hash(directory/relative)
    for field, module in (('path', 'oracle'), ('performance_path', 'performance_baseline')):
        reference = cfg.get('reference', {}).get(field)
        if reference:
            from importlib import import_module
            manifest = import_module('runtime.'+module).load_manifest(reference)
            contract = manifest['contract']
            for group in ('model', 'inputs'):
                if group in identity and semantic(identity[group]) != semantic(contract[group]):
                    raise ValueError('reference contract differs from captured '+group)
            for saved, expected in (('model_files','model_files'), ('seed','seed'), ('input_file_sha256','input_sha256')):
                if saved in identity and identity[saved] != contract.get(expected):
                    raise ValueError('reference contract differs from captured '+saved)
            if field == 'performance_path' and identity.get('parallelism', {}).get('world_size', 1) != contract['tp_degree']:
                raise ValueError('reference TP degree differs from capture')
            inputs['reference:'+field] = manifest.get('oracle_id', manifest.get('baseline_id'))
    verify_records(captures)
    return digest({'inputs': inputs, 'capture': pack(captures, root/'result.json'),
                   'analysis_sources': sources, 'options': {k: cfg.get(k) for k in ('accuracy', 'performance', 'reference')}})


class Checkpoints:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root/'analysis-checkpoints.json'
        self.value = read_json(self.path) if self.path.is_file() else {'storage_version': 1, 'nodes': {}}
        self.reused, self.executed = [], []

    def reusable(self, name, key, outputs):
        node = self.value['nodes'].get(name)
        if not node or node.get('key') != key or set(node.get('outputs', {})) != set(outputs): return False
        for relative, checksum in node['outputs'].items():
            path = self.root/relative
            if not path.is_file() or file_hash(path) != checksum: return False
            try:
                verify_records(read_json(path))
            except (OSError, ValueError, KeyError):
                return False
        self.reused.append(name)
        return True

    def save(self, name, key, outputs):
        self.value['nodes'][name] = {'key': key, 'outputs': {name: file_hash(self.root/name) for name in outputs}}
        self.executed.append(name)
        write_json(self.path, self.value)
