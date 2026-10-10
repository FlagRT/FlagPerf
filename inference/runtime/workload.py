"""Model-neutral tensor boundaries, sample views and fixed-input metadata."""
from dataclasses import asdict, dataclass
from abc import ABC, abstractmethod
import time
from pathlib import Path

from runtime.common import read_json


@dataclass(frozen=True)
class BoundarySpec:
    name: str
    level: str
    sample_axis: int = 0
    valid_region: object = None  # {input: tensor name, axis: axis AFTER selecting sample}
    layout: str = 'replicated'
    sample_shape: object = None  # positive dimensions, null, or "valid_tokens"
    output_path: tuple = ()
    phase_shapes: object = None  # optional sealed sample shapes for stepped execution

    def metadata(self):
        value = asdict(self)
        value['output_path'] = list(self.output_path)
        if self.phase_shapes is None:
            value.pop('phase_shapes')
        return value


@dataclass
class LoadedModel:
    model: object
    total_seconds: float
    cpu_load_seconds: object = None
    transfer_seconds: object = None
    transfer_bytes: object = None
    transfer_scope: str = 'not measured by this model loader'
    unmeasured_reason: object = 'loader does not expose separate CPU load and device transfer'


class ModelAdapter(ABC):
    """Implementations own input/output semantics; orchestration owns evidence lifetime."""
    @abstractmethod
    def validate_config(self, cfg): ...
    @abstractmethod
    def validate_assets(self, cfg): ...
    @abstractmethod
    def asset_fingerprints(self, cfg): ...
    @abstractmethod
    def prepare_inputs(self, cfg): ...
    @abstractmethod
    def load(self, cfg, device, parallel=None): ...
    @abstractmethod
    def forward(self, model, inputs): ...
    @abstractmethod
    def boundary_specs(self, cfg, model=None, names=None): ...
    @abstractmethod
    def selected_modules(self, model, names): ...

    def normalize_config(self, cfg):
        return cfg

    def input_layout(self, cfg):
        """None retains the legacy batch-axis-zero input contract."""
        return None

    execution_kind = 'single_step'

    def open_execution(self, model, inputs, mode='accuracy'):
        from runtime.execution import SingleStepExecution
        return SingleStepExecution(self, model, inputs, mode)

    def execution_steps(self, cfg, mode='performance'):
        return 1

    def load_model(self, cfg, device, session, parallel=None):
        session.synchronize()
        start = time.perf_counter_ns()
        model = self.load(cfg, device, parallel)
        session.synchronize()
        return LoadedModel(model, (time.perf_counter_ns()-start)/1e9)

    def load_via_cpu(self, cfg, device, session):
        from runtime.performance import tensor_storage_bytes
        session.synchronize()
        start = time.perf_counter_ns()
        model = self.load_cpu(cfg)
        if any(t.device.type != 'cpu' for t in list(model.parameters())+list(model.buffers())):
            raise ValueError('CPU loader returned device-resident tensors')
        size = tensor_storage_bytes(model)
        transfer = time.perf_counter_ns()
        model = model.to(device)
        session.synchronize()
        end = time.perf_counter_ns()
        return LoadedModel(model, (end-start)/1e9, (transfer-start)/1e9, (end-transfer)/1e9,
                           size, 'one model.to(device) after CPU load; includes allocation and synchronization', None)

    def reference_contract(self, cfg, files=None):
        raise ValueError('model adapter does not support reference packages')

    def reference_boundaries(self, cfg, command='accuracy', metadata=None):
        levels = ['model', 'layer'] if command == 'preview' else cfg['accuracy']['levels']
        names = ['all'] if command == 'preview' else cfg['accuracy']['layers']
        specs = list(metadata['boundary_specs'].values()) if metadata and metadata.get('boundary_specs') else [
            s.metadata() for s in self.boundary_specs(cfg,names=names)]
        defaults = metadata.get('default_layer_boundaries') if metadata else None
        return [s['name'] for s in specs if s['level'] in levels and
                (s['level'] == 'model' or (names == ['all'] and (defaults is None or s['name'] in defaults))
                 or s['name'] in names)]

    def export_spec(self, cfg, batch):
        raise ValueError('model adapter does not support graph export')

    def validate_outputs(self, outputs):
        pass

    def comparison_outputs(self, cfg, outputs):
        """Select numeric model boundaries after validating the complete output tree."""
        return outputs

    boundary_relations = {'kind': 'independent', 'edges': []}
    def module_value(self, output, spec):
        import torch
        value = output
        for key in spec.output_path:
            value = value[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(spec.name + ': boundary extractor must return a tensor')
        return value

    def performance_output(self, outputs):
        return outputs

    def output_transfer(self, outputs):
        return outputs

    transfer_key = 'output'
    transfer_objects = ['model_output']
    transfer_scope = 'first prepared batch output tree copied after an extra forward; transfer only'

    def sample_counts(self, batch):
        return len(batch['ids']), None

    def valid_regions(self, batch):
        return {}

    def artifact_extras(self, batch):
        return {}

    def communication_roles(self, model):
        return {}

    def instrumentation(self, cfg):
        return [s.name for s in self.boundary_specs(cfg) if s.level == 'layer']

    def expected_collectives(self, model):
        return None

    def parallel_metadata(self, cfg, model):
        return {'strategy': self.parallel_strategy, 'rank_local_outputs': True}

    def metadata(self, cfg, model=None):
        value = {'schema_version': 1, 'model_adapter': cfg['model']['name'],
                'semantics': self.semantics, 'timing': self.timing,
                'parallel_strategy': self.parallel_strategy,
                'boundaries': [s.metadata() for s in self.boundary_specs(cfg, model)],
                'boundary_relations': self.boundary_relations,
                'throughput_unit': self.throughput_unit,
                'transfer_key': self.transfer_key, 'transfer_objects': self.transfer_objects}
        layout = self.input_layout(cfg)
        if layout is not None:
            from runtime.input_layout import validate_layout
            validate_layout(layout)
            value['input_layout'] = layout
        return value

    def forward_inputs(self, cfg, batches, model=None):
        from runtime.input_layout import tensor_metadata
        layout = self.input_layout(cfg)
        value = {'schema_version': 2, 'model_adapter': cfg['model']['name'],
                'batches': [self.batch_metadata(b, i) for i, b in enumerate(batches)],
                'input_tensors': [tensor_metadata(b['inputs'], layout) for b in batches],
                'boundaries': [s.name for s in self.boundary_specs(cfg, model)],
                'boundary_specs': {s.name: s.metadata() for s in self.boundary_specs(cfg, model)}}
        if layout is not None:
            value['input_layout'] = layout
        return value

    def prepared_specs(self, expected, names):
        specs = expected['boundary_specs']
        selected = [s for s in specs.values() if s['level'] == 'model']
        layers = expected.get('default_layer_boundaries', [n for n, s in specs.items() if s['level'] == 'layer']) if names == ['all'] else names
        return selected + [specs[name] for name in layers]

    def batch_metadata(self, batch, index=None):
        first = next(iter(batch['inputs'].values()))
        return {**({'batch_index': index} if index is not None else {}),
                'sample_ids': batch['ids'], 'input_shape': list(first.shape)}

    def validate_batches(self, batches, samples):
        import torch
        if not isinstance(batches, list) or not batches:
            raise ValueError('empty fixed input archive')
        ids = []
        for batch in batches:
            if not batch['ids'] or not batch['inputs'] or any(
                    not isinstance(t, torch.Tensor) or t.device.type != 'cpu' for t in batch['inputs'].values()):
                raise ValueError('fixed inputs must be nonempty CPU tensor mappings')
            ids.extend(batch['ids'])
        if ids != [s['id'] for s in samples] or len(set(ids)) != len(ids):
            raise ValueError('fixed input sample order/coverage mismatch')


def tensor_tree(value, fn):
    import torch
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: tensor_tree(v, fn) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensor_tree(v, fn) for v in value)
    raise TypeError('model output must be a tensor or a tree of tensors')


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in tensors(v)]
    raise TypeError('model output must be a tensor or a tree of tensors')


def sample_view(tensor, index, spec, regions):
    import torch
    if isinstance(spec, BoundarySpec):
        spec = spec.metadata()
    if spec['level'] not in ('model', 'layer') or spec['layout'] not in ('replicated', 'rank_local', 'sharded'):
        raise ValueError('invalid boundary level/layout')
    axis = spec.get('sample_axis', 0)
    if type(axis) is not int or not 0 <= axis < tensor.ndim:
        raise ValueError('invalid boundary sample axis')
    row = tensor.select(axis, index)
    region = spec.get('valid_region')
    if region:
        mask = regions[region['input']][index].bool()
        valid_axis = region['axis']
        if mask.ndim != 1 or mask.numel() != row.shape[valid_axis]:
            raise ValueError('boundary valid region shape differs from output')
        row = row.index_select(valid_axis, torch.nonzero(mask).flatten())
    return row


def legacy_specs(names):
    # Legacy artifact schema is explicitly the Qwen3 contract, defined by its adapter.
    from runtime.adapters import create
    return create('models', 'qwen3_embedding_0.6b').legacy_specs(names)


def artifact_specs(batch):
    return batch.get('boundary_specs') or legacy_specs(batch['outputs'])


def artifact_regions(batch):
    if 'valid_regions' in batch:
        return batch['valid_regions']
    from runtime.adapters import create
    return create('models', 'qwen3_embedding_0.6b').legacy_regions(batch)


def artifact_records(batch):
    specs, regions = artifact_specs(batch), artifact_regions(batch)
    return {sid: {name: sample_view(tensor, i, specs[name], regions).float().numpy()
                  for name, tensor in batch['outputs'].items()} for i, sid in enumerate(batch['ids'])}


def saved_metadata(prepared):
    path = Path(prepared) / 'workload.json'
    if path.is_file():
        return read_json(path)
    from runtime.adapters import model_adapter
    identity = read_json(Path(prepared) / 'identity.json')
    return model_adapter(identity).legacy_metadata(identity)


def report_metadata(root, identity):
    path = Path(root) / 'prepared/workload.json'
    if path.is_file():
        return read_json(path)
    if identity.get('model', {}).get('name') == 'qwen3_embedding_0.6b':
        from runtime.adapters import model_adapter
        adapter = model_adapter(identity)
        if hasattr(adapter, 'legacy_metadata'):
            return adapter.legacy_metadata(identity)
    return {}
