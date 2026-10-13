"""PyTorch execution with bounded module capture and auditable routing."""
from collections import defaultdict
from contextlib import nullcontext
import functools
import json
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from runtime.adapters import model_adapter, create
from runtime.workload import sample_view


def signature(value):
    if isinstance(value, torch.Tensor):
        return {'shape':list(value.shape), 'dtype':str(value.dtype), 'device':str(value.device),
                'stride':list(value.stride())}
    if isinstance(value,(list,tuple)):
        return [signature(v) for v in value]
    if isinstance(value,dict):
        return {str(k):signature(v) for k,v in value.items()}
    return value if isinstance(value,(int,float,str,bool,type(None))) else str(value)


class Inventory(TorchDispatchMode):
    def __init__(self, route=None, scopes=None):
        super().__init__()
        self.ops = {}
        self.route = route
        self.scopes = scopes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        nested = bool(self.route and self.route.depth)
        output = func(*args, **(kwargs or {}))
        if nested:
            return output
        def accelerator(value):
            if isinstance(value,torch.Tensor): return value.device.type != 'cpu'
            if isinstance(value,(tuple,list)): return any(accelerator(v) for v in value)
            if isinstance(value,dict): return any(accelerator(v) for v in value.values())
            return False
        if not accelerator((args,kwargs or {},output)):
            return output
        name = str(func).replace('aten.','',1)
        if name.endswith('.default'): name = name[:-8]
        if self.scopes: self.scopes.observe(name, 'aten')
        sig = signature({'args':args,'kwargs':kwargs or {}})
        row = self.ops.setdefault(name, {'calls':0,'signatures':[]})
        row['calls'] += 1
        if sig not in row['signatures'] and len(row['signatures']) < 32:
            row['signatures'].append(sig)
        return output


def GemsRoute(include, directory):
    return create('components', 'flaggems').route(include, directory)


def candidates(inventory):
    return create('components', 'flaggems').candidates(inventory)


class PyTorchEngine:
    def forward(self, *args, **kwargs):
        return forward(*args, **kwargs)

    def candidates(self, inventory):
        return candidates(inventory)


def forward(model, batches, cfg, device, directory, include=None, probe=False, evidence_mode=None, stages=None):
    root = Path(directory)
    adapter = model_adapter(cfg)
    route = GemsRoute(include, root) if include is not None else None
    captures, handles, errors = {}, [], []
    layer_audit = bool(probe and cfg.get('_layer_audit'))
    scopes = None
    if layer_audit:
        from runtime.layer_capture import RouteScopes
        scopes = RouteScopes(adapter.selected_modules(model, cfg['performance']['layers']))
        if route: route.scopes = scopes
    names = cfg['accuracy']['layers'] if not probe else ['all']
    specs = {s.name: s for s in adapter.boundary_specs(cfg, model, names=names)}
    if adapter.execution_kind != 'single_step':
        from runtime.step_capture import capture
        gen_inventory = Inventory(route,scopes)
        try:
            if route: route.enable()
            if scopes: scopes.__enter__()
            return capture(adapter,model,batches,cfg,device,root,inventory=gen_inventory,
                           route=route,scopes=scopes,probe=probe,evidence_mode=evidence_mode or 'full',stages=stages)
        finally:
            from runtime.common import write_json
            write_json(root/'inventory.json',gen_inventory.ops)
            if route:
                write_json(root/'route.json',route.evidence()); route.close()
            if scopes:
                write_json(root/'layer-routes.json',scopes.evidence()); scopes.__exit__(None,None,None)
    if not layer_audit and (probe or 'layer' in cfg['accuracy']['levels']):
        layer_names = [s.name for s in specs.values() if s.level == 'layer']
        for name, module in adapter.selected_modules(model, layer_names).items():
            def hook(_module, _args, output, name=name):
                value = adapter.module_value(output, specs[name])
                captures[name] = value.detach().cpu()
            handles.append(module.register_forward_hook(hook))
    inventory = Inventory(route, scopes)
    collect_inventory = bool(probe and (not cfg.get('_preview_trial') or
                                       cfg['_preview_trial']['phase'] == 'baseline'))
    artifacts = []
    checks = []
    measure = stages.measure if stages else lambda name: nullcontext()
    try:
        if route: route.enable()
        if scopes: scopes.__enter__()
        for index, batch in enumerate(batches):
            captures.clear()
            inputs = {k:v.to(device) for k,v in batch['inputs'].items()}
            if scopes: scopes.begin({'batch_index': index})
            try:
                with measure('forward_and_capture'), torch.inference_mode(), (inventory if collect_inventory else nullcontext()):
                    outputs = adapter.forward(model, inputs)
            finally:
                if scopes: scopes.active = False
            # All diagnostics run outside the monitored device scope.
            with measure('finite_checks'):
                complete_outputs = {name: tensor.detach().cpu() for name, tensor in outputs.items()}
                adapter.validate_outputs(complete_outputs)
                values = {**adapter.comparison_outputs(cfg, complete_outputs), **captures}
                regions = adapter.valid_regions(batch)
                batch_checks = []
                for i, sample in enumerate(batch['ids']):
                    for boundary, tensor in values.items():
                        if tensor.shape[specs[boundary].sample_axis] != len(batch['ids']):
                            raise ValueError(boundary + ': output sample count differs from fixed inputs')
                        row = sample_view(tensor, i, specs[boundary], regions)
                        finite = torch.isfinite(row)
                        good = bool(finite.all())
                        if not good: errors.append({'sample_id':sample,'boundary':boundary,'kind':'nonfinite'})
                        if evidence_mode:
                            bad = (~finite).nonzero() if not good else None
                            batch_checks.append({'sample_id':sample,'boundary':boundary,
                                'shape':list(row.shape),'dtype':str(row.dtype),'checked_elements':row.numel(),
                                'nonfinite_count':0 if good else int(bad.shape[0]),
                                'first_nonfinite':[] if good else [{'index':p.tolist(),'value':str(row[tuple(p.tolist())].item())} for p in bad[:16]]})
                if evidence_mode:
                    checks.append({'batch_index':index,'sample_ids':batch['ids'],
                        'input_shape':adapter.batch_metadata(batch)['input_shape'],'checks':batch_checks})
            with measure('evidence_write'):
                if evidence_mode != 'lightweight':
                    artifact = root / f'batch-{index:04}.pt'
                    torch.save({'ids':batch['ids'], 'outputs':values, **adapter.artifact_extras(batch),
                                'boundary_specs':{name:specs[name].metadata() for name in values},
                                'valid_regions':regions}, artifact)
                    artifacts.append(artifact.name)
                if evidence_mode:
                    # A killed worker leaves readable checks, but never a completed certificate.
                    from runtime.common import write_json
                    write_json(root/'forward-checks.json', {'schema_version':1,'mode':evidence_mode,
                               'complete':False,'batches':checks})
        evidence = {}
        if evidence_mode:
            from runtime.preview_evidence import seal_checks
            with measure('evidence_write'):
                evidence['forward_evidence'] = seal_checks(root,evidence_mode,checks)
        return {'numerical_anomalies':errors,'inventory':inventory.ops,'inventory_collected':collect_inventory,'artifacts':artifacts,
                **evidence,
                **({'layer_routes': scopes.evidence()} if scopes else {}),
                'route':route.evidence() if route else {'requested':False,'observed':False,'actual_function_calls':{}}}
    finally:
        from runtime.common import write_json
        if collect_inventory or not cfg.get('_preview_trial'):
            write_json(root/'inventory.json',inventory.ops)
        if route:
            write_json(root/'route.json',route.evidence())
            route.close()
        if scopes:
            write_json(root/'layer-routes.json', scopes.evidence())
            scopes.__exit__(None, None, None)
        for handle in handles: handle.remove()
