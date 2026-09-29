# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""PyTorch execution with bounded module capture and auditable routing."""
from collections import defaultdict
from contextlib import nullcontext
import functools
import json
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from models.qwen3_embedding.model import pool, selected_layers


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


class GemsRoute:
    """Wrap actual registered functions, not the surrounding ATen call count."""
    def __init__(self, include, directory):
        self.include, self.directory = include, Path(directory)
        self.calls = defaultdict(int)
        self.signatures = {}
        self.registered = []
        self.last_error = None
        self.lib = None
        self.log = None
        self.depth = 0
        self.scopes = None

    def enable(self):
        import flag_gems
        from flag_gems.runtime.op_registrar import GeneralOpRegistrar
        owner = self
        class EvidenceRegistrar(GeneralOpRegistrar):
            def register_impl(self, key, fn, extra_dispatch_keys=()):
                @functools.wraps(fn)
                def observed(*args, **kwargs):
                    owner.calls[fn.__name__] += 1
                    if owner.scopes: owner.scopes.observe(fn.__name__, 'function')
                    if fn.__name__ not in owner.signatures:
                        sig = signature({'args':args,'kwargs':kwargs})
                        owner.signatures[fn.__name__] = sig
                        owner.log.write(json.dumps({'function':fn.__name__,'aten_key':key,'signature':sig})+'\n')
                        owner.log.flush()
                    try:
                        owner.depth += 1
                        return fn(*args, **kwargs)
                    except Exception as error:
                        owner.last_error = {'function':fn.__name__, 'aten_key':key,
                                            'error':str(error), 'signature':signature({'args':args,'kwargs':kwargs})}
                        raise
                    finally:
                        owner.depth -= 1
                super().register_impl(key, observed, extra_dispatch_keys)
                owner.registered.append({'aten_key':key,'function':fn.__name__})
        self.log = (self.directory / 'gems-calls.jsonl').open('w')
        if not self.include:
            return
        self.lib = torch.library.Library('aten','IMPL')
        flag_gems.only_enable(include=self.include, lib=self.lib, registrar=EvidenceRegistrar,
                              record=True, once=True, path=str(self.directory/'gems.log'))
        if not self.registered:
            raise RuntimeError('requested FlagGems functions did not register')

    def evidence(self):
        return {'requested':True, 'allowed':self.include,'registered':self.registered,
                'actual_function_calls':dict(self.calls), 'signatures':self.signatures,
                'observed':bool(self.calls), 'last_error':self.last_error,
                'evidence_level':'Python FlagGems function invocation, not hardware kernel trace'}

    def close(self):
        if self.lib is not None: self.lib._destroy()
        if self.log is not None: self.log.close()


def candidates(inventory):
    import flag_gems
    by_function = {}
    for item in flag_gems._FULL_CONFIG:
        key, fn = item[:2]
        if key in inventory:
            row = by_function.setdefault(fn.__name__, {'function':fn.__name__, 'aten_keys':[], 'signatures':[]})
            row['aten_keys'].append(key)
            row['signatures'].extend(inventory[key]['signatures'])
    return [by_function[k] for k in sorted(by_function)]


def forward(model, batches, cfg, device, directory, include=None, probe=False, evidence_mode=None, stages=None):
    root = Path(directory)
    route = GemsRoute(include, root) if include is not None else None
    captures, handles, errors = {}, [], []
    layer_audit = bool(probe and cfg.get('_layer_audit'))
    scopes = None
    if layer_audit:
        from runtime.layer_capture import RouteScopes
        scopes = RouteScopes(selected_layers(model, cfg['performance']['layers']))
        if route: route.scopes = scopes
    names = cfg['accuracy']['layers'] if not probe else ['all']
    if not layer_audit and (probe or 'layer' in cfg['accuracy']['levels']):
        for name, module in selected_layers(model, names).items():
            def hook(_module, _args, output, name=name):
                value = output[0] if isinstance(output,(tuple,list)) else output
                if not isinstance(value,torch.Tensor) or value.ndim != 3:
                    raise ValueError(f'{name}: expected batch/sequence/hidden tensor')
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
                    hidden = model(**inputs, use_cache=False, return_dict=True).last_hidden_state
                    pooled, embedding = pool(hidden, inputs['attention_mask'])
            finally:
                if scopes: scopes.active = False
            # All diagnostics run outside the monitored device scope.
            with measure('finite_checks'):
                values = {'pooled':pooled.detach().cpu(), 'embedding':embedding.detach().cpu(), **captures}
                mask = batch['inputs']['attention_mask'].bool()
                batch_checks = []
                for i, sample in enumerate(batch['ids']):
                    for boundary, tensor in values.items():
                        row = tensor[i] if boundary in ['pooled','embedding'] else tensor[i][mask[i]]
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
                        'input_shape':list(batch['inputs']['input_ids'].shape),'checks':batch_checks})
            with measure('evidence_write'):
                if evidence_mode != 'lightweight':
                    artifact = root / f'batch-{index:04}.pt'
                    torch.save({'ids':batch['ids'],'mask':batch['inputs']['attention_mask'],'outputs':values}, artifact)
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
