# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Private subprocess protocol; every device worker has its own Python context."""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import traceback
from contextlib import nullcontext
import yaml
from runtime.common import ROOT, write_json, read_json, digest, file_hash, source_identity, source_snapshot, assert_source_snapshot


def identity(cfg, backend):
    model = Path(cfg['model']['path'])
    files = {str(p.relative_to(model)):file_hash(p) for p in sorted(model.rglob('*'))
             if p.is_file() and p.suffix in ['.json','.safetensors','.model','.txt']}
    import torch
    import triton
    from vendors.stack import compiler_identity, communication_backend
    spec = importlib.util.find_spec('flag_gems')
    gems = {}
    if spec and spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations)))
        gems = {str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*.py'))}
    env = {k:v for k,v in sorted(os.environ.items()) if k.startswith(('ASCEND','CANN','HCCL_','GEMS_','TRITON_','TASK_QUEUE','TORCH_','CUDA_','HF_','OMP_'))
           and k not in {'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_NCCL_ASYNC_ERROR_HANDLING'}}
    result = {'model_files':files, 'model':{k:v for k,v in cfg['model'].items() if k!='path'},
            'inputs':{k:v for k,v in cfg['inputs'].items() if k!='path'},
            'input_file_sha256':file_hash(cfg['inputs']['path']),
            'seed':cfg['runtime']['seed'],'vendor':cfg['runtime']['vendor'],
            'physical_device':cfg['runtime']['device'],'device_name':backend.get_device_name(0),
            'device_properties':{'name':backend.get_device_name(0),
                                 'total_memory':backend.get_device_properties(0).total_memory},
            'image_id':os.environ.get('FLAGPERF_IMAGE_ID','unknown-container-image'),
            'python':sys.version, 'packages':sorted([d.metadata['Name'],d.version] for d in importlib.metadata.distributions()),
            'torch_import':torch.__version__, 'triton_import':triton.__version__,
            'compiler':compiler_identity(cfg), 'communication_backend':communication_backend(cfg) if cfg['runtime'].get('parallelism')=='tp' else 'not_applicable',
            'gems_source_hash':digest(gems), 'environment':env,'source_files':source_identity()}
    if cfg['runtime'].get('parallelism') == 'tp':
        result.pop('physical_device')
        result['parallelism'] = {'mode':'tp','devices':cfg['runtime']['devices'],
                                'backend':communication_backend(cfg),'tp_plan':'transformers_qwen3_auto',
                                'world_size':len(cfg['runtime']['devices'])}
        result['devices'] = [{'rank':r,'physical_device':d,'logical_device':r,
                             'name':backend.get_device_name(r),
                             'total_memory':backend.get_device_properties(r).total_memory}
                            for r,d in enumerate(cfg['runtime']['devices'])]
    return result


def prepare(cfg, root, shared=None):
    from vendors.device import initialize
    from models.qwen3_embedding.model import tokenize
    import torch
    _, backend = initialize(cfg)
    if shared is None:
        batches, samples = tokenize(cfg)
        torch.save(batches, root/'inputs.pt')
    else:
        import shutil
        shutil.copyfile(shared/'inputs.pt',root/'inputs.pt')
        batches = torch.load(root/'inputs.pt',map_location='cpu',weights_only=True)
        samples = read_json(shared/'samples.json')
    write_json(root/'samples.json',samples)
    from runtime.preview_evidence import input_batches
    architecture = read_json(Path(cfg['model']['path'])/'config.json')
    write_json(root/'forward-inputs.json', {'schema_version': 1, 'batches': input_batches(batches),
        'hidden_size': architecture['hidden_size'],
        'boundaries': [f'layers.{i}' for i in range(architecture['num_hidden_layers'])]+['pooled','embedding']})
    ident = identity(cfg,backend)
    ident['tokenized_sha256'] = file_hash(root/'inputs.pt')
    write_json(root/'identity.json',ident)
    write_json(root/'presentation-source.json',
               {'reporting/model.py':file_hash(ROOT/'reporting/model.py')})
    write_json(root/'identity-key.json',{'key':digest(ident)})
    snapshot = source_snapshot()
    write_json(root/'analysis-identity.json', {'source_files':snapshot['analysis'], 'key':snapshot['analysis_key']})
    return {'status':'completed','identity_key':digest(ident),'samples':len(samples),
            'batch_shapes':[list(b['inputs']['input_ids'].shape) for b in batches]}


def execute_forward(cfg, root, prepared, include, probe):
    from runtime.preview_evidence import Stages, input_batches
    preview = cfg.get('_preview_trial')
    stages = Stages(root) if preview else None
    measure = stages.measure if stages else lambda name: nullcontext()
    with measure('initialization'):
        import torch
        from vendors.device import initialize
        from models.qwen3_embedding.model import load_model
        from engines.pytorch import forward, candidates
        device, backend = initialize(cfg)
    with measure('identity'):
        expected = read_json(prepared/'identity.json')
        # A complete content check still runs in every independent worker.
        current = identity(cfg,backend)
        current['tokenized_sha256'] = file_hash(prepared/'inputs.pt')
        if current != expected:
            write_json(root/'observed-identity.json',current)
            from runtime.common import identity_differences
            differences = identity_differences(expected,current)
            raise ValueError('worker identity differs from prepared inputs/environment: '+', '.join(sorted(differences)))
    with measure('model_load'):
        model = load_model(cfg,device)
    with measure('input_load'):
        batches = torch.load(prepared/'inputs.pt',map_location='cpu',weights_only=True)
        if preview:
            expected_checks = {'schema_version': 1, 'batches': input_batches(batches),
                'hidden_size': model.config.hidden_size,
                'boundaries': [f'layers.{i}' for i in range(len(model.layers))]+['pooled','embedding']}
            if expected_checks != read_json(prepared/'forward-inputs.json'):
                raise ValueError('prepared forward check workload differs from sealed inputs/model')
    from runtime.stack_audit import StackAudit
    with StackAudit(cfg,root,compiler=current['compiler'] if preview else None) as audit:
        result = forward(model,batches,cfg,device,root,include,probe,
                         evidence_mode=preview['evidence_mode'] if preview else None, stages=stages)
        backend.synchronize()
    result["components"] = audit.result
    backend.synchronize()
    if probe and include is None and (not preview or preview['phase'] == 'baseline'):
        result['candidates'] = candidates(result['inventory'])
    result['status'] = 'completed'
    result['identity_key'] = digest(expected)
    return result


def export_graphs(cfg, root, prepared):
    import torch
    from models.qwen3_embedding.model import load_model, ExportModel
    # Graph export is explicitly CPU/native with the requested weight dtype.
    model = ExportModel(load_model(cfg,'cpu')).eval()
    batch = torch.load(prepared/'inputs.pt',map_location='cpu',weights_only=True)[0]['inputs']
    inputs = (batch['input_ids'],batch['attention_mask'])
    results = {'status':'completed','device':'cpu','flaggems':'off','dtype':cfg['model']['dtype'],
               'scope':'fixed first prepared batch; structural export, not a FlagGems route trace',
               'input_shape':list(inputs[0].shape),'formats':{}}
    with torch.inference_mode():
        expected = model(*inputs)
        exported = torch.export.export(model,inputs,strict=False)
        for fmt in cfg['export']['formats']:
            try:
                if fmt == 'fx':
                    torch.export.save(exported,root/'model.pt2')
                    loaded = torch.export.load(root/'model.pt2')
                    actual = loaded.module()(*inputs)
                    from analysis.metrics import metrics
                    checks = [metrics(a.float().numpy(), b.float().numpy()) for a,b in zip(expected,actual)]
                    # Same native computation round-trip, exact equality is expected.
                    if not all(torch.equal(a,b) for a,b in zip(expected,actual)):
                        raise RuntimeError(f'FX round-trip output mismatch: {checks}')
                    (root/'model.fx.txt').write_text(str(exported.graph_module.graph))
                    write_json(root/'fx-nodes.json',[{'name':n.name,'op':n.op,'target':str(n.target)} for n in exported.graph.nodes])
                    results['formats'][fmt] = {'status':'completed','roundtrip':'exact_equal','metrics':checks}
                else:
                    import onnx
                    torch.onnx.export(exported,args=(),f=str(root/'model.onnx'),dynamo=True,
                                      external_data=True,input_names=['input_ids','attention_mask'],
                                      output_names=['pooled','embedding'],opset_version=18)
                    onnx.checker.check_model(str(root/'model.onnx'))
                    graph = onnx.load(str(root/'model.onnx'),load_external_data=False)
                    write_json(root/'onnx-nodes.json',[{'name':n.name,'op_type':n.op_type,'domain':n.domain} for n in graph.graph.node])
                    results['formats'][fmt] = {'status':'completed','structural_check':'passed','numerical_runtime_validation':'not_run'}
            except Exception:
                results['formats'][fmt] = {'status':'failed','error':traceback.format_exc()}
                results['status'] = 'partial'
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','forward','export','performance','profile',
                                   'layer_timing','layer_memory','layer_profile'])
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--prepared')
    p.add_argument('--include',help='JSON list; omitted means native off, [] means explicit empty on')
    p.add_argument('--probe',action='store_true')
    p.add_argument('--repeat-index',type=int,default=0)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.output)
    distributed = cfg['runtime'].get('parallelism') == 'tp' and args.action != 'prepare'
    if distributed:
        root = root / ('rank-'+os.environ['RANK'])
    root.mkdir(parents=True,exist_ok=True)
    result = {'status':'failed','stage':args.action}
    try:
        assert_source_snapshot(cfg.get('_source_snapshot'))
        if args.action == 'prepare': result = prepare(cfg,root,Path(args.prepared) if args.prepared else None)
        elif args.action == 'export': result = export_graphs(cfg,root,Path(args.prepared))
        elif args.action in ['performance','profile','layer_timing','layer_memory','layer_profile']:
            import json
            if distributed:
                from runtime.tp import run
            else:
                from runtime.performance import run
            result = run(cfg,root,Path(args.prepared),
                         json.loads(args.include) if args.include is not None else None,args.repeat_index,
                         **({'profile':args.action=='profile'} if distributed else {}),
                         **({'phase':args.action} if args.action.startswith('layer_') else {}))
        else:
            import json
            result = execute_forward(cfg,root,Path(args.prepared),json.loads(args.include) if args.include is not None else None,args.probe)
        assert_source_snapshot(cfg.get('_source_snapshot'))
        result['analysis_key'] = source_snapshot()['analysis_key']
    except Exception as error:
        result['status'] = 'failed'
        result.update(error=str(error),traceback=traceback.format_exc(),exception_type=type(error).__name__)
        if getattr(error, 'errno', None) is not None: result['errno'] = error.errno
        traceback.print_exc()
    finally:
        write_json(root/'result.json',result)
        if distributed:
            from runtime.tp import close_group
            from runtime.preview_evidence import Stages
            with Stages(root,append=True).measure('cleanup') if cfg.get('_preview_trial') else nullcontext():
                close_group()
            from vendors.stack import communication_backend
            result['cleanup'] = {'groups_deregistered':True,
                'native_backend_lifetime':'worker_process' if communication_backend(cfg)=='flagcx' else 'group'}
            write_json(root/'result.json',result)
    return 0 if result['status']=='completed' else 1

if __name__ == '__main__':
    raise SystemExit(main())
