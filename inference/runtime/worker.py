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
from runtime.adapters import model_adapter, vendor_adapter, required_components, descriptor, SCHEME, create
from runtime.common import ROOT, write_json, read_json, digest, file_hash, source_identity, source_snapshot, assert_source_snapshot


def identity(cfg, backend):
    adapter = model_adapter(cfg)
    files = adapter.asset_fingerprints(cfg)
    import torch
    from vendors.stack import compiler_identity, communication_backend
    required = required_components(cfg)
    compiler = compiler_identity(cfg)
    gems = create('components', 'flaggems').source_identity() if 'flaggems' in required else None
    if 'flagcx' in required and importlib.util.find_spec('flagcx') is None:
        raise ValueError('selected FlagCX component is not installed')
    component_sources = ({'flagcx': create('components', 'flagcx').source_identity()}
                         if 'flagcx' in required else {})
    packages = {'torch', 'pyyaml'}
    entries = [('models', cfg['model']['name']), ('vendors', cfg['runtime']['vendor'])]
    entries += [('components', name) for name in required]
    for kind, name in entries:
        packages.update(descriptor(kind, name).get('packages', []))
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
            'python':sys.version, 'packages':sorted([d.metadata['Name'],d.version] for d in importlib.metadata.distributions()
                if d.metadata and d.metadata.get('Name') and d.metadata['Name'].lower().replace('_', '-') in packages),
            'torch_import':torch.__version__, 'triton_import':compiler.get('import_version'),
            'cpu_threads':{'intraop':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),
                           'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None},
            'compiler':compiler, 'communication_backend':communication_backend(cfg) if cfg['runtime'].get('parallelism')=='tp' else 'not_applicable',
            'gems_source_hash':gems,
            'environment':env, 'identity_scheme':SCHEME, 'component_sources':component_sources,
            'source_files':source_identity(cfg)}
    if cfg['runtime'].get('parallelism') == 'tp':
        result.pop('physical_device')
        result['parallelism'] = {'mode':'tp','devices':cfg['runtime']['devices'],
                                'backend':communication_backend(cfg),'tp_plan':adapter.parallel_strategy,
                                'world_size':len(cfg['runtime']['devices'])}
        result['devices'] = [{'rank':r,'physical_device':d,'logical_device':r,
                             'name':backend.get_device_name(r),
                             'total_memory':backend.get_device_properties(r).total_memory}
                            for r,d in enumerate(cfg['runtime']['devices'])]
    if cfg.get('_performance_native'):
        settings = vendor_adapter(cfg).numerical_settings()
        if settings is not None:
            result['performance_numerics'] = settings
    return result


def export_identity(cfg):
    """CPU graph preparation has no accelerator or component runtime identity."""
    import torch
    adapter = model_adapter(cfg)
    return {'execution': 'cpu-native-export', 'device_name': 'cpu',
            'model_files': adapter.asset_fingerprints(cfg),
            'model': {k: v for k, v in cfg['model'].items() if k != 'path'},
            'inputs': {k: v for k, v in cfg['inputs'].items() if k != 'path'},
            'input_file_sha256': file_hash(cfg['inputs']['path']),
            'seed': cfg['runtime']['seed'], 'python': sys.version,
            'torch_import': torch.__version__,
            'image_id': os.environ.get('FLAGPERF_IMAGE_ID', 'unknown-container-image'),
            'source_files': source_identity(cfg)}


def prepare(cfg, root, shared=None):
    if cfg.get('reference',{}).get('path'):
        from runtime.oracle import validate_reference
        manifest = validate_reference(cfg)
        if shared is None:
            shared = Path(cfg['reference']['path'])/'prepared'
    if cfg.get('reference',{}).get('performance_path'):
        from runtime.performance_baseline import validate_reference
        manifest = validate_reference(cfg)
        if shared is None:
            shared = Path(cfg['reference']['performance_path'])/'prepared'
    adapter = model_adapter(cfg)
    import torch
    cpu_export = cfg.get('_command') == 'export'
    if cpu_export:
        torch.manual_seed(cfg['runtime']['seed'])
    else:
        from runtime.context import initialize
        _, backend = initialize(cfg)
    if shared is None:
        batches, samples = adapter.prepare_inputs(cfg)
        torch.save(batches, root/'inputs.pt')
    else:
        import shutil
        shutil.copyfile(shared/'inputs.pt',root/'inputs.pt')
        batches = torch.load(root/'inputs.pt',map_location='cpu',weights_only=True)
        samples = read_json(shared/'samples.json')
    write_json(root/'samples.json',samples)
    if cfg.get('reference',{}).get('path'):
        if file_hash(root/'inputs.pt') != manifest['files']['prepared/inputs.pt']['sha256']:
            raise ValueError('prepared tokens differ from the selected oracle')
    if cfg.get('reference',{}).get('performance_path'):
        if file_hash(root/'inputs.pt') != manifest['files']['prepared/inputs.pt']['sha256']:
            raise ValueError('prepared tokens differ from performance baseline')
    adapter.validate_batches(batches, samples)
    write_json(root/'workload.json', adapter.metadata(cfg))
    write_json(root/'forward-inputs.json', adapter.forward_inputs(cfg, batches))
    write_json(root/'environment-packages.json', sorted([d.metadata['Name'],d.version] for d in importlib.metadata.distributions()
               if d.metadata and d.metadata.get('Name')))
    ident = export_identity(cfg) if cpu_export else identity(cfg,backend)
    if cfg.get('_oracle_capture'):
        ident['oracle_numerics'] = vendor_adapter(cfg).numerical_settings()
    ident['tokenized_sha256'] = file_hash(root/'inputs.pt')
    write_json(root/'identity.json',ident)
    write_json(root/'presentation-source.json',
               {'runtime/reporting.py':file_hash(ROOT/'runtime/reporting.py')})
    write_json(root/'identity-key.json',{'key':digest(ident)})
    snapshot = source_snapshot(cfg)
    write_json(root/'analysis-identity.json', {'source_files':snapshot['analysis'], 'key':snapshot['analysis_key']})
    return {'status':'completed','identity_key':digest(ident),'samples':len(samples),
            'batch_shapes':[adapter.batch_metadata(b)['input_shape'] for b in batches]}


def execute_forward(cfg, root, prepared, include, probe):
    from runtime.preview_evidence import Stages, input_batches
    preview = cfg.get('_preview_trial')
    stages = Stages(root) if preview else None
    measure = stages.measure if stages else lambda name: nullcontext()
    with measure('initialization'):
        import torch
        from runtime.context import initialize
        adapter = model_adapter(cfg)
        from runtime.context import parallel_context
        engine = create('engines', 'pytorch')
        device, backend = initialize(cfg)
    with measure('identity'):
        expected = read_json(prepared/'identity.json')
        # A complete content check still runs in every independent worker.
        current = identity(cfg,backend)
        if cfg.get('_oracle_capture'):
            current['oracle_numerics'] = vendor_adapter(cfg).numerical_settings()
        current['tokenized_sha256'] = file_hash(prepared/'inputs.pt')
        if current != expected:
            write_json(root/'observed-identity.json',current)
            from runtime.common import identity_differences
            differences = identity_differences(expected,current)
            raise ValueError('worker identity differs from prepared inputs/environment: '+', '.join(sorted(differences)))
    with measure('model_load'):
        model = adapter.load_model(cfg, device, backend, parallel_context(cfg)).model
    with measure('input_load'):
        batches = torch.load(prepared/'inputs.pt',map_location='cpu',weights_only=True)
        if preview:
            expected_checks = adapter.forward_inputs(cfg, batches, model)
            if expected_checks != read_json(prepared/'forward-inputs.json'):
                raise ValueError('prepared forward check workload differs from sealed inputs/model')
    from runtime.stack_audit import StackAudit
    with StackAudit(cfg,root,compiler=current['compiler'] if preview else None) as audit:
        result = engine.forward(model,batches,cfg,device,root,include,probe,
                         evidence_mode=preview['evidence_mode'] if preview else 'full', stages=stages)
        backend.synchronize()
    result["components"] = audit.result
    if cfg.get('reference',{}).get('path') and cfg['runtime'].get('parallelism') == 'tp':
        result['oracle_layout'] = {**adapter.oracle_layout(cfg, model),
            'rank':int(os.environ['RANK']), 'world_size':len(cfg['runtime']['devices'])}
    backend.synchronize()
    if probe and include is None and cfg.get('_candidate_mapping',True) and (not preview or preview['phase'] == 'baseline'):
        result['candidates'] = engine.candidates(result['inventory'])
    result['status'] = 'completed'
    result['identity_key'] = digest(expected)
    return result


def export_graphs(cfg, root, prepared):
    import torch
    adapter = model_adapter(cfg)
    batch = torch.load(prepared/'inputs.pt',map_location='cpu',weights_only=True)[0]['inputs']
    model, inputs, input_names, output_names = adapter.export_spec(cfg, batch)
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
                    from runtime.metrics import metrics
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
                                      external_data=True,input_names=input_names,
                                      output_names=output_names,opset_version=18)
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
    from runtime.monitor_events import emit
    selected = cfg.get('_stack', cfg.get('performance', {}))
    emit(root, 'worker_start', action=args.action, repeat=args.repeat_index,
         side=next((s for s in root.parts if s in ('off','on')), None),
         components={k:selected.get(k, 'off') for k in ('flaggems','flagtree','flagcx')})
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
        emit(root, 'worker_end', status=result.get('status'))
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
