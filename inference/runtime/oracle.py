"""Versioned, self-contained A100 evidence; host validation never imports torch."""
from runtime.tensor_io import load_cpu
from pathlib import Path, PurePosixPath
from copy import deepcopy
import json
import os
import shutil
import time
from runtime.common import read_json, write_json, file_hash, digest, identity_differences

SCHEMA = 2
from runtime.adapters import model_adapter, create


def reference_adapter(contract=None):
    if contract is None:
        raise ValueError('reference adapter requires an explicit model contract')
    return model_adapter(contract)


def safe_file(root, name):
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or any(p in ('..','.') for p in name.split('/')):
        raise ValueError('unsafe oracle member: '+name)
    target = Path(root).joinpath(*path.parts)
    if any(p.is_symlink() for p in [target, *target.parents] if p != Path(root).parent):
        raise ValueError('oracle members must not be symlinks: '+name)
    if not target.is_file() or not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('missing or unsafe oracle member: '+name)
    return target


def model_files(cfg):
    return model_adapter(cfg).asset_fingerprints(cfg)


def contract(cfg, files=None):
    return model_adapter(cfg).reference_contract(cfg, files)


def selected_boundaries(cfg, command='accuracy', prepared=None, available=None):
    path = Path(prepared)/'forward-inputs.json' if prepared is not None else None
    metadata = (read_json(path) if path is not None and path.is_file()
                else {'boundaries': available} if available is not None else None)
    return model_adapter(cfg).reference_boundaries(cfg, command, metadata=metadata)


def load_manifest(root):
    root = Path(root).resolve()
    manifest = read_json(safe_file(root,'manifest.json'))
    if manifest.get('schema_version') not in (1, 2, 3, 4) or manifest.get('kind') != 'a100_accuracy' or manifest.get('status') != 'complete':
        raise ValueError('unsupported or incomplete oracle package')
    unsigned = {k:v for k,v in manifest.items() if k != 'oracle_id'}
    if manifest.get('oracle_id') != digest(unsigned):
        raise ValueError('oracle manifest digest mismatch')
    files = manifest.get('files')
    required = {'prepared/inputs.pt','prepared/samples.json','inputs.jsonl','outputs/index.json','provenance/identity.json'}
    if manifest.get('protocol_schema_version', manifest['schema_version']) in (2,3):
        required |= {'prepared/workload.json','prepared/forward-inputs.json'}
    if not isinstance(files,dict) or not required <= set(files):
        raise ValueError('oracle package lacks mandatory evidence')
    for name, info in files.items():
        p = safe_file(root,name)
        if p.stat().st_size != info['bytes'] or file_hash(p) != info['sha256']:
            raise ValueError('oracle member digest mismatch: '+name)
    from runtime.verification import semantic_validation
    if semantic_validation(root, manifest, 'oracle'): return manifest
    if manifest['schema_version']==3 or manifest.get('workload',{}).get('execution_kind') == 'autoregressive':
        if file_hash(root/'inputs.jsonl')!=manifest['contract']['input_sha256']:
            raise ValueError('generation reference input contract mismatch')
        from runtime.generation_reference import validate_index
        validate_index(root,manifest)
        semantic_validation(root, manifest, 'oracle', complete=True)
        return manifest
    index = read_json(root/'outputs/index.json')
    if not index or any(row['file'] not in files for row in index):
        raise ValueError('oracle output index is incomplete')
    if len({row['file'] for row in index}) != len(index):
        raise ValueError('duplicate oracle output files')
    if any(set(row['tensors']) != set(manifest['boundaries']) for row in index):
        raise ValueError('oracle boundary index differs from manifest')
    if file_hash(root/'inputs.jsonl') != manifest['contract']['input_sha256']:
        raise ValueError('oracle source input differs from contract')
    ids = [sid for row in index for sid in row['ids']]
    samples = read_json(root/'prepared/samples.json')
    if ids != [s['id'] for s in samples] or len(ids) != len(set(ids)):
        raise ValueError('oracle sample order/coverage mismatch')
    from runtime.reference_protocol import package_specs
    specs = package_specs(root,manifest)
    if not set(manifest['boundaries']) <= set(specs): raise ValueError('oracle lacks sealed boundary specifications')
    if manifest.get('protocol_schema_version', manifest['schema_version']) == 1 and manifest['contract']['semantics'] != reference_adapter(manifest['contract']).semantics:
        raise ValueError('unsupported legacy oracle model semantics')
    semantic_validation(root, manifest, 'oracle', complete=True)
    return manifest


def validate_reference(cfg, command='accuracy', *, manifest=None):
    root = Path(cfg['reference']['path'])
    manifest = load_manifest(root) if manifest is None else manifest
    expected = manifest['contract']
    actual = contract(cfg)
    differences = identity_differences(expected,actual)
    if differences:
        raise ValueError('oracle contract mismatch: '+', '.join(differences))
    required = selected_boundaries(cfg, command)
    if not set(required) <= set(manifest['boundaries']):
        raise ValueError('oracle lacks selected boundaries: '+', '.join(sorted(set(required)-set(manifest['boundaries']))))
    return manifest


def bind_reference(cfg, command, explicit_input=False):
    root = Path(cfg['reference']['path'])
    manifest = load_manifest(root)
    if explicit_input and file_hash(cfg['inputs']['path']) != manifest['contract']['input_sha256']:
        raise ValueError('explicit input file differs from oracle input')
    cfg['inputs']['path'] = str(root/'inputs.jsonl')
    validate_reference(cfg,command,manifest=manifest)


def configure_numerics(backend, seed):
    create('vendors', 'nvidia').configure_numerics({'_oracle_capture': True, 'runtime': {'seed': seed}}, backend)


def numerical_settings():
    return create('vendors', 'nvidia').numerical_settings()


def load_batches(prepared, adapter=None):
    import torch
    batches = load_cpu(Path(prepared)/'inputs.pt')
    samples = read_json(Path(prepared)/'samples.json')
    if adapter is not None:
        adapter.validate_batches(batches,samples)
    elif (Path(prepared)/'workload.json').is_file():
        from runtime.reference_protocol import validate_archive, validate_workload
        workload = read_json(Path(prepared)/'workload.json')
        validate_workload(workload)
        layout = workload.get('input_layout')
        validate_archive(batches,samples,layout)
        path = Path(prepared)/'forward-inputs.json'
        if path.is_file() and 'input_tensors' in read_json(path):
            forward = read_json(path)
            if forward.get('input_layout') != layout:
                raise ValueError('fixed input layout metadata mismatch')
            expected = forward['input_tensors']
            from runtime.input_layout import tensor_metadata
            actual = [tensor_metadata(b['inputs'],layout) for b in batches]
            if actual != expected: raise ValueError('fixed input tensor metadata mismatch')
    else:
        # The old Qwen archive is recognized by its exact token/mask input contract.
        if any(set(b['inputs']) != {'input_ids','attention_mask'} for b in batches):
            raise ValueError('legacy input archive is not the Qwen3 contract; workload metadata required')
        create('models','qwen3_embedding_0.6b').validate_batches(batches,samples)
    return batches


def output_batch(directory, name, batch, boundaries, hidden_size=None, adapter=None):
    import torch
    value = load_cpu(safe_file(directory, name))
    if adapter is not None:
        adapter.validate_reference_output(value,batch,boundaries,hidden_size)
    elif value.get('boundary_specs'):
        from runtime.reference_protocol import validate_outputs
        validate_outputs(value,batch,boundaries)
    else:
        if 'mask' not in value or set(batch['inputs']) != {'input_ids','attention_mask'}:
            raise ValueError('legacy output is not the Qwen3 contract; boundary metadata required')
        create('models','qwen3_embedding_0.6b').validate_reference_output(value,batch,boundaries,hidden_size)
    return value


def generate(cfg, root, progress, invoke):
    """Two isolated complete forwards qualify one immutable reference package."""
    from runtime.stack import run_stack
    import torch
    selected = deepcopy(cfg)
    selected['_oracle_capture'] = True
    result = run_stack('accuracy',selected,root,progress,invoke)
    if result['status'] != 'completed':
        return result
    prepared = root/'prepared'
    identity = read_json(prepared/'identity.json')
    if 'A100' not in identity['device_name'] or identity['vendor'] != 'nvidia':
        raise ValueError('not an A100 producer')
    progress.phase = 'oracle/independent-repeatability'
    repeat = invoke(selected,root,'forward',root/'verification',prepared)
    if repeat['status'] != 'completed' or repeat.get('numerical_anomalies'):
        raise ValueError('oracle repeatability worker failed')
    if model_adapter(cfg).execution_kind != 'single_step':
        from runtime.generation_reference import publish
        return publish(cfg,root,result,repeat)
    return publish_capture(cfg, root, result, repeat, progress)


def publish_capture(cfg, root, result, repeat, progress=None):
    import torch
    prepared = root/'prepared'
    identity = read_json(prepared/'identity.json')
    if (result.get('status') != 'completed' or repeat.get('status') != 'completed'
            or repeat.get('numerical_anomalies') or identity.get('vendor') != 'nvidia'
            or 'A100' not in identity.get('device_name', '')):
        raise ValueError('oracle requires complete finite A100 capture and independent repeat')
    if model_adapter(cfg).execution_kind != 'single_step':
        from runtime.generation_reference import publish
        return publish(cfg, root, result, repeat)
    batches = load_batches(prepared)
    boundaries = selected_boundaries(cfg, prepared=prepared)
    # Model outputs are always emitted by the existing engine. Keep selected scope only.
    artifacts = result['paths']['off']['artifacts']
    if len(artifacts) != len(batches) or len(repeat['artifacts']) != len(batches):
        raise ValueError('incomplete oracle batch coverage')
    checks = []
    for i,batch in enumerate(batches):
        first = output_batch(root/'off',artifacts[i],batch,boundaries)
        second = output_batch(root/'verification',repeat['artifacts'][i],batch,boundaries)
        for name in boundaries:
            a,b = first['outputs'][name],second['outputs'][name]
            if not bool(torch.isfinite(a).all()) or not torch.equal(a,b) or a.dtype != b.dtype:
                raise ValueError('oracle is nonfinite or not bitwise repeatable: '+name)
            checks.append({'batch':i,'boundary':name,'shape':list(a.shape),'dtype':str(a.dtype),'equal':True})
    evidence = {'status':'bitwise_equal','scope':'two independent processes, same A100/configuration; not cross-platform determinism',
                'checks':checks}
    write_json(root/'repeatability.json',evidence)
    if progress is not None: progress.phase = 'oracle/seal-package'
    package = root/'package.incomplete'
    package.mkdir(exist_ok=False)
    (package/'prepared').mkdir()
    (package/'outputs').mkdir()
    (package/'provenance').mkdir()
    for name in ['inputs.pt','samples.json','forward-inputs.json','workload.json']:
        shutil.copyfile(prepared/name,package/'prepared'/name)
    # A requested nested module may extend the adapter's default report boundaries.
    captured = output_batch(root/'off',artifacts[0],batches[0],boundaries)['boundary_specs']
    workload = read_json(package/'prepared/workload.json')
    workload['boundaries'] = [captured[n] for n in boundaries]
    write_json(package/'prepared/workload.json',workload)
    shutil.copyfile(cfg['inputs']['path'],package/'inputs.jsonl')
    shutil.copyfile(prepared/'identity.json',package/'provenance/identity.json')
    for name in ['effective.yaml','source-snapshot.json','repeatability.json','comparison-context.json']:
        shutil.copyfile(root/name,package/'provenance'/name)
    shutil.copytree(root/'source-code',package/'provenance/source-code')
    for folder in ['prepared','off','verification']:
        for name in ['result.json','run.log','command.json','exit.json','components.json']:
            if (root/folder/name).is_file():
                dest=package/'provenance'/folder/name; dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(root/folder/name,dest)
    from safetensors.torch import save_file
    index=[]
    for i,batch in enumerate(batches):
        value=output_batch(root/'off',artifacts[i],batch,boundaries)
        name=f'outputs/batch-{i:04}.safetensors'
        tensors={k:value['outputs'][k].contiguous().clone() for k in boundaries}
        save_file(tensors,str(package/name))
        index.append({'file':name,'ids':batch['ids'],'tensors':{k:{'shape':list(t.shape),'dtype':str(t.dtype)} for k,t in tensors.items()}})
    write_json(package/'outputs/index.json',index)
    manifest={'schema_version':SCHEMA,'kind':'a100_accuracy','status':'complete',
              'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'contract':contract(cfg,identity['model_files']),'boundaries':boundaries,
              'workload':read_json(package/'prepared/workload.json'),
              'producer':{'device_name':identity['device_name'],'identity_key':digest(identity),
                          'numerical_settings':identity['oracle_numerics']},
              'repeatability':{'status':'bitwise_equal','forward_processes':2},
              'files':{p.relative_to(package).as_posix():{'bytes':p.stat().st_size,'sha256':file_hash(p)}
                       for p in sorted(package.rglob('*')) if p.is_file()}}
    from runtime.reference_storage import seal_accuracy
    manifest = seal_accuracy(package, manifest)
    load_manifest(package)
    package.rename(root/'package')
    (root/'package-evidence.incomplete').rename(root/'package-evidence')
    result['oracle']={'role':'producer','oracle_id':manifest['oracle_id'],'package':'package',
                      'repeatability':evidence['status'],'status':'completed'}
    return result


def finalize_retention(cfg, root, result):
    """Compatibility entry; the ordinary accuracy path uses the shared policy."""
    from runtime.retention import finalize_retention as finalize
    return finalize(cfg, root, result, command='accuracy')
