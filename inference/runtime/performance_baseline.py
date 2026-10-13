"""Immutable performance references. Validation and comparison are CPU-only."""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess
import time

from runtime.common import read_json, write_json, digest, file_hash, identity_differences
from runtime.oracle import contract as workload_contract, safe_file

SCHEMA = 2
from runtime.adapters import model_adapter, create
LAYER_TIMING = 'inclusive-current-stream-event-independent-process-v1'


def contract(cfg, files=None):
    adapter = model_adapter(cfg)
    value = workload_contract(cfg, files)
    size = len(cfg['runtime']['devices']) if cfg['runtime']['parallelism'] == 'tp' else 1
    value.update(runtime_devices=size, tp_degree=size, timing=adapter.timing,
                 layer_timing=LAYER_TIMING, warmup_rounds=cfg['performance']['warmup_rounds'],
                 tp_strategy=adapter.parallel_strategy if size > 1 else 'unsharded',
                 instrumentation=adapter.instrumentation(cfg))
    if adapter.execution_kind == 'autoregressive':
        value.update(execution_kind=adapter.execution_kind, generation=adapter.metadata(cfg)['generation'])
    return value


def load_manifest(root):
    root = Path(root).resolve()
    m = read_json(safe_file(root,'manifest.json'))
    if m.get('schema_version') not in (1,SCHEMA,3,4) or (m.get('kind'),m.get('status')) != ('a100_performance','complete'):
        raise ValueError('unsupported or incomplete performance baseline')
    if m.get('baseline_id') != digest({k:v for k,v in m.items() if k != 'baseline_id'}):
        raise ValueError('performance manifest digest mismatch')
    required = {'prepared/inputs.pt','prepared/samples.json','prepared/forward-inputs.json',
                'inputs.jsonl','metrics.json','provenance/identity.json','evidence/result.json',
                'evidence/effective.yaml','evidence/source-snapshot.json','environment.json'}
    if m['schema_version'] == 4:
        required.remove('evidence/result.json')
    files=m.get('files')
    if m.get('protocol_schema_version', m['schema_version']) in (2,3): required.add('prepared/workload.json')
    if not isinstance(files,dict) or not required <= files.keys():
        raise ValueError('performance baseline lacks mandatory evidence')
    for name,info in files.items():
        p=safe_file(root,name)
        if p.stat().st_size != info['bytes'] or file_hash(p) != info['sha256']:
            raise ValueError('performance baseline member digest mismatch: '+name)
    if file_hash(root/'inputs.jsonl') != m['contract']['input_sha256']:
        raise ValueError('performance input contract mismatch')
    from runtime.verification import semantic_validation
    if semantic_validation(root, m, 'performance'): return m
    if m.get('protocol_schema_version', m['schema_version']) in (2,3):
        from runtime.reference_protocol import package_specs
        package_specs(root,m)
    elif m['contract']['model']['name'] != 'qwen3_embedding_0.6b' or m['contract']['timing'] != model_adapter(m['contract']).timing:
        raise ValueError('unsupported legacy performance timing semantics')
    if m['contract']['layer_timing'] != LAYER_TIMING:
        raise ValueError('unsupported performance timing semantics')
    from runtime.performance_comparison import validate_metrics
    metrics=read_json(root/'metrics.json')
    validate_metrics(metrics, m['contract'])
    if set(metrics.get('layer',{})) != {'off'} or set(metrics['total']) != {'off'}:
        raise ValueError('baseline requires native total and complete layer measurements')
    if metrics['input_batches'] != read_json(root/'prepared/forward-inputs.json')['batches']:
        raise ValueError('baseline prepared workload mismatch')
    semantic_validation(root, m, 'performance', complete=True)
    return m


def validate_reference(cfg,command='performance', *, manifest=None):
    m=load_manifest(cfg['reference']['performance_path']) if manifest is None else manifest
    actual=contract(cfg)
    differences=identity_differences(m['contract'],actual)
    if differences:
        raise ValueError('performance baseline contract mismatch: '+', '.join(differences))
    if command == 'performance' and cfg['performance']['level'] == 'layer':
        names=cfg['performance']['layers']
        if names != ['all'] and names != m['contract']['instrumentation']:
            raise ValueError('layer instrumentation must match baseline; select subsets with offline report')
    return m


def bind_reference(cfg,command,explicit_input=False):
    root=Path(cfg['reference']['performance_path'])
    m=load_manifest(root)
    if explicit_input and file_hash(cfg['inputs']['path']) != m['contract']['input_sha256']:
        raise ValueError('explicit input differs from performance baseline')
    cfg['inputs']['path']=str(root/'inputs.jsonl')
    validate_reference(cfg,command,manifest=m)


def configure_numerics(backend):
    create('vendors', 'nvidia').configure_numerics({'_performance_native': True}, backend)


def environment_snapshot():
    return create('vendors', 'nvidia').environment_snapshot()


def publish(cfg,root,result):
    if result.get('status') != 'completed' or result.get('analysis_status') != 'completed':
        raise ValueError('cannot publish incomplete performance baseline')
    from runtime.performance_comparison import extract, validate_metrics
    prepared=root/'prepared'
    ident=read_json(prepared/'identity.json')
    if ident['vendor'] != 'nvidia' or 'A100' not in ident['device_name']:
        raise ValueError('baseline producer is not A100')
    if any('A100' not in row['name'] for row in ident.get('devices',[])):
        raise ValueError('baseline contains a non-A100 rank')
    metrics=extract(root,result,cfg)
    c=contract(cfg, ident['model_files'])
    validate_metrics(metrics,c)
    if not metrics['layer'] or result.get('layer',{}).get('status') != 'completed':
        raise ValueError('baseline requires complete layer diagnostics')
    stage=root/'package.incomplete'
    stage.mkdir()
    (stage/'prepared').mkdir()
    for name in ('inputs.pt','samples.json','forward-inputs.json','workload.json'):
        shutil.copyfile(prepared/name,stage/'prepared'/name)
    shutil.copyfile(cfg['inputs']['path'],stage/'inputs.jsonl')
    write_json(stage/'provenance/identity.json',ident)
    write_json(stage/'metrics.json',metrics)
    write_json(stage/'environment.json',read_json(root/'performance-environment.json'))
    from runtime.reference_storage import seal, run_members
    # Small provenance remains in the comparison core; diagnostics are separate.
    for name in ('effective.yaml', 'source-snapshot.json'):
        (stage/'evidence').mkdir(exist_ok=True)
        shutil.copyfile(root/name, stage/'evidence'/name)
    m={'schema_version':3 if c.get('execution_kind') == 'autoregressive' else SCHEMA,
       'kind':'a100_performance','status':'complete', 'contract':c,
       'workload':read_json(prepared/'workload.json'),
       'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'sampling':deepcopy(cfg['performance'])}
    m = seal(stage, m, root/'package-evidence.incomplete', run_members(root))
    load_manifest(stage)
    stage.rename(root/'package')
    (root/'package-evidence.incomplete').rename(root/'package-evidence')
    result['performance_baseline']={'baseline_id':m['baseline_id'],'package':'package'}


def compare_run(cfg,root,result):
    if result.get('status') != 'completed':
        return
    m=validate_reference(cfg)
    reference=read_json(Path(cfg['reference']['performance_path'])/'metrics.json')
    from runtime.performance_comparison import extract, compare
    candidate=extract(root,result,cfg)
    output=compare(reference,candidate,m['contract'])
    output.update(baseline_id=m['baseline_id'],contract=m['contract'])
    write_json(root/'a100-performance-extract.json',reference)
    write_json(root/'candidate-performance-extract.json',candidate)
    write_json(root/'performance-comparison.json',output)
    result['performance_comparison']='performance-comparison.json'


def finalize_retention(cfg, root, result):
    """Compatibility entry for callers of the earlier baseline consumer API."""
    from runtime.retention import finalize_retention as finalize
    return finalize(cfg, root, result, command='performance')


from runtime.retention import update_retained_links
