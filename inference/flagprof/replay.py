# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Portable pure-operator replay. Preparation/reporting never imports torch."""
import hashlib
import json
import math
import struct
from pathlib import Path
import tempfile
from .store import StoreView
from .store.manifest import atomic_json
from .tensor_data import FORMATS, blob_path, verify_blob, compare

SUPPORTED = {('aten::mean','dim'),('aten::softmax','int'),('aten::rsqrt',''),('aten::pow','Tensor_Scalar')}


def digest(path):
    from .offline.checkpoints import fileinfo
    return fileinfo(path)


# Persist these stable codes in oracle-result/1. Preparing the package additionally
# verifies blob content, so exporting a row never needs to hash its inputs again.
READY = 'replay_ready'
TENSOR_BYTES = 64 << 20
REASONS = {
    'replay_partial': 'replay source read is partial or Store is incomplete',
    'replay_identity': 'replay source identity mismatch or incomplete execution identity',
    'replay_call': 'replay requires exactly one tensor per side for this call',
    'replay_operator': 'replay unsupported operator',
    'replay_config': 'replay call configuration mismatch or unknown',
    'replay_sources': 'replay unknown, misaligned or unsupported input sources',
    'replay_metadata': 'replay input metadata mismatch',
    'replay_layout': 'replay layout or argument kind cannot be reconstructed',
    'replay_blob': 'replay full blob missing',
    'replay_range': 'replay blob metadata, digest or range mismatch',
    'replay_size': 'replay input/output exceeds 64 MiB per tensor',
    'replay_same_input': 'replay target requires already different actual inputs',
}


def find(work, key):
    rows = work.db.execute("select payload from events where type='tensor' and json_extract(payload,'$.call_key')=? limit 2",(key,)).fetchall()
    return json.loads(rows[0][0]) if len(rows) == 1 else None


def source_status(stores, identities):
    from .analyzers.oracle import required_fields
    if any(ws.partial() or ws.view.run_status()['manifest_status'] != 'complete' for ws in stores.values()):
        return 'replay_partial'
    a, b = identities['ref'], identities['cmp']
    required = required_fields(a, b) + tuple(k for k in ('parallelism', 'rank', 'code_files', 'padding_side', 'adapter_names') if k in a or k in b)
    if any(k not in a or k not in b or a[k] != b[k] for k in required):
        return 'replay_identity'
    for value in identities.values():
        device, software, numerics = (value.get(k) for k in ('device', 'software', 'numerics'))
        gems = value['flaggems']
        if (not isinstance(device, str) or not device or not isinstance(software, dict) or not software.get('torch')
                or not isinstance(numerics, dict) or not numerics.get('mode') or gems.get('enabled') not in ('on', 'off')
                or gems['enabled'] == 'on' and (not gems.get('source_sha256') or not isinstance(value.get('include'), list))):
            return 'replay_identity'
    return READY


def blob_status(run, item):
    blob = item.get('blob')
    if not blob:
        return 'replay_blob'
    nbytes, offset = blob['nbytes'], blob['offset']
    if nbytes > TENSOR_BYTES:
        return 'replay_size'
    fmt = FORMATS.get(item['dtype'].removeprefix('torch.'))
    if (blob['hash'] != item['digest'] or offset < 0 or fmt is None
            or nbytes != math.prod(item['shape']) * struct.calcsize(fmt)):
        return 'replay_range'
    try:
        if blob_path(run, blob).stat().st_size < offset + nbytes:
            return 'replay_range'
    except (OSError, ValueError):
        return 'replay_blob'
    return READY


def assess_call(stores, outputs, lookup):
    """Use indexed events and blob descriptors from already ingested Stores."""
    from .analyzers.oracle import call_config
    if any(not event for event in outputs.values()):
        return 'replay_call', {}
    entries = {}
    for side, ws in stores.items():
        event = outputs[side]
        sig = ws.signature(event.get('sid'))
        if not sig or (sig['name'], sig['overload']) not in SUPPORTED:
            return 'replay_operator', {}
        sources = event.get('src') or []
        if len(sources) != 1 or sources[0] == 'unknown' or outputs['ref'].get('src') != outputs['cmp'].get('src'):
            return 'replay_sources', {}
        tensor = lookup(ws, sources[0])
        if tensor is None:
            return 'replay_sources', {}
        nodes = [n for n in sig['tree'] if n['k'] == 'tensor']
        if len(nodes) != 1 or nodes[0]['shape'] != tensor['shape'] or nodes[0]['dtype'] != tensor['dtype']:
            return 'replay_metadata', {}
        node = nodes[0]
        if node['layout'] != 'torch.strided' or any(n['k'] not in ('tensor','none','bool','int','float','list','tuple','dtype') for n in sig['tree']):
            return 'replay_layout', {}
        if not node.get('contiguous'):
            stride = node.get('stride', [])
            if len(stride) != len(tensor['shape']) or any(v <= 0 for v in stride):
                return 'replay_layout', {}
            span = 1 + sum((dim - 1) * step for dim, step in zip(tensor['shape'], stride))
            fmt = FORMATS.get(tensor['dtype'].removeprefix('torch.'))
            if fmt and span * struct.calcsize(fmt) > TENSOR_BYTES:
                return 'replay_size', {}
        for item in (event, tensor):
            status = blob_status(ws.view.run, item)
            if status != READY:
                return status, {}
        entries[side] = dict(signature=sig, input=tensor, expected_output=event)
    a, b = (entries[s]['signature'] for s in ('ref', 'cmp'))
    config = call_config(outputs['ref'])
    if (a['name'], a['overload']) != (b['name'], b['overload']) or not config or config != call_config(outputs['cmp']):
        return 'replay_config', {}
    if entries['ref']['input']['digest'] == entries['cmp']['input']['digest']:
        return 'replay_same_input', {}
    return READY, entries


def copy_blob(run, event, out, name):
    checksum=hashlib.sha256()
    blob=event['blob']
    target=out/name
    with blob_path(run,blob).open('rb') as source,target.open('xb') as dest:
        source.seek(blob['offset']);remaining=blob['nbytes']
        while remaining:
            block=source.read(min(65536,remaining))
            if not block:raise ValueError('truncated replay blob')
            checksum.update(block)
            dest.write(block);remaining-=len(block)
    if 'sha256:'+checksum.hexdigest()!=blob['hash']:
        raise ValueError('tensor blob hash mismatch: '+event['call_key'])
    return dict(event,blob=dict(blob,path=name,offset=0))


def prepare(ref, cmp, call_key, out):
    from .offline.worker import output_operation
    return output_operation(out, [ref, cmp], _prepare, ref, cmp, call_key)


def _prepare(ref, cmp, call_key, out):
    from .analyzers.workspace import Workspace
    from .analyzers.oracle import identity
    out=Path(out).resolve()
    for source in (Path(ref).resolve(),Path(cmp).resolve()):
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError('replay package must be outside source Stores')
    with StoreView(ref) as av,StoreView(cmp) as bv,tempfile.TemporaryDirectory(prefix='replay-') as temp:
        a=Workspace(av,Path(temp)/'a');b=Workspace(bv,Path(temp)/'b')
        try:
            a.ingest();b.ingest()
            stores = dict(ref=a, cmp=b)
            identities = {side: identity(ws) for side, ws in stores.items()}
            status = source_status(stores, identities)
            if status != READY:
                raise ValueError(REASONS[status])
            outputs = {side: find(ws, call_key) for side, ws in stores.items()}
            status, entries = assess_call(stores, outputs, find)
            if status != READY:
                raise ValueError(REASONS[status] + ': ' + call_key)
            for side, entry in entries.items():
                ws = stores[side]
                entry.update(source_run_id=ws.view.run_status()['run_id'], source_manifest=digest(ws.view.run/'manifest.json'),
                             config=outputs[side]['ext']['call_config'], identity=identities[side])
            from .offline.worker import new_output
            new_output(out)
            with tempfile.TemporaryDirectory(dir=out) as stage:
                staged = Path(stage)
                for side, ws in stores.items():
                    for field in ('input', 'expected_output'):
                        entries[side][field] = copy_blob(ws.view.run, entries[side][field], staged, side+'-'+field+'.bin')
                for blob in staged.iterdir():
                    blob.replace(out/blob.name)
            packet=dict(schema='operator-replay/1',call_key=call_key,sides=entries)
            atomic_json(out/'replay.json',packet)
            return dict(package=str(out),manifest=digest(out/'replay.json'),call_key=call_key)
        finally:a.close();b.close()


def arguments(tree, tensor):
    """Restore only the frozen allowlisted value kinds; no evaluation of code."""
    import torch
    nodes={n['p']:n for n in tree}
    def decode(path):
        n=nodes[path];kind=n['k']
        if kind=='tensor':return tensor
        if kind=='none':return None
        if kind in ('bool','int'):return n['v']
        if kind=='float':return float(n['v'])
        if kind=='dtype':return getattr(torch,n['v'].removeprefix('torch.'))
        if kind in ('list','tuple'):
            values=[decode(path+'['+str(i)+']') for i in range(n['n'])]
            return tuple(values) if kind=='tuple' else values
        raise ValueError('unsupported replay value kind: '+kind)
    roots=[n['p'] for n in tree if '[' not in n['p']]
    args=[decode(str(i)) for i in range(sum(p.isdigit() for p in roots))]
    kwargs={p[3:]:decode(p) for p in roots if p.startswith('kw:')}
    return args,kwargs


def run(package, *, side, device, store, vendor_compiler=None):
    from .offline.limits import read_json
    from .store import create_run,Writer,finalize
    package=Path(package).resolve();packet=read_json(package/'replay.json')
    if packet.get('schema')!='operator-replay/1' or side not in ('ref','cmp'):
        raise ValueError('invalid replay package or side')
    selected=packet['sides'][side];expected=selected['identity']
    if device.split(':')[0]!=expected['device'].split(':')[0]:
        raise ValueError('replay execution device family mismatch')
    compiler_expected=expected.get('compiler')
    if compiler_expected and compiler_expected.get('status')=='available':
        from .components.compiler import select, identity as compiler_identity
        import os
        provider=compiler_expected['provider']
        config=dict(flagtree='on' if provider=='flagtree' else 'off')
        if vendor_compiler is not None:
            config['vendor_compiler'] = vendor_compiler
        select(config, cache_root=Path(store).resolve(), target=device)
        actual_compiler=compiler_identity()
        if any(actual_compiler.get(k)!=compiler_expected.get(k) for k in ('provider','version','source_sha256')):
            raise ValueError('replay compiler source mismatch')
    import torch
    if device.startswith('npu'):__import__('torch_npu')
    from .runner.fixed import numerics,software
    actual=numerics(device,expected['numerics'].get('mode','default'))
    if actual!=expected['numerics']:raise ValueError('replay numerical configuration mismatch')
    environment=software()
    for key in ('torch','torch-npu'):
        if environment[key]!=expected['software'].get(key):raise ValueError('replay software mismatch: '+key)
    component=None;observed=[];observing=False
    def observe(stage,key,function,token):
        if observing and stage=='enter':
            if len(observed)>=256:raise ValueError('replay route observation budget')
            observed.append(dict(key=key,function=function))
    if expected['flaggems']['enabled']=='on':
        from .components.flaggems import FlagGems
        component=FlagGems(expected['include'],observer=observe).enable()
        if component.identity['source_sha256']!=expected['flaggems']['source_sha256']:
            component.close();raise ValueError('replay FlagGems source mismatch')
    path=create_run(store,identity=dict(kind='operator-replay',package_sha256=digest(package/'replay.json')['sha256']))
    state=dict(schema='operator-replay-result/1',status='incomplete',side=side,device=device,software=environment,
        package=digest(package/'replay.json'),call_key=packet['call_key'],original_sources={s:e['source_manifest'] for s,e in packet['sides'].items()},
        identity=expected,numerics=actual,cases={})
    atomic_json(path/'replay-result.json',state)
    try:
        with Writer(path) as writer,torch.no_grad():
            for input_side,entry in packet['sides'].items():
                sig=entry['signature'];tensor=entry['input']
                if (sig['name'],sig['overload']) not in SUPPORTED:raise ValueError('replay unsupported operator')
                verify_blob(package,tensor)
                raw=blob_path(package,tensor['blob']).read_bytes()
                dtype=getattr(torch,tensor['dtype'].removeprefix('torch.'))
                cpu=(torch.frombuffer(bytearray(raw),dtype=dtype).clone() if raw else torch.empty(0,dtype=dtype)).reshape(tensor['shape'])
                node=next(n for n in sig['tree'] if n['k']=='tensor')
                if node['contiguous']:
                    value=cpu.to(device)
                else:
                    value=torch.empty_strided(node['shape'],node['stride'],dtype=dtype,device=device)
                    value.copy_(cpu.to(device))
                actual_raw=value.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                actual_digest='sha256:'+hashlib.sha256(actual_raw).hexdigest()
                if actual_digest!=tensor['digest']:raise ValueError('replay execution input bytes mismatch')
                args,kwargs=arguments(sig['tree'],value)
                from .probes.argtree import describe_args
                if describe_args(args,kwargs)!=sig['tree']:raise ValueError('replay reconstructed arguments mismatch')
                operator=getattr(getattr(torch.ops.aten,sig['name'].removeprefix('aten::')),sig['overload'] or 'default')
                observed.clear();observing=True
                try:output=operator(*args,**kwargs)
                finally:observing=False
                from .components.flaggems import aten_key
                route=dict(condition=expected['flaggems']['enabled'],observed=list(observed),
                    registration=component.table.get(aten_key(sig['name'],sig['overload']),{}) if component else {},
                    observation_errors=component.observer_errors if component else [])
                if route['observation_errors']:raise ValueError('replay route observation failed')
                output_cpu=output.detach().to('cpu').contiguous()
                output_raw=output_cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
                output_digest='sha256:'+hashlib.sha256(output_raw).hexdigest()
                blob=path/'blobs'/(input_side+'-output.bin');blob.write_bytes(output_raw)
                writer.register_signature({k:sig[k] for k in ('name','overload','role','tree','sid')})
                event=dict(t='tensor',role='out',call_key='replay#0#out#'+input_side,boundary='out',
                    sid=sig['sid'],shape=list(output_cpu.shape),dtype=str(output_cpu.dtype),stats={},chunks=[],probe=[],probe_alg='p4v1',
                    digest=output_digest,blob=dict(path=str(blob.relative_to(path)),offset=0,nbytes=len(output_raw),hash=output_digest,
                    fmt='raw-le:'+str(output_cpu.dtype).removeprefix('torch.')+':'+str(list(output_cpu.shape))),
                    ext={'replay':dict(call_key=packet['call_key'],input_side=input_side,input_digest=actual_digest,config=entry['config'])})
                writer.emit(event)
                self_check=compare(package,entry['expected_output'],path,event) if input_side==side else None
                state['cases'][input_side]=dict(input_digest=actual_digest,signature=sig['sid'],output_digest=output_digest,
                    self_reproduction=self_check,call_key=event['call_key'],route=route)
        final=finalize(path)
        if final['status']!='complete':raise ValueError('replay Store incomplete')
        state['status']='complete' if state['cases'][side]['self_reproduction'].get('digest_equal') else 'self_reproduction_failed'
    except BaseException as error:
        state['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        if component:component.close()
        atomic_json(path/'replay-result.json',state)
    return dict(run=str(path),status=state['status'],cases=state['cases'])


def panel(work, ref, cmp, replay_ref, replay_cmp):
    from .analyzers import Result
    from .offline.limits import read_json
    states={s:read_json(Path(p)/'replay-result.json') for s,p in (('ref',replay_ref),('cmp',replay_cmp))}
    reason=[]
    for side,path in (('ref',ref),('cmp',cmp)):
        if states[side]['side']!=side:reason.append('execution_side_mismatch')
        if states[side]['status']!='complete':reason.append(side+':self_reproduction_failed_or_incomplete')
        if states[side]['original_sources'].get(side)!=digest(Path(path)/'manifest.json'):reason.append(side+':source_identity_mismatch')
    if states['ref']['package']!=states['cmp']['package']:reason.append('input_package_mismatch')
    if states['ref']['call_key']!=states['cmp']['call_key']:reason.append('target_mismatch')
    events={}
    for side,path in (('ref',replay_ref),('cmp',replay_cmp)):
        with StoreView(path) as view:
            events[side]={e.record['call_key']:e.record for e in view.iter_events(types=('tensor',))}
    invalid=False
    for source in ('ref','cmp'):
        left=states['ref']['cases'][source];right=states['cmp']['cases'][source]
        failures=list(reason)
        if left['input_digest']!=right['input_digest'] or left['signature']!=right['signature']:failures.append('actual_input_or_signature_mismatch')
        metrics=compare(replay_ref,events['ref'][left['call_key']],replay_cmp,events['cmp'][right['call_key']])
        if metrics['status']!='ok':failures.append('replay_output_metadata_mismatch')
        invalid |= bool(failures)
        row=dict(call_key=states['ref']['call_key'],input_side=source,input_digest=left['input_digest'],valid=not failures,
            reasons=failures,metrics=metrics,self_reproduction={s:states[s]['cases'][s]['self_reproduction'] for s in states},
            routes=dict(ref=left.get('route'),cmp=right.get('route')),
            performance_links=[json.loads(value[0]) for value in work.db.execute(
                "select payload from rows where panel='oracle.perf_links' and key in (?,?) order by key",
                tuple(s+':'+states['ref']['call_key'] for s in ('ref','cmp')))],
            evidence=dict(ref=str(replay_ref),cmp=str(replay_cmp),package=states['ref']['package']))
        work.row('oracle.replay',source,row)
    return Result('oracle.replay','unavailable' if invalid else 'ok',work.table('oracle.replay'),{},
        ['同一输入字节与配置在两侧实现执行；同侧封存输出复现后才解释隔离结果。'])
