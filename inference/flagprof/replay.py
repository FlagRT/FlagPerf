"""Portable pure-operator replay. Preparation/reporting never imports torch."""
import hashlib
import json
from pathlib import Path
import tempfile
from .store import StoreView
from .store.manifest import atomic_json
from .tensor_data import blob_path, verify_blob, compare

SUPPORTED = {('aten::mean','dim'),('aten::softmax','int'),('aten::rsqrt',''),('aten::pow','Tensor_Scalar')}


def digest(path):
    from .offline.checkpoints import fileinfo
    return fileinfo(path)


def find(work, key):
    rows = work.db.execute("select payload from events where type='tensor' and json_extract(payload,'$.call_key')=? limit 2",(key,)).fetchall()
    if len(rows)!=1:
        raise ValueError('replay requires exactly one tensor: '+key)
    return json.loads(rows[0][0])


def copy_blob(run, event, out, name):
    if 'blob' not in event:
        raise ValueError('replay requires a full blob: '+event['call_key'])
    verify_blob(run,event)
    blob=event['blob']
    if blob['hash']!=event['digest']:raise ValueError('replay blob/digest mismatch')
    if blob['nbytes']>64<<20:
        raise ValueError('replay input/output exceeds 64 MiB per tensor')
    target=out/name
    with blob_path(run,blob).open('rb') as source,target.open('xb') as dest:
        source.seek(blob['offset']);remaining=blob['nbytes']
        while remaining:
            block=source.read(min(65536,remaining))
            if not block:raise ValueError('truncated replay blob')
            dest.write(block);remaining-=len(block)
    return dict(event,blob=dict(blob,path=name,offset=0))


def prepare(ref, cmp, call_key, out, *, work=None):
    from .analyzers.workspace import Workspace
    from .analyzers.oracle import identity, REQUIRED, call_config
    out=Path(out).resolve()
    for source in (Path(ref).resolve(),Path(cmp).resolve()):
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError('replay package must be outside source Stores')
    if work:Path(work).mkdir(parents=True,exist_ok=True)
    with StoreView(ref) as av,StoreView(cmp) as bv,tempfile.TemporaryDirectory(prefix='replay-',dir=work) as temp:
        a=Workspace(av,Path(temp)/'a');b=Workspace(bv,Path(temp)/'b')
        try:
            a.ingest();b.ingest()
            if a.partial() or b.partial():raise ValueError('replay source read is partial')
            identities={side:identity(ws) for side,ws in (('ref',a),('cmp',b))}
            if any(k not in identities['ref'] or identities['ref'][k]!=identities['cmp'].get(k) for k in REQUIRED):
                raise ValueError('replay source identity mismatch')
            outputs={side:find(ws,call_key) for side,ws in (('ref',a),('cmp',b))}
            configs=[call_config(outputs[side]) for side in ('ref','cmp')]
            if not configs[0] or configs[0]!=configs[1]:raise ValueError('replay call configuration mismatch or unknown')
            entries={}
            for side,ws in (('ref',a),('cmp',b)):
                event=outputs[side];sig=ws.signature(event.get('sid'))
                if not sig or (sig['name'],sig['overload']) not in SUPPORTED:
                    raise ValueError('replay unsupported operator')
                sources=event.get('src',[])
                if len(sources)!=1 or sources[0]=='unknown':raise ValueError('replay unknown or unsupported input sources')
                tensor=find(ws,sources[0])
                nodes=[n for n in sig['tree'] if n['k']=='tensor']
                if len(nodes)!=1 or nodes[0]['shape']!=tensor['shape'] or nodes[0]['dtype']!=tensor['dtype']:
                    raise ValueError('replay input metadata mismatch')
                if nodes[0]['layout']!='torch.strided' or any(n['k'] not in ('tensor','none','bool','int','float','list','tuple','dtype') for n in sig['tree']):
                    raise ValueError('replay unsupported argument layout or kind')
                if not nodes[0].get('contiguous') and (len(nodes[0].get('stride',[]))!=len(tensor['shape']) or any(v<=0 for v in nodes[0]['stride'])):
                    raise ValueError('replay layout cannot be reconstructed')
                for item in (event,tensor):
                    if 'blob' not in item:raise ValueError('replay full blob missing')
                    verify_blob(ws.view.run,item)
                entries[side]=dict(source_run=str(ws.view.run.resolve()),source_manifest=digest(ws.view.run/'manifest.json'),
                    signature=sig,config=event['ext']['call_config'],input=tensor,expected_output=event,
                    identity=identities[side])
            if entries['ref']['input']['digest']==entries['cmp']['input']['digest']:
                raise ValueError('replay target requires already different actual inputs')
            out.mkdir(parents=True,exist_ok=False)
            for side,ws in (('ref',a),('cmp',b)):
                for field in ('input','expected_output'):
                    entries[side][field]=copy_blob(ws.view.run,entries[side][field],out,side+'-'+field+'.bin')
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


def run(package, *, side, device, store):
    from .offline.limits import read_json
    from .store import create_run,Writer,finalize
    package=Path(package).resolve();packet=read_json(package/'replay.json')
    if packet.get('schema')!='operator-replay/1' or side not in ('ref','cmp'):
        raise ValueError('invalid replay package or side')
    selected=packet['sides'][side];expected=selected['identity']
    if device.split(':')[0]!=expected['device'].split(':')[0]:
        raise ValueError('replay execution device family mismatch')
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
                cpu=torch.frombuffer(bytearray(raw),dtype=dtype).clone().reshape(tensor['shape'])
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
