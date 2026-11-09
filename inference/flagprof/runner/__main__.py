# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""HF embedding entrypoint, also usable as a no-probe component baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def model_files_identity(model_path):
    """Weight-file digests so accuracy runs can prove they used the same model."""
    files={}
    for path in sorted(Path(model_path).iterdir()):
        if path.suffix in ('.safetensors','.bin'):
            digest=hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1<<22), b''):
                    digest.update(chunk)
            files[path.name]=digest.hexdigest()
    if not files:
        raise ValueError('accuracy identity needs safetensors/bin weights under '+str(model_path))
    return files


POOLING_SEMANTICS={'last':'last-valid-token-fp32-l2','mean':'mean-fp32-l2','cls':'cls-token-fp32-l2'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--task', choices=['embedding','generate','rerank','ctr'], default='embedding')
    p.add_argument('--generation-mode', choices=['accuracy','performance'], default='accuracy')
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--inputs')
    p.add_argument('--inputs-from')
    p.add_argument('--samples')
    p.add_argument('--numerics',choices=['reference','default'],default='default')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--dtype', choices=['bfloat16','float16','float32'], default='bfloat16')
    p.add_argument('--attention', default='eager')
    p.add_argument('--padding-side', choices=['left','right'], default='left')
    p.add_argument('--pooling', choices=['last','mean','cls'], default='last')
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--forwards', type=int)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--outputs', required=True)
    p.add_argument('--flaggems', choices=['off','on'], default=None)
    p.add_argument('--flaggems-include')
    p.add_argument('--package')
    p.add_argument('--score', choices=['causal_yesno','sequence_logit'])
    p.add_argument('--arch', choices=['dlrm','din','widedeep','deepfm','dcn','esmm','mmoe','ple','bst','dien'])
    p.add_argument('--seed', type=int, default=531)
    args = p.parse_args()
    if args.forwards is None:args.forwards=1 if args.task=='generate' else 3
    if args.task=='generate' and __import__('os').environ.get('FLAGPROF_DEVICE'):
        args.device=__import__('os').environ['FLAGPROF_DEVICE']
        args.outputs=str(Path(args.outputs)/('rank-'+__import__('os').environ['RANK']))
    if args.task=='rerank':
        if not args.model.startswith('hf:') or not args.package or not args.score:
            p.error('--task rerank needs hf:<directory> model, --package and --score')
        from .rerank import execute
        return execute(args)
    if args.task=='ctr':
        if not args.package or not args.arch:
            p.error('--task ctr needs --package and --arch')
        from .ctr import execute
        return execute(args)
    if not args.model.startswith('hf:') or min(args.batch_size or 4, args.forwards, args.repeats) < 1 or args.warmup < 0:
        p.error('hf:<directory>, positive counts and nonnegative warmup required')
    if not args.inputs and not args.inputs_from:p.error('--inputs or --inputs-from required')
    if args.task=='generate':
        from .generate import execute
        return execute(args)
    import torch
    from .fixed import numerics, software
    numeric_settings=numerics(args.device,args.numerics)
    if args.device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(args.device)
        sync = torch.npu.synchronize
    elif args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        sync = torch.cuda.synchronize
    else:
        sync = lambda: None
    torch.set_num_threads(4)
    torch.manual_seed(531)
    from flagprof import current_attachment
    from flagprof.probes.config import resolve
    handle = current_attachment()
    component = None
    component_install_s=0.0
    if handle:
        effective = handle.state['effective_config']
        if args.flaggems is not None and args.flaggems != effective['flaggems']:
            p.error('runner component configuration conflicts with record')
        if args.flaggems_include:
            include = json.loads(Path(args.flaggems_include).read_text())
            if include != effective['flaggems_include']:
                p.error('runner include conflicts with record')
    else:
        _, effective = resolve('coverage', dict(flaggems=args.flaggems or 'off', flaggems_include=args.flaggems_include,probes='none'))
        if effective['flaggems'] == 'on':
            from flagprof.components.flaggems import FlagGems
            component_started=time.perf_counter()
            component = FlagGems(effective['flaggems_include']).enable()
            component_install_s=time.perf_counter()-component_started
    start = time.perf_counter()
    try:
        from transformers import AutoModel, AutoTokenizer
        model_path = args.model[3:]
        tokenizer = None if args.inputs_from else AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, padding_side=args.padding_side)
        model_started=time.perf_counter()
        model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, args.dtype), attn_implementation=args.attention).to(args.device).eval()
        model_load_and_device_s=time.perf_counter()-model_started
        input_started=time.perf_counter()
        if args.inputs_from:
            from .fixed import load
            batches,input_identity=load(args.inputs_from,samples=args.samples.split(',') if args.samples else None,batch_size=args.batch_size,model=model_path)
        else:
            args.batch_size=args.batch_size or 4
            rows=[json.loads(line) for line in Path(args.inputs).read_text().splitlines() if line.strip()]
            texts=[r if isinstance(r,str) else r.get('text',r.get('query',r.get('input'))) for r in rows[:args.batch_size]]
            if len(texts)!=args.batch_size or not all(isinstance(t,str) for t in texts):raise ValueError('inputs need batch-size text rows')
            batches=[([str(i) for i in range(args.batch_size)],dict(tokenizer(texts,padding=True,truncation=True,max_length=args.max_length,return_tensors='pt')))]
            input_identity=dict(inputs_sha256=hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
                model_files=model_files_identity(model_path),probe_alg='p4v1')
        import inspect
        if hasattr(model.config,'use_cache'):model.config.use_cache=False
        from . import run_prepared, pool_embedding
        if handle:handle.register_model(model)
        input_identity.update(dtype=args.dtype,attention=args.attention,pooling=POOLING_SEMANTICS[args.pooling],
            device=args.device,software=software(),numerics=numeric_settings,flaggems=effective['flaggems'],include=effective['flaggems_include'])
        # Direct inputs carry the same oracle identity contract as prepared packages
        # (runner/fixed.load): model plus input digest pin the must_equal fields.
        input_identity.setdefault('samples',[s for ids,_ in batches for s in ids])
        input_identity.setdefault('batches',[dict(samples=ids,shape=list(v['input_ids'].shape),valid_tokens=v['attention_mask'].sum(1).tolist()) for ids,v in batches])
        if 'oracle_id' not in input_identity:
            input_identity['oracle_id']=hashlib.sha256(json.dumps(
                {k:input_identity[k] for k in ('model_files','inputs_sha256','dtype','attention','pooling')},
                sort_keys=True).encode()).hexdigest()
        input_prepare_s=time.perf_counter()-input_started
        load_s=time.perf_counter()-start
        out=Path(args.outputs);runs=[]
        for repeat in range(args.repeats):
            for batch_index,(sample_ids,values) in enumerate(batches):
                inputs={key:value.to(args.device) for key,value in values.items()}
                inputs.update(return_dict=True)
                if 'use_cache' in inspect.signature(model.forward).parameters:inputs['use_cache']=False
                if handle:handle.bind_tensor_inputs(sample_ids,inputs,identity=input_identity)
                def transform(output):
                    pooled=pool_embedding(output.last_hidden_state,inputs['attention_mask'],mean=args.pooling=='mean',cls=args.pooling=='cls',normalize=False)
                    result=torch.nn.functional.normalize(pooled.float(),p=2,dim=1)
                    if handle:
                        handle.record_tensor('pooled',pooled);handle.record_tensor('output',result)
                    return result
                folder=out/(str(repeat)+'-'+str(batch_index)) if len(batches)>1 or args.repeats>1 else out
                runs.append(run_prepared(model,inputs,warmup=args.warmup,forwards=args.forwards,outputs=folder,sync=sync,transform=transform))
        component_close_s=0.0
        if component:
            close_started=time.perf_counter();component.close();component_close_s=time.perf_counter()-close_started
        result = dict(model=model_path, load_s=load_s, effective_config=effective,
                      model_load_and_device_s=model_load_and_device_s,input_prepare_s=input_prepare_s,
                      component_owner='attachment' if handle else 'runner',
                      component_install_s=handle.state['component'].get('install_ns',0)/1e9 if handle else component_install_s,
                      component_close_s=None if handle else component_close_s,
                      semantics=POOLING_SEMANTICS[args.pooling]+'; use_cache=False; eager',
                      instrumentation='attachment' if handle else 'none',
                      component_identity=component.identity if component else handle.state.get('flaggems_identity') if handle else None,
                      batch_size=args.batch_size, oracle_identity=input_identity, runs=runs)
        (out/'runner.json').write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
        print(json.dumps(dict(output=str(out), attachment=bool(handle), load_s=load_s)))
    finally:
        if component:
            component.close()


if __name__ == '__main__':
    main()
