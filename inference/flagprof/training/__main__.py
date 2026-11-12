# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Standard PyTorch training examples, observed by the existing flagprof record."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('classification','causal-lm','ctr','embedding','rerank'),default='classification')
    p.add_argument('--model',help='Trusted local HF model directory; no network download')
    p.add_argument('--inputs',help='JSONL of token IDs (causal-lm) or triplet groups (embedding)')
    p.add_argument('--ctr-arch',help='CTR architecture (dlrm din widedeep deepfm dcn sasrec lightgcn bst dien dcnmix esmm mmoe ple dssm); required')
    p.add_argument('--package',help='Sealed package directory (ctr/rerank task); digests verified, corpus unchanged; ctr may omit it to generate the deterministic default corpus')
    p.add_argument('--pooling',choices=('last','cls','mean'),default=None,help='Embedding pooling; optional for adapted models in targets/model_registry.json, otherwise required')
    p.add_argument('--temperature',type=float,default=.05,help='InfoNCE temperature (embedding task)')
    p.add_argument('--rerank-score',choices=('causal_yesno','sequence_logit'),default=None,help='Rerank scoring semantics; optional for adapted models in targets/model_registry.json, otherwise required; must match the package manifest')
    p.add_argument('--outputs',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=3,help='Total optimizer steps, including resumed steps')
    p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=4,help='Per-rank microbatch size')
    p.add_argument('--seed',type=int,default=531)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--optimizer',choices=('sgd','adamw'),default='sgd')
    p.add_argument('--amp',choices=('off','bfloat16','float16'),default='off')
    p.add_argument('--mode',choices=('accuracy','performance'),default='accuracy')
    p.add_argument('--checkpoint',help='Explicit checkpoint output; rank suffix added for DDP')
    p.add_argument('--resume',help='Trusted local pickle checkpoint; rank suffix added for DDP')
    p.add_argument('--stop-after',type=int,help='Stop after this absolute optimizer step, at a checkpoint boundary')
    p.add_argument('--ddp',action='store_true',help='Use torchrun-provided ranks and DistributedDataParallel')
    p.add_argument('--backend',choices=('gloo','nccl'),default=None)
    return p


def local_files(model):
    h=hashlib.sha256(); root=Path(model)
    for path in sorted(root.rglob('*')):
        if path.is_file():
            h.update(path.relative_to(root).as_posix().encode())
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def ensure_generated(kind, outputs, seed):
    """Generate the deterministic default corpus or CTR package next to --outputs.

    The generators are the sealed, seeded CLI tools under inference/targets/;
    an existing result is reused untouched. Single-process only: DDP ranks
    must share an explicit --inputs/--package instead of racing on creation.
    """
    base=Path(outputs)
    suffix='-default-corpus' if kind=='corpus' else '-default-package'
    target=base.with_name(base.name+suffix)
    marker='groups.jsonl' if kind=='corpus' else 'manifest.json'
    if (target/marker).exists(): return str(target)
    script=Path(__file__).resolve().parents[2]/'targets'/(
        'make_embedding_corpus.py' if kind=='corpus' else 'make_ctr_package.py')
    if not script.is_file(): raise ValueError('generator script missing: %s' % script)
    target.mkdir(parents=True, exist_ok=True)
    import subprocess, sys
    cmd=[sys.executable,'-B',str(script),'--seed',str(seed)]
    if kind=='corpus':
        cmd += ['--groups','32','--negs','3','--out',str(target/'groups.jsonl')]
    else:
        cmd += ['--samples','1024','--out',str(target)]
    done=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if done.returncode!=0:
        raise ValueError('default %s generation failed: %s' % (
            kind, done.stderr.decode(errors='replace')[-400:]))
    return str(target)


def prepared_lm(path, *, batch_size, rank=0, world_size=1):
    import torch
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line); ids=r['input_ids']; mask=r.get('attention_mask',[1]*len(ids))
        labels=r.get('labels',[v if m else -100 for v,m in zip(ids,mask)])
        if len(ids)<2 or len(ids)!=len(mask) or len(ids)!=len(labels):
            raise ValueError('LM token, mask and label lengths must match and contain at least two tokens')
        if any(type(v) is not int or v<0 for v in ids) or any(v not in (0,1) for v in mask):
            raise ValueError('invalid token IDs or attention mask')
        if any(type(v) is not int or v<0 and v!=-100 for v in labels): raise ValueError('invalid labels')
        if not any(v!=-100 for v in labels[1:]): raise ValueError('LM batch has no supervised next token')
        rows.append(dict(input_ids=ids,attention_mask=mask,labels=labels))
    group=batch_size*world_size
    if not rows or len(rows)%group:
        raise ValueError('input row count must be a positive multiple of global microbatch size; no silently dropped data')
    batches=[]
    for start in range(0,len(rows),group):
        selected=rows[start+rank*batch_size:start+(rank+1)*batch_size]
        lengths={len(r['input_ids']) for r in selected}
        if len(lengths)!=1: raise ValueError('pretokenized rows within a microbatch require equal length; supply explicit padding')
        batches.append({k:torch.tensor([r[k] for r in selected],dtype=torch.long) for k in selected[0]})
    return batches


def execute(args):
    import torch
    from .engine import train
    import torch.distributed as dist
    if args.steps<=0 or args.batch_size<=0 or args.accumulation<=0:
        raise ValueError('steps, batch-size and accumulation must be positive')
    if not 0<args.lr<float('inf'): raise ValueError('lr must be positive and finite')
    if args.ddp and args.task=='embedding' and not args.inputs:
        raise ValueError('DDP ranks must share an explicit --inputs; default corpus generation is single-process only')
    if args.ddp and args.task=='ctr' and not args.package:
        raise ValueError('DDP ranks must share an explicit --package; default corpus generation is single-process only')
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if world!=1 and not args.ddp: raise ValueError('torchrun requires explicit --ddp')
    if args.ddp and world<2: raise ValueError('DDP requires at least two torchrun workers')
    device=args.device
    if args.ddp and device.startswith('cuda'): device='cuda:'+str(local_rank)
    if device.startswith('cuda'): torch.cuda.set_device(device)
    torch.manual_seed(args.seed);random.seed(args.seed)
    owned=False
    try:
        if args.ddp:
            from datetime import timedelta
            dist.init_process_group(args.backend or ('nccl' if device.startswith('cuda') else 'gloo'),timeout=timedelta(seconds=120))
            owned=True
        protocol=dict(task=args.task,seed=args.seed,lr=args.lr,batch_size_per_rank=args.batch_size,
                      data='synthetic labelled classification' if args.task=='classification' else 'explicit pretokenized local input',
                      numerical_policy='framework defaults; no TF32/determinism/reduction override',
                      dtype='float32 parameters; optional explicit AMP',backend=dist.get_backend() if args.ddp else None)
        if args.task=='classification':
            if args.model or args.inputs: raise ValueError('classification demo does not accept model or inputs')
            model=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.ReLU(),torch.nn.Linear(32,4))
            generator=torch.Generator().manual_seed(args.seed+1)
            batches=[]
            for _ in range(4):
                x=torch.randn(args.batch_size*world,16,generator=generator)
                y=torch.randint(0,4,(args.batch_size*world,),generator=generator)
                sl=slice(rank*args.batch_size,(rank+1)*args.batch_size)
                batches.append(dict(features=x[sl],labels=y[sl]))
            def loss_fn(m,b):return torch.nn.functional.cross_entropy(m(b['features']),b['labels'])
        elif args.task=='ctr':
            if args.model or args.inputs: raise ValueError('ctr task does not accept model or inputs')
            from .ctr import TASK_ARCHS
            if not args.ctr_arch:
                raise ValueError('ctr task requires --ctr-arch with one of: '+', '.join(TASK_ARCHS))
            if not args.package and args.ddp:
                raise ValueError('DDP ranks must share an explicit --package')
            package=args.package
            if not package:
                package=ensure_generated('ctr-package',args.outputs,args.seed)
                protocol['data']='deterministic sealed CTR corpus generated next to --outputs (make_ctr_package, 1024 samples)'
                protocol['default_package_generated']=True
            from .ctr import prepare_ctr
            model,batches,loss_fn,ctr_protocol=prepare_ctr(arch=args.ctr_arch,package=package,
                seed=args.seed,batch_size=args.batch_size,rank=rank,world_size=world)
            protocol.update(ctr_protocol)
        elif args.task=='embedding':
            if args.package or args.ctr_arch: raise ValueError('embedding task does not accept package or ctr-arch')
            if not args.model: raise ValueError('embedding requires --model')
            from .registry import resolve
            pooling,pooling_source=resolve(args.pooling,args.model,kind='pooling')
            if not pooling:
                raise ValueError('embedding requires --pooling (last|cls|mean); model %r is not in '
                                 'inference/targets/model_registry.json, so no default pooling is known' % args.model)
            protocol['pooling_source']=pooling_source
            inputs=args.inputs
            if not inputs:
                inputs=ensure_generated('corpus',args.outputs,args.seed)
                protocol['data']='deterministic synthetic triplet corpus generated next to --outputs (make_embedding_corpus, 32 groups, 3 negatives)'
            from transformers import AutoModel,AutoModelForCausalLM
            source=args.model.removeprefix('hf:')
            if not Path(source).is_dir(): raise ValueError('local model directory missing')
            protocol['model_files_sha256']=local_files(source)
            protocol['attention']='eager'
            try:
                model=AutoModel.from_pretrained(source,local_files_only=True,trust_remote_code=False,
                                                torch_dtype=torch.float32,attn_implementation='eager')
                protocol['model_class']='AutoModel'
            except Exception:
                model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,
                                                           torch_dtype=torch.float32,attn_implementation='eager').model
                protocol['model_class']='AutoModelForCausalLM.base'
            from .embedding import prepare_embedding
            model,batches,loss_fn,emb_protocol=prepare_embedding(model=model,inputs=inputs,
                pooling=pooling,temperature=args.temperature,batch_size=args.batch_size,
                rank=rank,world_size=world)
            protocol.update(emb_protocol)
        elif args.task=='rerank':
            if args.inputs: raise ValueError('rerank task does not accept inputs')
            if not args.package or not args.model: raise ValueError('rerank task requires --package and --model')
            from .registry import resolve
            score,score_source=resolve(args.rerank_score,args.model,kind='rerank_score')
            if not score:
                raise ValueError('rerank requires --rerank-score (causal_yesno|sequence_logit); model %r is not in '
                                 'inference/targets/model_registry.json, so no default scoring semantics is known' % args.model)
            protocol['rerank_score_source']=score_source
            from transformers import AutoTokenizer
            source=args.model.removeprefix('hf:')
            if not Path(source).is_dir(): raise ValueError('local model directory missing')
            protocol['model_files_sha256']=local_files(source)
            protocol['attention']='eager'
            tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
            yes_id=no_id=None
            if score=='causal_yesno':
                yes_id=tokenizer.convert_tokens_to_ids('yes')
                no_id=tokenizer.convert_tokens_to_ids('no')
                if yes_id==tokenizer.unk_token_id or no_id==tokenizer.unk_token_id:
                    raise ValueError('tokenizer lacks literal yes/no tokens')
            cls_name='AutoModelForCausalLM' if score=='causal_yesno' else 'AutoModelForSequenceClassification'
            import transformers
            model=getattr(transformers,cls_name).from_pretrained(source,local_files_only=True,trust_remote_code=False,
                                                                torch_dtype=torch.float32,attn_implementation='eager')
            protocol['model_class']=cls_name
            if hasattr(model.config,'use_cache'): model.config.use_cache=False
            from .rerank import prepare_rerank
            model,batches,loss_fn,rerank_protocol=prepare_rerank(model=model,package=args.package,
                score=score,batch_size=args.batch_size,yes_id=yes_id,no_id=no_id,
                rank=rank,world_size=world)
            protocol.update(rerank_protocol)
        else:
            if not args.model or not args.inputs: raise ValueError('causal-lm requires --model and --inputs')
            from transformers import AutoModelForCausalLM
            source=args.model.removeprefix('hf:')
            if not Path(source).is_dir(): raise ValueError('local model directory missing')
            protocol['model_files_sha256']=local_files(source)
            model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,
                                                       torch_dtype=torch.float32,attn_implementation='eager')
            protocol['attention']='eager'
            batches=prepared_lm(args.inputs,batch_size=args.batch_size,rank=rank,world_size=world)
            def loss_fn(m,b):return m(**b).loss
        model=model.to(device)
        if args.ddp:
            model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank] if device.startswith('cuda') else None)
        optimizer=(torch.optim.SGD(model.parameters(),lr=args.lr) if args.optimizer=='sgd' else
                   torch.optim.AdamW(model.parameters(),lr=args.lr))
        def rank_path(path):return str(path)+'.rank-'+str(rank) if path and args.ddp else path
        outputs=Path(args.outputs)/('rank-'+str(rank)) if args.ddp else Path(args.outputs)
        result=train(model,optimizer,batches,loss_fn,outputs=outputs,steps=args.steps,
                     accumulation=args.accumulation,device=device,amp=args.amp,mode=args.mode,protocol=protocol,
                     checkpoint=rank_path(args.checkpoint),resume=rank_path(args.resume),stop_after=args.stop_after,
                     rank=rank,world_size=world)
        print(json.dumps(dict(output=str(outputs),status=result['status'],next_step=result['next_step'],
                              final_model_sha256=result['final_model_sha256'])),flush=True)
        return 0
    finally:
        if owned: dist.destroy_process_group()


def main(argv=None):
    args=parser().parse_args(argv)
    return execute(args)


if __name__=='__main__':
    raise SystemExit(main())
