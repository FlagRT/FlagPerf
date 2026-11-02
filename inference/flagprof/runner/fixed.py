"""Read the existing Qwen reference inputs and record effective run identity."""
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def load(package,*,samples=None,batch_size=None,model=None):
    import torch
    root=Path(package);manifest=json.loads((root/'manifest.json').read_text())
    for name in ('prepared/inputs.pt','prepared/forward-inputs.json','prepared/samples.json'):
        expected=manifest['files'][name]['sha256']
        if sha(root/name)!=expected:raise ValueError('fixed input SHA mismatch: '+name)
    if model:
        for name,expected in manifest['contract']['model_files'].items():
            if sha(Path(model)/name)!=expected:raise ValueError('model SHA mismatch: '+name)
    batches=torch.load(root/'prepared/inputs.pt',map_location='cpu',weights_only=True)
    forward=json.loads((root/'prepared/forward-inputs.json').read_text())['batches']
    rows=[(info['sample_ids'],batch['inputs']) for batch,info in zip(batches,forward)]
    if samples or batch_size:
        selected=samples or [s for ids,_ in rows for s in ids]
        found={}
        for ids,inputs in rows:
            for i,sample in enumerate(ids):
                mask=inputs['attention_mask'][i].bool()
                found[sample]={key:value[i][mask].clone() for key,value in inputs.items()}
        if len(selected)!=len(set(selected)) or any(s not in found for s in selected):raise ValueError('unknown or duplicate sample ID')
        rows=[];size=batch_size or manifest['contract']['inputs']['batch_size']
        for offset in range(0,len(selected),size):
            ids=selected[offset:offset+size];width=max(found[s]['input_ids'].numel() for s in ids)
            inputs={key:torch.zeros((len(ids),width),dtype=found[ids[0]][key].dtype) for key in found[ids[0]]}
            for i,sample in enumerate(ids):
                n=found[sample]['input_ids'].numel()
                for key in inputs:inputs[key][i,-n:]=found[sample][key]
            rows.append((ids,inputs))
    identity=dict(oracle_id=manifest['oracle_id'],inputs_sha256=manifest['files']['prepared/inputs.pt']['sha256'],
        model_files=manifest['contract']['model_files'],samples=[s for ids,_ in rows for s in ids],
        batches=[dict(samples=ids,shape=list(v['input_ids'].shape),valid_tokens=v['attention_mask'].sum(1).tolist()) for ids,v in rows],probe_alg='p4v1')
    return rows,identity


def numerics(device,mode):
    import torch
    if mode=='reference':
        if not device.startswith('cuda'):raise ValueError('reference numerics require CUDA')
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
    return dict(mode=mode,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        cudnn_benchmark=torch.backends.cudnn.benchmark)


def software():
    result={}
    for name in ('torch','torch-npu','transformers','flag-gems','triton'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result
