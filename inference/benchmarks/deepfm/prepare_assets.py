"""Initialize once on CPU and seal DeepFM weights plus fixed categorical inputs."""
import argparse
import random
from pathlib import Path

from benchmarks.deepfm.adapter import DEFAULTS
from runtime.common import write_json, file_hash, ROOT


def prepare(output, seed=2026, samples=128):
    import torch
    from safetensors.torch import save_file
    from benchmarks.deepfm.model import DeepFM
    root = Path(output)
    root.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    model = DeepFM(**DEFAULTS).eval()
    write_json(root/'config.json',DEFAULTS)
    save_file({k:v.contiguous() for k,v in model.state_dict().items()},str(root/'model.safetensors'))
    rng = random.Random(seed)
    dims = DEFAULTS['field_dims']
    rows = [{'id':f'deepfm-{i:04}', 'features': [0]*len(dims) if i == 0 else
             [d-1 for d in dims] if i == 1 else [rng.randrange(d) for d in dims]} for i in range(samples)]
    import json
    (root/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(root/'assets.json',{'schema_version':1,'model':'deepfm','seed':seed,
               'parameter_count':sum(p.numel() for p in model.parameters()),
               'files':{n:file_hash(root/n) for n in ('config.json','model.safetensors')},
               'input_sha256':file_hash(root/'inputs.jsonl'),
               'source':__import__('json').loads((ROOT/'benchmarks/deepfm/origin.json').read_text()),
               'purpose':'synthetic inference benchmark; no CTR/AUC claim'})
    return root


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=2026)
    args = p.parse_args()
    print(prepare(args.output,args.seed))
