"""Initialize once on CPU and seal configurable DLRM weights plus fixed ragged inputs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

from benchmarks.dlrm.config import DEFAULTS, validate_architecture
from runtime.common import ROOT, write_json, file_hash


def prepare(output, seed=2026, samples=128, config=None):
    import torch
    from safetensors.torch import save_file
    from benchmarks.dlrm.model import DLRM
    config = validate_architecture(deepcopy(DEFAULTS if config is None else config))
    if type(seed) is not int or type(samples) is not int or samples < 1:
        raise ValueError('seed must be an integer and samples must be positive')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = DLRM(**config).eval()
    write_json(root / 'config.json', config)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(root / 'model.safetensors'))
    rng = random.Random(seed)
    rows = []
    for i in range(samples):
        dense = [rng.random() for _ in range(config['dense_dim'])]
        bags = [[rng.randrange(d) for _ in range(rng.randrange(5))] for d in config['field_dims']]
        if i == 0:
            dense, bags = [0.] * config['dense_dim'], [[] for _ in config['field_dims']]
        elif i == 1:
            bags = [[d-1, d-1] for d in config['field_dims']]
        rows.append({'id': f'dlrm-{i:04}', 'dense': dense, 'sparse': bags})
    (root / 'inputs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_json(root / 'assets.json', {'schema_version': 1, 'model': 'dlrm', 'seed': seed,
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'files': {n: file_hash(root / n) for n in ('config.json', 'model.safetensors')},
               'input_sha256': file_hash(root / 'inputs.jsonl'),
               'source': json.loads((ROOT / 'benchmarks/dlrm/origin.json').read_text()),
               'purpose': 'synthetic inference benchmark; no CTR/AUC claim'})
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--architecture', help='Architecture JSON; widths and field cardinalities are sealed with weights')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--samples', type=int, default=128)
    args = parser.parse_args()
    config = json.loads(Path(args.architecture).read_text()) if args.architecture else None
    print(prepare(args.output, args.seed, args.samples, config))
