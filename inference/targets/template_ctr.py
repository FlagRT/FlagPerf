# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Vendor-agnostic CTR attach target: sealed ctr-package-v1 -> click scores.

Architectures mirror canonical reference implementations:

  dlrm       Meta facebookresearch/dlrm (dlrm_s_pytorch.py): bottom MLP on
             the dense vector projected to emb_dim, per-feature embeddings,
             dot interaction (upper triangle, diagonal excluded, dense
             vector included), top MLP -> sigmoid.
  din        torch-rechub DIN (models/ranking/din.py): ActivationUnit target
             attention ([target, history, target-history, target*history]
             -> MLP -> multiplicative mask -> weighted sum), dice MLP.
  widedeep   torch-rechub WideDeep (models/ranking/widedeep.py): wide linear
             over 1-dim sparse embeddings + raw dense, deep MLP over 16-dim
             embeddings + dense, sigmoid(wide + deep).

Weights are random-initialized from torch.manual_seed(--seed, default 531);
CTR templates have no pretrained upstream weights, so the run identity pins
the actual initialized weights (weight_sha256 over the state dict), the
architecture config canonical JSON and the package inputs digest instead of
weight files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_common import sha


def load_ctr_package(package, *, batch_size, samples=None):
    """Load a sealed ctr-package-v1, verify digests, return (batches, manifest).

    Batches are list[(sample_ids, dict_of_stacked_cpu_tensors)].
    """
    import torch
    root = Path(package)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('kind') != 'ctr-package-v1':
        raise ValueError('not a ctr-package-v1 package: ' + str(root))
    for rel, spec in sorted(manifest['files'].items()):
        if sha(root / rel) != spec['sha256']:
            raise ValueError('digest mismatch: ' + rel)
    rows = torch.load(root / 'prepared' / 'inputs.pt', weights_only=False)
    wanted = samples if samples is not None else [str(i) for i in range(len(rows))]
    for sample in wanted:
        if sample not in {str(i) for i in range(len(rows))}:
            raise ValueError('unknown sample: ' + sample)
    order = [int(s) for s in wanted]
    batches = []
    fields = ('dense', 'sparse', 'hist_ids', 'hist_mask')
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        stacked = {}
        for field in fields:
            values = [rows[i][field] for i in chunk]
            stacked[field] = torch.stack(values)
        batches.append(([str(i) for i in chunk], stacked))
    return batches, manifest


def weight_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Mlp:
    """Linear(+BN)+activation stack mirroring torch-rechub MLP defaults."""

    @staticmethod
    def build(input_dim, dims, activation='relu', output_layer=True, batchnorm=True):
        import torch
        layers = []
        prev = input_dim
        for dim in dims:
            layers.append(torch.nn.Linear(prev, dim))
            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(_activation(activation))
            prev = dim
        if output_layer:
            layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers)


def _activation(name):
    import torch
    if name == 'relu':
        return torch.nn.ReLU()
    if name == 'dice':
        return Dice()
    raise ValueError('unsupported activation ' + name)


class Dice(torch.nn.Module):
    """torch-rechub basic/activation.py Dice, mirrored exactly."""

    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        avg = x.mean(dim=1).unsqueeze(dim=1)
        var = torch.pow(x - avg, 2) + self.epsilon
        var = var.sum(dim=1).unsqueeze(dim=1)
        ps = (x - avg) / torch.sqrt(var)
        ps = torch.nn.Sigmoid()(ps)
        return ps * x + (1 - ps) * self.alpha * x


ARCH_CONFIG = dict(
    emb_dim=16, vocab=1000, dense_dim=13, sparse_dim=26, hist_len=20,
    dlrm=dict(bottom_dims=[128, 16], top_dims=[256, 128, 64], activation='relu', batchnorm=False),
    din=dict(profile_sparse=10, attn_dims=[36], mlp_dims=[128, 64], activation='dice', batchnorm=True),
    widedeep=dict(mlp_dims=[128, 64], activation='relu', batchnorm=True),
)


class Dlrm(torch.nn.Module):
    """Meta DLRM single-device semantics with the template's fixed config."""

    def __init__(self, config):
        super().__init__()
        c = ARCH_CONFIG
        self.dense_dim, self.sparse_dim = c['dense_dim'], c['sparse_dim']
        emb = c['emb_dim']
        spec = config['dlrm']
        layers = []
        prev = self.dense_dim
        for dim in spec['bottom_dims']:
            layers += [torch.nn.Linear(prev, dim), torch.nn.ReLU()]
            prev = dim
        self.bottom = torch.nn.Sequential(*layers)
        self.emb_dim = emb
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(c['vocab'], emb) for _ in range(self.sparse_dim)])
        self.top = Mlp.build(prev + self._interaction_dim(), spec['top_dims'],
                             activation=spec['activation'], output_layer=True,
                             batchnorm=spec['batchnorm'])

    def _interaction_dim(self):
        n = self.sparse_dim + 1
        return n * (n - 1) // 2

    def forward(self, dense, sparse):
        dense_out = self.bottom(dense)
        vectors = [dense_out]
        for index in range(self.sparse_dim):
            vectors.append(self.embeddings[index](sparse[:, index]))
        stacked = torch.stack(vectors, dim=1)
        dots = torch.bmm(stacked, stacked.transpose(1, 2))
        tril = torch.tril_indices(dots.size(1), dots.size(2), offset=-1,
                                  device=dots.device)
        interaction = dots[:, tril[0], tril[1]]
        return self.top(torch.cat([dense_out, interaction], dim=1))


class ActivationUnit(torch.nn.Module):
    """torch-rechub ActivationUnit: multiplicative mask (use_softmax False)."""

    def __init__(self, emb_dim, dims):
        super().__init__()
        self.attention = Mlp.build(4 * emb_dim, dims, activation='dice',
                                   output_layer=True, batchnorm=True)

    def forward(self, history, target, mask):
        seq_length = history.size(1)
        expanded = target.unsqueeze(1).expand(-1, seq_length, -1)
        att_input = torch.cat([expanded, history, expanded - history, expanded * history], dim=-1)
        weight = self.attention(att_input.view(-1, att_input.size(-1))).view(-1, seq_length)
        weight = weight * mask.to(weight.dtype)
        return (weight.unsqueeze(-1) * history).sum(dim=1)


class Din(torch.nn.Module):
    """torch-rechub DIN semantics with the template's fixed config."""

    def __init__(self, config):
        super().__init__()
        c = ARCH_CONFIG
        emb = c['emb_dim']
        spec = config['din']
        self.profile_n = spec['profile_sparse']
        self.embedding = torch.nn.Embedding(c['vocab'], emb)
        self.attention = ActivationUnit(emb, spec['attn_dims'])
        mlp_in = 2 * emb + self.profile_n * emb
        self.mlp = Mlp.build(mlp_in, spec['mlp_dims'], activation=spec['activation'],
                             output_layer=True, batchnorm=spec['batchnorm'])

    def forward(self, dense, sparse, hist_ids, hist_mask):
        profile = self.embedding(sparse[:, :self.profile_n]).flatten(start_dim=1)
        target = self.embedding(sparse[:, self.profile_n])
        history = self.embedding(hist_ids)
        attended = self.attention(history, target, hist_mask)
        return self.mlp(torch.cat([attended, target, profile], dim=1))


class WideDeep(torch.nn.Module):
    """torch-rechub WideDeep semantics: wide LR over concatenated per-feature
    one-hot embeddings + raw dense; deep MLP over 16-dim embeddings + dense."""

    def __init__(self, config):
        super().__init__()
        c = ARCH_CONFIG
        spec = config['widedeep']
        self.wide_linear = torch.nn.Linear(c['sparse_dim'] * c['vocab'] + c['dense_dim'], 1)
        self.sparse_embedding = torch.nn.Embedding(c['vocab'], c['emb_dim'])
        self.mlp = Mlp.build(c['sparse_dim'] * c['emb_dim'] + c['dense_dim'],
                             spec['mlp_dims'], activation=spec['activation'],
                             output_layer=True, batchnorm=spec['batchnorm'])

    def forward(self, dense, sparse):
        onehot = torch.nn.functional.one_hot(sparse, ARCH_CONFIG['vocab']).to(dense.dtype)
        wide_feat = onehot.flatten(start_dim=1)
        y_wide = self.wide_linear(torch.cat([wide_feat, dense], dim=1))
        deep_in = torch.cat([self.sparse_embedding(sparse).flatten(start_dim=1), dense], dim=1)
        return self.mlp(deep_in) + y_wide


class CtrCallable(torch.nn.Module):
    """Adapter so run_prepared can call every arch with the full input dict."""

    def __init__(self, arch, net):
        super().__init__()
        self.arch = arch
        self.net = net

    def forward(self, dense, sparse, hist_ids=None, hist_mask=None):
        if self.arch == 'din':
            return self.net(dense, sparse, hist_ids, hist_mask)
        return self.net(dense, sparse)


def build_model(arch, seed, device, dtype=None):
    import torch
    torch.manual_seed(seed)
    if arch == 'dlrm':
        model = Dlrm(ARCH_CONFIG)
    elif arch == 'din':
        model = Din(ARCH_CONFIG)
    elif arch == 'widedeep':
        model = WideDeep(ARCH_CONFIG)
    else:
        raise ValueError('template arch not implemented yet: ' + arch)
    if dtype is not None:
        model = model.to(dtype)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', required=True)
    p.add_argument('--arch', choices=['dlrm', 'din', 'widedeep'], required=True)
    p.add_argument('--seed', type=int, default=531)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--outputs', required=True)
    p.add_argument('--samples')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--forwards', type=int, default=3)
    args = p.parse_args()

    import torch
    from flagprof import current_attachment
    from flagprof.runner import run_prepared
    handle = current_attachment()
    sync = torch.cuda.synchronize if args.device.startswith('cuda') else (torch.npu.synchronize if args.device.startswith('npu') else (lambda: None))
    if args.device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(args.device)

    batches, manifest = load_ctr_package(args.package, batch_size=args.batch_size,
        samples=args.samples.split(',') if args.samples else None)
    if args.arch not in manifest['contract']['archs']:
        p.error('arch not supported by package contract')
    dtype = getattr(torch, args.dtype) if args.dtype != 'float32' else torch.float32
    net = build_model(args.arch, args.seed, args.device, dtype=dtype)
    model = CtrCallable(args.arch, net)
    config_sha = hashlib.sha256(json.dumps(ARCH_CONFIG, sort_keys=True,
        default=str).encode()).hexdigest()
    identity = dict(kind='ctr-template-v1', arch=args.arch,
        config_sha256=config_sha, weight_sha256=weight_sha256(net),
        seed=args.seed, init='manual_seed+default-init',
        inputs_sha256=manifest['files']['prepared/inputs.pt']['sha256'],
        oracle_id=manifest['oracle_id'], dtype=args.dtype, device=args.device,
        samples=[s for ids_, _ in batches for s in ids_])

    out = Path(args.outputs)
    run_rows = []
    for batch_index, (sample_ids, cpu) in enumerate(batches):
        inputs = {key: value.to(args.device) for key, value in cpu.items()}
        def transform(output):
            scores = torch.sigmoid(output.float().squeeze(1))
            if handle:
                handle.record_tensor('scores', scores)
            return scores
        if handle:
            handle.bind_tensor_inputs(sample_ids, inputs, identity=identity)
        folder = out / str(batch_index) if len(batches) > 1 else out
        run_rows.append(dict(samples=sample_ids,
            run=run_prepared(model, inputs, warmup=args.warmup,
                             forwards=args.forwards, outputs=folder, sync=sync,
                             transform=transform)))
    score_sha = hashlib.sha256(json.dumps([r['run']['outputs'] for r in run_rows],
        sort_keys=True).encode()).hexdigest()
    result = dict(model='template:' + args.arch, arch=args.arch,
        semantics='{ctr-template ' + args.arch + ' sigmoid scores; random-init seed=' +
                  str(args.seed) + '; weight_sha256 pinned}',
        oracle_identity=identity, batch_size=args.batch_size,
        score_batches_sha256=score_sha, runs=run_rows)
    (out / 'target-result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), attachment=bool(handle),
        score_batches_sha256=score_sha)))


if __name__ == '__main__':
    main()
