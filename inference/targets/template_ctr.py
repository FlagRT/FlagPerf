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
  deepfm     torch-rechub DeepFM (models/ranking/deepfm.py): shared embedding,
             LR + FM (0.5*(sum^2 - sum_of_squares)) + deep MLP -> sigmoid.
  dcn        torch-rechub DCN (models/ranking/dcn.py): CrossNetwork
             (x = x0 * Linear_i(x) + b_i + x) + MLP -> LR over concat -> sigmoid.
  esmm       torch-rechub ESMM (multi_task/esmm.py): shared embedding, CVR/CTR
             towers sigmoided, ctcvr = ctr * cvr, output [cvr, ctr, ctcvr].
  mmoe       torch-rechub MMOE (multi_task/mmoe.py): shared embedding, expert
             MLPs, per-task softmax gates, towers, sigmoid; output [batch, 2].
  ple        torch-rechub PLE (multi_task/ple.py): stacked CGC levels (task-
             specific + shared experts, softmax gates), towers; output [batch, 2].
  bst        torch-rechub BST (models/ranking/bst.py): history+target through
             TransformerEncoder (positional embedding, src_key_padding_mask),
             target-position output -> MLP -> sigmoid.

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
        raw = tensor.detach().cpu().contiguous()
        if raw.dtype == torch.bfloat16:
            raw = raw.view(torch.int16)  # numpy() rejects bfloat16; keep raw bytes
        digest.update(raw.numpy().tobytes())
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
    if name == 'softmax':
        return torch.nn.Softmax(dim=-1)
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
    deepfm=dict(mlp_dims=[128, 64], activation='relu', batchnorm=True),
    dcn=dict(n_cross_layers=2, mlp_dims=[128, 64], activation='relu', batchnorm=True),
    esmm=dict(user_sparse=13, cvr_dims=[128, 64], ctr_dims=[128, 64], activation='relu', batchnorm=True),
    mmoe=dict(n_task=2, n_expert=2, expert_dims=[64, 16], tower_dims=[64],
              activation='relu', batchnorm=True),
    ple=dict(n_task=2, n_level=2, n_expert_specific=1, n_expert_shared=1,
             expert_dims=[64, 16], tower_dims=[64], activation='relu', batchnorm=True),
    bst=dict(profile_sparse=10, nhead=8, num_layers=1, dropout=0.0,
             mlp_dims=[128, 64], activation='relu', batchnorm=True),
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


class Fm(torch.nn.Module):
    """torch-rechub FM(reduce_sum=True), mirrored exactly."""

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class DeepFm(torch.nn.Module):
    """torch-rechub DeepFM semantics: shared embedding, LR + FM + deep MLP."""

    def __init__(self, config):
        super().__init__()
        c = ARCH_CONFIG
        spec = config['deepfm']
        emb = c['emb_dim']
        self.embedding = torch.nn.Embedding(c['vocab'], emb)
        self.linear = torch.nn.Linear(c['sparse_dim'] * emb, 1)
        self.fm = Fm()
        self.mlp = Mlp.build(c['sparse_dim'] * emb, spec['mlp_dims'],
                             activation=spec['activation'], output_layer=True,
                             batchnorm=spec['batchnorm'])

    def forward(self, dense, sparse):
        fields = self.embedding(sparse)
        flat = fields.flatten(start_dim=1)
        y_linear = self.linear(flat)
        y_fm = self.fm(fields)
        y_deep = self.mlp(flat)
        return y_linear + y_fm + y_deep


class CrossNetwork(torch.nn.Module):
    """torch-rechub CrossNetwork: x = x0 * Linear(x) + b + x per layer."""

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, 1, bias=False)
                                      for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(input_dim))
                                         for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(len(self.w)):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class Dcn(torch.nn.Module):
    """torch-rechub DCN semantics: cross network + MLP + LR over concat."""

    def __init__(self, config):
        super().__init__()
        c = ARCH_CONFIG
        spec = config['dcn']
        dims = c['sparse_dim'] * c['emb_dim']
        self.embedding = torch.nn.Embedding(c['vocab'], c['emb_dim'])
        self.cn = CrossNetwork(dims, spec['n_cross_layers'])
        self.mlp = Mlp.build(dims, spec['mlp_dims'], activation=spec['activation'],
                             output_layer=False, batchnorm=spec['batchnorm'])
        self.linear = torch.nn.Linear(dims + spec['mlp_dims'][-1], 1)

    def forward(self, dense, sparse):
        embed = self.embedding(sparse).flatten(start_dim=1)
        cn_out = self.cn(embed)
        mlp_out = self.mlp(embed)
        return self.linear(torch.cat([cn_out, mlp_out], dim=1))


class Esmm(torch.nn.Module):
    """torch-rechub ESMM semantics: shared embedding, two towers, ctcvr product."""

    def __init__(self, config):
        super().__init__()
        c = ARCH_CONFIG
        spec = config['esmm']
        self.user_n = spec['user_sparse']
        dims = c['sparse_dim'] * c['emb_dim']
        self.embedding = torch.nn.Embedding(c['vocab'], c['emb_dim'])
        self.tower_cvr = Mlp.build(dims, spec['cvr_dims'], activation=spec['activation'],
                                   output_layer=True, batchnorm=spec['batchnorm'])
        self.tower_ctr = Mlp.build(dims, spec['ctr_dims'], activation=spec['activation'],
                                   output_layer=True, batchnorm=spec['batchnorm'])

    def forward(self, dense, sparse):
        embed = self.embedding(sparse).flatten(start_dim=1)
        cvr_pred = torch.sigmoid(self.tower_cvr(embed))
        ctr_pred = torch.sigmoid(self.tower_ctr(embed))
        ctcvr_pred = torch.mul(ctr_pred, cvr_pred)
        return torch.cat([cvr_pred, ctr_pred, ctcvr_pred], dim=1)


class Mmoe(torch.nn.Module):
    """torch-rechub MMOE semantics: shared embedding, experts, per-task softmax
    gates, towers, sigmoid per task; output [batch, n_task]."""

    def __init__(self, config):
        super().__init__()
        c = ARCH_CONFIG
        spec = config['mmoe']
        dims = c['sparse_dim'] * c['emb_dim']
        self.embedding = torch.nn.Embedding(c['vocab'], c['emb_dim'])
        self.experts = torch.nn.ModuleList([
            Mlp.build(dims, spec['expert_dims'], activation=spec['activation'],
                      output_layer=False, batchnorm=spec['batchnorm'])
            for _ in range(spec['n_expert'])])
        self.gates = torch.nn.ModuleList([
            Mlp.build(dims, [spec['n_expert']], activation='softmax',
                      output_layer=False, batchnorm=spec['batchnorm'])
            for _ in range(spec['n_task'])])
        self.towers = torch.nn.ModuleList([
            Mlp.build(spec['expert_dims'][-1], spec['tower_dims'],
                      activation=spec['activation'], output_layer=True,
                      batchnorm=spec['batchnorm'])
            for _ in range(spec['n_task'])])

    def forward(self, dense, sparse):
        embed = self.embedding(sparse).flatten(start_dim=1)
        expert_outs = torch.stack([expert(embed) for expert in self.experts], dim=1)
        ys = []
        for gate, tower in zip(self.gates, self.towers):
            gate_out = gate(embed).unsqueeze(-1)
            expert_pooling = torch.sum(torch.mul(gate_out, expert_outs), dim=1)
            ys.append(torch.sigmoid(tower(expert_pooling)))  # PredictionLayer
        return torch.cat(ys, dim=1)


class Cgc(torch.nn.Module):
    """torch-rechub CGC layer, mirrored exactly (shared gate only below top)."""

    def __init__(self, cur_level, spec):
        super().__init__()
        c = ARCH_CONFIG
        self.n_task = spec['n_task']
        self.n_specific = spec['n_expert_specific']
        self.n_shared = spec['n_expert_shared']
        self.cur_level = cur_level
        self.n_level = spec['n_level']
        self.n_all = self.n_specific * self.n_task + self.n_shared
        input_dims = (c['sparse_dim'] * c['emb_dim'] if cur_level == 1
                      else spec['expert_dims'][-1])
        self.experts_specific = torch.nn.ModuleList([
            Mlp.build(input_dims, spec['expert_dims'], activation=spec['activation'],
                      output_layer=False, batchnorm=spec['batchnorm'])
            for _ in range(self.n_task * self.n_specific)])
        self.experts_shared = torch.nn.ModuleList([
            Mlp.build(input_dims, spec['expert_dims'], activation=spec['activation'],
                      output_layer=False, batchnorm=spec['batchnorm'])
            for _ in range(self.n_shared)])
        self.gates_specific = torch.nn.ModuleList([
            Mlp.build(input_dims, [self.n_specific + self.n_shared], activation='softmax',
                      output_layer=False, batchnorm=spec['batchnorm'])
            for _ in range(self.n_task)])
        if cur_level < self.n_level:
            self.gate_shared = Mlp.build(input_dims, [self.n_all], activation='softmax',
                                         output_layer=False, batchnorm=spec['batchnorm'])

    def forward(self, x_list):
        expert_specific_outs = []
        for i in range(self.n_task):
            experts = self.experts_specific[i * self.n_specific:(i + 1) * self.n_specific]
            expert_specific_outs.extend([expert(x_list[i]).unsqueeze(1) for expert in experts])
        expert_shared_outs = [expert(x_list[-1]).unsqueeze(1) for expert in self.experts_shared]
        gate_specific_outs = [gate(x_list[i]).unsqueeze(-1)
                              for i, gate in enumerate(self.gates_specific)]
        cgc_outs = []
        for i, gate_out in enumerate(gate_specific_outs):
            cur = expert_specific_outs[i * self.n_specific:(i + 1) * self.n_specific] + expert_shared_outs
            expert_concat = torch.cat(cur, dim=1)
            expert_pooling = torch.sum(torch.mul(gate_out, expert_concat), dim=1)
            cgc_outs.append(expert_pooling)
        if self.cur_level < self.n_level:
            gate_shared_out = self.gate_shared(x_list[-1]).unsqueeze(-1)
            expert_concat = torch.cat(expert_specific_outs + expert_shared_outs, dim=1)
            expert_pooling = torch.sum(torch.mul(gate_shared_out, expert_concat), dim=1)
            cgc_outs.append(expert_pooling)
        return cgc_outs


class Ple(torch.nn.Module):
    """torch-rechub PLE semantics: stacked CGC levels then per-task towers."""

    def __init__(self, config):
        super().__init__()
        c = ARCH_CONFIG
        spec = config['ple']
        self.n_task = spec['n_task']
        self.n_level = spec['n_level']
        self.embedding = torch.nn.Embedding(c['vocab'], c['emb_dim'])
        self.cgc_layers = torch.nn.ModuleList([
            Cgc(i + 1, spec) for i in range(spec['n_level'])])
        self.towers = torch.nn.ModuleList([
            Mlp.build(spec['expert_dims'][-1], spec['tower_dims'],
                      activation=spec['activation'], output_layer=True,
                      batchnorm=spec['batchnorm'])
            for _ in range(self.n_task)])

    def forward(self, dense, sparse):
        embed = self.embedding(sparse).flatten(start_dim=1)
        ple_inputs = [embed] * (self.n_task + 1)
        ple_outs = []
        for i in range(self.n_level):
            ple_outs = self.cgc_layers[i](ple_inputs)
            ple_inputs = ple_outs
        ys = []
        for ple_out, tower in zip(ple_outs, self.towers):
            ys.append(torch.sigmoid(tower(ple_out)))  # PredictionLayer
        return torch.cat(ys, dim=1)


class Bst(torch.nn.Module):
    """torch-rechub BST semantics: history+target through a TransformerEncoder
    with positional embedding and src_key_padding_mask; target-position output
    -> MLP -> sigmoid (reference sigmoids inside forward)."""

    def __init__(self, config):
        super().__init__()
        c = ARCH_CONFIG
        spec = config['bst']
        emb = c['emb_dim']
        self.profile_n = spec['profile_sparse']
        self.max_seq_len = c['hist_len'] + 1
        self.embedding = torch.nn.Embedding(c['vocab'], emb)
        self.pos_embedding = torch.nn.Embedding(self.max_seq_len, emb)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=emb, nhead=spec['nhead'], dropout=spec['dropout'],
            activation=torch.nn.LeakyReLU(), batch_first=True)
        self.transformer_layers = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=spec['num_layers'])
        all_dims = self.profile_n * emb + 2 * emb
        self.mlp = Mlp.build(all_dims, spec['mlp_dims'], activation=spec['activation'],
                             output_layer=True, batchnorm=spec['batchnorm'])

    def forward(self, dense, sparse, hist_ids, hist_mask):
        profile = self.embedding(sparse[:, :self.profile_n]).flatten(start_dim=1)
        target = self.embedding(sparse[:, self.profile_n])
        history = self.embedding(hist_ids)
        seq = torch.cat([history, target.unsqueeze(1)], dim=1)
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        seq = seq + self.pos_embedding(positions)
        pad_mask = hist_ids == 0
        tgt_mask = torch.zeros(pad_mask.size(0), 1, dtype=torch.bool,
                               device=pad_mask.device)
        src_key_padding_mask = torch.cat([pad_mask, tgt_mask], dim=1)
        out = self.transformer_layers(seq, src_key_padding_mask=src_key_padding_mask)
        interest = out[:, -1, :]
        mlp_in = torch.cat([interest, target, profile], dim=1)
        return torch.sigmoid(self.mlp(mlp_in).squeeze(1))


class CtrCallable(torch.nn.Module):
    """Adapter so run_prepared can call every arch with the full input dict."""

    def __init__(self, arch, net):
        super().__init__()
        self.arch = arch
        self.net = net

    def forward(self, dense, sparse, hist_ids=None, hist_mask=None):
        if self.arch in ('din', 'bst'):
            return self.net(dense, sparse, hist_ids, hist_mask)
        return self.net(dense, sparse)


def build_model(arch, seed, device, dtype=None):
    import torch
    torch.manual_seed(seed)
    builders = dict(dlrm=Dlrm, din=Din, widedeep=WideDeep, deepfm=DeepFm,
                    dcn=Dcn, esmm=Esmm, mmoe=Mmoe, ple=Ple, bst=Bst)
    if arch not in builders:
        raise ValueError('template arch not implemented yet: ' + arch)
    model = builders[arch](ARCH_CONFIG)
    if dtype is not None:
        model = model.to(dtype)
    return model.to(device).eval()


def prepare_inputs(cpu, device, dtype):
    """Move a cpu input dict to device; float inputs are cast to the compute
    dtype (package dense is float32, model weights follow --dtype). Integer
    id fields are never cast."""
    out = {}
    for key, value in cpu.items():
        if value.is_floating_point():
            out[key] = value.to(device, dtype)
        else:
            out[key] = value.to(device)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', required=True)
    p.add_argument('--arch', choices=['dlrm', 'din', 'widedeep', 'deepfm', 'dcn', 'esmm',
                                      'mmoe', 'ple', 'bst'], required=True)
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
        inputs = prepare_inputs(cpu, args.device, dtype)
        def transform(output):
            if args.arch in ('esmm', 'mmoe', 'ple', 'bst'):
                # reference already applies sigmoid; esmm/mmoe/ple are [batch, n_task]
                scores = output.float()
            else:
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
