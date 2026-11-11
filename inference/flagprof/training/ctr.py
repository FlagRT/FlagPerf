# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""CTR training task over the sealed ctr-package-v1 inference corpus.

The sealed package stays byte-identical to the inference contract: digests are
verified by the shared loader and training targets are derived deterministically
from the sample ids, never stored in the package. Every architecture keeps its
inference forward semantics unchanged (sigmoid/cosine stay inside forward where
the reference places them); the task-defined losses consume those outputs
directly. Importing this module does not import torch.
"""
import hashlib
import json


# Forward returns a raw logit [batch, 1]; the inference transform applies the sigmoid.
LOGIT_ARCHS = ('dlrm', 'din', 'widedeep', 'deepfm', 'dcn', 'sasrec', 'lightgcn')
# Forward returns sigmoided scores [batch]; the inference transform passes them through.
PROB_ARCHS = ('bst', 'dien', 'dcnmix')
# Forward returns per-task sigmoided columns; esmm additionally emits ctcvr.
MULTI_ARCHS = ('esmm', 'mmoe', 'ple')
TASK_ARCHS = LOGIT_ARCHS + PROB_ARCHS + MULTI_ARCHS + ('dssm',)

LOSS_PRECISION = 'float32 loss region (autocast disabled inside the task loss)'

# Device types whose autocast context manager can safely disable an enclosing
# autocast region; anything else relies on the engine running with amp=off.
AUTOCAST_TYPES = ('cpu', 'cuda', 'xpu', 'npu')


def label_bit(salt, sample_id):
    """Deterministic 0/1 target derived from the sample id; no RNG, no package change."""
    digest = hashlib.sha256(('ctr-train/' + salt + '/' + sample_id).encode()).digest()
    return digest[0] & 1


def loss_definition(arch):
    """Canonical per-arch loss description recorded in the training protocol."""
    if arch in LOGIT_ARCHS:
        base = 'binary_cross_entropy_with_logits(raw forward output, click_label)'
    elif arch in PROB_ARCHS:
        base = 'binary_cross_entropy(forward sigmoid output, click_label)'
    elif arch == 'esmm':
        base = ('binary_cross_entropy(cvr column, cvr_label) + binary_cross_entropy(ctr column, '
                'click_label) + binary_cross_entropy(ctcvr column, click_label * cvr_label)')
    elif arch in ('mmoe', 'ple'):
        base = ('binary_cross_entropy(task 0 column, click_label) + '
                'binary_cross_entropy(task 1 column, cvr_label)')
    elif arch == 'dssm':
        base = 'mse(cosine output, 2 * click_label - 1)'
    else:
        raise ValueError('no training loss defined for architecture: ' + arch)
    return base + '; ' + LOSS_PRECISION


def build_loss_fn(arch):
    """Task loss: consumes the architecture's inference forward output unchanged."""
    import torch
    from contextlib import nullcontext
    import torch.nn.functional as F

    def loss_fn(model, batch):
        click = batch['click_label']
        cvr = batch.get('cvr_label')
        inputs = {k: v for k, v in batch.items() if not k.endswith('_label')}
        output = model(**inputs)
        device_type = batch['dense'].device.type
        if device_type in AUTOCAST_TYPES:
            region = torch.autocast(device_type, enabled=False)
        else:
            region = nullcontext()
        with region:
            output = output.float()
            if arch in LOGIT_ARCHS:
                return F.binary_cross_entropy_with_logits(output.reshape(-1), click)
            if arch in PROB_ARCHS:
                return F.binary_cross_entropy(output.reshape(-1), click)
            if arch == 'esmm':
                columns = output.reshape(-1, 3)  # [cvr, ctr, ctcvr]
                return (F.binary_cross_entropy(columns[:, 0], cvr)
                        + F.binary_cross_entropy(columns[:, 1], click)
                        + F.binary_cross_entropy(columns[:, 2], click * cvr))
            if arch in ('mmoe', 'ple'):
                columns = output.reshape(-1, 2)
                return (F.binary_cross_entropy(columns[:, 0], click)
                        + F.binary_cross_entropy(columns[:, 1], cvr))
            if arch == 'dssm':
                return F.mse_loss(output.reshape(-1), click * 2 - 1)
        raise ValueError('no training loss defined for architecture: ' + arch)
    return loss_fn


def prepare_ctr(*, arch, package, seed, batch_size, samples=None, rank=0, world_size=1):
    """Build the CTR training task from a sealed package; returns (net, batches, loss_fn, protocol).

    Batches are fixed CPU dicts carrying the package rows plus derived label
    tensors; the engine moves them and calls the loss. DDP splits the microbatch
    list interleaved by rank, mirroring the per-rank sharding of the
    classification task.
    """
    import torch
    from ..runner.ctr import (ARCH_CHOICES, ARCH_CONFIG, build_model, load_ctr_package,
                              weight_sha256, CtrCallable)
    if arch not in ARCH_CHOICES:
        raise ValueError('unknown CTR architecture: ' + arch + ' (valid: ' + ', '.join(ARCH_CHOICES) + ')')
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError('ctr training requires batch-size >= 2 (BatchNorm1d runs in train mode)')
    ordered, manifest = load_ctr_package(package, batch_size=batch_size, samples=samples)
    total = sum(len(ids) for ids, _ in ordered)
    if total % batch_size:
        raise ValueError('package sample count must be a positive multiple of batch-size '
                         '(equal microbatches); got ' + str(total) + ' samples for batch-size '
                         + str(batch_size))
    multi = arch in MULTI_ARCHS
    batches = []
    for sample_ids, stacked in ordered:
        fields = dict(stacked)
        fields['click_label'] = torch.tensor([label_bit('click', s) for s in sample_ids],
                                             dtype=torch.float32)
        if multi:
            fields['cvr_label'] = torch.tensor([label_bit('cvr', s) for s in sample_ids],
                                               dtype=torch.float32)
        if arch == 'lightgcn':
            fields['user_idx'] = torch.tensor([int(s) for s in sample_ids], dtype=torch.long)
        batches.append(fields)
    if world_size > 1:
        batches = batches[rank::world_size]
    if not batches:
        raise ValueError('rank ' + str(rank) + ' has no microbatches for world_size ' + str(world_size))
    net = build_model(arch, seed, 'cpu', dtype=None, package=package)
    config_sha = hashlib.sha256(json.dumps(ARCH_CONFIG, sort_keys=True, default=str).encode()).hexdigest()
    protocol = dict(
        arch=arch,
        ctr_package_oracle_id=manifest['oracle_id'],
        ctr_inputs_sha256=manifest['files']['prepared/inputs.pt']['sha256'],
        ctr_config_sha256=config_sha,
        # Same seed and builder as the inference runner, so this digest also ties
        # the training run to the inference oracle_identity weight_sha256.
        ctr_weight_sha256=weight_sha256(net),
        labels=('sha256-derived per-sample bits (salt ctr-train/{click,cvr}/<sample_id>, '
                'first byte bit 0); the sealed package is unchanged'),
        loss=loss_definition(arch),
        data='sealed ctr-package-v1 rows (dense, sparse, hist_ids, hist_mask) with derived labels',
    )
    return CtrCallable(arch, net), batches, build_loss_fn(arch), protocol
