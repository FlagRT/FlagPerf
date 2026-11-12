# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Reranker fine-tune task over the sealed rerank-package-v1 corpus.

The sealed package stays byte-identical to the inference contract (digests
verified by the shared runner loader); relevance targets are derived
deterministically from the sample ids and never written into the package.
Two scoring semantics, mirroring the inference runner:

- causal_yesno: causal LM logits at each row's last valid position, a 2-way
  cross entropy over the literal ``no``/``yes`` token columns (the same
  softmax the inference scorer applies).
- sequence_logit: sequence-classification head logit, BCEWithLogits.

The loss region is pinned to float32 (autocast disabled inside the loss,
same policy as the other tasks); importing this module does not import torch.
"""
import hashlib
import json

SCORES = ('causal_yesno', 'sequence_logit')


def relevance_bit(sample_id):
    """Deterministic 0/1 target derived from the sample id; no RNG, no package change."""
    digest = hashlib.sha256(('rerank-train/' + sample_id).encode()).digest()
    return digest[0] & 1


def loss_definition(score):
    if score == 'causal_yesno':
        return ('cross_entropy over the [no, yes] token columns of the last valid position '
                '(target bit 1 = yes)')
    return 'binary_cross_entropy_with_logits(sequence-classification logit, relevance bit)'


def build_loss_fn(score, yes_id, no_id):
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext

    def loss_fn(model, batch):
        mask = batch['attention_mask']
        target = batch['rerank_target']
        output = model(input_ids=batch['input_ids'], attention_mask=mask)
        logits = getattr(output, 'logits', None)
        if logits is None:
            logits = output[0]
        device_type = logits.device.type
        if device_type in ('cpu', 'cuda', 'xpu', 'npu'):
            region = torch.autocast(device_type, enabled=False)
        else:
            region = nullcontext()
        with region:
            if score == 'causal_yesno':
                positions = torch.arange(logits.shape[1], device=logits.device).expand_as(mask)
                last = positions.masked_fill(mask == 0, -1).max(dim=1).values
                picked = logits[torch.arange(logits.shape[0], device=logits.device), last]
                picked = picked[:, [no_id, yes_id]]
                return F.cross_entropy(picked.float(), target.long())
            return F.binary_cross_entropy_with_logits(logits[:, 0].float(), target.float())
    return loss_fn


def prepare_rerank(*, model, package, score, batch_size, yes_id=None, no_id=None,
                   samples=None, rank=0, world_size=1):
    """Build the rerank training task from a sealed package; returns (model, batches, loss_fn, protocol).

    Batches are fixed CPU dicts (padded input_ids/attention_mask from the
    shared loader plus a derived relevance bit); the engine moves them and
    calls the loss. DDP splits the microbatch list interleaved by rank.
    """
    if score not in SCORES:
        raise ValueError('rerank score must be one of ' + ', '.join(SCORES))
    if score == 'causal_yesno' and (yes_id is None or no_id is None):
        raise ValueError('causal_yesno requires the tokenizer yes/no token ids')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch-size must be a positive integer')
    from ..runner.rerank_common import load_package
    from ..runner.ctr import weight_sha256
    ordered, manifest = load_package(package, batch_size=batch_size, samples=samples)
    if world_size > 1:
        ordered = ordered[rank::world_size]
        if not ordered:
            raise ValueError('rank ' + str(rank) + ' has no microbatches for world_size ' + str(world_size))
    import torch
    batches = []
    for sample_ids, stacked in ordered:
        fields = dict(stacked)
        fields['rerank_target'] = torch.tensor([relevance_bit(s) for s in sample_ids],
                                               dtype=torch.float32)
        batches.append(fields)
    config_sha = hashlib.sha256(json.dumps(manifest['contract'], sort_keys=True, default=str).encode()).hexdigest()
    protocol = dict(
        score_semantics=score,
        rerank_package_oracle_id=manifest['oracle_id'],
        rerank_inputs_sha256=manifest['files']['prepared/inputs.pt']['sha256'],
        rerank_contract_sha256=config_sha,
        rerank_template_sha256=manifest.get('template_sha'),
        rerank_weight_sha256=weight_sha256(model),
        labels=('sha256-derived per-sample relevance bits (salt rerank-train/<sample_id>, '
                'first byte bit 0); the sealed package is unchanged'),
        loss=loss_definition(score),
        data='sealed rerank-package-v1 rows with derived relevance labels',
    )
    return model, batches, build_loss_fn(score, yes_id, no_id), protocol
