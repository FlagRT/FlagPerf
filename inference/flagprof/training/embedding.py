# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Embedding fine-tune task over pretokenized triplet groups.

Rows carry token ids (anchor, positive, K negatives); no tokenizer runs here
and no text corpus is downloaded. The task loss is in-batch + hard-negative
InfoNCE over L2-normalised pooled embeddings, with the loss region pinned to
float32 (autocast disabled inside the loss, same policy as the CTR task).
Pooling is explicit and recorded in the protocol; importing this module does
not import torch.
"""
import json

POOLS = ('last', 'cls', 'mean')


def pool_description(pool):
    if pool == 'last':
        return 'last non-pad token hidden state (attention-mask index)'
    if pool == 'cls':
        return 'first token hidden state (position 0)'
    if pool == 'mean':
        return 'attention-mask weighted mean of token hidden states'
    raise ValueError('unknown pooling: ' + pool)


def loss_definition(pool, temperature):
    return ('info_nce(l2_normalise(pool(hidden)), temperature=' + repr(temperature) +
            '; columns = per-group positive then its negatives; target = own positive)')


def build_loss_fn(pool, temperature):
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext

    def pool_hidden(hidden, mask):
        if pool == 'cls':
            return hidden[:, 0]
        if pool == 'mean':
            weight = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weight).sum(1) / weight.sum(1).clamp(min=1)
        index = (mask.long().sum(1) - 1).clamp(min=0)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), index]

    def loss_fn(model, batch):
        anchor_mask = batch['anchor_mask']
        output = model(input_ids=batch['anchor_ids'], attention_mask=anchor_mask)
        hidden = getattr(output, 'last_hidden_state', None)
        if hidden is None:
            hidden = output[0]
        anchor = F.normalize(pool_hidden(hidden, anchor_mask), dim=-1)
        cand_mask = batch['cand_mask']
        output = model(input_ids=batch['cand_ids'], attention_mask=cand_mask)
        hidden = getattr(output, 'last_hidden_state', None)
        if hidden is None:
            hidden = output[0]
        candidates = F.normalize(pool_hidden(hidden, cand_mask), dim=-1)
        device_type = anchor.device.type
        if device_type in ('cpu', 'cuda', 'xpu', 'npu'):
            region = torch.autocast(device_type, enabled=False)
        else:
            region = nullcontext()
        with region:
            scores = anchor @ candidates.t() / float(temperature)
            stride = candidates.shape[0] // anchor.shape[0]
            target = torch.arange(anchor.shape[0], device=scores.device) * stride
            return F.cross_entropy(scores.float(), target)
    return loss_fn


def prepare_embedding(*, model, inputs, pooling, temperature, batch_size,
                      rank=0, world_size=1):
    """Build the embedding task from a pretokenized triplet-group JSONL.

    Returns (model, batches, loss_fn, protocol). Each JSONL row is one group:
    {"anchor": [ids], "pos": [ids], "negs": [[ids] x K]} with K identical
    across rows. Sequences are padded per microbatch to that microbatch's
    maximum; candidate rows are ordered per group (positive first, then its
    negatives); groups are split across ranks interleaved, mirroring the
    other tasks.
    """
    if pooling not in POOLS:
        raise ValueError('pooling must be one of ' + ', '.join(POOLS))
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    groups = []
    neg_k = None
    with open(inputs, encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            anchor, pos, negs = row['anchor'], row['pos'], row['negs']
            if not anchor or not pos or not negs:
                raise ValueError('each group needs non-empty anchor, pos and negs')
            if any(not n for n in negs):
                raise ValueError('empty negative sequence in group')
            if neg_k is None:
                neg_k = len(negs)
            elif len(negs) != neg_k:
                raise ValueError('every group must carry the same negative count')
            groups.append((anchor, pos, negs))
    if not groups:
        raise ValueError('no triplet groups in inputs')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch-size must be a positive integer')
    if len(groups) % batch_size:
        raise ValueError('group count must be a positive multiple of batch-size '
                         '(equal microbatches); got ' + str(len(groups)))
    if world_size > 1:
        groups = groups[rank::world_size]
        if not groups:
            raise ValueError('rank ' + str(rank) + ' has no groups for world_size ' + str(world_size))
    import torch
    batches = []
    for start in range(0, len(groups), batch_size):
        chunk = groups[start:start + batch_size]
        size = len(chunk)
        width = max(len(s) for anchor, pos, negs in chunk for s in (anchor, pos, *negs))
        anchor_ids = torch.zeros((size, width), dtype=torch.long)
        anchor_mask = torch.zeros((size, width), dtype=torch.long)
        cand_ids = torch.zeros((size * (1 + neg_k), width), dtype=torch.long)
        cand_mask = torch.zeros((size * (1 + neg_k), width), dtype=torch.long)
        for gi, (anchor, pos, negs) in enumerate(chunk):
            n = len(anchor)
            anchor_ids[gi, :n] = torch.tensor(anchor, dtype=torch.long)
            anchor_mask[gi, :n] = 1
            rows = [(pos, gi * (1 + neg_k))]
            rows.extend((neg, gi * (1 + neg_k) + 1 + ni) for ni, neg in enumerate(negs))
            for ids, target_row in rows:
                n = len(ids)
                cand_ids[target_row, :n] = torch.tensor(ids, dtype=torch.long)
                cand_mask[target_row, :n] = 1
        batches.append(dict(anchor_ids=anchor_ids, anchor_mask=anchor_mask,
                            cand_ids=cand_ids, cand_mask=cand_mask))
    protocol = dict(pooling=pooling,
                    pooling_detail=pool_description(pooling),
                    temperature=temperature,
                    groups_per_rank=len(groups),
                    negatives_per_group=neg_k,
                    loss=loss_definition(pooling, temperature),
                    data='pretokenized triplet groups (anchor, pos, K negatives); '
                         'padded per microbatch; sealed by the engine input digest')
    return model, batches, build_loss_fn(pooling, temperature), protocol
