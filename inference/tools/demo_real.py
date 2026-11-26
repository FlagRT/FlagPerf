# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Best-practice demo: a FlagScale-style training loop (not the FlagPerf engine)
over a real model, wrapped by the observation adapter.

- model: Qwen3-Reranker-0.6B loaded with transformers, FSDP2 per-block
- data:  the sealed qwen3-reranker-causal package (pre-tokenized rows)
- loop:  20 steps of a standard LM loss; the loop is hand-written, exactly the
         shape a model-layer training script would have
- observation: enabled by FLAGPROF_TRAIN_OBSERVE (adapter handles everything)
- output: final model file + its sha256 (for the zero-drift comparison)

Run under torchrun --nproc-per-node=2.
"""
import argparse
import hashlib
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flagscale_adapter import ObservationSession


def load_batches(package, batch_size):
    from flagprof.runner.rerank_common import load_package
    batches, manifest = load_package(package, batch_size=batch_size)
    return batches, manifest


def block_layers(model):
    inner = getattr(model, 'model', None)
    layers = getattr(inner, 'layers', None) if inner is not None else None
    return list(layers) if layers else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--package', required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--save', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                 trust_remote_code=False,
                                                 torch_dtype=torch.float32).to('cuda')
    if world > 1:
        from torch.distributed.fsdp import fully_shard
        blocks = block_layers(model)
        for block in blocks:
            fully_shard(block)
        fully_shard(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    batches, manifest = load_batches(args.package, args.batch_size)
    per_rank = batches[rank::world] if world > 1 else batches
    assert per_rank, 'rank has no microbatches'
    losses = []

    obs = ObservationSession()
    with obs:
        for step in range(args.steps):
            with obs.phase('data', step=step):
                _sample_ids, fields = per_rank[step % len(per_rank)]
                fields = dict(fields)
                ids = fields['input_ids'].to('cuda')
                mask = fields['attention_mask'].to('cuda')
            with obs.phase('forward', step=step):
                logits = model(input_ids=ids, attention_mask=mask).logits
                shift_logits = logits[:, :-1].reshape(-1, logits.size(-1)).float()
                shift_labels = ids[:, 1:].reshape(-1).clone()
                shift_labels[mask[:, 1:].reshape(-1) == 0] = -100
                loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
            with obs.phase('backward', step=step):
                loss.backward()
            with obs.phase('optimizer', step=step):
                opt.step()
                opt.zero_grad(set_to_none=True)
            with obs.phase('step', step=step):
                pass
            losses.append(round(float(loss.detach()), 6))

    # save final weights (rank0 full state; for FSDP use full state dict gather)
    if rank == 0:
        if world > 1:
            from torch.distributed.checkpoint.state_dict import get_model_state_dict
            state = get_model_state_dict(model)
        else:
            state = model.state_dict()
        tmp = args.save + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, args.save)
        sha = hashlib.sha256(open(args.save, 'rb').read()).hexdigest()
        import json
        print(json.dumps(dict(demo='flagscale-best-practice', world=world,
                              steps=args.steps, losses=losses,
                              final_model_sha256=sha,
                              observe=bool(os.environ.get('FLAGPROF_TRAIN_OBSERVE')))),
              flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
