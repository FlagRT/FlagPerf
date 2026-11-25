# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Parallel-strategy demos for the best-practice document (2 ranks).

--mode ddp : classic DistributedDataParallel loop (gradient allreduce happens
             in the C++ reducer, which is deliberately invisible to the
             python-level probe - this demo documents that boundary).
--mode p2p : Megatron-style pipeline stage split: rank0 runs a stage, ships
             activations to rank1 via batch_isend_irecv, rank1 computes the
             loss and ships the input gradient back. Exercises the p2p family
             of the communication probe.

Both wrap the loop with the same ObservationSession used in the docs.
"""
import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flagscale_adapter import ObservationSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ddp', 'p2p'], required=True)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    generator = torch.Generator(device='cuda').manual_seed(531)
    losses = []

    def new_linear(a, b):
        lin = torch.nn.Linear(a, b).to('cuda')
        torch.nn.init.normal_(lin.weight, generator=generator)
        torch.nn.init.normal_(lin.bias, generator=generator)
        return lin

    if args.mode == 'ddp':
        model = torch.nn.Sequential(new_linear(32, 64), torch.nn.ReLU(),
                                    new_linear(64, 1))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with ObservationSession() as obs:
            for step in range(args.steps):
                with obs.phase('data', step=step):
                    x = torch.randn(8, 32, device='cuda', generator=generator)
                    y = torch.randn(8, 1, device='cuda', generator=generator)
                with obs.phase('forward', step=step):
                    out = model(x)
                    loss = torch.nn.functional.mse_loss(out, y)
                with obs.phase('backward', step=step):
                    loss.backward()
                with obs.phase('optimizer', step=step):
                    opt.step(); opt.zero_grad(set_to_none=True)
                with obs.phase('step', step=step):
                    pass
                losses.append(round(float(loss.detach()), 6))
    else:  # p2p pipeline: stage0 = first linear, stage1 = second linear + loss
        stage = new_linear(32, 64) if rank == 0 else new_linear(64, 1)
        opt = torch.optim.AdamW(stage.parameters(), lr=1e-3)
        act_size = 64  # activator width; rank1 receives the same width
        with ObservationSession() as obs:
            for step in range(args.steps):
                with obs.phase('data', step=step):
                    if rank == 0:
                        x = torch.randn(8, 32, device='cuda', generator=generator)
                with obs.phase('forward', step=step):
                    act = torch.zeros(8, act_size, device='cuda')
                    if rank == 0:
                        a = stage(x)
                        ops = [dist.P2POp(dist.isend, a, 1)]
                    else:
                        ops = [dist.P2POp(dist.irecv, act, 0)]
                    for work in dist.batch_isend_irecv(ops):
                        work.wait()
                    if rank == 1:
                        with obs.phase('data', step=step):
                            y = torch.randn(8, 1, device='cuda', generator=generator)
                        loss = torch.nn.functional.mse_loss(stage(act), y)
                with obs.phase('backward', step=step):
                    if rank == 1:
                        loss.backward()
                        grad = act.grad if act.grad is not None else torch.ones_like(act)
                        ops = [dist.P2POp(dist.isend, grad.contiguous(), 0)]
                        for work in dist.batch_isend_irecv(ops):
                            work.wait()
                    else:
                        gradbuf = torch.zeros(8, 64, device='cuda')
                        ops = [dist.P2POp(dist.irecv, gradbuf, 1)]
                        for work in dist.batch_isend_irecv(ops):
                            work.wait()
                        a.backward(gradbuf)
                with obs.phase('optimizer', step=step):
                    opt.step(); opt.zero_grad(set_to_none=True)
                with obs.phase('step', step=step):
                    pass
                if rank == 1:
                    losses.append(round(float(loss.detach()), 6))

    if rank == 0:
        print(json.dumps(dict(demo='parallel-strategy', mode=args.mode, world=world,
                              steps=args.steps, rank0_losses=losses,
                              observe=bool(os.environ.get('FLAGPROF_TRAIN_OBSERVE')))),
              flush=True)
    if rank == 1 and args.mode == 'p2p':
        print(json.dumps(dict(demo='parallel-strategy', mode=args.mode, world=world,
                              steps=args.steps, rank1_losses=losses,
                              observe=bool(os.environ.get('FLAGPROF_TRAIN_OBSERVE')))),
              flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
