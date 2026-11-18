# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""适配器验证 demo：模拟"外部训练循环"（非 flagprof 引擎）挂适配器。

用法（torchrun 2 进程）：
    FLAGPROF_TRAIN_OBSERVE=/store torchrun --standalone --nproc-per-node=2 \
        python flagscale_demo.py --device cuda --steps 2

脚本用一个两卡 DDP 的小模型跑标准训练步（梯度 allreduce 产生集合通信事件），
全程通过 ObservationSession 标记 data/forward/backward/optimizer/step 相段；
结束后打印 JSON 标记供断言，store 供 flagprof 报表消费。
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flagscale_adapter import ObservationSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')

    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1)).to('cuda')
    if world > 1:
        # FSDP2 分片（与检索组训练同形态）：funcol 通路对探针可见，
        # 每步产生 all-gather / reduce-scatter 事件；传统 DDP 的 C++ Reducer
        # 不可见（框架已知边界），故此处用 fully_shard。
        from torch.distributed.fsdp import fully_shard
        fully_shard(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.functional.mse_loss
    generator = torch.Generator(device='cuda').manual_seed(531)

    losses = []
    with ObservationSession() as obs:
        for step in range(args.steps):
            with obs.phase('data', step=step):
                x = torch.randn(8, 32, device='cuda', generator=generator)
                y = torch.randn(8, 1, device='cuda', generator=generator)
            with obs.phase('forward', step=step):
                out = model(x)
                loss = loss_fn(out, y)
            with obs.phase('backward', step=step):
                loss.backward()
            with obs.phase('optimizer', step=step):
                opt.step()
                opt.zero_grad(set_to_none=True)
            with obs.phase('step', step=step):
                pass
            losses.append(round(float(loss.detach()), 6))

    if rank == 0:
        print(json.dumps(dict(demo='flagscale-adapter', world=world, steps=args.steps,
                              losses=losses, observe=bool(os.environ.get('FLAGPROF_TRAIN_OBSERVE')))),
              flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
