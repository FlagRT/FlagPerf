# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""FlagScale 检索训练的外部观察适配器（运行时层外挂，不改训练语义）。

在既有 torch.distributed / FSDP2 / Megatron 风格训练循环外使用：

    from flagscale_adapter import ObservationSession
    obs = ObservationSession()          # 读 FLAGPROF_TRAIN_OBSERVE；未设置则完全 no-op
    with obs:                           # attach 官方 profiler + 通信探针 + 进程内 finalize
        for step in range(steps):
            with obs.phase('data', step=step):      batch = next(loader)
            with obs.phase('forward', step=step):   loss = loss_fn(model(**batch))
            with obs.phase('backward', step=step):  loss.backward()
            with obs.phase('optimizer', step=step): opt.step(); opt.zero_grad()
            with obs.phase('step', step=step):      pass

要求：训练脚本使用标准 torch.distributed（默认进程组）；多卡时适配器会在 gloo
control group 上安装集合通信探针，fsdp2 funcol 与 p2p 入口均被覆盖。
输出：<FLAGPROF_TRAIN_OBSERVE>/rank-N/ 下的 writer store 与官方 trace，
供 flagprof 报表（通信分列 / 按 rank 归集 / 等待-带宽 / 相段占比）消费。
"""
import os
from contextlib import nullcontext


class ObservationSession:
    def __init__(self):
        self.enabled = bool(os.environ.get('FLAGPROF_TRAIN_OBSERVE'))
        self.handle = None
        self.run = None

    def _detect_vendor(self):
        """npu → ascend；PPU → thead；其余 NVIDIA → nvidia；显式变量优先。"""
        override = os.environ.get('FLAGPROF_TRAIN_OBSERVE_VENDOR')
        if override:
            return override
        import torch
        try:
            if torch.npu.is_available():
                return 'ascend'
        except (AttributeError, RuntimeError):
            pass
        try:
            name = torch.cuda.get_device_name().upper()
            if ('PPU' in name or 'P800' in name or 'XPU_VISIBLE_DEVICES' in os.environ):
                return 'thead'
            return 'nvidia'
        except (RuntimeError, AssertionError):
            return 'thead'

    def __enter__(self):
        if not self.enabled:
            return self
        from types import SimpleNamespace
        from flagprof.training.__main__ import observation_start
        vendor = self._detect_vendor()
        device = 'npu' if vendor == 'ascend' else 'cuda'
        self.observation = observation_start(SimpleNamespace(task='external', device=device),
                                             default_vendor=vendor, diagnostic='flagscale-integration/1')
        self.handle = self.observation['handle']
        self.run = self.observation['run']
        return self

    def phase(self, name, *, step, microstep=None):
        if not self.enabled:
            return nullcontext()
        from flagprof.training import phase as training_phase
        return training_phase(name, step=step, microstep=microstep)

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        from flagprof.training.__main__ import observation_close
        observation_close(self.observation, failed=exc_type is not None, report=False)
        return False
