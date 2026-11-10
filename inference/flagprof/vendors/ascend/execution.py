# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""NPU worker setup, imported only inside an executing worker."""


def prepare_components():
    # Component registration uses torch.npu even before a runner selects a device.
    import torch_npu


def initialize(rank):
    import torch
    import torch_npu
    device = 'npu:' + str(rank)
    torch.npu.set_device(device)
    return device
