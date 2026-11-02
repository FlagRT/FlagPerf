# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CUDA worker setup, imported only inside an executing worker."""


def initialize(rank):
    import torch
    device = 'cuda:' + str(rank)
    torch.cuda.set_device(device)
    return device
