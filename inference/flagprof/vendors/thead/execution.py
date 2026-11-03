# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""PPU worker device binding; no tracing gate variable exists on the locked stack."""
def initialize(rank):
    import torch
    device='cuda:'+str(rank)
    torch.cuda.set_device(device)
    return device
