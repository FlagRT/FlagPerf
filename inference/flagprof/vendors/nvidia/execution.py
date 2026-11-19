# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CUDA worker setup, imported only inside an executing worker."""


TORCH_DEVICE = 'cuda'
SMI = 'nvidia-smi'
BACKENDS = ('nccl',)


def initialize(rank):
    import torch
    device = 'cuda:' + str(rank)
    torch.cuda.set_device(device)
    return device


def allocator_memory(device):
    import torch
    return {'allocated_bytes': torch.cuda.memory_allocated(device),
            'reserved_bytes': torch.cuda.memory_reserved(device)}


def physical_identity(device):
    import torch
    props = torch.cuda.get_device_properties(device)
    uuid = str(props.uuid)
    if not uuid.startswith('GPU-'):
        uuid = 'GPU-'+uuid
    return dict(entity='chip:'+uuid, uuid=uuid, power_domain='chip:'+uuid)
