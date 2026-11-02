"""CUDA worker setup, imported only inside an executing worker."""


def initialize(rank):
    import torch
    device = 'cuda:' + str(rank)
    torch.cuda.set_device(device)
    return device
