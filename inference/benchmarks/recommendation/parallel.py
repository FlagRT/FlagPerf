"""Weight shards use the caller's process group and restore full numeric boundaries."""
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from benchmarks.recommendation.modules import SumEmbeddingBag


def gather_last_dim(local, degree, group):
    local = local.contiguous()
    parts = [torch.empty_like(local) for _ in range(degree)]
    dist.all_gather(parts, local, group=group)
    return torch.cat(parts, dim=-1)


class GatherEmbedding(nn.Module):
    def __init__(self, original, rank, degree, group):
        super().__init__()
        if original.weight.shape[1] % degree:
            raise ValueError('embedding dimension must be divisible by TP degree')
        width = original.weight.shape[1] // degree
        self.weight = nn.Parameter(original.weight[:, rank*width:(rank+1)*width].clone(), requires_grad=False)
        self.degree, self.group = degree, group

    def forward(self, x):
        return gather_last_dim(F.embedding(x, self.weight), self.degree, self.group)


class GatherEmbeddingBag(SumEmbeddingBag):
    def __init__(self, original, rank, degree, group):
        width = original.weight.shape[1]
        if width % degree:
            raise ValueError('embedding dimension must be divisible by TP degree')
        super().__init__(original.weight.shape[0], width // degree)
        self.weight = nn.Parameter(original.weight[:, rank*(width//degree):(rank+1)*(width//degree)].clone(), requires_grad=False)
        self.degree, self.group = degree, group

    def forward(self, values, offsets):
        return gather_last_dim(super().forward(values, offsets), self.degree, self.group)


class ReducedLinear(nn.Module):
    """Uneven input slices; sum contributions before adding bias once."""
    def __init__(self, original, rank, degree, group):
        super().__init__()
        if original.in_features < degree:
            raise ValueError('TP Linear requires at least one input column per rank')
        quotient, remainder = divmod(original.in_features, degree)
        self.width = quotient + int(rank < remainder)
        self.start = rank*quotient + min(rank, remainder)
        self.weight = nn.Parameter(original.weight[:, self.start:self.start+self.width].clone(), requires_grad=False)
        self.bias = nn.Parameter(original.bias.clone(), requires_grad=False) if original.bias is not None else None
        self.in_features, self.out_features = original.in_features, original.out_features
        self.group = group

    def forward(self, x):
        contribution = F.linear(x[..., self.start:self.start+self.width].contiguous(), self.weight)
        dist.all_reduce(contribution, group=self.group)
        return contribution + self.bias if self.bias is not None else contribution
