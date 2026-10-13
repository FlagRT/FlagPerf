"""Inference-only DeepFM TP; collectives use the caller's model process group."""
import torch
from torch import nn
import torch.distributed as dist
from benchmarks.deepfm.config import validate_tp_architecture
from benchmarks.recommendation.parallel import GatherEmbedding, ReducedLinear


def shard(model, config, parallel):
    validate_tp_architecture(config, parallel.world_size)
    if not dist.is_initialized() or dist.get_world_size() != parallel.world_size:
        raise ValueError('DeepFM TP requires the initialized matching model process group')
    rank, degree, group = dist.get_rank(), parallel.world_size, dist.group.WORLD
    model.embedding.embedding = GatherEmbedding(model.embedding.embedding, rank, degree, group)
    plan = {'embedding.embedding': 'embedding_dim_all_gather'}
    for index, layer in enumerate(model.mlp.mlp):
        if isinstance(layer, nn.Linear):
            model.mlp.mlp[index] = ReducedLinear(layer, rank, degree, group)
            plan[f'mlp.mlp.{index}'] = 'input_dim_all_reduce'
    model._tp_plan = plan
    model._tp_rank = rank
    model._tp_degree = degree
    return model.eval().requires_grad_(False)
