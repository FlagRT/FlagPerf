"""Classic DLRM TP=2 with complete per-table and MLP diagnostic outputs."""
from torch import nn
import torch.distributed as dist

from benchmarks.dlrm.config import validate_tp_architecture
from benchmarks.recommendation.parallel import GatherEmbeddingBag, ReducedLinear


def shard(model, config, parallel):
    validate_tp_architecture(config, parallel.world_size)
    if not dist.is_initialized() or dist.get_world_size() != parallel.world_size:
        raise ValueError('DLRM TP requires the initialized matching model process group')
    rank, degree, group = dist.get_rank(), parallel.world_size, dist.group.WORLD
    plan = {}
    for i, original in enumerate(model.embeddings):
        model.embeddings[i] = GatherEmbeddingBag(original, rank, degree, group)
        plan[f'embeddings.{i}'] = 'embedding_dim_all_gather'
    for name in ('bottom_mlp', 'top_mlp'):
        block = getattr(model, name)
        for i, layer in enumerate(block):
            if isinstance(layer, nn.Linear):
                block[i] = ReducedLinear(layer, rank, degree, group)
                plan[f'{name}.{i}'] = 'input_dim_all_reduce'
    model._tp_plan, model._tp_rank, model._tp_degree = plan, rank, degree
    return model.eval().requires_grad_(False)
