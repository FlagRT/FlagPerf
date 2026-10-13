# Copyright (c) Meta Platforms, Inc. and affiliates.
# DLRM mathematics adapted from facebookresearch/dlrm; see LICENSE and origin.json.
"""Inference-only classic DLRM; all field counts and network widths are configurable."""
import torch
from torch import nn

from benchmarks.dlrm.config import interaction_width
from benchmarks.recommendation.modules import HiddenSequential, SumEmbeddingBag


def mlp(input_dim, widths, head=False):
    layers = []
    for width in widths:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    if head:
        layers.append(nn.Linear(input_dim, 1))
    return HiddenSequential(*layers) if head else nn.Sequential(*layers)


class PooledEmbeddings(nn.ModuleList):
    def __init__(self, field_dims, embed_dim):
        super().__init__([SumEmbeddingBag(n, embed_dim) for n in field_dims])

    def forward(self, sparse):
        return torch.stack([bag(sparse[f'values_{i}'], sparse[f'offsets_{i}'])
                            for i, bag in enumerate(self)], dim=1)


class FeatureInteraction(nn.Module):
    def __init__(self, fields, mode):
        super().__init__()
        self.mode = mode
        indices = [i*(fields+1)+j for i in range(fields+1) for j in range(i)]
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.int64), persistent=False)

    def forward(self, dense, sparse):
        features = torch.cat((dense.unsqueeze(1), sparse), dim=1)
        if self.mode == 'cat':
            return features.flatten(1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return torch.cat((dense, gram.flatten(1).index_select(1, self.indices)), dim=1)


class DLRM(nn.Module):
    def __init__(self, dense_dim, field_dims, embed_dim, bottom_mlp_dims, top_mlp_dims, interaction):
        super().__init__()
        self.bottom_mlp = mlp(dense_dim, bottom_mlp_dims)
        self.embeddings = PooledEmbeddings(field_dims, embed_dim)
        self.interaction = FeatureInteraction(len(field_dims), interaction)
        config = {'field_dims': field_dims, 'embed_dim': embed_dim, 'interaction': interaction}
        self.top_mlp = mlp(interaction_width(config), top_mlp_dims, head=True)

    def forward(self, dense, **sparse):
        bottom = self.bottom_mlp(dense)
        embeddings = self.embeddings(sparse)
        interacted = self.interaction(bottom, embeddings)
        score, hidden = self.top_mlp(interacted, return_hidden=True)
        return {'hidden': hidden, 'probability': torch.sigmoid(score.squeeze(1))}
