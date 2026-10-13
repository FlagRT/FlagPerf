"""Head-input capture and exact empty-bag semantics, independent of hardware."""
import torch
from torch import nn


class HiddenSequential(nn.Sequential):
    def forward(self, x, return_hidden=False):
        if not return_hidden:
            return super().forward(x)
        hidden = None
        for index, layer in enumerate(self):
            x = layer(x)
            if index == len(self) - 2:
                hidden = x
        return x, hidden


class SumEmbeddingBag(nn.EmbeddingBag):
    """No padding or ID remapping; offsets include the terminal values length."""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim, mode='sum', include_last_offset=True)

    def forward(self, values, offsets):
        if values.numel() == 0:
            return self.weight.new_zeros((offsets.shape[0] - 1, self.embedding_dim))
        result = super().forward(values, offsets)
        return torch.where((offsets[1:] > offsets[:-1]).unsqueeze(1), result, torch.zeros_like(result))
