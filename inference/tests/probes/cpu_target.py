# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Small production-entry target, no pretrained model or additional package."""
import argparse
import torch
from flagprof import scope
from flagprof.runner import run_prepared


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, value):
        return self.norm(self.linear(value).relu()+value)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.layers = torch.nn.ModuleList([Layer() for _ in range(6)])
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, ids):
        value = self.embedding(ids)
        for layer in self.layers:
            value = layer(value)
        return self.norm(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(531)
    model = Model().eval()
    with scope('request'):
        run_prepared(model, torch.arange(8).reshape(2, 4), forwards=3, outputs=args.outputs)


if __name__ == '__main__':
    main()
