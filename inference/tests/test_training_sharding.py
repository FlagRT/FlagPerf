# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""FSDP sharding option: CLI surface, wrapping discovery and error paths.

The multi-rank numerical verification runs in Linux containers via torchrun
(see docs/training/v2-validation.md); these tests cover what a single process
can honestly check.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flagprof.training.__main__ import block_layers, parser


class ShardingOptionTests(unittest.TestCase):
    def test_sharding_defaults_to_none_and_task_defaults_to_none(self):
        args = parser().parse_args(['--outputs', 'out'])
        self.assertEqual(args.sharding, 'none')
        self.assertIsNone(args.task)
        args = parser().parse_args(['--outputs', 'out', '--sharding', 'fsdp', '--ddp'])
        self.assertEqual(args.sharding, 'fsdp')


try:
    import torch
except ImportError:
    torch = None


if torch is not None:
    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

    class _Inner(torch.nn.Module):
        def __init__(self, attribute, count):
            super().__init__()
            blocks = torch.nn.ModuleList(_Block() for _ in range(count))
            setattr(self, attribute, blocks)

        def forward(self, x):
            return x

    class _Outer(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

        def forward(self, x):
            return self.model(x)


@unittest.skipIf(torch is None, 'model wrapping discovery requires torch')
class BlockLayerDiscoveryTests(unittest.TestCase):
    def test_qwen3_style_layers_are_found_in_order(self):
        model = _Outer(_Inner('layers', 3))
        found = block_layers(model)
        self.assertEqual(len(found), 3)
        for layer in found:
            self.assertIsInstance(layer, _Block)

    def test_roberta_style_encoder_layer_is_found(self):
        inner = torch.nn.Module()
        encoder = torch.nn.Module()
        encoder.add_module('layer', torch.nn.ModuleList(_Block() for _ in range(2)))
        inner.add_module('encoder', encoder)
        model = _Outer(inner)
        self.assertEqual(len(block_layers(model)), 2)

    def test_model_without_repeated_blocks_returns_empty(self):
        model = _Outer(torch.nn.Sequential(torch.nn.Linear(4, 4)))
        self.assertEqual(block_layers(model), [])


@unittest.skipIf(torch is None, 'training error paths require torch')
class ExecuteErrorPathTests(unittest.TestCase):
    def parse(self, extra):
        return parser().parse_args(extra)

    def execute(self, extra):
        from flagprof.training.__main__ import execute
        execute(self.parse(extra))

    def test_fsdp_requires_ddp(self):
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'run')
            with self.assertRaisesRegex(ValueError, 'requires --ddp'):
                self.execute(['--outputs', out, '--sharding', 'fsdp'])

    def test_fsdp_rejects_checkpoint_and_resume(self):
        import os
        from unittest import mock
        environment = {'WORLD_SIZE': '2', 'RANK': '0', 'LOCAL_RANK': '0'}
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'run')
            checkpoint = str(Path(directory) / 'ck.pt')
            with mock.patch.dict(os.environ, environment):
                with self.assertRaisesRegex(ValueError, 'checkpoint/resume'):
                    self.execute(['--outputs', out, '--sharding', 'fsdp', '--ddp',
                                  '--checkpoint', checkpoint])

    def test_unadapted_model_without_task_lists_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'run')
            with self.assertRaisesRegex(ValueError, '--task is required'):
                self.execute(['--model', '/models/mystery-net', '--outputs', out])


@unittest.skipIf(torch is None, 'training default path requires torch')
class DefaultTaskTests(unittest.TestCase):
    def test_omitting_task_without_model_runs_classification_demo(self):
        from flagprof.training.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'cls-run')
            code = main(['--outputs', out, '--steps', '1', '--device', 'cpu'])
            self.assertEqual(code, 0)
            import json
            result = json.loads((Path(out) / 'training-result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['protocol']['task'], 'classification')
            self.assertEqual(result['protocol']['sharding'], 'none')


if __name__ == '__main__':
    unittest.main()
