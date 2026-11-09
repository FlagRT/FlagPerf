# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Vendor-code package loader and embed target helpers, CPU only."""
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'targets'))
from target_embedder import code_files_identity, load_with_vendor_code


STUB_FILES = {
    # two-level relative-import nesting: top module -> mid module -> leaf modules
    'deep.py': 'LEAF = "leaf"\n',
    'configuration_stub.py': (
        'from transformers import PretrainedConfig\n'
        'class StubConfig(PretrainedConfig):\n'
        '    model_type = "stub"\n'),
    'modeling_mid.py': (
        'from .configuration_stub import StubConfig\n'
        'from .deep import LEAF\n'
        'class StubMid:\n'
        '    marker = LEAF\n'),
    'modeling_stub.py': (
        'import torch\n'
        'from .configuration_stub import StubConfig\n'
        'from .modeling_mid import StubMid\n'
        'class StubModel(torch.nn.Module):\n'
        '    config_class = StubConfig\n'
        '    mid_marker = StubMid.marker\n'
        '    def __init__(self, config=None):\n'
        '        super().__init__()\n'
        '        self.config = config\n'),
}


def build_stub(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, body in STUB_FILES.items():
        (root / name).write_text(body)
    (root / 'config.json').write_text(json.dumps(dict(
        architectures=['StubModel'],
        auto_map={'AutoConfig': 'configuration_stub.StubConfig',
                  'AutoModel': 'modeling_stub.StubModel'},
        model_type='stub')))
    return root


class VendorCodeTests(unittest.TestCase):
    def test_two_level_package_load_resolves_leaf_modules(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            build_stub(Path(tmp) / 'stub-model')
            model = load_with_vendor_code(str(Path(tmp) / 'stub-model'), torch.float32)
            self.assertEqual(model.mid_marker, 'leaf')
            self.assertTrue(model.__class__.__module__.startswith('flagprof_vendor_'))

    def test_auto_map_remote_reference_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = build_stub(Path(tmp) / 'stub-model')
            cfg = json.loads((root / 'config.json').read_text())
            cfg['auto_map']['AutoModel'] = 'some/repo--modeling_stub.StubModel'
            (root / 'config.json').write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                load_with_vendor_code(str(root), torch.float32)

    def test_code_files_identity_lists_python_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_stub(Path(tmp) / 'stub-model')
            files = code_files_identity(str(root))
            self.assertEqual(sorted(files), sorted(STUB_FILES))


if __name__ == '__main__':
    unittest.main()
