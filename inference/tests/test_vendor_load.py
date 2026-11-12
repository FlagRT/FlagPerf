# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Vendor-code loader tests: identity pinning, auto_map contract, package bypass."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flagprof.runner.vendor import code_files_identity, load_with_vendor_code  # noqa: E402


MODELING_SRC = '''
import json

class FakeConfig:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cfg = cls()
        cfg.path = path
        cfg.local_files_only = kwargs.get('local_files_only')
        cfg.use_flash_attn = True  # vendor default; loader must force False
        cfg.from_vendor_config = True
        return cfg

class FakeModel:
    def __init__(self, config=None, **kwargs):
        self.config = config
        self.kwargs = kwargs

    @classmethod
    def from_pretrained(cls, path, config=None, **kwargs):
        if config is None:
            raise AssertionError('loader must pass the AutoConfig-built config')
        return cls(config=config, **kwargs)

class PlainModel:
    """No from_pretrained classmethod: loader must fall back to ctor path."""
    def __init__(self, config=None):
        self.config = config
'''

CONFIG_JSON = {
    'architectures': ['FakeModel'],
    'auto_map': {
        'AutoConfig': 'configuration_fake.FakeConfig',
        'AutoModel': 'modeling_fake.FakeModel',
    },
}


def _vendor_dir(root, auto_map=None):
    root = Path(root)
    # write bytes: keep source newlines verbatim so digests are deterministic
    (root / 'configuration_fake.py').write_bytes(MODELING_SRC.encode())
    (root / 'modeling_fake.py').write_bytes(MODELING_SRC.encode())
    (root / 'weights.bin').write_bytes(b'w')
    cfg = dict(CONFIG_JSON)
    if auto_map is not None:
        cfg['auto_map'] = auto_map
    elif auto_map is False:
        cfg.pop('auto_map')
    import json
    (root / 'config.json').write_text(json.dumps(cfg))
    return root


class CodeIdentityTests(unittest.TestCase):
    def test_pins_only_py_files_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _vendor_dir(tmp)
            root_join = Path(tmp)
            (root_join / 'README.md').write_text('x')
            ident = code_files_identity(root_join)
            self.assertEqual(list(ident), ['configuration_fake.py', 'modeling_fake.py'])
            self.assertEqual(ident['modeling_fake.py'],
                             hashlib.sha256(MODELING_SRC.encode()).hexdigest())

    def test_empty_dir_gives_falsy_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(code_files_identity(tmp), {})


class VendorLoadTests(unittest.TestCase):
    def test_package_bypass_forces_use_flash_attn_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _vendor_dir(tmp)
            model = load_with_vendor_code(root, 'float32')
            self.assertEqual(model.config.use_flash_attn, False)
            self.assertEqual(model.config.local_files_only, True)
            self.assertEqual(model.kwargs.get('torch_dtype'), 'float32')
            pkg = type(model).__module__.split('.')[0]
            self.assertIn(pkg, sys.modules)
            # cleanup so later tests do not see the fake package
            del sys.modules[pkg]

    def test_rejects_remote_prefixed_auto_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _vendor_dir(tmp, auto_map={'AutoModel': 'some-repo--modeling_fake.FakeModel'})
            with self.assertRaises(ValueError):
                load_with_vendor_code(root, 'float32')

    def test_rejects_missing_auto_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _vendor_dir(tmp, auto_map=False)
            with self.assertRaises(ValueError):
                load_with_vendor_code(root, 'float32')

    def test_ctor_fallback_without_from_pretrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _vendor_dir(tmp, auto_map={'AutoModel': 'configuration_fake.PlainModel'})
            model = load_with_vendor_code(tmp, 'float32')
            self.assertEqual(type(model).__name__, 'PlainModel')
            pkg = type(model).__module__.split('.')[0]
            del sys.modules[pkg]


if __name__ == '__main__':
    unittest.main()
