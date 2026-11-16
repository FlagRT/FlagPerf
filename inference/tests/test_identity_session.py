import importlib.util
import unittest
from pathlib import Path


def _load(name, relpath):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen_identity = _load('gen_identity_under_test', 'tools/gen_identity.py')


class SourcedIdentityTests(unittest.TestCase):
    """Pairing-table validation for sourced host identities (P800/Ascend-ready)."""

    def test_admitted_pairs(self):
        from flagprof.vendors.thead.session import validate_sourced_identity, SOURCED_IDENTITY
        self.assertIn('ppu-smi', SOURCED_IDENTITY)
        self.assertIn('xpu-smi', SOURCED_IDENTITY)
        self.assertIn('npu-smi', SOURCED_IDENTITY)
        device = dict(model='P800 OAM', pci_bdf='00000000:1c:00.0', driver='5.0.21.47',
                      host_device_node='/dev/xpu3')
        validate_sourced_identity(dict(source='xpu-smi'), device, 3)

    def test_model_source_mismatch_rejected(self):
        from flagprof.vendors.thead.session import validate_sourced_identity
        with self.assertRaises(ValueError):
            validate_sourced_identity(dict(source='xpu-smi'),
                                      dict(model='PPU-ZW810E', pci_bdf='x', driver='d',
                                           host_device_node='/dev/x'), 0)

    def test_unknown_source_rejected(self):
        from flagprof.vendors.thead.session import validate_sourced_identity
        with self.assertRaises(ValueError):
            validate_sourced_identity(dict(source='made-up-smi'),
                                      dict(model='Anything', pci_bdf='x', driver='d',
                                           host_device_node='/dev/x'), 0)

    def test_missing_sourced_field_rejected(self):
        from flagprof.vendors.thead.session import validate_sourced_identity
        with self.assertRaises(ValueError):
            validate_sourced_identity(dict(source='xpu-smi'),
                                      dict(model='P800 OAM', driver='d',
                                           host_device_node='/dev/x'), 0)


class GenIdentityMergeTests(unittest.TestCase):
    """Generator device-list merge (torch properties joined to smi metadata)."""

    def test_build_devices_merges_fields(self):
        class Props:
            name = 'P800 OAM'
            uuid = 'GPU-ABCD-1234'

        devices = gen_identity.build_devices([Props()], '5.0.21.47', '/dev/xpu',
                                             ['00000000:1c:00.0'])
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]['uuid'], 'abcd-1234')
        self.assertEqual(devices[0]['model'], 'P800 OAM')
        self.assertEqual(devices[0]['pci_bdf'], '00000000:1c:00.0')
        self.assertEqual(devices[0]['host_device_node'], '/dev/xpu0')


if __name__ == '__main__':
    unittest.main()
