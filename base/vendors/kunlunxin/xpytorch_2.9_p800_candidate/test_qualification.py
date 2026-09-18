"""Offline checks; no torch imports and no accelerator access."""
from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from hardware_probe import validate_binding, run_isolated
from verify_runtime import ROOT, validate_manifest, validate_packages
from launch_qualification import selected_device, normalize_bdf


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT/'image-manifest.json').read_text())
        self.inspect = json.loads((ROOT/'evidence/image-inspect.json').read_text())
        self.binding = dict(schema_version=1, host_physical_id=3, container_node='/dev/xpu3',
                            logical_device=0, pci_bdf='0000:83:00.0', reservation_reference='offline-fixture',
                            reservation_start='2026-09-18T00:00:00Z', reservation_end='2026-09-18T02:00:00Z')
        self.now = datetime(2026, 9, 18, 1, tzinfo=timezone.utc)

    def binding_check(self, binding=None, environ=None, nodes=None, now=None):
        return validate_binding(binding or self.binding, environ or {'CUDA_VISIBLE_DEVICES': '3'},
                                nodes or ['/dev/xpu3', '/dev/xpuctrl'], now or self.now)

    def test_exact_image(self):
        validate_manifest(self.manifest, self.inspect)

    def test_reject_image_substitution(self):
        observed = copy.deepcopy(self.inspect)
        observed['Id'] = 'sha256:'+'0'*64
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            validate_manifest(self.manifest, observed)

    def test_no_self_promotion(self):
        self.manifest['validated'] = True
        with self.assertRaises(ValueError):
            validate_manifest(self.manifest, self.inspect)

    def test_static_evidence_matches_lock(self):
        audit = json.loads((ROOT/'evidence/static-audit.json').read_text())['audit']
        lock = json.loads((ROOT/'stack.lock.yaml').read_text())
        self.assertEqual(validate_packages(audit['packages'], lock['packages']), [])
        self.assertEqual(audit['python'], lock['python'])

    def test_reject_package_drift(self):
        self.assertTrue(validate_packages({'torch': {'version': '2.8'}}, {'torch': '2.9'}))

    def test_single_card_binding(self):
        self.binding_check()

    def test_reject_extra_card(self):
        with self.assertRaisesRegex(ValueError, 'device nodes'):
            self.binding_check(nodes=['/dev/xpu3', '/dev/xpu4', '/dev/xpuctrl'])

    def test_reject_wrong_visibility(self):
        with self.assertRaisesRegex(ValueError, 'visibility'):
            self.binding_check(environ={'CUDA_VISIBLE_DEVICES': '0'})

    def test_reject_expired_reservation(self):
        with self.assertRaisesRegex(ValueError, 'reservation'):
            self.binding_check(now=datetime(2026, 9, 18, 3, tzinfo=timezone.utc))

    def test_reject_kl3(self):
        with self.assertRaisesRegex(ValueError, 'KL3'):
            self.binding_check(environ={'CUDA_VISIBLE_DEVICES': '3', 'XPU_EVENT_KL3_ENABLE': '1'})

    def test_worker_exit_and_logs(self):
        with tempfile.TemporaryDirectory() as path:
            result = run_isolated([sys.executable, '-S', '-c', 'import sys; print("failed"); sys.exit(7)'],
                                  Path(path), 'fixture', 5)
            self.assertEqual(result['returncode'], 7)
            self.assertFalse(result['timed_out'])
            self.assertIn('failed', (Path(path)/'fixture.stdout.log').read_text())

    def test_xpu_inventory_quoted_product_and_pci_domain(self):
        line = '00000000:16:00.0 1 1 SERIAL 37 0 0 0 93 1450 1450 1450 1450 1450 1450 0 96 0 98304 0 FW "P800 OAM" 0'
        card = selected_device(line, 1)
        self.assertEqual(card['pci_bdf'], '0000:16:00.0')
        self.assertEqual(card['used_memory_mib'], 0)
        self.assertEqual(card['total_memory_mib'], 98304)

    def test_xpu_inventory_missing_or_duplicate_card(self):
        line = '00000000:16:00.0 1 1 SERIAL 37 0 0 0 93 1450 1450 1450 1450 1450 1450 0 96 0 98304 0 FW "P800 OAM" 0'
        for raw, card in [(line, 2), (line+'\n'+line, 1)]:
            with self.assertRaises(ValueError):
                selected_device(raw, card)

    def test_xpu_inventory_rejects_unknown_format(self):
        with self.assertRaises(ValueError):
            selected_device('unexpected output', 1)

    def test_pci_domain_does_not_truncate(self):
        with self.assertRaises(ValueError):
            normalize_bdf('12345678:16:00.0')

    @unittest.skipUnless(os.name == 'posix', 'process-group timeout requires Linux')
    def test_hung_worker_is_killed(self):
        with tempfile.TemporaryDirectory() as path:
            result = run_isolated([sys.executable, '-S', '-c', 'import time; time.sleep(30)'],
                                  Path(path), 'fixture', 0.1)
            self.assertTrue(result['timed_out'])
            self.assertNotEqual(result['returncode'], 0)


if __name__ == '__main__':
    unittest.main()
