"""CPU-only provider, runtime and lock compatibility contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from executors.common import BaseRunContext, DeviceLease, DeviceLeaseError, load_host_config, runtime_artifact_paths
from base.vendors.protocol import ConfigurationError
from base.vendors.registry import get_provider


class VendorTests(unittest.TestCase):
    def test_shared_helpers_do_not_import_another_domains_vendor_package(self):
        import subprocess
        script = """
import sys, types
sys.path.insert(0, BASE_PATH)
foreign = types.ModuleType('vendors')
sys.modules['vendors'] = foreign
from executors.common import runtime_lock_record
assert runtime_lock_record()['vendor'] == 'ascend'
assert sys.modules['vendors'] is foreign
assert 'torch' not in sys.modules
""".replace('BASE_PATH', repr(str(BASE)))
        proc = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_registry_does_not_guess_unknown_vendor(self):
        for name in ("unknown", "../ascend", None):
            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                get_provider(name)

    def test_runtime_path_is_contained_even_for_symlinks(self):
        for profile in ("..", "../torch_fl_2.10", "/tmp/profile", "a/b"):
            with self.subTest(profile=profile), self.assertRaises(ConfigurationError):
                runtime_artifact_paths(profile)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vendors/ascend").mkdir(parents=True)
            (root / "outside").mkdir()
            (root / "vendors/ascend/escape").symlink_to(root / "outside", target_is_directory=True)
            with self.assertRaises(ConfigurationError):
                get_provider("ascend").runtime_root(root, "escape")

    def test_unknown_environment_rejected(self):
        config = json.loads((BASE / "configs/ascend910_cann9_local.yaml").read_text())
        config["runtime_environment"] = {"UNREVIEWED_OPTION": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "host.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ConfigurationError, "unsupported keys"):
                load_host_config(path)

    def test_physical_selection_preserves_request_order_and_rejects_aliases(self):
        context = BaseRunContext(physical_device_ids="6,2")
        context.validate(require_selection=True)
        self.assertEqual(context.selection_request()["requested_ids"], [6, 2])
        for other in ({"npu_ids": "2"}, {"device_ids": "2"}):
            with self.assertRaises(ConfigurationError):
                BaseRunContext(physical_device_ids="2", **other).validate(require_selection=True)

    def test_bindings_separate_physical_logical_and_rank(self):
        bindings = get_provider("ascend").bindings({"actual_device_map": [
            {"npu_id": 7, "chip_id": 1, "logic_id": 15},
            {"npu_id": 7, "chip_id": 0, "logic_id": 14},
        ]}, [15, 14])
        self.assertEqual([b.host_physical_id for b in bindings], [7, 7])
        self.assertEqual([b.framework_local_rank for b in bindings], [0, 1])
        self.assertEqual([b.legacy_logical_id for b in bindings], [15, 14])
        self.assertNotEqual(bindings[0].resource_key, bindings[1].resource_key)

    def test_physical_preflight_expands_in_request_order_without_rewriting_raw(self):
        provider = get_provider("ascend")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.json"
            raw.write_text(json.dumps({"actual_device_map": [
                {"npu_id": 2, "chip_id": 0, "logic_id": 4},
                {"npu_id": 2, "chip_id": 1, "logic_id": 5},
                {"npu_id": 6, "chip_id": 0, "logic_id": 12},
                {"npu_id": 6, "chip_id": 1, "logic_id": 13}],
                "selection": {"source": "npu-ids", "requested_ids": [2, 6],
                              "selected_device_ids": [4, 5, 12, 13]}}))
            original = raw.read_bytes()
            with patch("base.vendors.ascend.provider.run_host_preflight", return_value=raw):
                path = provider.preflight(root, {"expected_device_ids": [4, 5, 12, 13]},
                                          BaseRunContext(physical_device_ids="6,2"))
                record = json.loads(path.read_text())
                selected = record["selection"]["selected_device_ids"]
                self.assertEqual(selected, [12, 13, 4, 5])
                bindings = provider.bindings(record, selected)
                self.assertEqual([b.request_index for b in bindings], [0, 0, 1, 1])
                self.assertEqual([b.framework_local_rank for b in bindings], [0, 1, 2, 3])
                args = provider.docker_args({"required_devices": [], "host_mounts": []}, bindings)
                self.assertIn("ASCEND_RT_VISIBLE_DEVICES=12,13,4,5", args)
                self.assertEqual(raw.read_bytes(), original)
                self.assertEqual(provider.preflight(root, {"expected_device_ids": [4, 5, 12, 13]},
                                                   BaseRunContext(npu_ids="6,2")), raw)

    def test_new_lease_interlocks_with_legacy_and_releases_partial_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = DeviceLease([14], run_id="toolkit", kind="toolkit", root=root)
            legacy.acquire()
            blocked = DeviceLease([13, 14], resource_keys=["ascend/npu-7/chip-0"], run_id="new", kind="benchmark", root=root)
            try:
                with self.assertRaises(DeviceLeaseError):
                    blocked.acquire()
                with DeviceLease([13], run_id="released", kind="benchmark", root=root):
                    pass
            finally:
                legacy.release()
            with blocked:
                with self.assertRaises(DeviceLeaseError):
                    legacy.acquire()
            with legacy:
                pass

    def test_resource_identity_conflicts_across_aliases_but_not_vendors(self):
        with tempfile.TemporaryDirectory() as tmp:
            def lease(key):
                return DeviceLease([], resource_keys=[key], run_id="fixture", kind="benchmark", root=Path(tmp))
            with lease("fixture/uuid-a"):
                with self.assertRaises(DeviceLeaseError):
                    lease("fixture/uuid-a").acquire()
                with lease("other/uuid-a"):
                    pass

    def test_unknown_driver_vendor_never_calls_cuda(self):
        import subprocess
        script = """
import sys
sys.path.insert(0, BENCHMARKS)
from unittest.mock import patch
from drivers import utils
with patch.object(utils.torch.cuda,'synchronize',side_effect=AssertionError('CUDA fallback')) as sync:
    for operation in (utils.bootstrap_vendor, utils.host_device_sync, utils.multi_device_sync,
                      utils.set_ieee_float32, utils.unset_ieee_float32):
        try: operation('unknown')
        except ValueError: pass
        else: raise AssertionError('unknown vendor was accepted')
    sync.assert_not_called()
""".replace('BENCHMARKS',repr(str(BASE/'benchmarks')))
        proc=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)


if __name__ == "__main__":
    unittest.main()
