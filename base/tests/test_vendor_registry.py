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
from vendors.protocol import ConfigurationError
from vendors.registry import get_provider


class VendorTests(unittest.TestCase):
    def test_registry_does_not_guess_unknown_vendor(self):
        for name in ("kunlunxin", "unknown", "../ascend", None):
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


if __name__ == "__main__":
    unittest.main()
