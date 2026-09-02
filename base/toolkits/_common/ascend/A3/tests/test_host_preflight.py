from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "host_preflight.py"
SPEC = importlib.util.spec_from_file_location("ascend_host_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

class HostPreflightTests(unittest.TestCase):
    def test_parse_id_spec_accepts_ranges_and_rejects_ambiguity(self) -> None:
        self.assertEqual(MODULE.parse_id_spec("2,4-6,9"), [2, 4, 5, 6, 9])
        for value in ("", "1,,2", "3-1", "-1", "1,1", "1-3,3"):
            with self.assertRaises(MODULE.PreflightError):
                MODULE.parse_id_spec(value)

    def test_parse_expected_device_ids(self) -> None:
        self.assertEqual(MODULE.parse_expected_device_ids("2,0,1"), [0, 1, 2])
        for value in ("", "0,0", "0,x", "-1,0"):
            with self.assertRaises(MODULE.PreflightError):
                MODULE.parse_expected_device_ids(value)

    def test_parse_map_logic_ids_ignores_mcu_rows(self) -> None:
        text = """NPU ID Chip ID Chip Logic ID Chip Phy-ID Chip Name
0 0 0 0 Ascend910
0 1 1 1 Ascend910
0 2 - - Mcu
1 0 2 2 Ascend910
"""
        self.assertEqual(MODULE.parse_map_logic_ids(text), [0, 1, 2])

    def test_resolve_physical_and_logical_selection(self) -> None:
        mapping = MODULE.parse_device_map(
            "0 0 0 0 Ascend910\n0 1 1 1 Ascend910\n"
            "1 0 2 2 Ascend910\n1 1 3 3 Ascend910\n"
        )
        physical = MODULE.resolve_selection(
            [0, 1, 2, 3], mapping, requested_npu_ids=[1]
        )
        self.assertEqual(physical["selected_device_ids"], [2, 3])
        self.assertEqual(physical["excluded_device_ids"], [0, 1])
        logical = MODULE.resolve_selection(
            [0, 1, 2, 3], mapping, requested_device_ids=[1, 3]
        )
        self.assertEqual(logical["selected_npu_ids"], [0, 1])
        with self.assertRaises(MODULE.PreflightError):
            MODULE.resolve_selection([0, 1, 2, 3], mapping, requested_npu_ids=[2])

    def test_inventory_must_match_nodes_and_npu_smi_map(self) -> None:
        MODULE.validate_inventory([0, 1], [0, 1], [0, 1])
        with self.assertRaises(MODULE.PreflightError):
            MODULE.validate_inventory([0, 1], [0], [0, 1])
        with self.assertRaises(MODULE.PreflightError):
            MODULE.validate_inventory([0, 1], [0, 1], [0])

    def test_run_preflight_archives_passed_inventory_and_occupancy(self) -> None:
        map_text = "0 0 0 0 Ascend910\n0 1 1 1 Ascend910\n"

        def record(_root, _command, label, timeout=120):
            del timeout
            if label == "npu-smi-map":
                return {"returncode": 0, "stdout_text": map_text, "stderr_text": ""}
            if label == "occupancy":
                return {"returncode": 1, "stdout_text": "", "stderr_text": ""}
            return {"returncode": 0, "stdout_text": "ok", "stderr_text": ""}

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(MODULE, "discover_device_ids", return_value=[0, 1]), \
                mock.patch.object(MODULE, "command_record", side_effect=record), \
                mock.patch.object(MODULE.shutil, "which", return_value="/usr/bin/fuser"), \
                mock.patch.object(MODULE.socket, "gethostname", return_value="node-a"):
            summary = MODULE.run_preflight(Path(temporary), [0, 1])
            saved = MODULE.json.loads(
                (Path(temporary) / "node-a" / "summary.json").read_text()
            )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(saved["actual_map_logic_ids"], [0, 1])
        self.assertEqual(saved["occupancy"]["status"], "idle")

    def test_run_preflight_checks_only_selected_devices_for_occupancy(self) -> None:
        map_text = (
            "0 0 0 0 Ascend910\n0 1 1 1 Ascend910\n"
            "1 0 2 2 Ascend910\n1 1 3 3 Ascend910\n"
        )
        occupancy_commands = []

        def record(_root, command, label, timeout=120):
            del timeout
            if label == "npu-smi-map":
                return {"returncode": 0, "stdout_text": map_text, "stderr_text": ""}
            if label == "occupancy":
                occupancy_commands.append(command)
                return {"returncode": 1, "stdout_text": "", "stderr_text": ""}
            return {"returncode": 0, "stdout_text": "ok", "stderr_text": ""}

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(MODULE, "discover_device_ids", return_value=[0, 1, 2, 3]), \
                mock.patch.object(MODULE, "command_record", side_effect=record), \
                mock.patch.object(MODULE.shutil, "which", return_value="/usr/bin/fuser"), \
                mock.patch.object(MODULE.socket, "gethostname", return_value="node-c"):
            summary = MODULE.run_preflight(
                Path(temporary), [0, 1, 2, 3], requested_npu_ids=[1]
            )
        self.assertEqual(summary["occupancy"]["checked_device_ids"], [2, 3])
        self.assertEqual(occupancy_commands[0][1:], ["/dev/davinci2", "/dev/davinci3"])

    def test_run_preflight_fails_closed_and_preserves_summary(self) -> None:
        map_text = "0 0 0 0 Ascend910\n"

        def record(_root, _command, label, timeout=120):
            del timeout
            if label == "npu-smi-map":
                return {"returncode": 0, "stdout_text": map_text, "stderr_text": ""}
            if label == "occupancy":
                return {"returncode": 0, "stdout_text": "1234", "stderr_text": ""}
            return {"returncode": 0, "stdout_text": "ok", "stderr_text": ""}

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(MODULE, "discover_device_ids", return_value=[0]), \
                mock.patch.object(MODULE, "command_record", side_effect=record), \
                mock.patch.object(MODULE.shutil, "which", return_value="/usr/bin/fuser"), \
                mock.patch.object(MODULE.socket, "gethostname", return_value="node-b"):
            with self.assertRaises(MODULE.PreflightError):
                MODULE.run_preflight(Path(temporary), [0])
            saved = MODULE.json.loads(
                (Path(temporary) / "node-b" / "summary.json").read_text()
            )
        self.assertEqual(saved["status"], "failed")
        self.assertIn("1234", saved["error"])
if __name__ == "__main__":
    unittest.main()
