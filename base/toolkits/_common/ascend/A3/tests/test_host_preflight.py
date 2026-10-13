# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
# Patch dependencies where the implementation now lives. The legacy path is a
# re-export shim, so patching its copied names would not intercept device calls.
MODULE_PATH = HERE.parents[4] / "vendors" / "ascend" / "preflight.py"
SPEC = importlib.util.spec_from_file_location("ascend_host_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EMPTY_PROCESS_TABLE = """\
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
"""

class HostPreflightTests(unittest.TestCase):
    def test_legacy_cli_preserves_help_and_early_error_contract(self) -> None:
        legacy = HERE.parent / "host_preflight.py"
        help_result = subprocess.run([sys.executable, str(legacy), "--help"],
                                     capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--npu-ids", help_result.stdout)
        with tempfile.TemporaryDirectory() as temporary:
            failure = subprocess.run([sys.executable, str(legacy), "--output", temporary,
                                      "--expected-device-ids", "invalid"],
                                     capture_output=True, text=True)
            self.assertEqual(failure.returncode, 1)
            self.assertTrue(failure.stderr.startswith("ERROR:"), failure.stderr)
            self.assertNotIn("Traceback", failure.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])

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

    def test_parse_npu_processes_and_map_selected_devices(self) -> None:
        text = EMPTY_PROCESS_TABLE + """\
| 7       0                 | 1444272       | sglangschedul            | 60970                   |
| 7       1                 | 1444275       | sglangschedul            | 60970                   |
"""
        processes = MODULE.parse_npu_processes(text)
        mapping = MODULE.parse_device_map(
            "7 0 14 14 Ascend910\n7 1 15 15 Ascend910\n"
        )
        selected = MODULE.selected_npu_processes(processes, mapping, [14, 15])
        self.assertEqual([item["logic_id"] for item in selected], [14, 15])
        self.assertEqual(selected[0]["process_id"], 1444272)
        with self.assertRaisesRegex(MODULE.PreflightError, "table is missing"):
            MODULE.parse_npu_processes("unstructured output")

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
            if label == "npu-smi-info":
                return {"returncode": 0, "stdout_text": EMPTY_PROCESS_TABLE, "stderr_text": ""}
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
            if label == "npu-smi-info":
                return {"returncode": 0, "stdout_text": EMPTY_PROCESS_TABLE, "stderr_text": ""}
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
            if label == "npu-smi-info":
                return {"returncode": 0, "stdout_text": EMPTY_PROCESS_TABLE, "stderr_text": ""}
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

    def test_run_preflight_rejects_npu_smi_process_missed_by_fuser(self) -> None:
        map_text = "7 0 14 14 Ascend910\n7 1 15 15 Ascend910\n"
        process_text = EMPTY_PROCESS_TABLE + (
            "| 7 0 | 1444272 | sglangschedul | 60970 |\n"
            "| 7 1 | 1444275 | sglangschedul | 60970 |\n"
        )

        def record(_root, _command, label, timeout=120):
            del timeout
            if label == "npu-smi-info":
                return {"returncode": 0, "stdout_text": process_text, "stderr_text": ""}
            if label == "npu-smi-map":
                return {"returncode": 0, "stdout_text": map_text, "stderr_text": ""}
            if label == "occupancy":
                self.fail("fuser must not run after npu-smi proves occupancy")
            return {"returncode": 0, "stdout_text": "ok", "stderr_text": ""}

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(MODULE, "discover_device_ids", return_value=[14, 15]), \
                mock.patch.object(MODULE, "command_record", side_effect=record), \
                mock.patch.object(MODULE.socket, "gethostname", return_value="node-d"):
            with self.assertRaisesRegex(MODULE.PreflightError, "sglangschedul"):
                MODULE.run_preflight(
                    Path(temporary), [14, 15], requested_device_ids=[14, 15],
                )
            saved = MODULE.json.loads(
                (Path(temporary) / "node-d" / "summary.json").read_text()
            )
        self.assertEqual(saved["occupancy"]["status"], "occupied")
        self.assertEqual(
            [item["logic_id"] for item in saved["occupancy"]["processes"]],
            [14, 15],
        )
if __name__ == "__main__":
    unittest.main()
