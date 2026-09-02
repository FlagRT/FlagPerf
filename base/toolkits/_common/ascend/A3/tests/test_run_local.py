from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parents[4] / "run_local.py"
SPEC = importlib.util.spec_from_file_location("base_run_local", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def args(**overrides: object) -> argparse.Namespace:
    values = {
        "suite": "ascend-toolkit",
        "legacy_probe": False,
        "npu_ids": None,
        "device_ids": None,
        "compute_monitor": "on",
        "data_movement_monitor": "on",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class StaticArgumentTests(unittest.TestCase):
    def test_cli_defaults_to_toolkit_and_rejects_removed_workload_suite(self) -> None:
        with patch.object(sys, "argv", ["run_local.py"]):
            self.assertEqual(MODULE.parse_args().suite, "ascend-toolkit")
        with (
            patch.object(sys, "argv", ["run_local.py", "--suite", "workload"]),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            MODULE.parse_args()

    def test_p2p_latency_toolkit_shares_host_pid_namespace(self) -> None:
        self.assertEqual(
            MODULE.container_namespace_args(None),
            ["--ipc=host", "--pid=host"],
        )
        self.assertEqual(
            MODULE.container_namespace_args(
                ["interconnect-P2P_intraserver-latency"]
            ),
            ["--ipc=host", "--pid=host"],
        )

    def test_non_latency_toolkit_run_keeps_private_pid_namespace(self) -> None:
        self.assertEqual(
            MODULE.container_namespace_args(["interconnect-P2P_intraserver"]),
            ["--ipc=host"],
        )

    def test_toolkit_accepts_each_selector_and_default_all(self) -> None:
        for namespace in (args(), args(npu_ids="1-3"), args(device_ids="2,3")):
            MODULE.validate_static_args(namespace)

    def test_selection_is_rejected_with_legacy_probe(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot be combined"):
            MODULE.validate_static_args(args(legacy_probe=True, device_ids="2"))

    def test_report_fallback_uses_current_schema_and_preserves_status(self) -> None:
        def fail_generation(_root: Path) -> None:
            raise RuntimeError("fixture report failure")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(
                json.dumps({"schema_version": 1, "status": "partial"}) + "\n",
                encoding="utf-8",
            )
            fake_module = SimpleNamespace(generate_and_record=fail_generation)
            with patch.dict(sys.modules, {"generate_toolkit_report": fake_module}):
                metadata = MODULE.generate_report_safely(root)

            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 6)
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(summary["status"], "partial")


if __name__ == "__main__":
    unittest.main()
