# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest


BASE_DIR = Path(__file__).resolve().parents[1]
import sys

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import run as base_run
from benchmark_worker import build_torchrun_command
from executors.benchmark import (
    BenchmarkRunRequest,
    aggregate_benchmark_status,
    generate_benchmark_report,
    parse_benchmark_results,
)
from executors.common import (
    BaseRunContext,
    ConfigurationError,
    DeviceLease,
    DeviceLeaseError,
    runtime_lock_record,
    validate_runtime_identity,
)


class UnifiedFacadeTests(unittest.TestCase):
    def test_no_argument_entry_is_safe_and_does_not_run_legacy_cluster(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(base_run.main([]), 2)
        self.assertIn("legacy/cluster_run.py", stderr.getvalue())

    def test_benchmark_static_dry_run_requires_explicit_selection(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = base_run.main([
                "benchmark", "run", "--case", "computation-FP16", "--dry-run",
            ])
        self.assertEqual(code, 2)
        self.assertIn("explicit --npu-ids or --device-ids", stderr.getvalue())

    def test_benchmark_static_dry_run_has_no_runtime_side_effect(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = base_run.main([
                "benchmark", "run", "--case", "computation-FP16",
                "--device-ids", "14", "--dry-run",
            ])
        self.assertEqual(code, 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["kind"], "benchmark")
        self.assertEqual(plan["worker"], "benchmark_worker.py")
        self.assertEqual(plan["selection_request"]["selected_device_ids"], [14])
        self.assertEqual(
            plan["runtime_identity"]["image_id"],
            "deferred-until-image-inspection",
        )
        self.assertTrue(plan["monitoring"]["enabled"])
        self.assertFalse(plan["monitoring"]["automatic_workload_extension"])

    def test_benchmark_monitor_can_be_explicitly_disabled(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = base_run.main([
                "benchmark", "run", "--case", "computation-FP16",
                "--device-ids", "14", "--monitor", "off", "--dry-run",
            ])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(stdout.getvalue())["monitoring"]["enabled"])

    def test_toolkit_static_dry_run_uses_separate_permissions(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = base_run.main([
                "toolkit", "run", "--case", "computation-FP16",
                "--device-ids", "14", "--dry-run",
                "--allow-disruptive-dmi", "--allow-privileged-root",
            ])
        self.assertEqual(code, 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["kind"], "toolkit")
        self.assertTrue(plan["permissions"]["active_dmi"])
        self.assertNotIn("worker", plan)

    def test_high_risk_benchmark_requires_separate_authorization(self) -> None:
        request = BenchmarkRunRequest(
            context=BaseRunContext(device_ids="14", dry_run=True),
            case="main_memory-capacity",
            case_config=None,
            nproc_per_node=1,
            master_port=29721,
            log_level="INFO",
            allow_privileged_root=False,
            allow_high_risk_case=False,
        )
        with self.assertRaisesRegex(ConfigurationError, "high risk"):
            request.validate()


class ResultContractTests(unittest.TestCase):
    def test_requested_monitor_gap_is_partial_but_measurement_failure_is_fatal(self) -> None:
        self.assertEqual(
            aggregate_benchmark_status(
                "passed", "passed", "partial", monitor_enabled=True
            ),
            "partial",
        )
        self.assertEqual(
            aggregate_benchmark_status(
                "passed", "passed", "not-run", monitor_enabled=False
            ),
            "passed",
        )
        self.assertEqual(
            aggregate_benchmark_status(
                "passed", "failed", "passed", monitor_enabled=True
            ),
            "failed",
        )

    def test_benchmark_result_parser_requires_every_rank(self) -> None:
        parsed = parse_benchmark_results(
            "[FlagPerf Result]Rank 0's computation-FP16=1.25TFLOPS\n"
            "[FlagPerf Result]Rank 1's computation-FP16=1.20TFLOPS\n",
            2,
        )
        self.assertEqual(parsed["status"], "passed")
        self.assertEqual(parsed["observed_ranks"], [0, 1])

        partial = parse_benchmark_results(
            "[FlagPerf Result]Rank 0's computation-FP16=1.25TFLOPS\n",
            2,
        )
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["missing_ranks"], [1])

    def test_report_generation_does_not_change_experiment_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(
                json.dumps({
                    "run_id": "benchmark-fixture",
                    "case": "computation-FP16",
                    "status": "partial",
                    "execution_status": "passed",
                    "measurement_status": "partial",
                    "runtime": {"image": "fixture", "image_id": "sha256:test"},
                    "selection": {"selected_device_ids": [14]},
                }) + "\n",
                encoding="utf-8",
            )
            (root / "benchmark-result.json").write_text(
                json.dumps({"metrics": []}) + "\n", encoding="utf-8"
            )
            metadata = generate_benchmark_report(root)
            summary = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "passed")
            self.assertEqual(summary["status"], "partial")
            self.assertTrue((root / "report.md").is_file())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = base_run.main([
                    "report", "--run-id", root.name,
                    "--result-root", str(root.parent),
                ])
            self.assertEqual(code, 0)
            self.assertIn("Report directory:", stdout.getvalue())


class ResourceAndWorkerTests(unittest.TestCase):
    def test_runtime_identity_fails_closed_on_image_id_drift(self) -> None:
        lock = runtime_lock_record()
        image = lock["image_manifest"]["image"]
        expected_id = lock["image_manifest"]["image_id"]
        validated = validate_runtime_identity(
            {"image": image}, {"Id": expected_id}
        )
        self.assertEqual(
            validated["image_manifest"]["validation_status"], "partial"
        )
        with self.assertRaisesRegex(ConfigurationError, "does not match"):
            validate_runtime_identity(
                {"image": image}, {"Id": "sha256:unexpected"}
            )

    def test_device_lease_is_exclusive_and_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = DeviceLease([14], run_id="first", kind="benchmark", root=root)
            second = DeviceLease([14], run_id="second", kind="toolkit", root=root)
            first.acquire()
            try:
                with self.assertRaises(DeviceLeaseError):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()

    def test_benchmark_worker_builds_argv_without_shell_string(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "benchmarks" / "fixture"
            vendor_dir = case_dir / "ascend"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "case_config.yaml").touch()
            (case_dir / "main.py").touch()
            namespace = type("Args", (), {
                "perf_path": str(root),
                "case_name": "fixture",
                "vendor": "ascend",
                "nproc_per_node": 1,
                "nnodes": 1,
                "node_rank": 0,
                "master_addr": "127.0.0.1",
                "master_port": 29721,
                "log_dir": str(root / "result"),
                "host_addr": "127.0.0.1",
            })()
            command, _, _ = build_torchrun_command(namespace)
            self.assertIsInstance(command, list)
            self.assertEqual(command[0], "torchrun")
            self.assertIn("--vendor=ascend", command)


if __name__ == "__main__":
    unittest.main()
