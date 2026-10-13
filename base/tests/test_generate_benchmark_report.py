from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import generate_benchmark_report as report_generator
from executors.benchmark import (
    BenchmarkRunRequest,
    snapshot_case_configuration,
)
from executors.common import BaseRunContext


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_result(root: Path, *, status: str = "passed") -> None:
    generic = root / "case-config" / "generic.yaml"
    ascend = root / "case-config" / "ascend.yaml"
    generic.parent.mkdir(parents=True, exist_ok=True)
    generic.write_text(
        "M: 4096\nN: 4096\nK: 4096\nWARMUP: 100\nITERS: 50000\n"
        'DIST_BACKEND: "mpi"\n',
        encoding="utf-8",
    )
    ascend.write_text(
        'M: 8192\nN: 8192\nK: 8192\nDIST_BACKEND: "gloo"\n',
        encoding="utf-8",
    )
    benchmark_log = root / "computation-FP16" / "host_noderank0" / "benchmark.log.txt"
    benchmark_log.parent.mkdir(parents=True, exist_ok=True)
    benchmark_log.write_text(
        "[FlagPerf Result]Rank 0's computation-FP16=291.3TFLOPS\n"
        "[FlagPerf Result]Rank 1's computation-FP16=287.1TFLOPS\n",
        encoding="utf-8",
    )
    (root / "runner.log").write_text("runner fixture\n", encoding="utf-8")
    write_json(root / "resolved-plan.json", {"schema_version": 1, "kind": "benchmark"})
    write_json(root / "host-preflight" / "fixture" / "summary.json", {"status": "passed"})
    write_json(root / "host-postflight" / "fixture" / "summary.json", {"status": "passed"})

    summary = {
        "schema_version": 1,
        "kind": "benchmark",
        "run_id": "benchmark-fixture",
        "case": "computation-FP16",
        "status": status,
        "execution_status": "passed" if status != "failed" else "failed",
        "measurement_status": status,
        "started_at": "2026-08-30T03:28:11Z",
        "finished_at": "2026-08-30T03:31:57Z",
        "wall_clock_duration_s": 226.5,
        "container_returncode": 0 if status != "failed" else 1,
        "timed_out": False,
        "selection": {
            "source": "npu-ids",
            "selected_npu_ids": [7],
            "selected_device_ids": [14, 15],
        },
        "runtime": {
            "image": "flagrt/ascend-operator-runtime:test",
            "image_id": "sha256:fixture",
            "lock": {
                "image_manifest": {
                    "sha256": "image-manifest-sha",
                    "validation_status": "partial",
                    "validation_scope": "fixture validation scope",
                },
                "stack_lock": {"sha256": "stack-lock-sha"},
            },
        },
        "resolved_plan": {
            "image": "flagrt/ascend-operator-runtime:test",
            "nproc_per_node": 2,
            "permissions": {
                "privileged_root": True,
                "network": "host",
                "ipc_namespace": "host",
                "pid_namespace": "private",
            },
        },
        "case_config": {
            "generic": {
                "path": "/source/generic.yaml",
                "artifact_path": "case-config/generic.yaml",
                "bytes": generic.stat().st_size,
                "sha256": digest(generic),
            },
            "ascend": {
                "path": "/source/ascend.yaml",
                "artifact_path": "case-config/ascend.yaml",
                "bytes": ascend.stat().st_size,
                "sha256": digest(ascend),
            },
        },
        "device_lease": {
            "backend": "flock",
            "device_ids": [14, 15],
            "acquired_at": "2026-08-30T03:28:14Z",
            "released_at": "2026-08-30T03:31:54Z",
        },
        "host_preflight": {
            "path": "host-preflight/fixture/summary.json",
            "result": {
                "status": "passed",
                "host": "fixture-host",
                "occupancy": {"status": "idle", "checked_device_ids": [14, 15]},
            },
        },
        "host_postflight": {
            "status": "passed",
            "path": "host-postflight/fixture/summary.json",
            "result": {
                "status": "passed",
                "host": "fixture-host",
                "occupancy": {"status": "idle", "checked_device_ids": [14, 15]},
            },
        },
        "report_generation": {"status": "not_started"},
    }
    write_json(root / "summary.json", summary)
    metrics = [
        {
            "rank": 0,
            "metric": "computation-FP16",
            "value": 291.3,
            "unit": "TFLOPS",
            "raw": "[FlagPerf Result]Rank 0's computation-FP16=291.3TFLOPS",
        },
        {
            "rank": 1,
            "metric": "computation-FP16",
            "value": 287.1,
            "unit": "TFLOPS",
            "raw": "[FlagPerf Result]Rank 1's computation-FP16=287.1TFLOPS",
        },
    ]
    if status == "partial":
        metrics = metrics[:1]
    if status == "failed":
        metrics = []
    write_json(root / "benchmark-result.json", {
        "schema_version": 1,
        "status": status,
        "case": "computation-FP16",
        "container_returncode": summary["container_returncode"],
        "timed_out": False,
        "fallback_count": 0,
        "expected_ranks": [0, 1],
        "observed_ranks": [item["rank"] for item in metrics],
        "missing_ranks": [rank for rank in (0, 1) if rank not in {item["rank"] for item in metrics}],
        "metrics": metrics,
        "benchmark_log": benchmark_log.relative_to(root).as_posix(),
    })


def fixture_monitor(root: Path) -> None:
    monitor_dir = root / "benchmark-monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    raw = []
    for sequence in range(10):
        source_fields = {
            "Aicore Usage Rate(%)": str(80 + sequence),
            "Aivector Usage Rate(%)": "3",
            "HBM Usage Rate(%)": "21",
            "HBM Bandwidth Usage Rate(%)": str(40 + sequence),
            "NPU Utilization(%)": str(75 + sequence),
        }
        samples.append({
            "sequence": sequence,
            "npu_id": 7,
            "chip_id": 0,
            "logical_device_id": 14,
            "sample_started_at": f"2026-08-30T03:30:{sequence:02d}Z",
            "sample_finished_at": f"2026-08-30T03:30:{sequence + 1:02d}Z",
            "started_offset_s": 5.0 + sequence,
            "finished_offset_s": 5.8 + sequence,
            "duration_s": 0.8,
            "source_fields": source_fields,
            "values": {
                "aicore_usage_rate_pct": 80 + sequence,
                "aivector_usage_rate_pct": 3,
                "hbm_usage_rate_pct": 21,
                "hbm_bandwidth_usage_rate_pct": 40 + sequence,
                "npu_utilization_pct": 75 + sequence,
            },
            "missing_or_invalid_fields": [],
            "valid": True,
        })
        raw.append({
            "sequence": sequence,
            "npu_id": 7,
            "command": ["npu-smi", "info", "-t", "usages", "-i", "7"],
            "returncode": 0,
            "stdout": "raw fixture",
            "stderr": "",
        })
    parsed_path = monitor_dir / "samples.jsonl"
    raw_path = monitor_dir / "samples.raw.jsonl"
    parsed_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in samples),
        encoding="utf-8",
    )
    raw_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in raw),
        encoding="utf-8",
    )
    window = {
        "role": "measurement",
        "rank": 0,
        "local_rank": 0,
        "logical_device_id": 14,
        "npu_id": 7,
        "chip_id": 0,
        "started_at": "2026-08-30T03:30:00Z",
        "finished_at": "2026-08-30T03:30:20Z",
        "started_offset_s": 4.9,
        "finished_offset_s": 15.9,
    }
    monitor = {
        "schema_version": 1,
        "status": "passed",
        "collector": "npu-smi info -t usages",
        "policy": {
            "enabled": True,
            "coverage_window": "rank-local exact measurement window",
            "required_samples_per_target": 10,
        },
        "target_interval_s": 1.0,
        "command_timeout_s": 5,
        "required_samples_per_chip": 10,
        "targets": [{"npu_id": 7, "chip_id": 0, "logic_id": 14}],
        "rank_device_map": [{
            "rank": 0, "local_rank": 0, "npu_id": 7,
            "chip_id": 0, "logic_id": 14,
        }],
        "lifecycle_windows": [{
            "role": "container", "started_at": "2026-08-30T03:29:00Z",
            "finished_at": "2026-08-30T03:31:00Z",
            "started_offset_s": 0.0, "finished_offset_s": 120.0,
        }],
        "workload_windows": [window],
        "sample_counts_by_target": {"7/0/14": 10},
        "primary_sample_counts_by_target": {"7/0/14": 10},
        "parsed_samples": {
            "path": parsed_path.relative_to(root).as_posix(),
            "bytes": parsed_path.stat().st_size,
            "sha256": digest(parsed_path),
        },
        "raw_samples": {
            "path": raw_path.relative_to(root).as_posix(),
            "bytes": raw_path.stat().st_size,
            "sha256": digest(raw_path),
        },
        "automatic_workload_extension": False,
        "reasons": [],
    }
    write_json(monitor_dir / "summary.json", monitor)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = 2
    summary["monitoring_status"] = "passed"
    summary["monitoring"] = monitor
    write_json(summary_path, summary)


class BenchmarkReportTests(unittest.TestCase):
    def test_monitor_report_keeps_raw_values_windows_and_svg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root)
            fixture_monitor(root)

            first = report_generator.generate_and_record(root)
            first_monitor = (root / "report_monitor.md").read_bytes()
            second = report_generator.generate_and_record(root)
            monitor_report = (root / "report_monitor.md").read_text(
                encoding="utf-8"
            )
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(first["monitor_sha256"], second["monitor_sha256"])
            self.assertEqual(first_monitor, (root / "report_monitor.md").read_bytes())
            self.assertEqual(summary["status"], "passed")
            self.assertIn("精确测量窗覆盖", monitor_report)
            self.assertIn("rank-local exact measurement window", monitor_report)
            self.assertIn("89", monitor_report)
            self.assertIn("HBM 占用", monitor_report)
            monitor_assets = [
                item for item in first["assets"]
                if item["path"].endswith("benchmark-monitor-usage.svg")
            ]
            self.assertEqual(len(monitor_assets), 1)
            ET.parse(root / monitor_assets[0]["path"])
            for target in re.findall(
                r"!?\[[^]]*\]\(([^)]+)\)", monitor_report
            ):
                self.assertTrue((root / target).is_file(), target)

    def test_unknown_future_monitor_schema_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root, status="partial")
            fixture_monitor(root)
            monitor_path = root / "benchmark-monitor" / "summary.json"
            monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
            monitor["schema_version"] = max(report_generator.SUPPORTED_MONITOR_SCHEMA_VERSIONS) + 1
            write_json(monitor_path, monitor)

            with self.assertRaisesRegex(
                report_generator.ReportError,
                "unsupported Benchmark monitor schema",
            ):
                report_generator.generate_and_record(root)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["report_generation"]["status"], "failed")

    def test_passed_report_is_human_readable_linked_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root)

            first = report_generator.generate_and_record(root)
            report_bytes = (root / "report.md").read_bytes()
            asset_bytes = {
                item["path"]: (root / item["path"]).read_bytes()
                for item in first["assets"]
            }
            second = report_generator.generate_and_record(root)
            report = (root / "report.md").read_text(encoding="utf-8")
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(first["schema_version"], 3)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(report_bytes, (root / "report.md").read_bytes())
            self.assertEqual(asset_bytes, {
                item["path"]: (root / item["path"]).read_bytes()
                for item in second["assets"]
            })
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["report_generation"]["status"], "passed")
            self.assertIn("结论", report)
            self.assertIn("状态总览", report)
            self.assertIn("291.3", report)
            self.assertIn("287.1", report)
            self.assertIn("Rank 差异率", report)
            self.assertIn("顶层标量的最终覆盖结果", report)
            self.assertRegex(report, r"`M`\s*\|\s*`8192`\s*\|\s*`ascend`")
            self.assertRegex(report, r"`WARMUP`\s*\|\s*`100`\s*\|\s*`generic`")
            self.assertIn("Canonical summary SHA-256", report)
            self.assertIn("不等于稳定性能基线", report)
            self.assertTrue((root / "report_monitor.md").is_file())
            self.assertEqual(len(first["assets"]), 1)
            ET.parse(root / first["assets"][0]["path"])

            for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", report):
                self.assertTrue((root / target).is_file(), target)

    def test_partial_report_explains_missing_rank_without_changing_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root, status="partial")

            report_generator.generate_and_record(root)
            report = (root / "report.md").read_text(encoding="utf-8")
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "partial")
        self.assertIn("仅部分完成", report)
        self.assertIn("缺少 Rank [1]", report)
        self.assertIn("1/2", report)

    def test_failed_report_is_still_generated_without_metric_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root, status="failed")

            metadata = report_generator.generate_and_record(root)
            report = (root / "report.md").read_text(encoding="utf-8")
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(metadata["assets"], [])
        self.assertIn("应先处理失败原因", report)
        self.assertIn("没有可展示的有限数值指标", report)

    def test_unknown_future_result_schema_fails_closed_and_preserves_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root, status="partial")
            result = json.loads((root / "benchmark-result.json").read_text(encoding="utf-8"))
            result["schema_version"] = 2
            write_json(root / "benchmark-result.json", result)

            with self.assertRaisesRegex(
                report_generator.ReportError, "unsupported Benchmark result schema",
            ):
                report_generator.generate_and_record(root)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["report_generation"]["status"], "failed")
        self.assertEqual(summary["report_generation"]["schema_version"], 3)

    def test_missing_result_artifact_produces_honest_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "summary.json", {
                "schema_version": 1,
                "kind": "benchmark",
                "run_id": "early-failure",
                "case": "computation-FP16",
                "status": "failed",
                "execution_status": "failed",
                "measurement_status": "not_available",
                "failure_stage": "image-inspection",
                "error": "locked image mismatch",
            })

            metadata = report_generator.generate_and_record(root)
            report = (root / "report.md").read_text(encoding="utf-8")

        self.assertIsNone(metadata["input_schemas"]["benchmark_result"])
        self.assertIn("没有可展示的有限数值指标", report)
        self.assertIn("locked image mismatch", report)

    def test_executor_snapshots_exact_case_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = BenchmarkRunRequest(
                context=BaseRunContext(device_ids="14"),
                case="computation-FP16",
                case_config=None,
                nproc_per_node=1,
                master_port=29721,
                log_level="INFO",
                allow_privileged_root=False,
                allow_high_risk_case=False,
            )

            records = snapshot_case_configuration(request, root)

            self.assertIn("generic", records)
            self.assertIn("ascend", records)
            for record in records.values():
                artifact = root / record["artifact_path"]
                self.assertTrue(artifact.is_file())
                self.assertEqual(digest(artifact), record["sha256"])


if __name__ == "__main__":
    unittest.main()
