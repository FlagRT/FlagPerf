from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parents[4]
MODULE_PATH = BASE_DIR / "generate_toolkit_report.py"
SPEC = importlib.util.spec_from_file_location("generate_toolkit_report", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command(path: str) -> dict:
    return {
        "command": ["ascend-dmi"],
        "returncode": 0,
        "timed_out": False,
        "stdout": {"path": path + "/microbenchmark.stdout"},
        "stderr": {"path": path + "/microbenchmark.stderr"},
    }


def metric(value: float, unit: str, **identity: object) -> dict:
    return {"value": value, "unit": unit, **identity}


def fixture_result(root: Path) -> None:
    summary = {
        "schema_version": 1,
        "run_id": "20260825T081245Z",
        "suite": "ascend-toolkit",
        "status": "partial",
        "started_at": "2026-08-25T08:12:45Z",
        "finished_at": "2026-08-25T08:22:45Z",
        "image": "flagrt/ascend-operator-runtime:test",
        "privileged_root": True,
        "host_preflight": {
            "result": {
                "status": "passed",
                "host": "fixture-host",
                "expected_device_ids": [0, 1, 2, 3],
                "actual_device_node_ids": [0, 1, 2, 3],
                "occupancy": {"status": "idle"},
            }
        },
    }
    cases = {
        "computation-FP16": {
            # Deliberately old-schema: measurement_status is absent and must be
            # inferred without confusing unsupported diagnosis with bad data.
            "status": "partial",
            "diagnosis_status": "unsupported",
            "error": "vendor diagnosis aiflops is unsupported or lacks a threshold for this device",
            "command": command("cases/computation-FP16"),
            "metrics": [metric(6019.71826171875, "TFLOPS", device="all")],
            "duration_s": 60.25,
        },
        "computation-INT8": {
            "status": "passed",
            "measurement_status": "passed",
            "diagnosis_status": "passed",
            "command": command("cases/computation-INT8"),
            "metrics": [metric(11677.6064453125, "TOPS", device="all")],
            "duration_s": 61.0,
        },
        "main_memory-bandwidth": {
            "status": "passed",
            "measurement_status": "passed",
            "diagnosis_status": "passed",
            "commands": [command("cases/main_memory-bandwidth")],
            "metrics": [
                metric(1518.0, "GB/s", device="0"),
                metric(1522.0, "GB/s", device="0"),
                metric(1531.0, "GB/s", device="1"),
                metric(1537.0, "GB/s", device="1"),
            ],
        },
        "main_memory-capacity": {
            "status": "passed",
            "measurement_status": "passed",
            "diagnosis_status": "passed",
            "commands": [command("cases/main_memory-capacity")],
            "capacity_scope": {
                "metric_scope": "chip", "aggregation": "none", "unit": "MB",
                "target_count": 2,
            },
            "metrics": [
                metric(65536, "MB", card=0, chip=0, device="0", scope="chip"),
                metric(65536, "MB", card=0, chip=1, device="1", scope="chip"),
            ],
        },
        "interconnect-P2P_intraserver": {
            "status": "passed",
            "measurement_status": "passed",
            "diagnosis_status": "passed",
            "command": command("cases/interconnect-P2P_intraserver"),
            "metrics": [
                metric(391.2, "GB/s", direction="unidirectional", source_device="0", destination_device="1"),
                metric(390.8, "GB/s", direction="unidirectional", source_device="1", destination_device="0"),
                metric(745.5, "GB/s", direction="bidirectional", source_device="0", destination_device="1"),
                metric(744.9, "GB/s", direction="bidirectional", source_device="1", destination_device="0"),
            ],
        },
    }
    manifest = {
        "schema_version": 1,
        "status": "partial",
        "started_at": summary["started_at"],
        "finished_at": summary["finished_at"],
        "discovered_device_ids": [0, 1, 2, 3],
        "cases": cases,
        "health": {},
        "diagnostics": {},
    }
    write_json(root / "summary.json", summary)
    write_json(root / "toolkit-evidence" / "manifest.json", manifest)


class ReportTests(unittest.TestCase):
    def test_schema_three_manifest_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root)
            manifest_path = root / "toolkit-evidence" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 3
            write_json(manifest_path, manifest)

            metadata = MODULE.generate_and_record(root)
            repeated = MODULE.generate_and_record(root)

            self.assertEqual(metadata["schema_version"], 6)
            self.assertEqual(metadata["sha256"], repeated["sha256"])
            self.assertEqual(metadata["monitor_sha256"], repeated["monitor_sha256"])
            self.assertTrue((root / "report.md").is_file())
            self.assertTrue((root / "report_monitor.md").is_file())

    def test_unknown_future_manifest_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root)
            manifest_path = root / "toolkit-evidence" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 4
            write_json(manifest_path, manifest)

            with self.assertRaisesRegex(MODULE.ReportError, "unsupported toolkit manifest schema"):
                MODULE.generate_and_record(root)

    def test_repeated_dmi_errors_are_grouped_once_with_raw_evidence_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root)
            detail = (
                "DMI error 2: Error code [0x2] is displayed. "
                "A software or internal error occurs."
            )
            errors = "; ".join(
                f"{source}->{destination}, {size} bytes: {detail}"
                for source, destination in ((2, 3), (2, 4), (3, 2))
                for size in (512, 4096, 65536, 1048576)
            )
            manifest_path = root / "toolkit-evidence" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            case_name = "interconnect-P2P_intraserver-latency"
            manifest["cases"][case_name] = {
                "status": "partial",
                "measurement_status": "partial",
                "measurement_error": errors,
                "error": errors,
                "diagnosis_status": "passed",
                "metrics": [],
                "sweep_scope": {"expected_points": 12, "completed_points": 0},
            }
            write_json(manifest_path, manifest)
            metrics_path = root / "toolkit-evidence" / "cases" / case_name / "metrics.json"
            write_json(metrics_path, manifest["cases"][case_name])

            MODULE.generate_and_record(root)
            report = (root / "report.md").read_text(encoding="utf-8")

        self.assertEqual(report.count("DMI error 2:"), 1)
        self.assertIn("重复 12 次", report)
        self.assertIn("示例范围：2->3, 512 bytes", report)
        self.assertIn("完整错误证据", report)
        self.assertIn("cases/interconnect-P2P_intraserver-latency/metrics.json", report)
        self.assertNotIn("3->2, 1048576 bytes: DMI error", report)
        self.assertIn("聚合原因见上表，逐命令事实见原始证据索引", report)

    def test_many_distinct_errors_obey_inline_budget(self) -> None:
        rendered = MODULE.human_error(
            "; ".join(f"target-{index}: distinct failure {index}" for index in range(30)),
            evidence_path="toolkit-evidence/manifest.json",
        )

        self.assertLess(len(rendered), 1200)
        self.assertIn("另有", rendered)
        self.assertIn("完整错误证据", rendered)

    def test_short_error_is_not_rewritten(self) -> None:
        self.assertEqual(MODULE.human_error("single actionable error"), "single actionable error")

    def test_partial_report_separates_measurement_and_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root)

            metadata = MODULE.generate_and_record(root)
            report = (root / "report.md").read_text(encoding="utf-8")
            updated = json.loads((root / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(updated["status"], "partial")
            self.assertEqual(updated["report_generation"]["schema_version"], 6)
            self.assertEqual(updated["report_generation"]["status"], "passed")
            self.assertEqual(updated["report_generation"]["monitor_path"], "report_monitor.md")
            self.assertTrue((root / "report_monitor.md").is_file())
            self.assertIn("测量层", report)
            self.assertIn("通过*", report)
            self.assertIn("6019.71826171875", report)
            self.assertIn("TFLOPS", report)
            self.assertIn("11677.6064453125", report)
            self.assertIn("TOPS", report)
            self.assertIn("逐 chip 原值，未聚合", report)
            self.assertIn("不乘二、不改标 MiB", report)
            self.assertEqual(len(metadata["assets"]), 4)
            for asset in metadata["assets"]:
                ET.parse(root / asset["path"])

    def test_selected_device_report_uses_resolved_scope_and_all_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root)
            summary = json.loads((root / "summary.json").read_text())
            summary["host_preflight"]["result"]["selection"] = {
                "source": "npu-ids",
                "requested_ids": [1],
                "selected_npu_ids": [1],
                "selected_device_ids": [2, 3],
                "excluded_device_ids": [0, 1],
            }
            summary["host_preflight"]["result"]["occupancy"] = {
                "status": "idle", "checked_device_ids": [2, 3],
            }
            manifest_path = root / "toolkit-evidence" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["selection"] = {
                "source": "npu-ids", "selected_device_ids": [2, 3],
                "excluded_device_ids": [0, 1],
            }
            manifest["cases"]["computation-FP16"]["metrics"] = [
                metric(700.0, "TFLOPS", device="2"),
                metric(710.0, "TFLOPS", device="3"),
            ]
            manifest["cases"]["interconnect-P2P_intraserver"]["p2p_scope"] = {
                "mode": "selected-combinations", "expected_pair_count": 1,
                "reverse_pairs_inferred": False,
            }
            write_json(root / "summary.json", summary)
            write_json(manifest_path, manifest)

            MODULE.generate_and_record(root)
            report = (root / "report.md").read_text(encoding="utf-8")

        self.assertIn("选中逻辑 Device", report)
        self.assertIn("[2, 3]", report)
        self.assertIn("700", report)
        self.assertIn("710", report)
        self.assertIn("P2P 覆盖所选逻辑 Device 组合", report)
        self.assertIn("不推断、不镜像", report)
        self.assertNotIn("全部已发现逻辑设备", report)

    def test_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root)
            first = MODULE.generate_and_record(root)
            first_bytes = (root / "report.md").read_bytes()
            first_assets = {
                item["path"]: (root / item["path"]).read_bytes()
                for item in first["assets"]
            }

            second = MODULE.generate_and_record(root)

            self.assertEqual(first_bytes, (root / "report.md").read_bytes())
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["monitor_sha256"], second["monitor_sha256"])
            self.assertEqual(first_assets, {
                item["path"]: (root / item["path"]).read_bytes()
                for item in second["assets"]
            })

    def test_monitor_report_keeps_raw_dmi_and_renders_usage_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root)
            manifest_path = root / "toolkit-evidence" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = 2
            sample_path = (
                root / "toolkit-evidence" / "cases" / "computation-FP16"
                / "monitor" / "all-devices" / "samples.jsonl"
            )
            raw_path = sample_path.with_name("samples.raw.jsonl")
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            samples = []
            for sequence in range(10):
                for chip_id, logic_id in ((0, 2), (1, 3)):
                    samples.append({
                        "sequence": sequence, "npu_id": 1, "chip_id": chip_id,
                        "logical_device_id": logic_id,
                        "sample_started_at": f"2026-08-25T08:11:{sequence:02d}Z",
                        "sample_finished_at": f"2026-08-25T08:12:{sequence:02d}Z",
                        "started_offset_s": sequence + 0.1,
                        "finished_offset_s": sequence + 0.2,
                        "valid": True,
                        "missing_or_invalid_fields": [],
                        "source_fields": {
                            "Aicore Usage Rate(%)": "100",
                            "Aivector Usage Rate(%)": "0",
                            "HBM Bandwidth Usage Rate(%)": "025",
                            "NPU Utilization(%)": "100",
                        },
                        "values": {
                            "aicore_usage_rate_pct": 100,
                            "aivector_usage_rate_pct": 0,
                            "hbm_bandwidth_usage_rate_pct": 25,
                            "npu_utilization_pct": 100,
                        },
                    })
            sample_path.write_text(
                "".join(json.dumps(item) + "\n" for item in samples), encoding="utf-8"
            )
            raw_path.write_text("{}\n", encoding="utf-8")
            unit = {
                "status": "passed",
                "targets": [
                    {"npu_id": 1, "chip_id": 0, "logic_id": 2},
                    {"npu_id": 1, "chip_id": 1, "logic_id": 3},
                ],
                "workload_windows": [{
                    "role": "primary", "started_offset_s": 0,
                    "finished_offset_s": 10.5,
                }],
                "sample_counts_by_target": {"1/0/2": 10, "1/1/3": 10},
                "primary_sample_counts_by_target": {"1/0/2": 10, "1/1/3": 10},
                "parsed_samples": {"path": str(sample_path.relative_to(root / "toolkit-evidence"))},
                "raw_samples": {"path": str(raw_path.relative_to(root / "toolkit-evidence"))},
                "extensions": [], "reasons": [],
            }
            manifest["cases"]["computation-FP16"].update(
                monitoring_status="passed",
                monitor={"status": "passed", "units": [unit]},
            )
            manifest["cases"]["computation-FP16"]["metrics"][0]["value_raw"] = (
                "6019.7182617187501"
            )
            write_json(manifest_path, manifest)

            metadata = MODULE.generate_and_record(root)
            report = (root / "report_monitor.md").read_text(encoding="utf-8")
            asset = root / "report-assets" / "monitor-fp16-unit-1.svg"
            tree = ET.parse(asset)
            asset_text = asset.read_text(encoding="utf-8")

        self.assertIn("6019.7182617187501", report)
        self.assertIn("1/0/2", report)
        self.assertIn("不自动判定硬件达到理论峰值", report)
        self.assertIn("## 快速导航", report)
        self.assertIn("(#monitor-fp16-unit-1-timeline)", report)
        self.assertIn('<a id="monitor-fp16-unit-1-timeline"></a>', report)
        self.assertIn("#### 逐样本原值", report)
        self.assertEqual(len(re.findall(r"^\| S\d+", report, re.M)), 20)
        self.assertIn("| 100 | 0 | 025 | 100 | 有效 |", report)
        self.assertNotIn("悬停", report)
        self.assertNotIn("悬停", asset_text)
        self.assertIn(">S00</text>", asset_text)
        self.assertIn(">025</text>", asset_text)
        titles = tree.findall(".//{http://www.w3.org/2000/svg}title")
        self.assertEqual(len(titles), 1)
        svg_text = tree.findall(".//{http://www.w3.org/2000/svg}text")
        self.assertEqual(sum("sample-id" in item.get("class", "") for item in svg_text), 10)
        self.assertEqual(sum("sample-value" in item.get("class", "") for item in svg_text), 80)
        self.assertIn("report-assets/monitor-fp16-unit-1.svg", {
            item["path"] for item in metadata["assets"]
        })

    def test_monitor_rows_keep_invalid_samples_and_window_roles(self) -> None:
        samples = [{
            "sequence": 7, "npu_id": 1, "chip_id": 0, "logical_device_id": 2,
            "sample_started_at": "2026-08-25T08:12:00Z",
            "sample_finished_at": "2026-08-25T08:12:01Z",
            "started_offset_s": 9.5, "finished_offset_s": 10.5,
            "valid": False,
            "missing_or_invalid_fields": ["HBM Bandwidth Usage Rate(%)"],
            "source_fields": {
                "Aicore Usage Rate(%)": "007",
                "Aivector Usage Rate(%)": "0",
                "NPU Utilization(%)": "42",
            },
            "values": {
                "aicore_usage_rate_pct": 7,
                "aivector_usage_rate_pct": 0,
                "npu_utilization_pct": 42,
            },
        }]
        windows = [
            {"role": "primary", "started_offset_s": 0, "finished_offset_s": 10},
            {"role": "extension-1", "started_offset_s": 10, "finished_offset_s": 20},
        ]

        rows = MODULE.monitor_sample_rows(samples, windows, (1, 0, 2))

        self.assertEqual(rows[0][0], "S07")
        self.assertEqual(rows[0][1], "primary + extension-1")
        self.assertEqual(rows[0][4:8], ["007", "0", "—", "42"])
        self.assertIn("HBM Bandwidth Usage Rate(%)", rows[0][8])

    def test_monitor_svg_keeps_twenty_one_visible_sample_columns(self) -> None:
        samples = []
        for sequence in range(21):
            value = 100 if sequence == 10 else sequence
            samples.append({
                "sequence": sequence, "npu_id": 1, "chip_id": 0,
                "logical_device_id": 2,
                "started_offset_s": float(sequence),
                "finished_offset_s": float(sequence) + 0.8,
                "valid": True, "missing_or_invalid_fields": [],
                "source_fields": {
                    "Aicore Usage Rate(%)": str(value),
                    "Aivector Usage Rate(%)": "0",
                    "HBM Bandwidth Usage Rate(%)": "0",
                    "NPU Utilization(%)": str(value),
                },
                "values": {
                    "aicore_usage_rate_pct": value,
                    "aivector_usage_rate_pct": 0,
                    "hbm_bandwidth_usage_rate_pct": 0,
                    "npu_utilization_pct": value,
                },
            })

        svg = MODULE.monitor_heatmap_svg(
            "21 samples", samples,
            [{"role": "primary", "started_offset_s": 0, "finished_offset_s": 20.8}],
        )

        self.assertIsNotNone(svg)
        assert svg is not None
        root = ET.fromstring(svg)
        self.assertIn(">S00</text>", svg)
        self.assertIn(">S20</text>", svg)
        self.assertIn(">100</text>", svg)
        self.assertNotIn("悬停", svg)
        self.assertEqual(len(root.findall(".//{http://www.w3.org/2000/svg}title")), 1)
        svg_text = root.findall(".//{http://www.w3.org/2000/svg}text")
        self.assertEqual(sum("sample-id" in item.get("class", "") for item in svg_text), 21)
        self.assertEqual(sum("sample-value" in item.get("class", "") for item in svg_text), 84)

    def test_new_latency_and_hccl_sections_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_result(root)
            manifest_path = root / "toolkit-evidence" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["cases"].update({
                "interconnect-d2h": {
                    "status": "passed", "measurement_status": "passed",
                    "diagnosis_status": "passed", "metrics": [metric(55.0, "GB/s", device="2")],
                },
                "interconnect-h2d-latency": {
                    "status": "passed", "measurement_status": "passed",
                    "diagnosis_status": "not-run", "metrics": [
                        metric(3.2, "ns", device="2", size_bytes=512),
                        metric(8.5, "ns", device="2", size_bytes=4096),
                    ],
                },
                "interconnect-P2P_intraserver-latency": {
                    "status": "passed", "measurement_status": "passed",
                    "diagnosis_status": "passed", "metrics": [
                        metric(2.1, "ns", source_device="2", destination_device="3", size_bytes=512),
                    ],
                },
                "interconnect-MPI_intraserver": {
                    "status": "passed", "measurement_status": "passed",
                    "diagnosis_status": "not-run", "hccl_scope": {
                        "rank_count": 2, "selected_device_ids": [2, 3],
                        "min_bytes": 8192, "max_bytes": 16384, "factor": 2,
                        "warmup": 10, "iterations": 20,
                    }, "metrics": [{
                        "message_size_bytes": 8192, "avg_time_us": 12.5,
                        "value": 18.2, "unit": "GB/s", "verification_passed": True,
                    }],
                },
            })
            write_json(manifest_path, manifest)
            metadata = MODULE.generate_and_record(root)
            report = (root / "report.md").read_text(encoding="utf-8")
            paths = {item["path"] for item in metadata["assets"]}

        self.assertIn("D2H：512 MiB", report)
        self.assertIn("H2D / D2H / P2P 时延扫点", report)
        self.assertIn("MPI/HCCL AllReduce", report)
        self.assertIn("不派生 bus bandwidth", report)
        self.assertIn("report-assets/transfer-latency.svg", paths)
        self.assertIn("report-assets/hccl-allreduce.svg", paths)

    def test_hccs_timeline_displays_raw_values_without_hover(self) -> None:
        samples = [{
            "sequence": 0, "npu_id": 1, "chip_id": 0,
            "logical_device_id": 2, "link": "total",
            "started_offset_s": 0.0, "finished_offset_s": 1.0,
            "source_fields": {
                "rx_bandwidth(GB/S)": "12.500",
                "tx_bandwidth(GB/S)": "0.000",
            },
            "values": {"rx_gb_s": 12.5, "tx_gb_s": 0.0},
            "valid": True,
        }]
        svg = MODULE.hccs_timeline_svg("HCCS raw", samples, [])
        self.assertIsNotNone(svg)
        assert svg is not None
        ET.fromstring(svg)
        self.assertIn(">12.500</text>", svg)
        self.assertIn(">0.000</text>", svg)
        self.assertNotIn("悬停", svg)

    def test_degraded_report_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "summary.json", {
                "schema_version": 1,
                "suite": "ascend-toolkit",
                "status": "failed",
                "failure_stage": "host-preflight",
                "error": "device inventory mismatch | <unsafe>",
            })

            metadata = MODULE.generate_and_record(root)
            report = (root / "report.md").read_text(encoding="utf-8")

            self.assertEqual(metadata["assets"], [])
            self.assertIn("未形成 Case manifest", report)
            self.assertIn("host-preflight", report)
            self.assertIn("device inventory mismatch \\| <unsafe>", report)

    def test_generator_failure_preserves_experiment_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "summary.json", {
                "schema_version": 1,
                "suite": "workload",
                "status": "passed",
            })

            with self.assertRaises(MODULE.ReportError):
                MODULE.generate_and_record(root)

            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["report_generation"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
