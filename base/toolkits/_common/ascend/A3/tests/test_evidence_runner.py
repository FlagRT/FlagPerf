from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "evidence_runner.py"
SPEC = importlib.util.spec_from_file_location("ascend_a3_evidence", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(name: str) -> str:
    return (HERE / "fixtures" / name).read_text(encoding="utf-8")


class P2PDefaultTests(unittest.TestCase):
    def test_default_pairs_and_sizes_for_selected_and_all_devices(self):
        for selection in (list(range(8, 16)), None):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as tmp:
                with mock.patch.object(MODULE, "discovered_devices", return_value=list(range(8, 16))):
                    runner = MODULE.Runner(Path(tmp), [], False, selected_devices=selection)
                self.assertEqual(runner.latency_sizes, (512, 4096, 65536, 1048576))
                def command(root, directory, argv, **kwargs):
                    src, dst = argv[argv.index("--ds") + 1], argv[argv.index("--dd") + 1]
                    self.assertLess(int(src), int(dst))
                    self.assertEqual(argv[argv.index("-s") + 1], "65536")
                    return {"returncode": 0, "stdout_text": json.dumps({"Latency": [{
                        "src_device_id": src, "dst_device_id": dst,
                        "latency": "500 ns", "size": "65536 Bytes", "type": "Peer to Peer Test"
                    }]}), "stderr_text": ""}
                with mock.patch.object(MODULE, "command_record", side_effect=command) as call:
                    metrics, commands, coverage = runner.run_p2p_latency_sweep(Path(tmp))
                self.assertEqual(call.call_count, 28)
                self.assertEqual(len(metrics), 28)
                self.assertIsNone(coverage["coverage_error"])
                self.assertFalse(coverage["sweep_scope"]["reverse_pairs_inferred"])

    def test_explicit_size_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(MODULE, "discovered_devices", return_value=[8, 9]):
                runner = MODULE.Runner(Path(tmp), [], False, latency_sizes=(512, 4096))
            self.assertEqual(runner.p2p_latency_sizes, (512, 4096))
            self.assertEqual(runner.latency_sizes, (512, 4096))


class ParserTests(unittest.TestCase):
    def test_npu_smi_usages_parses_two_chips_by_labels(self) -> None:
        text = """
NPU ID                         : 1
Chip Count                     : 2
Chip ID                        : 0
Aicore Usage Rate(%)           : 100
Aivector Usage Rate(%)         : 0
HBM Bandwidth Usage Rate(%)    : 27
NPU Utilization(%)             : 100
Chip ID                        : 1
NPU Utilization(%)             : 99
HBM Bandwidth Usage Rate(%)    : 28
Aivector Usage Rate(%)         : 1
Aicore Usage Rate(%)           : 98
"""
        parsed = MODULE.parse_npu_smi_usages(text, 1)
        self.assertEqual([(item["npu_id"], item["chip_id"]) for item in parsed], [(1, 0), (1, 1)])
        self.assertEqual(parsed[0]["values"]["aicore_usage_rate_pct"], 100)
        self.assertEqual(parsed[1]["values"]["hbm_bandwidth_usage_rate_pct"], 28)
        self.assertTrue(all(item["valid"] for item in parsed))

    def test_npu_smi_usages_keeps_missing_field_partial(self) -> None:
        parsed = MODULE.parse_npu_smi_usages(
            "Aicore Usage Rate(%) : 100\nChip ID : 0\n", 3
        )
        self.assertEqual(len(parsed), 1)
        self.assertFalse(parsed[0]["valid"])
        self.assertIn("Aivector Usage Rate(%)", parsed[0]["missing_or_invalid_fields"])

    def test_diagnosis_threshold_gap_is_unsupported(self) -> None:
        text = '{"item_result":"FAIL","item_info":"Read AI flops threshold failed from configuration file!"}'
        self.assertEqual(MODULE.diagnosis_status(text, 0), "unsupported")

    def test_real_diagnosis_failure_is_failed(self) -> None:
        self.assertEqual(MODULE.diagnosis_status('{"item_result":"FAIL"}', 0), "failed")

    def test_health_separates_passed_result_from_lost_card_coverage(self) -> None:
        text = (
            '{"DiagnosisItems":[{"sub_items":['
            '{"item_name":"driver","item_result":"HEALTH"},'
            '{"item_name":"cann","item_result":"PASS"},'
            '{"item_name":"hbm","item_result":"PASS"},'
            '{"item_name":"device","item_result":"HEALTH","item_info":['
            '"Notice: Lost card diagnosis is not support on docker and virtual machine"]}]}]}'
        )
        self.assertTrue(MODULE.diagnosis_passed(text))
        self.assertEqual(MODULE.health_coverage(text), {
            "status": "partial",
            "uncovered": ["lost-card-diagnosis"],
            "reason": "DMI reports lost-card diagnosis unsupported in this context",
        })

    def test_structured_health_without_unsupported_notice_is_complete(self) -> None:
        text = (
            '{"sub_items":['
            '{"item_name":"driver","item_result":"HEALTH"},'
            '{"item_name":"cann","item_result":"PASS"},'
            '{"item_name":"device","item_result":"HEALTH"},'
            '{"item_name":"hbm","item_result":"PASS"}]}'
        )
        self.assertEqual(MODULE.health_coverage(text)["status"], "complete")

    def test_structured_health_missing_requested_item_is_partial(self) -> None:
        text = '{"item_name":"device","item_result":"HEALTH"}'
        coverage = MODULE.health_coverage(text)
        self.assertEqual(coverage["status"], "partial")
        self.assertIn("health-item-driver", coverage["uncovered"])

    def test_fp16_real_26_1_output(self) -> None:
        metrics = MODULE.parse_compute(fixture("compute_fp16_26.1.txt"), "fp16", "TFLOPS")
        self.assertEqual(metrics[0]["value"], 752.465)
        self.assertEqual(metrics[0]["value_raw"], "752.465")
        self.assertEqual(metrics[0]["device"], "0/1")

    def test_compute_preserves_vendor_decimal_lexeme(self) -> None:
        text = '{"computing_power":{"device_id":"2/3","tflops_fp16":730.1680297851563}}'
        metric = MODULE.parse_compute(text, "fp16", "TFLOPS")[0]
        self.assertEqual(metric["value_raw"], "730.1680297851563")
        self.assertEqual(metric["value"], float("730.1680297851563"))

    def test_int8_is_tops(self) -> None:
        text = fixture("compute_int8_26.1.txt")
        self.assertEqual(MODULE.parse_compute(text, "int8", "TOPS")[0]["value"], 1462.402)
        with self.assertRaises(MODULE.EvidenceError):
            MODULE.parse_compute(text, "int8", "TFLOPS")

    def test_compute_ignores_prepended_warning_and_blank_lines(self) -> None:
        text = "warning: layout changed\n\n" + fixture("compute_fp16_26.1.txt")
        self.assertEqual(MODULE.parse_compute(text, "fp16", "TFLOPS")[0]["value"], 752.465)

    def test_bandwidth_uses_labelled_column(self) -> None:
        metrics = MODULE.parse_tabular_bandwidth(fixture("bandwidth_table_26.1.txt"))
        self.assertEqual([item["value"] for item in metrics], [58.125, 57.875])

    def test_bandwidth_json_does_not_parse_elapsed_time(self) -> None:
        text = ('{"bandwidths":[{"device_id":0,"results":['
                '{"bandwidth":"1535.5 GB/s","elapsed_time":"16384 us"}]}]}')
        metrics = MODULE.parse_tabular_bandwidth(text)
        self.assertEqual([item["value"] for item in metrics], [1535.5])

    def test_d2h_bandwidth_keeps_protocol_context(self) -> None:
        runner = object.__new__(MODULE.Runner)
        metrics = runner.parse_case(
            "interconnect-d2h",
            '{"bandwidths":[{"device_id":2,"bandwidth":"42.8 GB/s"}]}',
        )
        self.assertEqual(metrics[0]["direction"], "d2h")
        self.assertEqual(metrics[0]["transfer_size_bytes"], 536870912)
        self.assertEqual(metrics[0]["execute_times"], 50)

    def test_p2p_keeps_direction_and_pairs(self) -> None:
        metrics = MODULE.parse_p2p(fixture("p2p_card_26.1.txt"))
        self.assertEqual(len(metrics), 12)
        self.assertEqual({item["direction"] for item in metrics}, {"unidirectional", "bidirectional"})
        self.assertIn(("0", "1", 391.2), {
            (item["source_device"], item["destination_device"], item["value"])
            for item in metrics
        })

    def test_p2p_fixture_is_a_complete_three_card_matrix(self) -> None:
        runner = MODULE.Runner(Path("unused"), ["interconnect-P2P_intraserver"], False)
        runner.devices = list(range(6))
        runner.validate_coverage(
            "interconnect-P2P_intraserver",
            MODULE.parse_p2p(fixture("p2p_card_26.1.txt")),
        )

    def test_p2p_pair_json_keeps_requested_order_without_reverse(self) -> None:
        text = (
            '{"results":{"unidirectional_bandwidth":"391.2 GB/s",'
            '"bidirectional_bandwidth":"745.5 GB/s"}}'
        )
        metrics = MODULE.parse_p2p_pair(text, 2, 5)
        self.assertEqual({item["direction"] for item in metrics}, {
            "unidirectional", "bidirectional",
        })
        self.assertEqual({
            (item["source_device"], item["destination_device"])
            for item in metrics
        }, {("2", "5")})

    def test_p2p_pair_dmi_26_1_uses_only_largest_transfer_bandwidth(self) -> None:
        text = json.dumps({"bandwidths": [
            {
                "type": "Unidirectional Peer to Peer Test",
                "source_device_id": 2,
                "target_device_id": 3,
                "results": [
                    {"size": "2 Bytes", "bandwidth": "0.0004 GB/s",
                     "elapsed_time": "4.8 us", "execute_times": 5},
                    {"size": "33554432 Bytes", "bandwidth": "198.9 GB/s",
                     "elapsed_time": "168.7 us", "execute_times": 5},
                ],
            },
            {
                "type": "Bidirectional Peer to Peer Test",
                "source_device_id": 2,
                "target_device_id": 3,
                "results": [
                    {"size": "2 Bytes", "bandwidth": "0.0009 GB/s"},
                    {"size": "33554432 Bytes", "bandwidth": "377.6 GB/s"},
                ],
            },
        ]})
        metrics = MODULE.parse_p2p_pair(text, 2, 3)
        self.assertEqual(len(metrics), 2)
        self.assertEqual(
            {item["direction"]: item["value"] for item in metrics},
            {"unidirectional": 198.9, "bidirectional": 377.6},
        )
        self.assertEqual({item["transfer_size_bytes"] for item in metrics}, {33554432})
        self.assertNotIn("elapsed_time", metrics[1])

    def test_capacity_preserves_npu_smi_mb_label(self) -> None:
        metric = MODULE.parse_capacity("HBM Capacity(MB) : 65536", 0, 0)
        self.assertEqual(metric["unit"], "MB")
        self.assertEqual(metric["value_raw"], "65536")
        self.assertEqual(metric["scope"], "chip")
        self.assertEqual(metric["source_field"], "HBM Capacity(MB)")

    def test_json_field_order_does_not_matter(self) -> None:
        text = '{"results":[{"TFLOPS@FP32":85.5,"device_id":"all"}]}'
        metrics = MODULE.parse_compute(text, "fp32", "TFLOPS")
        self.assertEqual(metrics[0]["value"], 85.5)

    def test_compute_parses_dmi_26_1_json_shape(self) -> None:
        text = (
            '{"computing_power":{"device_id":"all",'
            '"tops_int8":11679.2646484375}}'
        )
        metrics = MODULE.parse_compute(text, "int8", "TOPS")
        self.assertEqual(metrics[0]["device"], "all")
        self.assertEqual(metrics[0]["unit"], "TOPS")
        self.assertEqual(metrics[0]["value"], 11679.2646484375)

    def test_compute_context_preserves_vendor_fields_without_derivation(self) -> None:
        contexts = MODULE.parse_compute_context(
            '{"computing_power":{"device_id":"all","execute_times":432000000,'
            '"duration":"1703 ms","power":"432.413 W","tflops_fp16":6019.7}}'
        )
        self.assertEqual(contexts[0]["execute_times"], 432000000)
        self.assertEqual(contexts[0]["duration"], "1703 ms")
        self.assertEqual(contexts[0]["power"], "432.413 W")

    def test_malformed_and_non_finite_values_fail(self) -> None:
        for text in ("", "Bandwidth(GB/s)\n0 nan", "TOPS@INT8\n0 1 2 nan"):
            with self.assertRaises(MODULE.EvidenceError):
                if "Bandwidth" in text:
                    MODULE.parse_tabular_bandwidth(text)
                else:
                    MODULE.parse_compute(text, "int8", "TOPS")

    def test_latency_requires_explicit_ns_and_keeps_context(self) -> None:
        metrics = MODULE.parse_latency(
            '{"results":{"latency_ns":"123.5 ns"}}', direction="d2h",
            size_bytes=4096, device=2,
        )
        self.assertEqual(metrics, [{
            "value": 123.5, "unit": "ns", "size_bytes": 4096,
            "direction": "d2h", "source": "json",
            "field": "results/latency_ns", "device": "2",
        }])
        with self.assertRaises(MODULE.EvidenceError):
            MODULE.parse_latency('{"latency": 3.0}', direction="h2d", size_bytes=512)

    def test_latency_surfaces_structured_dmi_error_even_with_zero_rc(self) -> None:
        with self.assertRaisesRegex(MODULE.EvidenceError, "DMI error 2: internal error"):
            MODULE.parse_latency(
                '{"error":{"code":2,"description":"internal error"}}',
                direction="p2p", size_bytes=512, source=2, destination=3,
            )

    def test_hccl_parser_keeps_time_bandwidth_and_correctness(self) -> None:
        text = (
            "data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:\n"
            "8192 | 12.50 | 18.25 | success\n"
            "16384 | 13.00 | 31.50 | failed\n"
        )
        metrics = MODULE.parse_hccl_allreduce(text)
        self.assertEqual(metrics[0]["message_size_bytes"], 8192)
        self.assertEqual(metrics[0]["avg_time_us"], 12.5)
        self.assertEqual(metrics[0]["value"], 18.25)
        self.assertTrue(metrics[0]["verification_passed"])
        self.assertFalse(metrics[1]["verification_passed"])

    def test_byte_size_protocol(self) -> None:
        self.assertEqual(MODULE.parse_byte_size("8K"), 8192)
        self.assertEqual(MODULE.parse_latency_sizes("512,4K,1M"), (512, 4096, 1048576))
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.parse_latency_sizes("512,512")


class ArtifactTests(unittest.TestCase):
    def test_compute_command_uses_extended_single_workload(self) -> None:
        runner = MODULE.Runner(Path("unused"), ["computation-FP16"], False)
        command = runner.optimized_command("computation-FP16")
        self.assertEqual(command[command.index("--et") + 1], "80")

    def test_command_record_preserves_streams_and_returncode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = MODULE.command_record(
                root, root / "case",
                ["/bin/sh", "-c", "echo stdout; echo stderr >&2; exit 7"],
                timeout=5, label="probe", required=False,
            )
            self.assertEqual(record["returncode"], 7)
            self.assertEqual((root / record["stdout"]["path"]).read_text().strip(), "stdout")
            self.assertEqual((root / record["stderr"]["path"]).read_text().strip(), "stderr")
            self.assertEqual(len(record["stdout"]["sha256"]), 64)

    def test_fixed_position_probe_documents_layout_dependency(self) -> None:
        text = "warning inserted\n" + fixture("compute_int8_26.1.txt")
        self.assertNotEqual(
            MODULE.legacy_fixed_value("computation-INT8", text), "1462.402"
        )

    def test_capacity_fixed_value_no_longer_multiplies_or_relabels(self) -> None:
        self.assertEqual(
            MODULE.legacy_fixed_value(
                "main_memory-capacity", "HBM Capacity(MB) : 65536"
            ),
            "65536",
        )

    def test_capacity_keeps_each_chip_raw_and_records_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = MODULE.Runner(root, ["main_memory-capacity"], False)
            runner.devices = [0, 1]
            runner.device_map = [
                {"npu_id": 0, "chip_id": 0, "logic_id": 0},
                {"npu_id": 0, "chip_id": 1, "logic_id": 1},
            ]

            def record(_root, _directory, command, **_kwargs):
                chip = int(command[command.index("-c") + 1])
                return {
                    "command": command, "returncode": 0, "timed_out": False,
                    "stdout": {"path": f"chip-{chip}.stdout"},
                    "stderr": {"path": f"chip-{chip}.stderr"},
                    "stdout_text": "HBM Capacity(MB) : 65536\n", "stderr_text": "",
                }

            with mock.patch.object(MODULE, "command_record", side_effect=record):
                metrics, extra = runner.run_capacity(root)

        self.assertEqual([item["value"] for item in metrics], [65536.0, 65536.0])
        self.assertEqual([item["value_raw"] for item in metrics], ["65536", "65536"])
        self.assertEqual([item["scope"] for item in metrics], ["chip", "chip"])
        self.assertEqual([item["unit"] for item in metrics], ["MB", "MB"])
        self.assertNotIn("flagperf_result", extra)
        self.assertEqual(extra["capacity_scope"], {
            "metric_scope": "chip", "aggregation": "none", "unit": "MB",
            "target_count": 2,
            "targets": [
                {"card": 0, "chip": 0, "device": "0"},
                {"card": 0, "chip": 1, "device": "1"},
            ],
        })

    def test_emit_results_keeps_raw_d2d_values_without_multiplier(self) -> None:
        runner = object.__new__(MODULE.Runner)
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            runner.emit_results("main_memory-bandwidth", [{
                "device": "0", "scope": "logical-device",
                "value": 1539.627197, "value_raw": "1539.627197", "unit": "GB/s",
            }], {})
        self.assertEqual(
            output.getvalue().strip(),
            "[FlagPerf Result] main_memory-bandwidth[0]=1539.627197 GB/s",
        )
        self.assertNotIn("3079", output.getvalue())

    def test_default_d2d_case_records_raw_logical_device_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = object.__new__(MODULE.Runner)
            runner.root = root
            runner.legacy_probe = False
            runner.explicit_selection = False
            runner.compute_monitor = False
            runner.devices = [0, 1]
            runner.manifest = {"cases": {}}
            runner.save = lambda: None
            runner.emit_results = lambda case, metrics, extra: None
            metrics = [
                {"device": "0", "value": 1539.627197, "unit": "GB/s"},
                {"device": "1", "value": 1536.5, "unit": "GB/s"},
            ]
            with mock.patch.object(
                runner, "run_bandwidth_with_fallback",
                return_value=(
                    metrics, [{"command": ["ascend-dmi"]}],
                    {"status": "not-run", "reasons": []},
                ),
            ):
                runner.run_case("main_memory-bandwidth")

        result = runner.manifest["cases"]["main_memory-bandwidth"]
        self.assertEqual(result["measurement_status"], "passed")
        self.assertEqual(
            [item["scope"] for item in result["metrics"]],
            ["logical-device", "logical-device"],
        )
        self.assertEqual(result["bandwidth_scope"], {
            "metric_scope": "logical-device", "aggregation": "none",
            "unit": "GB/s", "source": "ascend-dmi",
            "selected_device_ids": [0, 1],
        })
        self.assertNotIn("flagperf_result", result)

    def test_case_preserves_measurement_status_before_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = object.__new__(MODULE.Runner)
            runner.root = root
            runner.legacy_probe = False
            runner.explicit_selection = False
            runner.compute_monitor = False
            runner.devices = [0, 1]
            runner.device_map = []
            runner.manifest = {"cases": {}}
            runner.save = lambda: None
            runner.optimized_command = lambda case: ["ascend-dmi"]
            runner.parse_case = lambda case, text: [
                {"value": 1.0, "unit": "TFLOPS", "device": "all"}
            ]
            runner.validate_coverage = lambda case, metrics: None
            runner.emit_results = lambda case, metrics, extra: None
            record = {
                "command": ["ascend-dmi"],
                "returncode": 0,
                "timed_out": False,
                "stdout": {"path": "stdout"},
                "stderr": {"path": "stderr"},
                "stdout_text": "valid",
                "stderr_text": "",
            }

            with mock.patch.object(MODULE, "command_record", return_value=record):
                runner.run_case("computation-FP16")

            result = runner.manifest["cases"]["computation-FP16"]
            self.assertEqual(result["measurement_status"], "passed")
            self.assertEqual(result["status"], "passed")
            archived = json.loads(
                (root / "cases" / "computation-FP16" / "metrics.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(archived["measurement_status"], "passed")

    def test_explicit_commands_target_only_selected_device(self) -> None:
        runner = MODULE.Runner(
            Path("unused"), ["computation-FP16"], False,
            selected_devices=[2, 3], selection_source="npu-ids",
        )
        for case in ("computation-FP16", "main_memory-bandwidth", "interconnect-h2d", "interconnect-d2h"):
            for device in (2, 3):
                command = runner.device_command(case, device)
                self.assertIn("-d", command)
                self.assertEqual(command[command.index("-d") + 1], str(device))
                self.assertNotIn("--all", command)
        diagnosis = MODULE.diagnosis_command("bandwidth", [2, 3])
        self.assertEqual(diagnosis[diagnosis.index("-d") + 1], "2,3")

    def test_compute_runs_once_per_complete_physical_npu_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = MODULE.Runner(
                root, ["computation-FP16"], False,
                selected_devices=[2, 3], selection_source="npu-ids",
                compute_monitor=False,
            )
            runner.device_map = [
                {"npu_id": 1, "chip_id": 0, "logic_id": 2, "phy_id": 2},
                {"npu_id": 1, "chip_id": 1, "logic_id": 3, "phy_id": 3},
            ]
            record = {
                "command": ["ascend-dmi"], "returncode": 0, "timed_out": False,
                "stdout": {"path": "stdout"}, "stderr": {"path": "stderr"},
                "stdout_text": "valid", "stderr_text": "",
            }
            with mock.patch.object(MODULE, "command_record", return_value=record), \
                    mock.patch.object(runner, "parse_case", return_value=[{
                        "device": "2/3", "value": 752.0, "unit": "TFLOPS",
                    }]):
                metrics, commands, monitor = runner.run_compute_groups("computation-FP16", root)
        self.assertEqual(len(commands), 1)
        self.assertEqual(metrics[0]["device"], "2/3")
        self.assertEqual(metrics[0]["selected_device_ids"], [2, 3])
        self.assertEqual(metrics[0]["npu_id"], 1)
        self.assertEqual(monitor["status"], "not-run")

    def test_compute_rejects_half_physical_npu_without_running_command(self) -> None:
        runner = MODULE.Runner(
            Path("unused"), ["computation-FP16"], False,
            selected_devices=[2], selection_source="device-ids",
            compute_monitor=False,
        )
        runner.device_map = [
            {"npu_id": 1, "chip_id": 0, "logic_id": 2, "phy_id": 2},
            {"npu_id": 1, "chip_id": 1, "logic_id": 3, "phy_id": 3},
        ]
        with mock.patch.object(MODULE, "command_record") as command:
            with self.assertRaisesRegex(MODULE.PartialEvidence, "incomplete"):
                runner.run_compute_groups("computation-FP16", Path("unused"))
        command.assert_not_called()

    def test_compute_monitor_adds_one_unaggregated_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = MODULE.Runner(root, ["computation-FP16"], False)
            runner.device_map = [
                {"npu_id": 1, "chip_id": 0, "logic_id": 2, "phy_id": 2},
                {"npu_id": 1, "chip_id": 1, "logic_id": 3, "phy_id": 3},
            ]
            record = {
                "command": ["ascend-dmi"], "returncode": 0, "timed_out": False,
                "started_at": "2026-08-25T00:00:00Z",
                "finished_at": "2026-08-25T00:00:01Z",
                "started_monotonic_s": 1.0, "finished_monotonic_s": 2.0,
                "stdout": {"path": "stdout"}, "stderr": {"path": "stderr"},
                "stdout_text": '{"computing_power":{"device_id":"2/3","tflops_fp16":752.0}}',
                "stderr_text": "",
            }

            class FakeMonitor:
                def __init__(self, targets):
                    self.targets = targets

                def start(self):
                    return None

                def relative_window(self, _record, role):
                    return {"role": role, "started_offset_s": 0, "finished_offset_s": 1}

                def valid_counts(self, _windows):
                    return {"1/0/2": 5, "1/1/3": 5}

                def finish(self, _root, _directory, windows, extensions):
                    return {"status": "passed", "workload_windows": windows,
                            "extensions": extensions, "reasons": []}

            with mock.patch.object(MODULE, "UsageMonitor", FakeMonitor), \
                    mock.patch.object(MODULE, "command_record", side_effect=[record, record]) as command:
                metrics, primary, monitor = runner.run_compute_unit(
                    "computation-FP16", root / "primary", root / "monitor",
                    ["ascend-dmi"], [2, 3],
                )

        self.assertEqual(command.call_count, 2)
        self.assertEqual(metrics[0]["value"], 752.0)
        self.assertEqual(primary["stdout_text"], record["stdout_text"])
        self.assertEqual(monitor["extensions"][0]["raw_dmi_metrics"][0]["value"], 752.0)

    def test_p2p_explicit_selection_runs_each_unordered_pair_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = MODULE.Runner(
                root, ["interconnect-P2P_intraserver"], False,
                selected_devices=[2, 3, 4], selection_source="device-ids",
                data_movement_monitor=False,
            )
            calls = []

            def record(_root, _directory, command, **_kwargs):
                calls.append(command)
                return {
                    "command": command, "returncode": 0, "timed_out": False,
                    "stdout": {"path": "stdout"}, "stderr": {"path": "stderr"},
                    "stdout_text": "valid", "stderr_text": "",
                }

            with mock.patch.object(MODULE, "command_record", side_effect=record), \
                    mock.patch.object(MODULE, "parse_p2p_pair", return_value=[{
                        "value": 1.0, "unit": "GB/s", "direction": "pair",
                        "source_device": "2", "destination_device": "3",
                    }]):
                _metrics, _commands, scope, monitor = runner.run_p2p_combinations(root)

        pairs = [(cmd[cmd.index("--ds") + 1], cmd[cmd.index("--dd") + 1]) for cmd in calls]
        self.assertEqual(pairs, [("2", "3"), ("2", "4"), ("3", "4")])
        self.assertEqual(scope["expected_pair_count"], 3)
        self.assertFalse(scope["reverse_pairs_inferred"])
        self.assertEqual(monitor["status"], "not-run")

    def test_hccs_bandwidth_parser_preserves_zero_and_raw_lexemes(self) -> None:
        parsed = MODULE.parse_npu_smi_hccs_bw(
            "hccs link  rx_bandwidth(GB/S)  tx_bandwidth(GB/S)\n"
            "0  12.500  0.000\n"
            "total  12.500  0.000\n",
            1, 0,
        )
        self.assertEqual([item["link"] for item in parsed], ["0", "total"])
        self.assertEqual(parsed[0]["source_fields"]["rx_bandwidth(GB/S)"], "12.500")
        self.assertEqual(parsed[0]["values"]["tx_gb_s"], 0.0)

    def test_topology_parser_preserves_sio_and_hccs_switch_routes(self) -> None:
        routes = MODULE.parse_npu_smi_topology(
            "Phy-ID0 Phy-ID1 Phy-ID2\n"
            "Phy-ID0 X SIO HCCS_SW\n"
            "Phy-ID1 SIO X HCCS\n"
            "Phy-ID2 HCCS_SW HCCS X\n"
        )
        self.assertIn({"source_device": 0, "destination_device": 1, "relation": "SIO"}, routes)
        self.assertIn({"source_device": 0, "destination_device": 2, "relation": "HCCS_SW"}, routes)

    def test_hccs_monitor_counts_commands_not_link_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = {"npu_id": 1, "chip_id": 0, "logic_id": 2}
            monitor = MODULE.HccsBandwidthMonitor([target])
            monitor.stop = lambda: None
            for sequence in range(10):
                monitor.raw_records.append({
                    "sequence": sequence, **target, "returncode": 0,
                    "started_offset_s": float(sequence),
                    "stdout": (
                        "hccs link rx_bandwidth(GB/S) tx_bandwidth(GB/S)\n"
                        "0 1.0 2.0\ntotal 1.0 2.0\n"
                    ),
                })
                for link in ("0", "total"):
                    monitor.samples.append({
                        "sequence": sequence, **target,
                        "logical_device_id": 2, "link": link, "valid": True,
                        "started_offset_s": float(sequence),
                        "finished_offset_s": float(sequence) + 0.5,
                        "values": {"rx_gb_s": 1.0, "tx_gb_s": 2.0},
                        "source_fields": {
                            "rx_bandwidth(GB/S)": "1.0",
                            "tx_bandwidth(GB/S)": "2.0",
                        },
                    })
            windows = [{"role": "primary", "started_offset_s": 0.0, "finished_offset_s": 10.0}]
            result = monitor.finish(root, root / "monitor", windows, [])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["sample_counts_by_target"]["1/0/2"], 10)

    def test_sio_route_forces_monitor_partial_without_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = object.__new__(MODULE.Runner)
            runner.root = root
            runner.topology_routes = [{
                "source_device": 2, "destination_device": 3, "relation": "SIO",
            }]
            result = runner.aggregate_data_movement_units(
                root, [{"status": "passed", "reasons": []}], [2, 3]
            )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["route_coverage"]["sio_count"], 1)
        self.assertIn("no dynamic SIO", result["reasons"][0])

    def test_single_selected_device_makes_p2p_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = MODULE.Runner(
                Path(temporary), ["interconnect-P2P_intraserver"], False,
                selected_devices=[2], selection_source="device-ids",
            )
            runner.save = lambda: None
            runner.run_case("interconnect-P2P_intraserver")
            result = runner.manifest["cases"]["interconnect-P2P_intraserver"]
        self.assertEqual(result["status"], "partial")
        self.assertIn("at least two", result["error"])


if __name__ == "__main__":
    unittest.main()
