"""PPU Toolkit contracts and failure paths, independent of accelerator availability."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))
from executors.toolkit import ToolkitExecutor, ToolkitRunRequest, parse_args
from toolkits._common.thead.PPU import contract as c
from toolkits._common.thead.PPU import evidence as e
from toolkits._common.thead.PPU import evidence_runner as er
from toolkits._common.thead.PPU.report import generate_and_record
from executors.thead_toolkit import compose_status, finalize_monitor, settings_for

# Captured stdout fixtures (probes/day9/toolkit t9/t9b evidence, cards 14/15).
# Byte-faithful to t9b_p2p_confirmed.txt: the tool pads every matrix row with
# trailing spaces, and the column-0 anchored parser keys on that exact layout.
SAIL_STDOUT = """PPU SDK banner lines precede the JSON document.
{
\t"sailbandwidth" :
\t{
\t\t"GPU Device list" :
\t\t[
\t\t\t"0: PPU-ZW810E (00000000:a3:00): (135c7a8580c7)",
\t\t\t"1: PPU-ZW810E (00000000:a2:00): (135c7a8580c7)"
\t\t],
\t\t"testcases" :
\t\t[
\t\t\t{
\t\t\t\t"bandwidth_matrix" :
\t\t\t\t[
\t\t\t\t\t[
\t\t\t\t\t\t"29.0122",
\t\t\t\t\t\t"29.0079"
\t\t\t\t\t]
\t\t\t\t],
\t\t\t\t"name" : "host_to_device_memcpy_ce",
\t\t\t\t"status" : "Passed",
\t\t\t\t"sum" : 58.020153344000001
\t\t\t}
\t\t]
\t}
}
"""
P2P_HEAD = """Host 0 Proc 0 on 5fe37f71456c, 2 devices:
 Dev0, pciBus: 0000:A3:00.0, pciDeviceID: 0, pciDomainID: 0
 Dev1, pciBus: 0000:A2:00.0, pciDeviceID: 0, pciDomainID: 0
"""
UNI_WRITE_OUT = P2P_HEAD + """(EnabledP2P CE WRITE Unidirectional) P2P Transport Bandwidth Matrix (40000000 Bytes) (GB/s)
          DEV0      DEV1      
DEV0       x        46.00     
DEV1      46.73      x        
Total:   0.71 s
P2P test finished, data matched.
"""
UNI_READ_OUT = P2P_HEAD + """(EnabledP2P CE READ Unidirectional) P2P Transport Bandwidth Matrix (40000000 Bytes) (GB/s)
          DEV0      DEV1      
DEV0       x        47.37     
DEV1      45.19      x        
Total:   0.70 s
P2P test finished, data matched.
"""
LATENCY_OUT = P2P_HEAD + """(EnabledP2P SM WRITE) GPU P2P Transmission Latency Matrix (US)
          DEV0      DEV1      
DEV0      0.00      5.28      
DEV1      5.39      0.00      
(EnabledP2P SM WRITE) CPU P2P Transmission Latency Matrix (US)
          DEV0      DEV1      
DEV0      0.00      8.14      
DEV1      8.12      0.00      
Total:   0.52 s
P2P test finished.
"""
BIBW_OUT = P2P_HEAD + """(EnabledP2P CE WRITE Bidirectional) P2P Transport Bandwidth Matrix (40000000 *2 Bytes) (GB/s)
          DEV0      DEV1      
DEV0       x        60.95     
DEV1      88.03      x        
Total:   0.83 s
Host 0 Proc 0 on f8387f457991, 2 devices:
 Dev0, pciBus: 0000:A3:00.0, pciDeviceID: 0, pciDomainID: 0
 Dev1, pciBus: 0000:A2:00.0, pciDeviceID: 0, pciDomainID: 0
(EnabledP2P CE WRITE Bidirectional) P2P Transport Bandwidth Matrix (40000000 *2 Bytes) (GB/s)
          DEV0      DEV1      
DEV0       x        60.79     
DEV1      88.11      x        
Total:   0.84 s
P2P test finished, data matched.
"""


class ContractTests(unittest.TestCase):
    def test_twelve_cases_and_locked_sail_tables(self):
        self.assertEqual(len(c.CASES), 12)
        for (case, mode), (index, name, unit) in c.SAIL_EXPECTED_NAMES.items():
            self.assertEqual(c.SAIL_TESTCASES[case][mode], index)
            self.assertIn(unit, ("GB/s", "ns"))
        self.assertEqual(c.SAIL_EXPECTED_NAMES[("interconnect-h2d", "ce")][1], "host_to_device_memcpy_ce")

    def test_every_point_mode_is_routable(self):
        routed = {"gemm", "capacity-query", "uni-write", "uni-read", "bi-write", "latency"}
        routed.update(c.SAIL_ROUTED_MODES)
        for case in c.CASES:
            for point in c.case_points(case, [0, 4, 14, 15]):
                self.assertIn(point["mode"], routed, case)
        self.assertEqual({p["mode"] for p in c.case_points("interconnect-h2d", [14])}, {"ce", "sm"})
        self.assertEqual({p["mode"] for p in c.case_points("interconnect-d2h", [14])}, {"ce", "sm"})
        self.assertEqual({p["mode"] for p in c.case_points("main_memory-bandwidth", [14])}, {"d2d-local"})
        self.assertEqual({p["mode"] for p in c.case_points("interconnect-host_device_latency", [14])}, {"access"})

    def test_point_cardinality(self):
        self.assertEqual(len(c.case_points("interconnect-h2d", [0, 4])), 4)
        self.assertEqual({p["mode"] for p in c.case_points("interconnect-h2d", [0, 4])}, {"ce", "sm"})
        self.assertEqual(len(c.case_points("interconnect-d2h", [0, 4])), 4)
        self.assertEqual(len(c.case_points("main_memory-bandwidth", [0, 4])), 2)
        self.assertEqual(len(c.case_points("main_memory-capacity", [0, 4])), 2)
        self.assertEqual(len(c.case_points("interconnect-host_device_latency", [0, 4])), 2)
        self.assertEqual(len(c.case_points("computation-FP32", [0, 4])), 2)

    def test_p2p_points_are_ordered_and_directed(self):
        points = c.case_points("interconnect-P2P_intraserver", [6, 0, 4, 5])
        self.assertEqual(len(points), 36)
        for pair in ((0, 4), (4, 0)):
            modes = {p["mode"] for p in points if (p["source"], p["destination"]) == pair}
            self.assertEqual(modes, {"uni-write", "uni-read", "bi-write"})
        self.assertEqual(len(c.case_points("interconnect-P2P_intraserver", [4])), 0)
        self.assertEqual(len(c.case_points("interconnect-P2P_intraserver-latency", [6, 0, 4, 5])), 12)
        self.assertEqual(len(c.case_points("interconnect-P2P_intraserver-latency", [4])), 0)

    def test_target_names_unique_within_case(self):
        for case in c.CASES:
            names = [c.target_name(p, r) for p in c.case_points(case, [0, 4, 14, 15]) for r in (1, 2)]
            self.assertEqual(len(names), len(set(names)), case)

    def test_smoke_scope_is_bounded(self):
        options = settings_for(SimpleNamespace(thead_options={"smoke": True}))
        self.assertEqual((options["payload_mib"], options["matrix_size"],
                          options["compute_samples"], options["transfer_samples"],
                          options["p2p_samples"]), (64, 256, 25, 5, 5))
        self.assertEqual(options["minimum_seconds"], 0.0)

    def test_invalid_selection(self):
        for ids in ([4, 4], [], [-1], [16], [0, "4"]):
            with self.assertRaises(ValueError):
                c.validate_selection(ids)
        c.validate_selection([0, 15])
        c.validate_selection([14])

    def test_tool_bdf_joins_both_tool_forms(self):
        self.assertEqual(er.tool_bdf("00000000:a3:00"), "00000000:a3:00.0")
        self.assertEqual(er.tool_bdf("0000:A3:00.0"), "00000000:a3:00.0")
        for bad in ("garbage", "0000:a3:00.9", "0000:a3:gg.0", ""):
            with self.assertRaises(ValueError):
                er.tool_bdf(bad)

    def test_sail_fixture_replay(self):
        payload = er.parse_json_stdout(SAIL_STDOUT, "sailbandwidth")
        observed = [er.tool_bdf(re.search(r"\(([0-9A-Fa-f:.]+)\)", entry)[1])
                    for entry in payload["GPU Device list"]]
        self.assertEqual(observed, ["00000000:a3:00.0", "00000000:a2:00.0"])
        testcase = payload["testcases"][0]
        self.assertEqual(testcase["name"], c.SAIL_EXPECTED_NAMES[("interconnect-h2d", "ce")][1])
        self.assertEqual(testcase["status"], "Passed")
        self.assertEqual(float(testcase["bandwidth_matrix"][0][0]), 29.0122)

    def test_p2p_fixture_replay_and_direction_gate(self):
        dev = er.parse_dev_bdfs(UNI_WRITE_OUT)
        self.assertEqual(dev, {0: "00000000:a3:00.0", 1: "00000000:a2:00.0"})
        matrices = er.parse_matrix_sections(UNI_WRITE_OUT, "Unidirectional) P2P Transport Bandwidth Matrix", "(GB/s)", 2)
        self.assertEqual(len(matrices), 1)
        self.assertEqual(matrices[0]["values"], [[None, 46.0], [46.73, None]])
        er.expected_bytes_token(matrices[0]["header"], 40000000, False)
        with self.assertRaises(ValueError):
            er.expected_bytes_token(matrices[0]["header"], 40000000, True)
        wanted = "WRITE"
        self.assertIn(wanted, matrices[0]["header"])
        read = er.parse_matrix_sections(UNI_READ_OUT, "Unidirectional) P2P Transport Bandwidth Matrix", "(GB/s)", 2)
        self.assertEqual(read[0]["values"], [[None, 47.37], [45.19, None]])
        self.assertNotIn("WRITE", read[0]["header"])

    def test_latency_fixture_excludes_cpu_matrix(self):
        matrices = er.parse_matrix_sections(LATENCY_OUT, "GPU P2P Transmission Latency Matrix", "(US)", 2)
        self.assertEqual(len(matrices), 1)
        self.assertEqual(matrices[0]["values"], [[None, 5.28], [5.39, None]])

    def test_bibw_fixture_keeps_both_rank_matrices(self):
        matrices = er.parse_matrix_sections(BIBW_OUT, "Bidirectional) P2P Transport Bandwidth Matrix", "(GB/s)", 2)
        self.assertEqual(len(matrices), 2)
        self.assertEqual(matrices[0]["values"], [[None, 60.95], [88.03, None]])
        self.assertEqual(matrices[1]["values"], [[None, 60.79], [88.11, None]])
        for matrix in matrices:
            er.expected_bytes_token(matrix["header"], 40000000, True)

    def test_tflops_formula_and_sample_cv(self):
        record = {"samples_ns": [1000000, 1000000, 1000000, 1000000, 2000000]}
        metric = c.metric_from_record(record, "computation-FP32", 64)
        self.assertAlmostEqual(metric["value"], 2.0 * 64 ** 3 / 1000000 / 1000.0)
        self.assertEqual(metric["unit"], "TFLOPS")
        self.assertEqual(metric["source_kind"], "acblas-acblasSgemm_v2")
        self.assertAlmostEqual(metric["sample_cv"], 0.2 ** 0.5 / 1.2, places=12)

    def test_tool_metric_rows_stay_compact(self):
        record = {"tool": "p2pBandwidthLatencyTest", "value": 46.0, "verification": "tool-verified",
                  "matrices": [{"values": [[None, 46.0]]}]}
        metric = c.metric_from_record(record, "interconnect-P2P_intraserver", 64)
        self.assertEqual(metric["unit"], "GB/s")
        self.assertEqual(metric["source_kind"], "official-tool")
        self.assertNotIn("matrices", metric)

    def tool_sample(self):
        return {"schema_version": 1, "point": {"mode": "uni-write", "bytes": 40000000, "source": 14, "destination": 15},
                "repetition": 1, "tool": "p2pBandwidthLatencyTest", "tool_status": "Passed",
                "started_monotonic_ns": 1000, "finished_monotonic_ns": 2000,
                "value": 46.0, "verification": "tool-verified", "identity": {"bdf_matches": True}}

    def test_tool_record_gates(self):
        point, record = self.tool_sample()["point"], self.tool_sample()
        c.validate_tool_record(record, point, 5)
        for changes in ({"schema_version": 2}, {"tool_status": "Failed"}, {"repetition": 0}, {"repetition": 6},
                        {"started_monotonic_ns": 0}, {"finished_monotonic_ns": 1000},
                        {"value": 0}, {"value": float("nan")}, {"verification": "guessed"},
                        {"identity": {"bdf_matches": False}}):
            with self.assertRaises(ValueError):
                c.validate_tool_record({**record, **changes}, point, 5)
        with self.assertRaises(ValueError):
            c.validate_tool_record(record, {**point, "mode": "uni-read"}, 5)

    def compute_sample(self):
        return {"schema_version": 1, "mode": "gemm", "dtype": "FP32", "matrix_size": 256,
                "visible_devices": 1, "uuid": "GPU-fixture", "correctness": {"passed": True},
                "samples_ns": [100, 200, 300], "started_monotonic_ns": 1000, "finished_monotonic_ns": 2000}

    def test_compute_record_gates(self):
        point = {"mode": "gemm", "dtype": "FP32", "source": 14}
        record = self.compute_sample()
        c.validate_compute_record(record, point, 256, 3)
        for changes in ({"schema_version": 2}, {"dtype": "FP16"}, {"matrix_size": 128},
                        {"visible_devices": 2}, {"uuid": ""}, {"correctness": {"passed": False}},
                        {"samples_ns": [100, 200]}, {"samples_ns": [100, 200, float("nan")]},
                        {"samples_ns": [100, 200, 0]}, {"started_monotonic_ns": 0},
                        {"finished_monotonic_ns": 1000}):
            with self.assertRaises(ValueError):
                c.validate_compute_record({**record, **changes}, point, 256, 3)

    def test_int8_excluded_and_reasons_recorded(self):
        excluded = c.contract_record({})["excluded_capabilities"]
        self.assertIn("computation-INT8", excluded)
        self.assertIn("interconnect-MPI_intraserver", excluded)
        self.assertNotIn("computation-INT8", c.COMPUTE_CASES)
        self.assertTrue(all(case in c.CASES for case in c.COMPUTE_CASES))

    def test_failure_propagates(self):
        self.assertEqual(c.layer_status(["passed", "failed"]), "failed")
        self.assertEqual(c.layer_status(["passed", "not-run"]), "partial")
        self.assertEqual(c.layer_status([]), "not-run")


class FacadeTests(unittest.TestCase):
    def request(self, *flags):
        return ToolkitRunRequest.from_namespace(parse_args(
            ["--config", str(BASE / "configs/thead_ppu_torch210.yaml"), *flags]))

    def test_static_plan_never_calls_subprocess(self):
        with patch("subprocess.run", side_effect=AssertionError("dry-run touched system")):
            plan = ToolkitExecutor().plan(self.request("--physical-device-ids", "14,15", "--dry-run"))
        self.assertEqual(plan["suite"], "thead-ppu-toolkit")
        self.assertEqual(plan["vendor"], "thead")
        self.assertEqual(plan["permissions"]["network"], "none")
        self.assertEqual(plan["logical_point_counts"]["interconnect-P2P_intraserver"], 6)
        self.assertEqual(plan["repeat_count"], 5)
        self.assertEqual(plan["settings"]["payload_mib"], 512)
        self.assertEqual(plan["settings"]["p2p_bytes"], 40000000)
        self.assertEqual(plan["point_counts"]["interconnect-P2P_intraserver"], 30)
        self.assertIn("computation-INT8", " ".join(plan["excluded_capabilities"]))
        expanded = ToolkitExecutor().plan(self.request(
            "--physical-device-ids", "14,15", "--payload-mib", "64",
            "--p2p-bytes", "80000000", "--repeat", "2", "--dry-run"))
        self.assertEqual(expanded["settings"]["payload_mib"], 64)
        self.assertEqual(expanded["settings"]["p2p_bytes"], 80000000)
        self.assertEqual(expanded["point_counts"]["interconnect-P2P_intraserver"], 12)
        self.assertEqual(expanded["logical_point_counts"]["interconnect-h2d"], 4)

    def test_candidate_runtime_is_explicit(self):
        from executors.common import ConfigurationError
        plan = ToolkitExecutor().plan(self.request("--physical-device-ids", "14,15", "--dry-run"))
        self.assertIs(plan["runtime_lock"]["image_manifest"]["validated"], False)
        request = self.request("--physical-device-ids", "14", "--result-root", "unused")
        with patch.object(Path, "mkdir", side_effect=AssertionError("write before candidate gate")):
            with self.assertRaisesRegex(ConfigurationError, "candidate"):
                ToolkitExecutor().execute(request)

    def test_image_fixture_failure_keeps_summary_consistent(self):
        from executors import thead_toolkit as executor
        with tempfile.TemporaryDirectory() as temp:
            request = self.request("--physical-device-ids", "14", "--result-root", temp,
                                   "--allow-candidate-runtime")
            plan = ToolkitExecutor().plan(request)
            commands = Mock()
            commands.checked.side_effect = RuntimeError("fixture image inspection failed")
            container = Mock()
            container.cleanup.return_value = {"status": "passed", "container_absent": True}
            with patch.object(executor, "HostCommands", return_value=commands), \
                 patch.object(executor, "ManagedContainer", return_value=container), \
                 patch("subprocess.run", side_effect=AssertionError("hardware access forbidden")):
                self.assertEqual(executor.execute(request, plan), 1)
            summary = json.loads(next(Path(temp).glob("*/summary.json")).read_text())
            self.assertEqual(summary["release_stage"], "candidate")
            self.assertIs(summary["validated"], False)
            self.assertEqual(summary["qualification_status"], "not-qualified")
            self.assertEqual(summary["measurement_status"], "not-run")
            self.assertEqual(summary["status"], "failed")
            self.assertFalse(summary["lease_released"])
            commands.checked.assert_called_once()
            container.create.assert_not_called()

    def test_aliases_and_foreign_flags_rejected(self):
        for flags in (("--npu-ids", "14"),
                      ("--device-ids", "4"),
                      ("--physical-device-ids", "14", "--allow-disruptive-dmi"),
                      ("--physical-device-ids", "14", "--allow-privileged-root"),
                      ("--physical-device-ids", "14", "--legacy-probe"),
                      ("--physical-device-ids", "14", "--case", "not-real"),
                      ("--physical-device-ids", "14", "--compute-samples", "-1"),
                      ("--physical-device-ids", "14", "--p2p-bytes", "40000001"),
                      ("--physical-device-ids", "14", "--hccl-min-bytes", "4K"),
                      ("--physical-device-ids", "14", "--repeat", "21"),
                      ("--physical-device-ids", "14", "--payload-mib", "0"),
                      ("--physical-device-ids", "14", "--allow-busy-devices"),
                      ("--physical-device-ids", "14", "--pinned-api", "alloc"),
                      ("--physical-device-ids", "14", "--samples", "60"),
                      ("--physical-device-ids", "14", "--minimum-seconds", "121")):
            with self.assertRaises(RuntimeError):
                ToolkitExecutor().plan(self.request(*flags))

    def test_shared_flags_apply_to_ppu_settings(self):
        plan = ToolkitExecutor().plan(self.request("--physical-device-ids", "14", "--command-timeout", "700",
                                                   "--warmup", "9", "--minimum-seconds", "15", "--dry-run"))
        self.assertEqual(plan["settings"]["command_timeout"], 700)
        self.assertEqual(plan["settings"]["warmup"], 9)
        self.assertEqual(plan["settings"]["minimum_seconds"], 15.0)

    def test_smoke_plan(self):
        plan = ToolkitExecutor().plan(self.request("--physical-device-ids", "14", "--smoke", "--dry-run"))
        self.assertEqual(plan["settings"]["matrix_size"], 256)
        self.assertEqual(plan["settings"]["payload_mib"], 64)


class EvidenceTests(unittest.TestCase):
    def context_file(self, root):
        context = {"schema_version": 1, "run_id": "fixture",
                   "host": {"devices": [
                       {"host_physical_id": 14, "pci_bdf": "00000000:a3:00.0", "uuid": "u14"},
                       {"host_physical_id": 15, "pci_bdf": "00000000:a2:00.0", "uuid": "u15"}]},
                   "settings": {"payload_mib": 512, "matrix_size": 8192, "compute_samples": 4200,
                                "transfer_samples": 25, "p2p_samples": 15, "p2p_bytes": 40000000,
                                "repeat": 5, "command_timeout": 300, "compute_timeout": 900, "warmup": 3},
                   "cases": list(c.CASES), "runtime_lock": {}, "image_identity": {}}
        path = root / "context.json"
        e.write_json(path, context)
        return path

    def test_setup_failure_is_failed_and_keeps_each_requested_case(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(er, "prepare", side_effect=RuntimeError("compiler failed")):
                self.assertEqual(er.main(["--context", str(self.context_file(root)),
                                          "--output", str(root / "evidence")]), 1)
            manifest = json.loads((root / "evidence/manifest.json").read_text())
            self.assertEqual(manifest["measurement_status"], "failed")
            self.assertEqual(set(manifest["cases"]), set(c.CASES))
            self.assertTrue(all(v["measurement_status"] == "failed" for v in manifest["cases"].values()))

    def monitor_fixture(self, device="thead/u14", count=10, base_offset=1.0):
        return SimpleNamespace(origin_monotonic_s=100,
                               samples=[{"device_id": device, "valid": True,
                                         "started_offset_s": base_offset + i * .5,
                                         "finished_offset_s": base_offset + .1 + i * .5} for i in range(count)],
                               finish=lambda *a, **k: None)

    def test_finalize_counts_only_complete_samples_inside_window(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            samples = self.monitor_fixture().samples
            monitor = SimpleNamespace(origin_monotonic_s=100, samples=samples, finish=lambda *a, **k: None)
            target = {"target": "point/uni-write-14-to-15/rep-1", "measurement_status": "passed",
                      "bindings": [{"uuid": "u14", "host_physical_id": 14}],
                      "measurement_window": {"started_monotonic_s": 101, "finished_monotonic_s": 107,
                                             "role": "measurement"}}
            manifest = {"cases": {"interconnect-P2P_intraserver": {"targets": [target]}}}
            request = SimpleNamespace(compute_monitor="on", data_movement_monitor="on")
            finalize_monitor(root, manifest, monitor, request)
            self.assertEqual(target["monitoring_status"], "passed")
            samples[0]["started_offset_s"] = .99
            finalize_monitor(root, manifest, monitor, request)
            self.assertEqual(target["monitor_sample_counts"], {"14": 9})
            self.assertEqual(target["monitoring_status"], "partial")
            samples[0]["started_offset_s"] = 1
            samples[-1]["valid"] = False
            finalize_monitor(root, manifest, monitor, request)
            self.assertEqual(target["monitoring_status"], "partial")
            self.assertEqual(target["invalid_monitor_samples"], 1)

    def test_capacity_query_monitoring_is_static(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            monitor = self.monitor_fixture(count=0)
            target = {"target": "point/capacity-query-14/rep-1", "measurement_status": "passed",
                      "bindings": [{"uuid": "u14", "host_physical_id": 14}],
                      "measurement_window": {"started_monotonic_s": 101, "finished_monotonic_s": 101.4,
                                             "role": "capacity-query"}}
            manifest = {"cases": {"main_memory-capacity": {"targets": [target]}}}
            request = SimpleNamespace(compute_monitor="on", data_movement_monitor="on")
            finalize_monitor(root, manifest, monitor, request)
            self.assertEqual(target["monitoring_status"], "passed")
            self.assertIn("static", target["monitor_scope"])

    def test_timeout_keeps_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = e.command([sys.executable, "-u", "-c",
                                "import time; print('started',flush=True); time.sleep(5)"],
                               root / "case", root, timeout=.1)
            self.assertEqual(result["returncode"], 124)
            self.assertTrue(result["timed_out"])
            self.assertIn("started", (root / result["stdout"]["path"]).read_text())
            with self.assertRaises(FileExistsError):
                e.command(["anything"], root / "case", root)

    def test_final_status_composes_layers(self):
        self.assertEqual(compose_status("passed", "passed", "passed"), "passed")
        self.assertEqual(compose_status("failed", "passed", "passed"), "failed")
        self.assertEqual(compose_status("passed", "partial", "passed"), "partial")
        self.assertEqual(compose_status("not-run", "not-run", "not-run"), "partial")

    def test_hash_index_detects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "test").write_text("original")
            e.index(root)
            self.assertEqual(e.validate_index(root), [])
            (root / "test").write_text("changed")
            self.assertEqual(e.validate_index(root), ["test"])

    def test_report_renders_excluded_capabilities_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            e.write_json(root / "summary.json", {"suite": "thead-ppu-toolkit", "status": "failed",
                                                 "run_id": "fixture"})
            e.write_json(root / "toolkit-evidence/manifest.json",
                         {"schema_version": 1, "cases": {}, "settings": {}})
            e.write_json(root / "toolkit-evidence/toolkit-contract.json", c.contract_record({}))
            generate_and_record(root)
            text = (root / "report.md").read_text()
            self.assertIn("排除能力：computation-INT8", text)
            self.assertIn("acblasGemmEx", text)
            self.assertEqual(json.loads((root / "summary.json").read_text())["status"], "failed")
            first = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            generate_and_record(root)
            self.assertEqual(first, {p.relative_to(root).as_posix(): p.read_bytes()
                                     for p in root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
