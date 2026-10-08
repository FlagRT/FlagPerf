"""CPU fixtures only: synthetic evidence roots for the acceptance report generator."""
import json
import re
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from generate_base_acceptance_report import (  # noqa: E402
    ACCEPTANCE_SCHEMA_VERSION, ReportError, collect, domain_of, generate_acceptance_report,
    load_root, parse_group, summarize_group, worst_status,
)


def metric(name="computation-FP32", value=100.0, unit="TFLOPS", rank=0, elapsed=20.0, **extra):
    entry = {"metric": name, "value": value, "unit": unit, "rank": rank,
             "elapsed_seconds": elapsed, "status": "passed"}
    entry.update(extra)
    return entry


def summary(case="computation-FP32:P800", status="passed", measurement="passed",
            monitoring="passed", health="observed-clean", mode="qualification",
            reps=5, min_seconds=15, max_cv=5, devices=(5,), cleanup="passed", lease=True):
    return {
        "schema_version": 3, "kind": "benchmark", "case": case, "status": status,
        "measurement_status": measurement, "monitoring_status": monitoring,
        "correctness_status": "passed", "health_status": health,
        "cleanup_status": cleanup, "lease_released": lease,
        "started_at": "2026-10-08T02:00:00Z", "finished_at": "2026-10-08T02:01:00Z",
        "runtime": {"image": "fixture:1", "image_id": "sha256:fixture",
                    "lock": {"runtime_profile": "fixture-profile",
                             "image_manifest": {"release_stage": "candidate", "validated": False}}},
        "device_bindings": [{"host_physical_id": device} for device in devices],
        "qualification": {"mode": mode, "repetitions_required": reps,
                          "max_cv_percent": max_cv, "min_measurement_seconds": min_seconds,
                          "scope": "fixture"},
    }


def write_run(root: Path, label: str, metrics, **summary_updates):
    run_dir = root / "base-runs" / label
    run_dir.mkdir(parents=True)
    s = summary(**summary_updates)
    (run_dir / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    (run_dir / "benchmark-result.json").write_text(
        json.dumps({"case": s["case"], "schema_version": 1, "metrics": metrics}),
        encoding="utf-8")
    return run_dir


def records_from(values, unit="GB/s", name="allreduce-algbw", mode="qualification", **updates):
    out = []
    for index, value in enumerate(values):
        out.append({
            "label": f"run-q{index + 1}", "path": Path("."), "group": "run", "rep": index + 1,
            "summary": {"qualification": {"mode": mode, "repetitions_required": 5,
                                          "max_cv_percent": 5, "min_measurement_seconds": 15},
                        **updates},
            "result": {"metrics": [metric(name=name, value=value, unit=unit, rank=0)]},
        })
    return out


class GroupParsingTests(unittest.TestCase):
    def test_repetition_labels(self):
        self.assertEqual(parse_group("FP32-q1"), ("FP32", 1))
        self.assertEqual(parse_group("h2d-pinned-blocking-r2-q3"),
                         ("h2d-pinned-blocking-r2", 3))

    def test_single_shot_and_smoke_labels(self):
        self.assertEqual(parse_group("capacity-final"), ("capacity", None))
        self.assertEqual(parse_group("MPI-eight-smoke"), ("MPI-eight-smoke", None))


class StatusAndDomainTests(unittest.TestCase):
    def test_worst_status_ordering(self):
        self.assertEqual(worst_status(["passed", "partial"]), "partial")
        self.assertEqual(worst_status(["passed", None]), "passed")
        self.assertEqual(worst_status([None, ""]), "not recorded")
        self.assertEqual(worst_status(["failed", "partial"]), "failed")

    def test_domain_classification(self):
        self.assertEqual(domain_of("computation-INT8:P800"), "算力")
        self.assertEqual(domain_of("interconnect-d2h:P800"), "传输")
        self.assertEqual(domain_of("interconnect-MPI_intraserver:P800"), "通信")
        self.assertEqual(domain_of("main_memory-capacity:P800"), "内存")
        self.assertEqual(domain_of("something-else:P800"), "其他")


class SummarizeTests(unittest.TestCase):
    def test_median_cv_and_slowest_rank(self):
        records = records_from([10.0, 11.0, 12.0, 13.0, 14.0])
        records[2]["result"]["metrics"].append(metric(value=12.5, rank=1))
        stats = summarize_group(records)
        self.assertEqual(stats["n"], 5)
        self.assertEqual(stats["median"], 12.0)  # rank0 12.0 < rank1 12.5
        expected_cv = statistics.stdev([10.0, 11.0, 12.0, 13.0, 14.0]) / 12.0 * 100
        self.assertAlmostEqual(stats["cv_percent"], expected_cv)
        self.assertEqual(stats["verdict"], "unstable")

    def test_slowest_rank_picks_min(self):
        records = records_from([10.0, 11.0, 12.0, 13.0, 14.0])
        records[2]["result"]["metrics"].append(metric(value=11.5, rank=1))
        stats = summarize_group(records)
        self.assertEqual(stats["median"], 11.5)  # rank1 11.5 is the slowest rank

    def test_passed_group(self):
        stats = summarize_group(records_from([28.19, 28.20, 28.21, 28.20, 28.22]))
        self.assertEqual(stats["verdict"], "passed")
        self.assertLess(stats["cv_percent"], 5.0)

    def test_capacity_and_smoke_verdicts(self):
        stats = summarize_group(records_from([102.9], mode="capacity"))
        self.assertEqual(stats["verdict"], "single-shot（窗口不适用）")
        stats = summarize_group(records_from([30.0], mode="smoke"))
        self.assertEqual(stats["verdict"], "smoke（预期部分状态）")

    def test_missing_repetitions(self):
        stats = summarize_group(records_from([28.2, 28.3]))
        self.assertEqual(stats["verdict"], "重复不足(2/5)")

    def test_near_threshold_flag_for_measured_mode(self):
        from generate_base_acceptance_report import limitation_flags
        records = records_from([22.9, 24.0, 24.4, 25.1, 25.6], mode="measured")
        for record in records:
            record["summary"]["health_status"] = "observed-clean"
            record["summary"]["cleanup_status"] = "passed"
            record["summary"]["lease_released"] = True
        group = {"root": "r", "group": "g", "records": records,
                 "case": "interconnect-d2h:P800", "domain": "传输",
                 "stats": summarize_group(records)}
        flags = limitation_flags([group], Path("r/acceptance-report.md"))
        self.assertTrue(any("贴近阈值" in line for line in flags), flags)

    def test_harness_qualification_error_surfaces_verbatim(self):
        records = records_from([14.7])
        records[0]["summary"]["qualification_error"] = \
            "measurement window shorter than 15 seconds"
        stats = summarize_group(records)
        self.assertEqual(
            stats["verdict"],
            "harness 资格错误(measurement window shorter than 15 seconds)")
        # a short recorded window alone must NOT fail the group
        stats = summarize_group(records_from([18.3, 18.3, 18.4, 18.3, 18.4]))
        self.assertEqual(stats["verdict"], "passed")

    def test_no_measurements(self):
        records = records_from([1.0])
        records[0]["result"]["metrics"] = []
        stats = summarize_group(records)
        self.assertEqual(stats["verdict"], "no-measurement")


class LoadRootTests(unittest.TestCase):
    def test_gaps_and_missing_base_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evidence"
            write_run(root, "FP32-q1", [metric()])
            (root / "base-runs" / "broken").mkdir()
            data = load_root(root)
            self.assertEqual(len(data["runs"]), 1)
            self.assertEqual(data["gaps"], ["broken"])
            with self.assertRaises(ReportError):
                load_root(Path(tmp) / "empty-root")
            with self.assertRaises(ReportError):
                load_root(Path(tmp))  # no base-runs at all


class ReportGenerationTests(unittest.TestCase):
    def build_roots(self, base: Path):
        first = base / "root-one"
        second = base / "root-two"
        for index, value in enumerate([118.4, 118.5, 118.6, 118.5, 118.7]):
            write_run(first, f"FP32-q{index + 1}", [metric(value=value)],
                      case="computation-FP32:P800")
        for index, value in enumerate([19.7, 19.8, 19.8, 19.9, 19.8]):
            write_run(first, f"h2d-pinned-blocking-q{index + 1}",
                      [metric(name="transfer-bandwidth", value=value, unit="GB/s")],
                      case="interconnect-h2d:P800")
        write_run(first, "capacity-final",
                  [metric(name="main_memory-capacity", value=102.898860032, unit="GB",
                          elapsed=0.1, held_mib=98132, allocation_count=32,
                          value_gib_s=95.83)],
                  case="main_memory-capacity:P800", mode="capacity", monitoring="partial",
                  status="partial")
        for index, value in enumerate([64.0, 64.1, 64.2]):
            write_run(second, f"MPI-eight-q{index + 1}",
                      [metric(name="allreduce-algbw", value=value, unit="GB/s")],
                      case="interconnect-MPI_intraserver:P800", reps=5, health="degraded")
        write_run(second, "P2P-pair-smoke",
                  [metric(name="p2p-one-way-bandwidth", value=31.0, unit="GB/s", elapsed=5.0)],
                  case="interconnect-P2P_intraserver:P800", mode="smoke", status="partial",
                  reps=5)
        return first, second

    def test_determinism_and_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = self.build_roots(Path(tmp))
            output = first / "acceptance-report.md"
            meta_one = generate_acceptance_report([first, second], output)
            text_one = output.read_text(encoding="utf-8")
            meta_two = generate_acceptance_report([first, second], output)
            self.assertEqual(text_one, output.read_text(encoding="utf-8"))
            self.assertEqual(meta_one["report_sha256"], meta_two["report_sha256"])
            self.assertEqual(meta_one["schema_version"], ACCEPTANCE_SCHEMA_VERSION)
            self.assertEqual(meta_one["runs"], 15)
            self.assertEqual(meta_one["groups"], 5)
            self.assertIn("# FlagPerf Base Acceptance Report", text_one)
            for section in ("## 技术摘要", "## 环境与运行边界",
                            "## 测量与资格是两个独立结论", "## 算力明细", "## 传输明细",
                            "## 通信明细", "## 设备内存带宽与容量",
                            "## 健康、拓扑与证据完整性", "## 方法与指标口径",
                            "## 限制与后续动作", "## 原始证据索引"):
                self.assertIn(section, text_one)
            for target in set(re.findall(r"\]\(([^)]+)\)", text_one)):
                self.assertTrue((output.parent / target).exists(),
                                f"broken evidence link: {target}")

    def test_dual_layer_and_flags_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = self.build_roots(Path(tmp))
            output = first / "acceptance-report.md"
            meta = generate_acceptance_report([first, second], output)
            text = output.read_text(encoding="utf-8")
            # capacity run: monitoring partial + run status partial surface in the table
            self.assertIn("single-shot（窗口不适用）", text)
            self.assertIn("degraded", text)
            # 3-rep MPI group must be flagged as short of 5 repetitions
            self.assertIn("重复不足(3/5)", text)
            self.assertEqual(meta["status"], "passed")  # no missing summary dirs

    def test_missing_summary_reports_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root-gap"
            write_run(root, "FP32-q1", [metric()])
            (root / "base-runs" / "torn-run").mkdir()
            output = root / "acceptance-report.md"
            meta = generate_acceptance_report([root], output)
            self.assertEqual(meta["status"], "passed-with-gaps")
            self.assertEqual(meta["missing_summaries"], 1)
            self.assertIn("torn-run", output.read_text(encoding="utf-8"))


class CollectTests(unittest.TestCase):
    def test_groups_span_and_sort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            for index in range(5):
                write_run(root, f"FP32-q{5 - index}", [metric(value=118.0 + index)])
            loaded = collect([root])
            self.assertEqual(len(loaded[0]["groups"]), 1)
            records = loaded[0]["groups"][0]["records"]
            self.assertEqual([r["rep"] for r in records], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
