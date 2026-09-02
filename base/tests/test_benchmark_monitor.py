# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch


BASE_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
for path in (BASE_DIR, BENCHMARKS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_monitor import (  # noqa: E402
    create_usage_monitor,
    finalize_benchmark_monitor,
    finalize_benchmark_monitor_safely,
    prepare_benchmark_event_exchange,
    resolve_monitor_targets,
    write_monitor_terminal_summary,
)
from benchmark_worker import write_json_atomic  # noqa: E402
from executors.benchmark import aggregate_benchmark_status  # noqa: E402
from monitoring.ascend_usage import (  # noqa: E402
    BENCHMARK_USAGE_FIELDS,
    parse_npu_smi_usages,
)

EVENTS_SPEC = importlib.util.spec_from_file_location(
    "flagperf_benchmark_events", BENCHMARKS_DIR / "drivers" / "events.py"
)
assert EVENTS_SPEC is not None and EVENTS_SPEC.loader is not None
EVENTS_MODULE = importlib.util.module_from_spec(EVENTS_SPEC)
sys.modules[EVENTS_SPEC.name] = EVENTS_MODULE
EVENTS_SPEC.loader.exec_module(EVENTS_MODULE)
benchmark_measurement_start = EVENTS_MODULE.benchmark_measurement_start
benchmark_measurement_finish = EVENTS_MODULE.benchmark_measurement_finish


USAGES = """
    NPU ID                         : 7
    Chip Count                     : 2
    Aivector Usage Rate(%)         : 3
    HBM Usage Rate(%)              : 21
    NPU Utilization(%)             : 75
    Aicore Usage Rate(%)           : 88
    HBM Bandwidth Usage Rate(%)    : 44
    Chip ID                        : 0
    HBM Bandwidth Usage Rate(%)    : 45
    Aicore Usage Rate(%)           : 89
    NPU Utilization(%)             : 76
    HBM Usage Rate(%)              : 22
    Aivector Usage Rate(%)         : 4
    Chip ID                        : 1
"""


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class ParserAndEventTests(unittest.TestCase):
    def test_current_ascend_cases_emit_exact_measurement_windows(self) -> None:
        paths = [
            "computation-BF16/main.py",
            "computation-FP16/main.py",
            "computation-FP32/main.py",
            "computation-INT8/ascend/main.py",
            "main_memory-bandwidth/main.py",
            "main_memory-capacity/main.py",
            "interconnect-h2d/main.py",
            "interconnect-d2h/main.py",
        ]
        for relative in paths:
            source = (BENCHMARKS_DIR / relative).read_text(encoding="utf-8")
            with self.subTest(case=relative):
                self.assertIn("benchmark_measurement_start()", source)
                self.assertIn("benchmark_measurement_finish(measurement_event)", source)
                event_start = source.index("benchmark_measurement_start()")
                event_finish = source.index(
                    "benchmark_measurement_finish(measurement_event)"
                )
                if relative == "main_memory-capacity/main.py":
                    self.assertLess(event_start, source.index("while byte_size >="))
                    self.assertLess(
                        source.index("while byte_size >="), event_finish
                    )
                    self.assertLess(event_finish, source.index("start = time.time()"))
                else:
                    timer_start = source.index("start_time = time.perf_counter()")
                    timer_finish = source.index("end_time = time.perf_counter()")
                    self.assertLess(event_start, timer_start)
                    self.assertLess(timer_finish, event_finish)

    def test_benchmark_parser_keeps_five_label_addressed_fields(self) -> None:
        parsed = parse_npu_smi_usages(USAGES, 7, BENCHMARK_USAGE_FIELDS)
        self.assertEqual(len(parsed), 2)
        self.assertTrue(all(item["valid"] for item in parsed))
        self.assertEqual(parsed[0]["values"]["aicore_usage_rate_pct"], 88)
        self.assertEqual(parsed[1]["values"]["hbm_usage_rate_pct"], 22)

        missing = parse_npu_smi_usages(
            USAGES.replace("    HBM Usage Rate(%)              : 21\n", "", 1),
            7,
            BENCHMARK_USAGE_FIELDS,
        )
        self.assertFalse(missing[0]["valid"])
        self.assertIn("HBM Usage Rate(%)", missing[0]["missing_or_invalid_fields"])

    def test_measurement_event_is_noop_when_monitoring_is_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(benchmark_measurement_start())

    def test_measurement_event_writes_one_atomic_rank_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                "FLAGPERF_BENCHMARK_EVENTS_DIR": str(root),
                "FLAGPERF_BENCHMARK_CASE": "computation-FP16",
                "RANK": "1",
                "LOCAL_RANK": "1",
                "WORLD_SIZE": "2",
            }
            with patch.dict(os.environ, environment, clear=True):
                token = benchmark_measurement_start()
                benchmark_measurement_finish(token)
            event_path = root / "measurement-rank-1.json"
            event = json.loads(event_path.read_text(encoding="utf-8"))
            event_mode = stat.S_IMODE(event_path.stat().st_mode)
        self.assertEqual(event["kind"], "measurement-window")
        self.assertEqual(event["case"], "computation-FP16")
        self.assertEqual(event["rank"], 1)
        self.assertGreaterEqual(event["duration_s"], 0)
        self.assertEqual(event_mode, 0o644)

    def test_worker_event_writer_uses_host_readable_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "torchrun-window.json"
            previous_umask = os.umask(0o077)
            try:
                write_json_atomic(path, {"kind": "torchrun-window"})
            finally:
                os.umask(previous_umask)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_host_prepares_one_sticky_per_run_event_exchange(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = prepare_benchmark_event_exchange(root)
            path = root / record["path"]
            self.assertTrue(path.is_dir())
            self.assertFalse(path.is_symlink())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o1777)
            with self.assertRaises(FileExistsError):
                prepare_benchmark_event_exchange(root)


class MonitorContractTests(unittest.TestCase):
    def _preflight(self) -> dict:
        return {
            "actual_device_map": [
                {"npu_id": 7, "chip_id": 0, "logic_id": 14},
                {"npu_id": 7, "chip_id": 1, "logic_id": 15},
            ]
        }

    def _populate_monitor(self, monitor, logic_ids: list[int]) -> None:
        for sequence in range(10):
            monitor.raw_records.append({
                "sequence": sequence,
                "npu_id": 7,
                "command": ["npu-smi", "info", "-t", "usages", "-i", "7"],
                "started_at": "2026-08-30T03:30:00Z",
                "finished_at": "2026-08-30T03:30:01Z",
                "started_offset_s": 4.1 + sequence,
                "finished_offset_s": 4.7 + sequence,
                "duration_s": 0.6,
                "returncode": 0,
                "timed_out": False,
                "stdout": USAGES,
                "stderr": "",
            })
            for logic_id in logic_ids:
                chip_id = 0 if logic_id == 14 else 1
                monitor.samples.append({
                    "sequence": sequence,
                    "npu_id": 7,
                    "chip_id": chip_id,
                    "logical_device_id": logic_id,
                    "sample_started_at": "2026-08-30T03:30:00Z",
                    "sample_finished_at": "2026-08-30T03:30:01Z",
                    "started_offset_s": 4.1 + sequence,
                    "finished_offset_s": 4.7 + sequence,
                    "duration_s": 0.6,
                    "source_fields": {
                        label: "50" for label in BENCHMARK_USAGE_FIELDS
                    },
                    "values": {
                        normalized: 50
                        for normalized in BENCHMARK_USAGE_FIELDS.values()
                    },
                    "missing_or_invalid_fields": [],
                    "valid": True,
                })

    def _window_record(self, monitor, start: float, finish: float, kind: str) -> dict:
        return {
            "schema_version": 1,
            "kind": kind,
            "started_at": "2026-08-30T03:30:00Z",
            "finished_at": "2026-08-30T03:30:20Z",
            "started_monotonic_ns": int((monitor.origin_monotonic_s + start) * 1e9),
            "finished_monotonic_ns": int((monitor.origin_monotonic_s + finish) * 1e9),
        }

    def _write_rank_event(
        self, root: Path, monitor, rank: int, local_rank: int, world_size: int,
    ) -> None:
        event = self._window_record(monitor, 4.0, 15.0, "measurement-window")
        event.update({
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "case": "computation-FP16",
        })
        write_json(root / "benchmark-events" / f"measurement-rank-{rank}.json", event)

    def test_monitor_targets_are_resolved_from_live_preflight_map(self) -> None:
        targets = resolve_monitor_targets(self._preflight(), [15])
        self.assertEqual(
            targets, [{"npu_id": 7, "chip_id": 1, "logic_id": 15}]
        )
        with self.assertRaisesRegex(RuntimeError, "did not map every"):
            resolve_monitor_targets(self._preflight(), [13])

    def test_finalize_passes_only_with_exact_rank_window_and_ten_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = resolve_monitor_targets(self._preflight(), [14])
            monitor = create_usage_monitor(targets)
            self._populate_monitor(monitor, [14])
            write_json(
                root / "benchmark-events" / "torchrun-window.json",
                self._window_record(monitor, 2.0, 18.0, "torchrun-window"),
            )
            self._write_rank_event(root, monitor, 0, 0, 1)
            result = finalize_benchmark_monitor(
                monitor,
                root,
                self._window_record(monitor, 1.0, 20.0, "container-window"),
                [14],
                1,
            )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["primary_sample_counts_by_target"]["7/0/14"], 10)
            self.assertFalse(result["automatic_workload_extension"])
            self.assertEqual(result["extensions"], [])
            self.assertTrue((root / result["summary"]["path"]).is_file())

    def test_missing_rank_event_is_partial_without_cross_device_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = resolve_monitor_targets(self._preflight(), [14, 15])
            monitor = create_usage_monitor(targets)
            self._populate_monitor(monitor, [14, 15])
            write_json(
                root / "benchmark-events" / "torchrun-window.json",
                self._window_record(monitor, 2.0, 18.0, "torchrun-window"),
            )
            self._write_rank_event(root, monitor, 0, 0, 2)
            result = finalize_benchmark_monitor(
                monitor,
                root,
                self._window_record(monitor, 1.0, 20.0, "container-window"),
                [14, 15],
                2,
            )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["primary_sample_counts_by_target"]["7/1/15"], 0)
        self.assertTrue(any("rank 1" in reason for reason in result["reasons"]))

    def test_monitor_off_summary_is_independent_and_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = write_monitor_terminal_summary(
                Path(temporary),
                status="not-run",
                enabled=False,
                targets=[],
                reason="disabled",
            )
        self.assertEqual(result["status"], "not-run")
        self.assertFalse(result["policy"]["enabled"])

    def test_finalization_permission_error_is_monitor_only(self) -> None:
        class FakeMonitor:
            def __init__(self) -> None:
                self.stopped = False

            def stop(self) -> None:
                self.stopped = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_benchmark_event_exchange(root)
            monitor = FakeMonitor()
            with patch(
                "benchmark_monitor.finalize_benchmark_monitor",
                side_effect=PermissionError("fixture permission denied"),
            ):
                result = finalize_benchmark_monitor_safely(
                    monitor,  # type: ignore[arg-type]
                    root,
                    {"schema_version": 1, "kind": "container-window"},
                    [14],
                    1,
                    [{"npu_id": 7, "chip_id": 0, "logic_id": 14}],
                    None,
                )
        self.assertTrue(monitor.stopped)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("PermissionError" in item for item in result["reasons"]))
        self.assertEqual(
            aggregate_benchmark_status(
                "passed", "passed", result["status"], monitor_enabled=True
            ),
            "partial",
        )


if __name__ == "__main__":
    unittest.main()
