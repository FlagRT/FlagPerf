# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Selected-device Ascend ``npu-smi usages`` sampling primitives.

This module deliberately knows nothing about Benchmark or Toolkit result
semantics.  Callers provide the fields, workload windows, coverage role, and
evidence threshold that belong to their own domain.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_INTERVAL_S = 1.0
DEFAULT_COMMAND_TIMEOUT_S = 5
DEFAULT_MIN_SAMPLES_PER_TARGET = 10

TOOLKIT_USAGE_FIELDS: dict[str, str] = {
    "Aicore Usage Rate(%)": "aicore_usage_rate_pct",
    "Aivector Usage Rate(%)": "aivector_usage_rate_pct",
    "HBM Bandwidth Usage Rate(%)": "hbm_bandwidth_usage_rate_pct",
    "NPU Utilization(%)": "npu_utilization_pct",
}
BENCHMARK_USAGE_FIELDS: dict[str, str] = {
    "Aicore Usage Rate(%)": "aicore_usage_rate_pct",
    "Aivector Usage Rate(%)": "aivector_usage_rate_pct",
    "HBM Usage Rate(%)": "hbm_usage_rate_pct",
    "HBM Bandwidth Usage Rate(%)": "hbm_bandwidth_usage_rate_pct",
    "NPU Utilization(%)": "npu_utilization_pct",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    temporary.replace(path)


def artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def _usage_rate(value: str) -> int | None:
    match = re.fullmatch(r"\s*(\d{1,3})(?:\.0+)?\s*%?\s*", value)
    if not match:
        return None
    number = int(match.group(1))
    return number if 0 <= number <= 100 else None


def parse_npu_smi_usages(
    text: str,
    expected_npu_id: int,
    field_specs: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Parse label-addressed per-Chip values without relying on row order."""
    specs = dict(field_specs or TOOLKIT_USAGE_FIELDS)
    npu_id = expected_npu_id
    current: dict[str, str] = {}
    parsed: list[dict[str, Any]] = []

    def commit() -> None:
        nonlocal current
        chip_text = current.get("Chip ID")
        if chip_text is None or not re.fullmatch(r"\d+", chip_text.strip()):
            current = {}
            return
        source_fields = {key: current[key] for key in specs if key in current}
        values = {
            normalized: _usage_rate(source_fields.get(label, ""))
            for label, normalized in specs.items()
        }
        missing = [
            label for label, normalized in specs.items()
            if values[normalized] is None
        ]
        parsed.append({
            "npu_id": npu_id,
            "chip_id": int(chip_text.strip()),
            "source_fields": source_fields,
            "values": values,
            "missing_or_invalid_fields": missing,
            "valid": not missing,
        })
        current = {}

    for line in text.splitlines():
        match = re.match(r"\s*([^:]+?)\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        key, value = match.group(1).strip(), match.group(2).strip()
        if key == "NPU ID" and re.fullmatch(r"\d+", value):
            npu_id = int(value)
            continue
        if key == "Chip Count":
            continue
        if key == "Chip ID":
            if "Chip ID" in current:
                commit()
            current["Chip ID"] = value
            if any(label in current for label in specs):
                commit()
            continue
        if key in specs:
            if key in current:
                commit()
            current[key] = value
    if current:
        commit()
    return parsed


class UsageMonitor:
    """Low-rate sampler scoped to explicit physical-NPU/Chip targets."""

    def __init__(
        self,
        targets: Sequence[dict[str, int]],
        required_fields: Sequence[str] | None = None,
        *,
        field_specs: Mapping[str, str] | None = None,
        interval_s: float = DEFAULT_INTERVAL_S,
        command_timeout_s: int = DEFAULT_COMMAND_TIMEOUT_S,
    ) -> None:
        self.field_specs = dict(field_specs or TOOLKIT_USAGE_FIELDS)
        self.required_fields = tuple(required_fields or self.field_specs)
        unknown = sorted(set(self.required_fields) - set(self.field_specs))
        if unknown:
            raise ValueError(f"unknown required usage fields: {unknown}")
        if interval_s <= 0:
            raise ValueError("monitor interval must be positive")
        if command_timeout_s <= 0:
            raise ValueError("monitor command timeout must be positive")
        self.interval_s = float(interval_s)
        self.command_timeout_s = int(command_timeout_s)
        self.targets = [dict(item) for item in targets]
        self.npu_ids = sorted({item["npu_id"] for item in self.targets})
        self.target_by_pair = {
            (item["npu_id"], item["chip_id"]): item for item in self.targets
        }
        self.origin_monotonic_s = time.monotonic()
        self.stop_event = threading.Event()
        self.ready = {npu_id: threading.Event() for npu_id in self.npu_ids}
        self.lock = threading.Lock()
        self.raw_records: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        for npu_id in self.npu_ids:
            thread = threading.Thread(
                target=self._worker,
                args=(npu_id,),
                name=f"flagperf-npu-smi-{npu_id}",
                daemon=True,
            )
            self.threads.append(thread)
            thread.start()
        deadline = time.monotonic() + self.command_timeout_s + 1
        for event in self.ready.values():
            event.wait(max(0.0, deadline - time.monotonic()))

    def _worker(self, npu_id: int) -> None:
        sequence = 0
        while not self.stop_event.is_set():
            command = ["npu-smi", "info", "-t", "usages", "-i", str(npu_id)]
            started_at = utc_now()
            started = time.monotonic()
            timed_out = False
            try:
                proc = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=self.command_timeout_s,
                )
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                returncode = proc.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                returncode = 124
                timed_out = True
            except OSError as exc:
                stdout = ""
                stderr = f"{type(exc).__name__}: {exc}"
                returncode = 127
            # Event files and sampler windows are compared across host
            # processes.  Keep every endpoint in the same CLOCK_MONOTONIC
            # domain instead of relying on perf_counter being an alias on a
            # particular Python/platform build.
            finished = time.monotonic()
            raw = {
                "sequence": sequence,
                "npu_id": npu_id,
                "command": command,
                "started_at": started_at,
                "finished_at": utc_now(),
                "started_offset_s": round(started - self.origin_monotonic_s, 6),
                "finished_offset_s": round(finished - self.origin_monotonic_s, 6),
                "duration_s": round(finished - started, 6),
                "returncode": returncode,
                "timed_out": timed_out,
                "stdout": stdout,
                "stderr": stderr,
            }
            normalized: list[dict[str, Any]] = []
            if returncode == 0:
                for item in parse_npu_smi_usages(
                    stdout, npu_id, self.field_specs
                ):
                    target = self.target_by_pair.get(
                        (item["npu_id"], item["chip_id"])
                    )
                    if target is None:
                        continue
                    normalized.append(item | {
                        "sequence": sequence,
                        "logical_device_id": target["logic_id"],
                        "sample_started_at": started_at,
                        "sample_finished_at": raw["finished_at"],
                        "started_offset_s": raw["started_offset_s"],
                        "finished_offset_s": raw["finished_offset_s"],
                        "duration_s": raw["duration_s"],
                    })
            with self.lock:
                self.raw_records.append(raw)
                self.samples.extend(normalized)
            self.ready[npu_id].set()
            sequence += 1
            remaining = self.interval_s - (time.monotonic() - started)
            if remaining > 0:
                self.stop_event.wait(remaining)

    def snapshot(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self.lock:
            return list(self.raw_records), list(self.samples)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(self.command_timeout_s + 1)

    def relative_window(self, record: dict[str, Any], role: str) -> dict[str, Any]:
        return {
            "role": role,
            "started_at": record["started_at"],
            "finished_at": record["finished_at"],
            "started_offset_s": round(
                record["started_monotonic_s"] - self.origin_monotonic_s, 6
            ),
            "finished_offset_s": round(
                record["finished_monotonic_s"] - self.origin_monotonic_s, 6
            ),
        }

    @staticmethod
    def overlaps(sample: dict[str, Any], window: dict[str, Any]) -> bool:
        return (
            sample["started_offset_s"] <= window["finished_offset_s"]
            and sample["finished_offset_s"] >= window["started_offset_s"]
        )

    def valid_counts(self, windows: Sequence[dict[str, Any]]) -> dict[str, int]:
        _, samples = self.snapshot()
        counts = {
            f"{item['npu_id']}/{item['chip_id']}/{item['logic_id']}": 0
            for item in self.targets
        }
        for sample in samples:
            matching_windows = [
                window for window in windows
                if window.get("logical_device_id") in (
                    None, sample["logical_device_id"]
                )
            ]
            if any(
                sample["values"].get(self.field_specs[label]) is None
                for label in self.required_fields
            ) or not any(
                self.overlaps(sample, window) for window in matching_windows
            ):
                continue
            key = (
                f"{sample['npu_id']}/{sample['chip_id']}/"
                f"{sample['logical_device_id']}"
            )
            if key in counts:
                counts[key] += 1
        return counts

    def finish(
        self,
        root: Path,
        directory: Path,
        windows: Sequence[dict[str, Any]],
        extensions: Sequence[dict[str, Any]],
        *,
        min_samples_per_target: int = DEFAULT_MIN_SAMPLES_PER_TARGET,
        primary_role: str = "primary",
        window_semantics: str = (
            "sample command intervals overlapping the DMI process wall window; "
            "the vendor-reported DMI duration remains a separate raw field"
        ),
        target_resource: str | None = None,
    ) -> dict[str, Any]:
        self.stop()
        raw, samples = self.snapshot()
        raw.sort(key=lambda item: (item["started_offset_s"], item["npu_id"]))
        samples.sort(key=lambda item: (
            item["started_offset_s"], item["npu_id"], item["chip_id"]
        ))
        raw_path = directory / "samples.raw.jsonl"
        parsed_path = directory / "samples.jsonl"
        write_jsonl(raw_path, raw)
        write_jsonl(parsed_path, samples)
        counts = self.valid_counts(windows)
        primary = [window for window in windows if window.get("role") == primary_role]
        primary_counts = self.valid_counts(primary)
        complete = bool(counts) and all(
            count >= min_samples_per_target for count in counts.values()
        ) and all(count >= 1 for count in primary_counts.values())
        successful_commands = sum(record["returncode"] == 0 for record in raw)
        reasons: list[str] = []
        if not successful_commands:
            reasons.append("npu-smi usages did not produce a successful sample command")
        if not primary:
            reasons.append(f"no {primary_role} workload window was recorded")
        for target, count in counts.items():
            if count < min_samples_per_target:
                reasons.append(
                    f"target {target} has {count}/{min_samples_per_target} "
                    "valid workload samples"
                )
        for target, count in primary_counts.items():
            if count < 1:
                reasons.append(
                    f"target {target} has no sample overlapping the "
                    f"{primary_role} workload window"
                )
        result = {
            "schema_version": 1,
            "status": "passed" if complete else "partial",
            "collector": "npu-smi info -t usages",
            "evidence_kind": "usage-timeline",
            "target_resource": target_resource or (
                "hbm-bandwidth" if self.required_fields == (
                    "HBM Bandwidth Usage Rate(%)",
                ) else "compute-units"
            ),
            "required_fields": list(self.required_fields),
            "target_interval_s": self.interval_s,
            "command_timeout_s": self.command_timeout_s,
            "required_samples_per_chip": min_samples_per_target,
            "window_semantics": window_semantics,
            "targets": self.targets,
            "workload_windows": list(windows),
            "sample_counts_by_target": counts,
            "primary_sample_counts_by_target": primary_counts,
            "successful_sample_commands": successful_commands,
            "total_sample_commands": len(raw),
            "extensions": list(extensions),
            "raw_samples": artifact_ref(raw_path, root),
            "parsed_samples": artifact_ref(parsed_path, root),
            "reasons": reasons,
        }
        write_json(directory / "summary.json", result)
        result["summary"] = artifact_ref(directory / "summary.json", root)
        return result
