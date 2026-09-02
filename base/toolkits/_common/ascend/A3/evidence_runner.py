#!/usr/bin/env python3
"""Evidence-first single-node Ascend A3 toolkit runner.

The legacy A3 toolkit cases scraped fixed rows from human-readable DMI output
and deleted that output.  This runner keeps the public ``main.sh`` entrypoints
while making raw output, exit status, topology, health, and vendor diagnosis
part of the result contract.

   Case        实际命令                                                     权威指标
  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   BF16        ascend-dmi -f -t bf16 --all --et 80 -q --fmt json            TFLOPS
  ──────────  ───────────────────────────────────────────────────────────  ────────────────────────────
   FP16        ascend-dmi -f -t fp16 --all --et 80 -q --fmt json            TFLOPS
  ──────────  ───────────────────────────────────────────────────────────  ────────────────────────────
   FP32        ascend-dmi -f -t fp32 --all --et 80 -q --fmt json            TFLOPS
  ──────────  ───────────────────────────────────────────────────────────  ────────────────────────────
   INT8        ascend-dmi -f -t int8 --all --et 80 -q --fmt json            TOPS
  ──────────  ───────────────────────────────────────────────────────────  ────────────────────────────
   H2D         ascend-dmi --bw -t h2d -s 536870912 --et 50 -q --fmt json    GB/s
  ──────────  ───────────────────────────────────────────────────────────  ────────────────────────────
   D2H         ascend-dmi --bw -t d2h -s 536870912 --et 50 -q --fmt json    GB/s
  ──────────  ───────────────────────────────────────────────────────────  ────────────────────────────
   传输时延    ascend-dmi -l -t h2d|d2h|p2p -s BYTES ...                    ns
  ──────────  ───────────────────────────────────────────────────────────  ────────────────────────────
   D2D         ascend-dmi --bw -t d2d [-d ID] -q --fmt json                 每 Device 的完整 GB/s 序列
  ──────────  ───────────────────────────────────────────────────────────  ────────────────────────────
   单机 P2P    ascend-dmi --bw -t p2p -m card -q                            每卡对、每方向 GB/s
  ──────────  ───────────────────────────────────────────────────────────  ────────────────────────────
   HCCL        mpirun ... hccl_test/bin/all_reduce_test ...                 alg_bandwidth GB/s

  主要参数：

  - -f：算力 microbenchmark。
  - -t bf16/fp16/fp32/int8：计算数据类型。
  - 默认全量模式继续使用 --all；显式选卡模式逐 Device 使用 -d ID，不混入未选设备。
  - 算力 --et 80：矩阵乘执行次数参数，不是 80 秒。实际执行次数应读 stdout 的 execute_times。
  - --bw：带宽测试。
  - -t h2d：Host 内存到 Device 内存。
  - -s 536870912：每次传输 536,870,912 Byte，即 512 MiB。
  - H2D --et 50：执行 50 次内存拷贝。
  - -t d2d：同一 Device 内存之间的数据搬运，主要反映 Device 内存带宽。
  - -d ID：指定逻辑 Device。显式选卡时，算力、D2D、H2D 均只补跑所选 ID。
  - -t p2p -m card：生成单机 card-mode P2P 矩阵。
  - --fmt json：优先获得结构化结果。
  - -q、P2P 的 -m card 在当前 ToolBox 随包 README 中没有完整参数说明；runner 已按 DMI 26.1 实机行为验证，但不应自行扩展其物理含义。

  D2D 不允许自行添加 -s 或 --et：当前 A3 DMI 固定这两个参数并会拒绝显式传值。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations, permutations
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
BASE_DIR = SCRIPT_DIR.parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from host_preflight import parse_device_map, parse_id_spec
from monitoring.ascend_usage import (  # noqa: E402
    TOOLKIT_USAGE_FIELDS as SHARED_MONITOR_REQUIRED_FIELDS,
    UsageMonitor as SharedUsageMonitor,
    parse_npu_smi_usages as shared_parse_npu_smi_usages,
)


SUPPORTED_CASES = (
    "computation-BF16",
    "computation-FP16",
    "computation-FP32",
    "computation-INT8",
    "main_memory-bandwidth",
    "main_memory-capacity",
    "interconnect-h2d",
    "interconnect-d2h",
    "interconnect-h2d-latency",
    "interconnect-d2h-latency",
    "interconnect-P2P_intraserver",
    "interconnect-P2P_intraserver-latency",
    "interconnect-MPI_intraserver",
)
COMPUTE_DTYPES = {
    "computation-BF16": ("bf16", "TFLOPS"),
    "computation-FP16": ("fp16", "TFLOPS"),
    "computation-FP32": ("fp32", "TFLOPS"),
    "computation-INT8": ("int8", "TOPS"),
}
DIAGNOSIS_FOR_CASE = {
    **{name: "aiflops" for name in COMPUTE_DTYPES},
    "main_memory-bandwidth": "bandwidth",
    "main_memory-capacity": "hbm",
    "interconnect-h2d": "bandwidth",
    "interconnect-d2h": "bandwidth",
    "interconnect-h2d-latency": None,
    "interconnect-d2h-latency": None,
    "interconnect-P2P_intraserver": "signalQuality",
    "interconnect-P2P_intraserver-latency": "signalQuality",
    "interconnect-MPI_intraserver": None,
}
DEFAULT_LATENCY_SIZES = (512, 4096, 65536, 1048576)
COMPUTE_EXECUTE_TIMES = 80
MONITOR_INTERVAL_S = 1.0
MONITOR_COMMAND_TIMEOUT_S = 5
MONITOR_MIN_SAMPLES_PER_CHIP = 10
MONITOR_MAX_EXTENSIONS = 1
HCCS_MONITOR_DURATION_MS = 1000
MONITOR_REQUIRED_FIELDS = SHARED_MONITOR_REQUIRED_FIELDS
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
DEVICE_RE = re.compile(r"(?:device|npu|card)(?:\s*(?:id|index))?\s*[:=/]?\s*(\d+)", re.I)


class EvidenceError(RuntimeError):
    """A required evidence item could not be collected or validated."""


class PartialEvidence(EvidenceError):
    """The command ran, but the evidence does not cover the requested scope."""


def parse_byte_size(value: str) -> int:
    """Parse an exact positive byte size; K/M/G are binary powers."""
    match = re.fullmatch(r"\s*(\d+)\s*([KMGT]?)(?:i?B)?\s*", value, re.I)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value}")
    number = int(match.group(1))
    multiplier = {"": 1, "K": 1 << 10, "M": 1 << 20,
                  "G": 1 << 30, "T": 1 << 40}[match.group(2).upper()]
    result = number * multiplier
    if result <= 0:
        raise argparse.ArgumentTypeError("byte size must be positive")
    return result


def parse_latency_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(parse_byte_size(item) for item in value.split(",") if item.strip())
    except argparse.ArgumentTypeError:
        raise
    if not sizes or len(sizes) != len(set(sizes)):
        raise argparse.ArgumentTypeError("latency sizes must be a non-empty unique list")
    return sizes


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def execution_context() -> str:
    configured = os.environ.get("FLAGPERF_EXECUTION_CONTEXT")
    if configured:
        return configured
    return "container" if Path("/.dockerenv").exists() else "host"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def command_record(
    root: Path,
    directory: Path,
    command: list[str],
    *,
    timeout: int,
    label: str,
    required: bool = True,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    stdout_path = directory / f"{label}.stdout"
    stderr_path = directory / f"{label}.stderr"
    started_at = utc_now()
    # Usage monitors and workload records must share one cross-process clock
    # domain; CLOCK_MONOTONIC is also used by Benchmark rank events.
    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
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
        stderr += f"\nFlagPerf timeout after {timeout}s\n"
        returncode = 124
        timed_out = True
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    finished = time.monotonic()
    record = {
        "command": command,
        "command_shell": shlex.join(command),
        "started_at": started_at,
        "finished_at": utc_now(),
        "started_monotonic_s": started,
        "finished_monotonic_s": finished,
        "duration_s": round(finished - started, 4),
        "timeout_s": timeout,
        "timed_out": timed_out,
        "returncode": returncode,
        "stdout": artifact_ref(stdout_path, root),
        "stderr": artifact_ref(stderr_path, root),
    }
    write_json(directory / f"{label}.command.json", record)
    if required and returncode != 0:
        raise EvidenceError(f"command failed ({returncode}): {shlex.join(command)}")
    return record | {"stdout_text": stdout, "stderr_text": stderr}




# Toolkit retains its result adapter and extension policy, while the actual
# selected-device usages collector/parser is shared with Benchmark.
UsageMonitor = SharedUsageMonitor
parse_npu_smi_usages = shared_parse_npu_smi_usages


def parse_npu_smi_hccs_bw(text: str, expected_npu_id: int, expected_chip_id: int) -> list[dict[str, Any]]:
    """Parse raw per-link and total HCCS Rx/Tx values without deriving metrics."""
    rows: list[dict[str, Any]] = []
    header_seen = False
    for line in text.splitlines():
        lowered = line.lower()
        if "rx_bandwidth" in lowered and "tx_bandwidth" in lowered:
            header_seen = True
            continue
        if not header_seen:
            continue
        columns = line.replace("|", " ").split()
        if len(columns) < 3:
            continue
        link = columns[0]
        if not (re.fullmatch(r"\d+", link) or link.lower() == "total"):
            continue
        rx_raw, tx_raw = columns[1], columns[2]
        try:
            rx, tx = float(rx_raw), float(tx_raw)
        except ValueError:
            continue
        if not (math.isfinite(rx) and math.isfinite(tx) and rx >= 0 and tx >= 0):
            continue
        rows.append({
            "npu_id": expected_npu_id,
            "chip_id": expected_chip_id,
            "link": link,
            "source_fields": {
                "rx_bandwidth(GB/S)": rx_raw,
                "tx_bandwidth(GB/S)": tx_raw,
            },
            "values": {"rx_gb_s": rx, "tx_gb_s": tx},
            "valid": True,
        })
    return rows


def parse_npu_smi_topology(text: str) -> list[dict[str, Any]]:
    """Parse the Phy-ID relationship matrix emitted by npu-smi topo."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header_ids = [int(value) for value in re.findall(r"Phy-ID(\d+)", lines[0])]
    routes: list[dict[str, Any]] = []
    for line in lines[1:]:
        match = re.match(r"\s*Phy-ID(\d+)\s+(.*)$", line)
        if not match:
            continue
        source = int(match.group(1))
        relations = match.group(2).split()
        if len(relations) < len(header_ids):
            continue
        for destination, relation in zip(header_ids, relations):
            if source != destination:
                routes.append({
                    "source_device": source,
                    "destination_device": destination,
                    "relation": relation,
                })
    return routes


class HccsBandwidthMonitor:
    """Sample vendor-reported raw HCCS link counters for selected chips."""

    def __init__(self, targets: list[dict[str, int]]) -> None:
        self.targets = targets
        self.origin_monotonic_s = time.monotonic()
        self.stop_event = threading.Event()
        self.ready = {self._key(item): threading.Event() for item in targets}
        self.lock = threading.Lock()
        self.raw_records: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self.threads: list[threading.Thread] = []

    @staticmethod
    def _key(target: dict[str, int]) -> str:
        return f"{target['npu_id']}/{target['chip_id']}/{target['logic_id']}"

    def start(self) -> None:
        for target in self.targets:
            thread = threading.Thread(
                target=self._worker, args=(target,),
                name=f"flagperf-hccs-{self._key(target)}", daemon=True,
            )
            self.threads.append(thread)
            thread.start()
        deadline = time.monotonic() + MONITOR_COMMAND_TIMEOUT_S + 1
        for event in self.ready.values():
            event.wait(max(0.0, deadline - time.monotonic()))

    def _worker(self, target: dict[str, int]) -> None:
        sequence = 0
        key = self._key(target)
        while not self.stop_event.is_set():
            command = [
                "npu-smi", "info", "-t", "hccs-bw",
                "-i", str(target["npu_id"]), "-c", str(target["chip_id"]),
                "-time", str(HCCS_MONITOR_DURATION_MS),
            ]
            started_at = utc_now()
            started = time.monotonic()
            timed_out = False
            try:
                proc = subprocess.run(
                    command, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, check=False,
                    timeout=MONITOR_COMMAND_TIMEOUT_S,
                )
                stdout, stderr, returncode = proc.stdout or "", proc.stderr or "", proc.returncode
            except subprocess.TimeoutExpired as exc:
                stdout, stderr, returncode, timed_out = exc.stdout or "", exc.stderr or "", 124, True
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
            finished = time.monotonic()
            raw = {
                "sequence": sequence, **target, "command": command,
                "started_at": started_at, "finished_at": utc_now(),
                "started_offset_s": round(started - self.origin_monotonic_s, 6),
                "finished_offset_s": round(finished - self.origin_monotonic_s, 6),
                "duration_s": round(finished - started, 6),
                "returncode": returncode, "timed_out": timed_out,
                "stdout": stdout, "stderr": stderr,
            }
            normalized = []
            if returncode == 0:
                normalized = [item | {
                    "sequence": sequence,
                    "logical_device_id": target["logic_id"],
                    "sample_started_at": started_at,
                    "sample_finished_at": raw["finished_at"],
                    "started_offset_s": raw["started_offset_s"],
                    "finished_offset_s": raw["finished_offset_s"],
                    "duration_s": raw["duration_s"],
                } for item in parse_npu_smi_hccs_bw(
                    stdout, target["npu_id"], target["chip_id"]
                )]
            with self.lock:
                self.raw_records.append(raw)
                self.samples.extend(normalized)
            self.ready[key].set()
            sequence += 1
            remaining = MONITOR_INTERVAL_S - (time.monotonic() - started)
            if remaining > 0:
                self.stop_event.wait(remaining)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(MONITOR_COMMAND_TIMEOUT_S + 1)

    def snapshot(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self.lock:
            return list(self.raw_records), list(self.samples)

    def relative_window(self, record: dict[str, Any], role: str) -> dict[str, Any]:
        return {
            "role": role, "started_at": record["started_at"],
            "finished_at": record["finished_at"],
            "started_offset_s": round(record["started_monotonic_s"] - self.origin_monotonic_s, 6),
            "finished_offset_s": round(record["finished_monotonic_s"] - self.origin_monotonic_s, 6),
        }

    @staticmethod
    def overlaps(sample: dict[str, Any], window: dict[str, Any]) -> bool:
        return sample["started_offset_s"] <= window["finished_offset_s"] and sample["finished_offset_s"] >= window["started_offset_s"]

    def valid_counts(self, windows: list[dict[str, Any]]) -> dict[str, int]:
        _, samples = self.snapshot()
        counts = {self._key(item): 0 for item in self.targets}
        seen: set[tuple[str, int]] = set()
        for sample in samples:
            key = f"{sample['npu_id']}/{sample['chip_id']}/{sample['logical_device_id']}"
            identity = (key, sample["sequence"])
            if key in counts and identity not in seen and sample["valid"] and any(
                self.overlaps(sample, window) for window in windows
            ):
                counts[key] += 1
                seen.add(identity)
        return counts

    def finish(self, root: Path, directory: Path, windows: list[dict[str, Any]], extensions: list[dict[str, Any]]) -> dict[str, Any]:
        self.stop()
        raw, samples = self.snapshot()
        raw.sort(key=lambda item: (item["started_offset_s"], item["npu_id"], item["chip_id"]))
        samples.sort(key=lambda item: (item["started_offset_s"], item["npu_id"], item["chip_id"], str(item["link"])))
        raw_path, parsed_path = directory / "samples.raw.jsonl", directory / "samples.jsonl"
        write_jsonl(raw_path, raw)
        write_jsonl(parsed_path, samples)
        counts, primary_counts = self.valid_counts(windows), self.valid_counts([w for w in windows if w["role"] == "primary"])
        successful = sum(record["returncode"] == 0 and bool(parse_npu_smi_hccs_bw(record["stdout"], record["npu_id"], record["chip_id"])) for record in raw)
        complete = bool(counts) and all(value >= MONITOR_MIN_SAMPLES_PER_CHIP for value in counts.values()) and all(value >= 1 for value in primary_counts.values())
        reasons: list[str] = []
        if not successful:
            reasons.append("npu-smi hccs-bw capability probe produced no parseable sample")
        for target, count in counts.items():
            if count < MONITOR_MIN_SAMPLES_PER_CHIP:
                reasons.append(f"target {target} has {count}/{MONITOR_MIN_SAMPLES_PER_CHIP} valid workload samples")
        for target, count in primary_counts.items():
            if count < 1:
                reasons.append(f"target {target} has no sample overlapping the primary workload command")
        result = {
            "schema_version": 2, "status": "passed" if complete else "partial",
            "evidence_kind": "link-bandwidth-timeline", "target_resource": "hccs-links",
            "collector": "npu-smi info -t hccs-bw", "targets": self.targets,
            "capability": {"supported": bool(successful), "successful_sample_commands": successful, "total_sample_commands": len(raw)},
            "required_fields": ["rx_bandwidth(GB/S)", "tx_bandwidth(GB/S)"],
            "target_interval_s": MONITOR_INTERVAL_S, "command_timeout_s": MONITOR_COMMAND_TIMEOUT_S,
            "required_samples_per_chip": MONITOR_MIN_SAMPLES_PER_CHIP,
            "workload_windows": windows, "sample_counts_by_target": counts,
            "primary_sample_counts_by_target": primary_counts, "extensions": extensions,
            "raw_samples": artifact_ref(raw_path, root), "parsed_samples": artifact_ref(parsed_path, root),
            "reasons": reasons,
        }
        write_json(directory / "summary.json", result)
        result["summary"] = artifact_ref(directory / "summary.json", root)
        return result


def extract_json(text: str, *, preserve_float_lexemes: bool = False) -> Any | None:
    decoder = json.JSONDecoder(parse_float=str) if preserve_float_lexemes else json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _walk_json(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, path + (str(index),))
    else:
        yield path, value


def _value_at_path(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for component in path:
        if isinstance(current, dict):
            current = current[component]
        elif isinstance(current, list):
            current = current[int(component)]
        else:
            return None
    return current


def _numeric_lexeme(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    match = re.fullmatch(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"(?:\s*[A-Za-z/]+)?\s*",
        str(value),
    )
    return match.group(1) if match else None


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, str):
        match = re.fullmatch(
            r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"(?:\s*[A-Za-z/]+)?\s*",
            value,
        )
        if not match:
            return None
        value = match.group(1)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _device_from_context(path: tuple[str, ...], document: Any) -> str | None:
    text = "/".join(path)
    match = DEVICE_RE.search(text)
    if match:
        return match.group(1)
    # JSON commonly stores the device ID next to the metric rather than in its key.
    if isinstance(document, dict):
        for key, value in document.items():
            if re.search(r"(?:device|npu|card).*id", str(key), re.I):
                if isinstance(value, (str, int)):
                    return str(value)
            found = _device_from_context(path, value) if isinstance(value, (dict, list)) else None
            if found:
                return found
    elif isinstance(document, list):
        for child in document:
            found = _device_from_context(path, child) if isinstance(child, (dict, list)) else None
            if found:
                return found
    return None


def parse_compute(text: str, dtype: str, expected_unit: str) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    document = extract_json(text)
    raw_document = extract_json(text, preserve_float_lexemes=True)
    if document is not None:
        marker = dtype.lower()
        for path, value in _walk_json(document):
            key = "/".join(path).lower()
            if marker not in key and expected_unit.lower() not in key:
                continue
            leaf = path[-1].lower() if path else ""
            if "power" in leaf or "duration" in leaf or "time" in leaf:
                continue
            number = _finite_positive(value)
            if number is None:
                continue
            metrics.append({
                "device": _device_from_context(path, document),
                "value": number,
                "value_raw": _numeric_lexeme(_value_at_path(raw_document, path)),
                "unit": expected_unit,
                "source": "json",
                "field": "/".join(path),
            })
    if metrics:
        return metrics

    header_lines = [line for line in text.splitlines() if dtype.lower() in line.lower()]
    for header in header_lines:
        unit_match = re.search(r"(TFLOPS|TOPS)\s*@?\s*" + re.escape(dtype), header, re.I)
        if not unit_match:
            continue
        actual_unit = unit_match.group(1).upper()
        if actual_unit != expected_unit:
            raise EvidenceError(
                f"{dtype.upper()} unit mismatch: expected {expected_unit}, got {actual_unit}"
            )
        lines = text.splitlines()
        header_index = lines.index(header)
        for row in lines[header_index + 1:]:
            if not row.strip() or set(row.strip()) <= {"-", "+", "="}:
                continue
            columns = row.split()
            if len(columns) < 4 or not re.fullmatch(r"\d+(?:/\d+)?", columns[0]):
                if metrics:
                    break
                continue
            number = _finite_positive(columns[3].replace(",", ""))
            if number is not None:
                metrics.append({
                    "device": columns[0], "value": number,
                    "value_raw": _numeric_lexeme(columns[3].replace(",", "")),
                    "unit": expected_unit, "source": "semantic-text",
                })
    if not metrics:
        raise EvidenceError(f"no {dtype.upper()} {expected_unit} metric found")
    return metrics


def parse_compute_context(text: str) -> list[dict[str, Any]]:
    """Keep DMI execution context fields verbatim; never derive a new rate."""
    document = extract_json(text)
    contexts: list[dict[str, Any]] = []

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            normalized = {
                re.sub(r"[^a-z0-9]", "", str(key).lower()): (str(key), child)
                for key, child in value.items()
            }
            context: dict[str, Any] = {}
            for canonical, candidates in {
                "device": ("device", "deviceid"),
                "execute_times": ("executetimes",),
                "duration": ("duration", "durationms"),
                "power": ("power", "powerw"),
            }.items():
                for candidate in candidates:
                    if candidate in normalized:
                        source_key, source_value = normalized[candidate]
                        context[canonical] = source_value
                        context.setdefault("source_fields", {})[canonical] = source_key
                        break
            if any(key in context for key in ("execute_times", "duration", "power")):
                context["source_path"] = "/".join(path) or "$"
                contexts.append(context)
            for key, child in value.items():
                visit(child, path + (str(key),))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + (str(index),))

    if document is not None:
        visit(document)
    return contexts


def parse_tabular_bandwidth(text: str) -> list[dict[str, Any]]:
    """Parse labelled DMI bandwidth tables without depending on line numbers."""
    metrics: list[dict[str, Any]] = []
    document = extract_json(text)
    if document is not None:
        for path, value in _walk_json(document):
            key = "/".join(path).lower()
            leaf = path[-1].lower() if path else ""
            if "bandwidth" not in leaf or "threshold" in leaf:
                continue
            number = _finite_positive(value)
            if number is not None:
                metrics.append({
                    "device": _device_from_context(path, document),
                    "value": number, "unit": "GB/s", "source": "json",
                    "field": "/".join(path),
                })
    if metrics:
        return metrics

    lines = text.splitlines()
    for index, header in enumerate(lines):
        if "bandwidth" not in header.lower() or "gb/s" not in header.lower().replace(" ", ""):
            continue
        normalized_header = re.sub(r"Execute\s+Times", "ExecuteTimes", header, flags=re.I)
        normalized_header = re.sub(r"Elapsed\s+Time", "ElapsedTime", normalized_header, flags=re.I)
        columns = normalized_header.replace("|", " ").split()
        bw_index = next(
            (i for i, token in enumerate(columns) if "bandwidth" in token.lower()),
            None,
        )
        if bw_index is None:
            continue
        for row in lines[index + 1:]:
            stripped = row.strip()
            if not stripped or set(stripped) <= {"-", "+", "=", "|"}:
                continue
            values = row.replace("|", " ").split()
            if len(values) <= bw_index:
                if metrics:
                    break
                continue
            number = _finite_positive(values[bw_index])
            if number is None:
                if metrics:
                    break
                continue
            metrics.append({
                "device": values[0], "value": number, "unit": "GB/s",
                "source": "semantic-text",
            })
    if not metrics:
        raise EvidenceError("no labelled bandwidth (GB/s) metric found")
    return metrics


def parse_p2p(text: str) -> list[dict[str, Any]]:
    """Parse P2P tables/matrices by section and device labels."""
    section: str | None = None
    matrix: list[dict[str, Any]] = []
    for line in text.splitlines():
        lowered = line.lower()
        if "unidirectional peer to peer" in lowered:
            section = "unidirectional"
            continue
        if "bidirectional peer to peer" in lowered:
            section = "bidirectional"
            continue
        if section is None:
            continue
        columns = line.replace("|", " ").split()
        if len(columns) < 3 or not columns[0].isdigit():
            continue
        if all(token.isdigit() and int(token) == index
               for index, token in enumerate(columns)):
            # Matrix destination-device header, not a result row.
            continue
        source = columns[0]
        for destination, token in enumerate(columns[1:]):
            number = _finite_positive(token)
            if number is None or str(destination) == source:
                continue
            matrix.append({
                "source_device": source,
                "destination_device": str(destination),
                "direction": section,
                "value": number,
                "unit": "GB/s",
                "source": "semantic-text-matrix",
            })
    if matrix:
        return matrix
    metrics = parse_tabular_bandwidth(text) if re.search(
        r"Bandwidth\s*\(GB/s\)", text, re.I
    ) else []
    if metrics:
        return [metric | {"direction": "table"} for metric in metrics]
    raise EvidenceError("no labelled P2P bandwidth table or matrix found")


def parse_p2p_pair(text: str, source: int, destination: int) -> list[dict[str, Any]]:
    """Parse one explicit DMI --ds/--dd run without inventing its reverse pair."""
    document = extract_json(text)
    metrics: list[dict[str, Any]] = []
    # DMI 26.1 emits one group per direction and one result per transfer size.
    # A selected-pair report needs one comparable value per direction, so retain
    # the largest transfer size from each group.  Do not walk every descendant
    # below "bandwidths": elapsed time, execute count and byte size are numeric
    # too, but they are not bandwidth measurements.
    groups = document.get("bandwidths") if isinstance(document, dict) else None
    if isinstance(groups, list):
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict) or not isinstance(group.get("results"), list):
                continue
            candidates: list[tuple[float, float, dict[str, Any], int]] = []
            for result_index, result in enumerate(group["results"]):
                if not isinstance(result, dict):
                    continue
                bandwidth = _finite_positive(result.get("bandwidth"))
                size = _finite_positive(result.get("size"))
                if bandwidth is not None and size is not None:
                    candidates.append((size, bandwidth, result, result_index))
            if not candidates:
                continue
            size, bandwidth, result, result_index = max(candidates, key=lambda item: item[0])
            type_text = str(group.get("type", "")).lower()
            if "bidirectional" in type_text:
                direction = "bidirectional"
            elif "unidirectional" in type_text:
                direction = "unidirectional"
            else:
                direction = "pair"
            reported_source = group.get("source_device_id")
            reported_destination = group.get("target_device_id")
            metric: dict[str, Any] = {
                "source_device": str(source),
                "destination_device": str(destination),
                "direction": direction,
                "value": bandwidth,
                "unit": "GB/s",
                "transfer_size_bytes": int(size),
                "source": "dmi-json-largest-transfer-pair",
                "field": f"bandwidths/{group_index}/results/{result_index}/bandwidth",
            }
            if reported_source is not None and str(reported_source) != str(source):
                metric["reported_source_device"] = str(reported_source)
            if reported_destination is not None and str(reported_destination) != str(destination):
                metric["reported_destination_device"] = str(reported_destination)
            elapsed = result.get("elapsed_time")
            if elapsed is not None:
                metric["elapsed_time"] = elapsed
            execute_times = result.get("execute_times")
            if execute_times is not None:
                metric["execute_times"] = execute_times
            metrics.append(metric)
    elif document is not None:
        # Accept compact semantic JSON used by older/alternate DMI builds, but
        # only fields whose own key is explicitly a bandwidth field.
        for path, value in _walk_json(document):
            field = path[-1].lower() if path else ""
            if "bandwidth" not in field:
                continue
            number = _finite_positive(value)
            if number is None:
                continue
            if "bidirectional" in field or "bi_direction" in field:
                direction = "bidirectional"
            elif "unidirectional" in field or "uni_direction" in field:
                direction = "unidirectional"
            else:
                direction = "pair"
            metrics.append({
                "source_device": str(source),
                "destination_device": str(destination),
                "direction": direction,
                "value": number,
                "unit": "GB/s",
                "source": "semantic-json-pair",
                "field": "/".join(path),
            })
    if metrics:
        return metrics
    parsed = parse_p2p(text)
    for metric in parsed:
        reported_source = metric.get("source_device")
        reported_destination = metric.get("destination_device")
        if reported_source not in (None, str(source)):
            metric["reported_source_device"] = reported_source
        if reported_destination not in (None, str(destination)):
            metric["reported_destination_device"] = reported_destination
        metric["source_device"] = str(source)
        metric["destination_device"] = str(destination)
        metric["source"] = "semantic-text-pair"
    return parsed


def parse_latency(
    text: str, *, direction: str, size_bytes: int,
    device: int | None = None, source: int | None = None,
    destination: int | None = None,
) -> list[dict[str, Any]]:
    """Parse only explicitly labelled nanosecond latency fields."""
    document = extract_json(text)
    metrics: list[dict[str, Any]] = []
    if document is not None:
        if isinstance(document, dict) and isinstance(document.get("error"), dict):
            error = document["error"]
            code = error.get("code", "unknown")
            description = " ".join(str(error.get("description", "")).split())
            raise EvidenceError(f"DMI error {code}: {description}".rstrip())
        leaves = list(_walk_json(document))
        types = [str(value).lower() for path, value in leaves
                 if path and path[-1].lower() == "type"]
        expected_type = {"h2d": "host to device", "d2h": "device to host",
                         "p2p": "peer to peer"}[direction]
        if types and not all(expected_type in item for item in types):
            raise EvidenceError(f"latency direction mismatch: expected {direction}, reported {types}")
        reported_sizes = [int(number) for path, value in leaves
                          if path and path[-1].lower() == "size"
                          if (number := _finite_positive(value)) is not None]
        if reported_sizes and any(item != size_bytes for item in reported_sizes):
            raise EvidenceError(
                f"latency size mismatch: requested {size_bytes}, reported {reported_sizes}"
            )
        endpoint_fields = {path[-1].lower(): str(value) for path, value in leaves if path}
        if device is not None and "device_id" in endpoint_fields \
                and endpoint_fields["device_id"] != str(device):
            raise EvidenceError("latency output reports a different Device")
        if source is not None and endpoint_fields.get("src_device_id", str(source)) != str(source):
            raise EvidenceError("P2P latency output reports a different source Device")
        if destination is not None and endpoint_fields.get("dst_device_id", str(destination)) != str(destination):
            raise EvidenceError("P2P latency output reports a different destination Device")
        for path, value in leaves:
            leaf = path[-1].lower() if path else ""
            full = "/".join(path).lower()
            if not any(marker in leaf for marker in ("latency", "delay")):
                continue
            if any(marker in leaf for marker in ("threshold", "min", "max")):
                continue
            number = _finite_positive(value)
            if number is None:
                continue
            # Unit must be explicit in the field/path or scalar value.
            scalar = str(value).lower()
            if "ns" not in full and "ns" not in scalar and "nanosecond" not in full:
                continue
            metrics.append({
                "value": number, "unit": "ns", "size_bytes": size_bytes,
                "direction": direction, "source": "json", "field": "/".join(path),
            })
    if not metrics:
        for line in text.splitlines():
            if not re.search(r"(?:latency|delay)", line, re.I) or not re.search(r"\bns\b", line, re.I):
                continue
            numbers = NUMBER_RE.findall(line)
            if numbers:
                number = _finite_positive(numbers[-1])
                if number is not None:
                    metrics.append({
                        "value": number, "unit": "ns", "size_bytes": size_bytes,
                        "direction": direction, "source": "semantic-text",
                    })
    if not metrics:
        raise EvidenceError(
            f"no explicitly labelled {direction} latency (ns) metric found for {size_bytes} bytes"
        )
    # One command describes one endpoint and size. Multiple latency leaves are
    # ambiguous and must not be silently averaged or selected.
    if len(metrics) != 1:
        raise EvidenceError(f"ambiguous latency output: found {len(metrics)} metrics")
    metric = metrics[0]
    if device is not None:
        metric["device"] = str(device)
    if source is not None and destination is not None:
        metric["source_device"] = str(source)
        metric["destination_device"] = str(destination)
    return metrics


def parse_hccl_allreduce(text: str) -> list[dict[str, Any]]:
    """Parse hccl_test AllReduce result rows by their labelled table header."""
    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        lowered = line.lower()
        if all(item in lowered for item in ("data_size", "aveg_time", "alg_bandwidth", "check_result")):
            header_index = index
            break
    if header_index is None:
        raise EvidenceError("HCCL output has no labelled result header")
    metrics: list[dict[str, Any]] = []
    for line in lines[header_index + 1:]:
        columns = [item.strip() for item in line.split("|")]
        if len(columns) < 4:
            continue
        size = _finite_positive(columns[0])
        avg_time = _finite_positive(columns[1])
        bandwidth = _finite_positive(columns[2])
        check = columns[3].strip()
        if size is None or avg_time is None or bandwidth is None:
            continue
        verified = bool(re.search(r"success|pass|ok", check, re.I)) and not bool(
            re.search(r"fail|error", check, re.I)
        )
        metrics.append({
            "message_size_bytes": int(size), "avg_time_us": avg_time,
            "value": bandwidth, "unit": "GB/s", "collective": "all_reduce",
            "datatype": "fp32", "op": "sum", "verification": check,
            "verification_passed": verified, "source": "semantic-text",
        })
    if not metrics:
        raise EvidenceError("HCCL output has no valid AllReduce rows")
    if len({item["message_size_bytes"] for item in metrics}) != len(metrics):
        raise EvidenceError("HCCL output contains duplicate message sizes")
    return metrics


def parse_p2p_latency_matrix(text: str, size_bytes: int) -> list[dict[str, Any]]:
    """Parse a labelled directed P2P latency matrix in nanoseconds."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if "latency" in line.lower() and re.search(r"\bns\b", line, re.I)), None)
    if start is None:
        raise EvidenceError("no labelled P2P latency (ns) matrix found")
    matrix: list[dict[str, Any]] = []
    for line in lines[start + 1:]:
        columns = line.replace("|", " ").split()
        if len(columns) < 3 or not columns[0].isdigit():
            continue
        if all(token.isdigit() and int(token) == index for index, token in enumerate(columns)):
            continue
        source = columns[0]
        for destination, token in enumerate(columns[1:]):
            value = _finite_positive(token)
            if value is None or source == str(destination):
                continue
            matrix.append({
                "source_device": source, "destination_device": str(destination),
                "direction": "p2p", "size_bytes": size_bytes,
                "value": value, "unit": "ns", "source": "semantic-text-matrix",
            })
    if not matrix:
        raise EvidenceError("P2P latency matrix contains no directed metrics")
    return matrix


def parse_capacity(text: str, card: int, chip: int) -> dict[str, Any]:
    match = re.search(r"HBM\s+Capacity\s*\(MB\)\s*[:|]?\s*(\d+)", text, re.I)
    if not match:
        for line in text.splitlines():
            if "HBM Capacity(MB)" in line:
                numbers = NUMBER_RE.findall(line)
                if numbers:
                    match_value = numbers[-1]
                    break
        else:
            raise EvidenceError(f"no HBM capacity for card {card}, chip {chip}")
    else:
        match_value = match.group(1)
    value = _finite_positive(match_value)
    if value is None:
        raise EvidenceError(f"invalid HBM capacity for card {card}, chip {chip}")
    # Preserve the unit exactly as npu-smi names the source field.  The original
    # FlagPerf wrapper relabelled a doubled value as MiB; the optimized protocol
    # emits only this per-chip MB source value.
    return {
        "scope": "chip",
        "card": card,
        "chip": chip,
        "value": value,
        "value_raw": _numeric_lexeme(match_value),
        "unit": "MB",
        "source_field": "HBM Capacity(MB)",
    }


def parse_labelled_fields(text: str) -> dict[str, str]:
    """Preserve npu-smi label/value pairs verbatim for static guard evidence."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"\s*([^:|]+?)\s*[:|]\s*(.*?)\s*$", line)
        if match and match.group(1).strip():
            fields[match.group(1).strip()] = match.group(2).strip()
    return fields


def discovered_devices() -> list[int]:
    devices = []
    for path in Path("/dev").glob("davinci[0-9]*"):
        match = re.fullmatch(r"davinci(\d+)", path.name)
        if match:
            devices.append(int(match.group(1)))
    if not devices:
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        devices = [int(item) for item in visible.split(",") if item.strip().isdigit()]
    return sorted(set(devices))


def check_idle_devices(devices: list[int]) -> None:
    paths = [f"/dev/davinci{device}" for device in devices if Path(f"/dev/davinci{device}").exists()]
    if not paths or not shutil_which("fuser"):
        return
    proc = subprocess.run(["fuser", *paths], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode == 0 and (proc.stdout.strip() or proc.stderr.strip()):
        raise EvidenceError(
            "Ascend device nodes are in use; refusing disruptive DMI tests: "
            + (proc.stdout + proc.stderr).strip()
        )


def shutil_which(command: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def diagnosis_command(item: str, devices: list[int] | None = None) -> list[str]:
    command = ["ascend-dmi", "--dg", "--items", item]
    if devices is not None:
        command.extend(["-d", ",".join(map(str, devices))])
    if item == "signalQuality":
        command.extend(["--lt", "hccs"])
    if item in ("aiflops", "bandwidth"):
        command.append("-q")
    command.extend(["--fmt", "json"])
    return command


def diagnosis_passed(text: str) -> bool:
    document = extract_json(text)
    if document is not None:
        values = [str(value).lower() for _, value in _walk_json(document)
                  if isinstance(value, (str, int, float, bool))]
        flattened = "\n".join(values)
    else:
        flattened = text.lower()
    failure = re.search(
        r"\b(?:fail|failed|failure|unhealthy|incompatible|invalid)\b|diagnosis test failed",
        flattened,
    )
    return bool(flattened.strip()) and failure is None


def diagnosis_status(text: str, returncode: int) -> str:
    lowered = text.lower()
    unsupported_markers = (
        "threshold failed from configuration file",
        "does not support",
        "not supported",
        "will not diagnose",
    )
    if any(marker in lowered for marker in unsupported_markers):
        return "unsupported"
    if returncode != 0 or not diagnosis_passed(text):
        return "failed"
    return "passed"


def health_coverage(text: str) -> dict[str, Any]:
    """Separate the result of executed health checks from capability coverage."""
    document = extract_json(text)
    if document is None:
        return {
            "status": "unknown",
            "uncovered": ["health-subitem-coverage"],
            "reason": "health output was not structured JSON",
        }
    leaves = list(_walk_json(document))
    flattened = "\n".join(
        str(value).lower() for _, value in leaves
        if isinstance(value, (str, int, float, bool))
    )
    expected_items = {"driver", "cann", "device", "hbm"}
    observed_items = {
        str(value).lower() for path, value in leaves
        if path and path[-1].lower() == "item_name"
    }
    missing_items = sorted(expected_items - observed_items)
    lost_card_unsupported = (
        ("lost card diagnosis" in flattened or "lost-card diagnosis" in flattened)
        and any(marker in flattened for marker in ("not support", "unsupported"))
    )
    uncovered = [f"health-item-{item}" for item in missing_items]
    if lost_card_unsupported:
        uncovered.append("lost-card-diagnosis")
    if uncovered:
        reasons = []
        if missing_items:
            reasons.append("requested health items absent: " + ",".join(missing_items))
        if lost_card_unsupported:
            reasons.append("DMI reports lost-card diagnosis unsupported in this context")
        return {
            "status": "partial",
            "uncovered": uncovered,
            "reason": "; ".join(reasons),
        }
    return {"status": "complete", "uncovered": []}


def legacy_command(case: str) -> list[str]:
    if case in COMPUTE_DTYPES:
        return ["ascend-dmi", "-f", "-t", COMPUTE_DTYPES[case][0], "-q"]
    if case == "main_memory-bandwidth":
        return ["ascend-dmi", "--bw", "-t", "d2d", "-q"]
    if case == "main_memory-capacity":
        return ["npu-smi", "info", "-t", "memory", "-i", "0", "-c", "0"]
    if case == "interconnect-h2d":
        return ["ascend-dmi", "--bw", "-t", "h2d", "-s", "536870912", "--et", "50", "-q"]
    return ["ascend-dmi", "--bw", "-t", "p2p", "-m", "card", "-q"]


def legacy_fixed_value(case: str, text: str) -> str | None:
    lines = text.splitlines()
    row_column = {
        "computation-BF16": (4, 4), "computation-FP16": (4, 4),
        "computation-FP32": (4, 4), "computation-INT8": (4, 4),
        "main_memory-bandwidth": (30, 3),
        "interconnect-h2d": (6, 4),
        "interconnect-P2P_intraserver": (22, 2),
    }.get(case)
    if row_column:
        row, column = row_column
        values = lines[row - 1].split() if len(lines) >= row else []
        return values[column - 1] if len(values) >= column else None
    if case == "main_memory-capacity":
        try:
            return str(int(parse_capacity(text, 0, 0)["value"]))
        except EvidenceError:
            return None
    return None


class Runner:
    def __init__(
        self, root: Path, cases: list[str], legacy_probe: bool,
        selected_devices: list[int] | None = None,
        selection_source: str = "default-all",
        latency_sizes: tuple[int, ...] = DEFAULT_LATENCY_SIZES,
        hccl_min_bytes: int = 8 << 10,
        hccl_max_bytes: int = 1 << 30,
        compute_monitor: bool = True,
        data_movement_monitor: bool = True,
    ) -> None:
        self.root = root
        self.cases = cases
        self.legacy_probe = legacy_probe
        self.inventory_devices = discovered_devices()
        self.explicit_selection = selected_devices is not None
        self.devices = sorted(selected_devices) if selected_devices is not None else list(self.inventory_devices)
        self.selection_source = selection_source
        self.latency_sizes = latency_sizes
        self.hccl_min_bytes = hccl_min_bytes
        self.hccl_max_bytes = hccl_max_bytes
        self.compute_monitor = compute_monitor
        self.data_movement_monitor = data_movement_monitor
        self.device_map: list[dict[str, Any]] = []
        self.topology_routes: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {
            "schema_version": 3,
            "run_id": root.name,
            "scope": (
                "single-node-selected-device" if self.explicit_selection
                else "single-node-full-device"
            ),
            "started_at": utc_now(),
            "status": "running",
            "requested_cases": cases,
            "discovered_device_ids": self.inventory_devices,
            "selection": {
                "source": selection_source,
                "selected_device_ids": self.devices,
                "excluded_device_ids": sorted(set(self.inventory_devices) - set(self.devices)),
            },
            "environment": {}, "health": {}, "topology": {},
            "diagnostics": {}, "cases": {},
            "legacy_probe": legacy_probe,
            "parameters": {
                "latency_sizes_bytes": list(latency_sizes),
                "hccl_min_bytes": hccl_min_bytes,
                "hccl_max_bytes": hccl_max_bytes,
                "hccl_factor": 2, "hccl_warmup": 10, "hccl_iterations": 20,
                "compute_monitor": "on" if compute_monitor else "off",
                "data_movement_monitor": "on" if data_movement_monitor else "off",
                "compute_execute_times": COMPUTE_EXECUTE_TIMES,
                "monitor_interval_s": MONITOR_INTERVAL_S,
                "monitor_min_samples_per_chip": MONITOR_MIN_SAMPLES_PER_CHIP,
                "monitor_max_extensions": MONITOR_MAX_EXTENSIONS,
            },
        }

    def save(self) -> None:
        write_json(self.root / "manifest.json", self.manifest)

    def collect_environment(self) -> None:
        directory = self.root / "environment"
        commands = {
            "dmi-version": ["ascend-dmi", "-v"],
            "dmi-compatibility": ["ascend-dmi", "-c", "--fmt", "json"],
            "dmi-info": ["ascend-dmi", "-i", "--fmt", "json"],
        }
        for label, command in commands.items():
            record = command_record(self.root, directory, command, timeout=180, label=label)
            public = {key: value for key, value in record.items() if not key.endswith("_text")}
            if label == "dmi-compatibility" and not diagnosis_passed(record["stdout_text"]):
                raise EvidenceError("DMI compatibility output reports a failure")
            self.manifest["environment"][label] = public

    def collect_topology(self) -> None:
        directory = self.root / "topology"
        commands = {
            "npu-smi-list": ["npu-smi", "info", "-l"],
            "npu-smi-map": ["npu-smi", "info", "-m"],
            "npu-smi-topology": ["npu-smi", "info", "-t", "topo"],
        }
        for label, command in commands.items():
            record = command_record(self.root, directory, command, timeout=120,
                                    label=label, required=False)
            self.manifest["topology"][label] = {
                key: value for key, value in record.items() if not key.endswith("_text")
            }
            if label == "npu-smi-map" and record["returncode"] == 0:
                self.device_map = parse_device_map(record["stdout_text"])
                self.manifest["topology"]["device_map"] = self.device_map
            if label == "npu-smi-topology" and record["returncode"] == 0:
                self.topology_routes = parse_npu_smi_topology(record["stdout_text"])
                self.manifest["topology"]["routes"] = self.topology_routes
        if all(
            self.manifest["topology"][label]["returncode"] != 0
            for label in commands
        ):
            raise PartialEvidence("npu-smi topology/list/map collection failed")

    def collect_health(self, phase: str) -> None:
        record = command_record(
            self.root, self.root / "health" / phase,
            diagnosis_command(
                "driver,cann,device,hbm",
                self.devices if self.explicit_selection else None,
            ),
            timeout=900, label="health", required=False,
        )
        public = {key: value for key, value in record.items() if not key.endswith("_text")}
        combined = record["stdout_text"] + record["stderr_text"]
        public["passed"] = record["returncode"] == 0 and diagnosis_passed(combined)
        public["result"] = "passed" if public["passed"] else "failed"
        public["coverage"] = health_coverage(combined)
        public["execution_context"] = execution_context()
        self.manifest["health"][phase] = public
        if not public["passed"]:
            raise EvidenceError(f"{phase}-test device health diagnosis failed")

    def collect_diagnostics(self) -> None:
        needed = sorted({item for case in self.cases
                         if (item := DIAGNOSIS_FOR_CASE[case]) is not None})
        for item in needed:
            related = [
                case for case in self.cases
                if DIAGNOSIS_FOR_CASE[case] == item
            ]
            measured = [
                case for case in related
                if self.manifest["cases"].get(case, {}).get("measurement_status") == "passed"
            ]
            if not measured:
                self.manifest["diagnostics"][item] = {
                    "status": "not-run",
                    "passed": False,
                    "reason": (
                        "diagnosis was not run because no related Case completed "
                        "its measurement safely"
                    ),
                }
                continue
            if self.explicit_selection and item == "signalQuality" and len(self.devices) < 2:
                self.manifest["diagnostics"][item] = {
                    "status": "not-run",
                    "passed": False,
                    "reason": "P2P diagnosis requires at least two selected logical Devices",
                }
                continue
            record = command_record(
                self.root, self.root / "diagnostics" / item,
                diagnosis_command(
                    item, self.devices if self.explicit_selection else None
                ), timeout=1800,
                label="diagnosis", required=False,
            )
            public = {key: value for key, value in record.items() if not key.endswith("_text")}
            combined = record["stdout_text"] + record["stderr_text"]
            public["status"] = diagnosis_status(combined, record["returncode"])
            public["passed"] = public["status"] == "passed"
            self.manifest["diagnostics"][item] = public

    def optimized_command(self, case: str) -> list[str]:
        if case in COMPUTE_DTYPES:
            dtype = COMPUTE_DTYPES[case][0]
            return [
                "ascend-dmi", "-f", "-t", dtype, "--all", "--et",
                str(COMPUTE_EXECUTE_TIMES), "-q", "--fmt", "json",
            ]
        if case == "main_memory-bandwidth":
            # On A3, DMI fixes D2D size and execute-times and rejects both flags.
            return ["ascend-dmi", "--bw", "-t", "d2d", "-q", "--fmt", "json"]
        if case == "interconnect-h2d":
            return ["ascend-dmi", "--bw", "-t", "h2d", "-s", "536870912", "--et", "50", "-q", "--fmt", "json"]
        if case == "interconnect-d2h":
            return ["ascend-dmi", "--bw", "-t", "d2h", "-s", "536870912", "--et", "50", "-q", "--fmt", "json"]
        if case == "interconnect-P2P_intraserver":
            # DMI 26.1 card-mode P2P may ignore --fmt; text is parsed semantically.
            return ["ascend-dmi", "--bw", "-t", "p2p", "-m", "card", "-q"]
        raise EvidenceError(f"no single command for {case}")

    def device_command(self, case: str, device: int) -> list[str]:
        if case in COMPUTE_DTYPES:
            dtype = COMPUTE_DTYPES[case][0]
            return [
                "ascend-dmi", "-f", "-t", dtype, "-d", str(device),
                "--et", str(COMPUTE_EXECUTE_TIMES), "-q", "--fmt", "json",
            ]
        if case == "main_memory-bandwidth":
            return [
                "ascend-dmi", "--bw", "-t", "d2d", "-d", str(device),
                "-q", "--fmt", "json",
            ]
        if case in ("interconnect-h2d", "interconnect-d2h"):
            direction = "h2d" if case == "interconnect-h2d" else "d2h"
            return [
                "ascend-dmi", "--bw", "-t", direction, "-d", str(device),
                "-s", "536870912", "--et", "50", "-q", "--fmt", "json",
            ]
        raise EvidenceError(f"case does not support a per-Device command: {case}")

    def run_per_device(
        self, case: str, directory: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        commands: list[dict[str, Any]] = []
        monitor_units: list[dict[str, Any]] = []
        for device in self.devices:
            device_metrics, record, monitor = self.run_data_movement_unit(
                case, directory / f"device-{device}",
                directory / "monitor" / f"device-{device}",
                self.device_command(case, device), [device],
            )
            monitor_units.append(monitor)
            commands.append({
                key: value for key, value in record.items()
                if not key.endswith("_text")
            })
            if record["returncode"] != 0:
                raise PartialEvidence(f"{case} failed for selected Device {device}")
            for metric in device_metrics:
                reported = metric.get("device")
                if reported not in (None, "", str(device)):
                    metric["reported_device"] = reported
                metric["device"] = str(device)
            metrics.extend(device_metrics)
        observed = {str(metric.get("device")) for metric in metrics}
        expected = {str(device) for device in self.devices}
        if observed != expected:
            raise PartialEvidence(
                f"{case} metrics cover selected Devices {sorted(observed)}, "
                f"expected {sorted(expected)}"
            )
        return metrics, commands, self.aggregate_data_movement_units(
            directory, monitor_units, [],
        )

    @staticmethod
    def public_command(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in record.items()
            if not key.endswith("_text")
        }

    def monitor_targets(self, logical_devices: list[int]) -> list[dict[str, int]]:
        selected = set(logical_devices)
        targets = [
            {
                "npu_id": int(item["npu_id"]),
                "chip_id": int(item["chip_id"]),
                "logic_id": int(item["logic_id"]),
            }
            for item in self.device_map if int(item["logic_id"]) in selected
        ]
        targets.sort(key=lambda item: (item["npu_id"], item["chip_id"], item["logic_id"]))
        if {item["logic_id"] for item in targets} != selected:
            raise EvidenceError(
                "npu-smi map does not fully resolve compute monitor targets"
            )
        return targets

    def run_compute_unit(
        self, case: str, command_directory: Path, monitor_directory: Path,
        command: list[str], logical_devices: list[int],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Run one authoritative DMI command with optional concurrent monitoring."""
        if not getattr(self, "compute_monitor", True):
            record = command_record(
                self.root, command_directory, command, timeout=3600,
                label="microbenchmark", required=False,
            )
            metrics = self.parse_case(case, record["stdout_text"]) if record["returncode"] == 0 else []
            contexts = parse_compute_context(record["stdout_text"])
            if contexts:
                for metric in metrics:
                    metric["dmi_context"] = contexts[0]
            return metrics, record, {
                "schema_version": 1,
                "status": "not-run",
                "reason": "compute monitoring disabled by --compute-monitor off",
                "targets": (
                    self.monitor_targets(logical_devices)
                    if getattr(self, "device_map", []) else []
                ),
            }

        monitor = UsageMonitor(self.monitor_targets(logical_devices))
        monitor.start()
        windows: list[dict[str, Any]] = []
        extensions: list[dict[str, Any]] = []
        try:
            record = command_record(
                self.root, command_directory, command, timeout=3600,
                label="microbenchmark", required=False,
            )
            windows.append(monitor.relative_window(record, "primary"))
            metrics = self.parse_case(case, record["stdout_text"]) if record["returncode"] == 0 else []
            contexts = parse_compute_context(record["stdout_text"])
            if contexts:
                for metric in metrics:
                    metric["dmi_context"] = contexts[0]
            counts = monitor.valid_counts(windows)
            needs_extension = (
                record["returncode"] == 0
                and bool(counts)
                and min(counts.values()) > 0
                and min(counts.values()) < MONITOR_MIN_SAMPLES_PER_CHIP
            )
            if needs_extension and MONITOR_MAX_EXTENSIONS:
                extension_record = command_record(
                    self.root, monitor_directory / "extensions" / "extension-1",
                    command, timeout=3600, label="microbenchmark", required=False,
                )
                windows.append(monitor.relative_window(extension_record, "extension-1"))
                extension_public = self.public_command(extension_record)
                if extension_record["returncode"] == 0:
                    try:
                        extension_public["raw_dmi_metrics"] = self.parse_case(
                            case, extension_record["stdout_text"]
                        )
                        extension_contexts = parse_compute_context(
                            extension_record["stdout_text"]
                        )
                        if extension_contexts:
                            for metric in extension_public["raw_dmi_metrics"]:
                                metric["dmi_context"] = extension_contexts[0]
                    except EvidenceError as exc:
                        extension_public["parse_error"] = str(exc)
                extensions.append(extension_public)
        finally:
            try:
                monitor_summary = monitor.finish(
                    self.root, monitor_directory, windows, extensions
                )
            except Exception as exc:
                monitor.stop()
                monitor_summary = {
                    "schema_version": 1,
                    "status": "partial",
                    "collector": "npu-smi info -t usages",
                    "targets": monitor.targets,
                    "workload_windows": windows,
                    "extensions": extensions,
                    "reasons": [
                        f"monitor evidence finalization failed: {type(exc).__name__}: {exc}"
                    ],
                }
        return metrics, record, monitor_summary

    def aggregate_monitor_units(
        self, directory: Path, units: list[dict[str, Any]]
    ) -> dict[str, Any]:
        statuses = [unit.get("status", "partial") for unit in units]
        if statuses and all(status == "not-run" for status in statuses):
            status = "not-run"
        else:
            status = "passed" if statuses and all(
                item == "passed" for item in statuses
            ) else "partial"
        result = {
            "schema_version": 1,
            "status": status,
            "collector": "npu-smi info -t usages",
            "required_samples_per_chip": MONITOR_MIN_SAMPLES_PER_CHIP,
            "units": units,
            "reasons": [
                reason for unit in units for reason in unit.get("reasons", [])
            ],
        }
        path = directory / "monitor" / "summary.json"
        write_json(path, result)
        result["summary"] = artifact_ref(path, self.root)
        return result

    def run_data_movement_unit(
        self, case: str, command_directory: Path, monitor_directory: Path,
        command: list[str], logical_devices: list[int],
        parser: Any | None = None, label: str = "microbenchmark",
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Run a DMI/HCCL command while collecting independent raw evidence."""
        parse = parser or (lambda text: self.parse_case(case, text))
        if not getattr(self, "data_movement_monitor", False):
            record = command_record(
                self.root, command_directory, command,
                timeout=7200 if case == "interconnect-MPI_intraserver" else 3600,
                label=label, required=False,
            )
            metrics = []
            if record["returncode"] == 0:
                try:
                    metrics = parse(record["stdout_text"])
                except EvidenceError as exc:
                    record["parse_error"] = str(exc)
            return metrics, record, {
                "schema_version": 2, "status": "not-run",
                "evidence_kind": "not-run", "targets": [],
                "reason": "data movement monitoring disabled by --data-movement-monitor off",
            }

        targets = self.monitor_targets(logical_devices)

        if case == "main_memory-bandwidth":
            monitor: Any = UsageMonitor(
                targets, ("HBM Bandwidth Usage Rate(%)",)
            )
        else:
            monitor = HccsBandwidthMonitor(targets)
        monitor.start()
        windows: list[dict[str, Any]] = []
        extensions: list[dict[str, Any]] = []
        try:
            record = command_record(
                self.root, command_directory, command,
                timeout=7200 if case == "interconnect-MPI_intraserver" else 3600,
                label=label, required=False,
            )
            windows.append(monitor.relative_window(record, "primary"))
            metrics = []
            if record["returncode"] == 0:
                try:
                    metrics = parse(record["stdout_text"])
                except EvidenceError as exc:
                    record["parse_error"] = str(exc)
            counts = monitor.valid_counts(windows)
            if (
                record["returncode"] == 0 and counts
                and min(counts.values()) > 0
                and min(counts.values()) < MONITOR_MIN_SAMPLES_PER_CHIP
                and MONITOR_MAX_EXTENSIONS
            ):
                extension_record = command_record(
                    self.root, monitor_directory / "extensions" / "extension-1",
                    command,
                    timeout=7200 if case == "interconnect-MPI_intraserver" else 3600,
                    label=label, required=False,
                )
                windows.append(monitor.relative_window(extension_record, "extension-1"))
                extension_public = self.public_command(extension_record)
                if extension_record["returncode"] == 0:
                    try:
                        extension_public["raw_dmi_metrics"] = parse(
                            extension_record["stdout_text"]
                        )
                    except EvidenceError as exc:
                        extension_public["parse_error"] = str(exc)
                extensions.append(extension_public)
        finally:
            try:
                summary = monitor.finish(
                    self.root, monitor_directory, windows, extensions
                )
            except Exception as exc:
                monitor.stop()
                summary = {
                    "schema_version": 2, "status": "partial",
                    "collector": (
                        "npu-smi info -t usages" if case == "main_memory-bandwidth"
                        else "npu-smi info -t hccs-bw"
                    ),
                    "targets": targets, "workload_windows": windows,
                    "extensions": extensions,
                    "reasons": [f"monitor evidence finalization failed: {type(exc).__name__}: {exc}"],
                }
        return metrics, record, summary

    def movement_routes(self, logical_devices: list[int]) -> list[dict[str, Any]]:
        selected = set(logical_devices)
        device_map = getattr(self, "device_map", [])
        phy_by_logic = {
            int(item["logic_id"]): int(item.get("phy_id", item["logic_id"]))
            for item in device_map if "logic_id" in item
        }
        logic_by_phy = {phy: logic for logic, phy in phy_by_logic.items()}
        selected_phy = {phy_by_logic.get(logic, logic) for logic in selected}
        routes: list[dict[str, Any]] = []
        for item in self.topology_routes:
            source_phy, destination_phy = item["source_device"], item["destination_device"]
            if source_phy not in selected_phy or destination_phy not in selected_phy:
                continue
            routes.append({
                "source_device": logic_by_phy.get(source_phy, source_phy),
                "destination_device": logic_by_phy.get(destination_phy, destination_phy),
                "source_phy_id": source_phy,
                "destination_phy_id": destination_phy,
                "relation": item["relation"],
            })
        return routes

    def aggregate_data_movement_units(
        self, directory: Path, units: list[dict[str, Any]],
        logical_devices: list[int],
    ) -> dict[str, Any]:
        statuses = [unit.get("status", "partial") for unit in units]
        if statuses and all(status == "not-run" for status in statuses):
            status = "not-run"
        else:
            status = "passed" if statuses and all(status == "passed" for status in statuses) else "partial"
        routes = self.movement_routes(logical_devices)
        sio_routes = [route for route in routes if route["relation"] == "SIO"]
        uncovered_routes = [
            route for route in routes
            if route["relation"] not in ("HCCS", "HCCS_SW", "SIO")
        ]
        reasons = [reason for unit in units for reason in unit.get("reasons", [])]
        if sio_routes and status != "not-run":
            status = "partial"
            reasons.append(
                "selected topology contains SIO routes; locked CANN 9 stack exposes no dynamic SIO bandwidth counter"
            )
        expected_route_count = len(logical_devices) * max(0, len(logical_devices) - 1)
        if logical_devices and len(routes) != expected_route_count and status != "not-run":
            status = "partial"
            reasons.append(
                f"topology route coverage is {len(routes)}/{expected_route_count} directed routes"
            )
        if uncovered_routes and status != "not-run":
            status = "partial"
            reasons.append("selected topology contains routes not covered by the HCCS bandwidth collector")
        result = {
            "schema_version": 2, "status": status,
            "evidence_kind": "data-movement-evidence",
            "units": units, "routes": routes,
            "route_coverage": {
                "selected_route_count": len(routes),
                "hccs_or_switch_count": sum(route["relation"] in ("HCCS", "HCCS_SW") for route in routes),
                "sio_count": len(sio_routes),
                "unknown_count": len(uncovered_routes),
            },
            "reasons": list(dict.fromkeys(reasons)),
        }
        path = directory / "monitor" / "summary.json"
        write_json(path, result)
        result["summary"] = artifact_ref(path, self.root)
        return result

    def run_compute_all(
        self, case: str, directory: Path
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        metrics, record, unit = self.run_compute_unit(
            case, directory / "all-devices", directory / "monitor" / "all-devices",
            self.optimized_command(case), list(self.devices),
        )
        if record["returncode"] != 0:
            raise PartialEvidence(f"{case} all-device DMI command failed")
        self.validate_coverage(case, metrics)
        return metrics, self.public_command(record), self.aggregate_monitor_units(
            directory, [unit]
        )

    def run_compute_groups(
        self, case: str, directory: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Run once per complete physical NPU execution group reported by A3 DMI."""
        selected = set(self.devices)
        groups: dict[int, set[int]] = {}
        for item in self.device_map:
            groups.setdefault(item["npu_id"], set()).add(item["logic_id"])
        targets: list[tuple[int, list[int]]] = []
        for npu_id, inventory_devices in sorted(groups.items()):
            chosen = selected & inventory_devices
            if not chosen:
                continue
            if chosen != inventory_devices:
                raise PartialEvidence(
                    f"{case} DMI executes physical NPU {npu_id} as logical Device "
                    f"group {sorted(inventory_devices)}; selection {sorted(chosen)} is "
                    "incomplete, so the compute command was not run"
                )
            targets.append((npu_id, sorted(inventory_devices)))
        if not targets or set().union(*(set(devices) for _, devices in targets)) != selected:
            raise EvidenceError(
                "npu-smi map does not fully resolve selected Devices into compute groups"
            )

        metrics: list[dict[str, Any]] = []
        commands: list[dict[str, Any]] = []
        monitor_units: list[dict[str, Any]] = []
        for npu_id, group_devices in targets:
            representative = group_devices[0]
            label = f"npu-{npu_id}-devices-{'-'.join(map(str, group_devices))}"
            group_metrics, record, monitor_unit = self.run_compute_unit(
                case, directory / label, directory / "monitor" / label,
                self.device_command(case, representative), group_devices,
            )
            commands.append(self.public_command(record))
            monitor_units.append(monitor_unit)
            if record["returncode"] != 0:
                raise PartialEvidence(f"{case} failed for selected physical NPU {npu_id}")
            for metric in group_metrics:
                reported = metric.get("device")
                metric["device"] = (
                    str(reported) if reported not in (None, "")
                    else "/".join(map(str, group_devices))
                )
                metric["npu_id"] = npu_id
                metric["selected_device_ids"] = group_devices
            metrics.extend(group_metrics)
        if {metric.get("npu_id") for metric in metrics} != {
            npu_id for npu_id, _ in targets
        }:
            raise PartialEvidence(f"{case} metrics do not cover every selected physical NPU")
        return metrics, commands, self.aggregate_monitor_units(directory, monitor_units)

    def run_capacity(self, directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        commands: list[dict[str, Any]] = []
        static_samples: list[dict[str, Any]] = []
        selected = set(self.devices)
        targets = sorted({
            (item["npu_id"], item["chip_id"], item["logic_id"])
            for item in self.device_map if item["logic_id"] in selected
        })
        if not targets:
            if self.explicit_selection:
                raise EvidenceError("npu-smi map does not cover the selected logical Devices")
            targets = [(device // 2, device % 2, device) for device in self.devices]
        for card, chip, logic_id in targets:
            record = command_record(
                self.root, directory / f"card-{card}-chip-{chip}",
                ["npu-smi", "info", "-t", "memory", "-i", str(card), "-c", str(chip)],
                timeout=120, label="microbenchmark", required=False,
            )
            commands.append({key: value for key, value in record.items() if not key.endswith("_text")})
            static_samples.append({
                "npu_id": card, "chip_id": chip, "logical_device_id": logic_id,
                "collector": "npu-smi info -t memory",
                "returncode": record["returncode"],
                "source_fields": parse_labelled_fields(record["stdout_text"]),
                "command": self.public_command(record),
            })
            if getattr(self, "data_movement_monitor", False):
                ecc_record = command_record(
                    self.root, directory / f"card-{card}-chip-{chip}" / "monitor",
                    ["npu-smi", "info", "-t", "ecc", "-i", str(card), "-c", str(chip)],
                    timeout=30, label="ecc-guard", required=False,
                )
                commands.append(self.public_command(ecc_record))
                static_samples[-1]["ecc_guard"] = {
                    "collector": "npu-smi info -t ecc",
                    "returncode": ecc_record["returncode"],
                    "source_fields": parse_labelled_fields(ecc_record["stdout_text"]),
                    "command": self.public_command(ecc_record),
                }
            if record["returncode"] == 0:
                try:
                    metric = parse_capacity(record["stdout_text"], card, chip)
                    metric["device"] = str(logic_id)
                    metrics.append(metric)
                except EvidenceError:
                    pass
        if not metrics:
            raise EvidenceError("no HBM capacity metric was collected")
        expected = {(card, chip) for card, chip, _ in targets}
        observed = {(metric["card"], metric["chip"]) for metric in metrics}
        monitor_enabled = getattr(self, "data_movement_monitor", False)
        ecc_failures = [
            f"{sample['npu_id']}/{sample['chip_id']}/{sample['logical_device_id']}"
            for sample in static_samples
            if monitor_enabled and sample.get("ecc_guard", {}).get("returncode") != 0
        ]
        monitor_status = (
            "not-run" if not monitor_enabled else
            "passed" if observed == expected and not ecc_failures else "partial"
        )
        monitor = {
            "schema_version": 2, "status": monitor_status,
            "evidence_kind": "static-capacity-guards",
            "target_resource": "hbm-capacity-and-health",
            "collector": "npu-smi memory plus run-level DMI HBM health",
            "timeline_applicable": False,
            "targets": [
                {"npu_id": card, "chip_id": chip, "logic_id": logic}
                for card, chip, logic in targets
            ],
            "samples": static_samples if monitor_enabled else [],
            "health_evidence": {
                "pre": self.manifest.get("health", {}).get("pre"),
                "post": "run-level health/post artifact is attached after all Cases finish",
            },
            "reasons": ([] if monitor_status in ("passed", "not-run") else [
                *([f"static capacity evidence covers {len(observed)}/{len(expected)} targets"]
                  if observed != expected else []),
                *(["npu-smi ecc guard is unavailable for targets " + ",".join(ecc_failures)]
                  if ecc_failures else []),
            ]),
        }
        monitor_path = directory / "monitor" / "summary.json"
        write_json(monitor_path, monitor)
        monitor["summary"] = artifact_ref(monitor_path, self.root)
        return metrics, {
            "commands": commands,
            "monitor": monitor,
            "monitoring_status": monitor_status,
            "capacity_scope": {
                "metric_scope": "chip",
                "aggregation": "none",
                "unit": "MB",
                "target_count": len(metrics),
                "targets": [
                    {
                        "card": metric["card"],
                        "chip": metric["chip"],
                        "device": metric["device"],
                    }
                    for metric in metrics
                ],
            },
            "coverage_error": (
                f"HBM capacity covers {len(observed)}/{len(expected)} card/chip targets"
                if observed != expected else None
            ),
        }

    def run_p2p_combinations(
        self, directory: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        pairs = list(combinations(self.devices, 2))
        if not pairs:
            raise PartialEvidence(
                "P2P requires at least two selected logical Devices; "
                f"selected={self.devices}"
            )
        metrics: list[dict[str, Any]] = []
        commands: list[dict[str, Any]] = []
        completed: list[list[int]] = []
        monitor_units: list[dict[str, Any]] = []
        for source, destination in pairs:
            command = [
                "ascend-dmi", "--bw", "-t", "p2p",
                "--ds", str(source), "--dd", str(destination),
                "-q", "--fmt", "json",
            ]
            pair_metrics, record, monitor = self.run_data_movement_unit(
                "interconnect-P2P_intraserver",
                directory / f"device-{source}-to-{destination}",
                directory / "monitor" / f"device-{source}-to-{destination}",
                command, [source, destination],
                parser=lambda text, source=source, destination=destination: parse_p2p_pair(
                    text, source, destination
                ),
            )
            monitor_units.append(monitor)
            commands.append({
                key: value for key, value in record.items()
                if not key.endswith("_text")
            })
            if record["returncode"] != 0:
                raise PartialEvidence(
                    f"P2P selected pair {source}->{destination} failed"
                )
            if not pair_metrics:
                raise PartialEvidence(
                    f"P2P selected pair {source}->{destination} produced no metric"
                )
            metrics.extend(pair_metrics)
            completed.append([source, destination])
        return metrics, commands, {
            "mode": "selected-combinations",
            "pair_order": "one ascending source-to-destination command per unordered pair",
            "expected_pair_count": len(pairs),
            "completed_pairs": completed,
            "reverse_pairs_inferred": False,
        }, self.aggregate_data_movement_units(
            directory, monitor_units, list(self.devices)
        )

    @staticmethod
    def _public_command(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if not key.endswith("_text")}

    def run_device_latency_sweep(
        self, case: str, directory: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        direction = "h2d" if case == "interconnect-h2d-latency" else "d2h"
        metrics: list[dict[str, Any]] = []
        commands: list[dict[str, Any]] = []
        errors: list[str] = []
        for device in self.devices:
            for size in self.latency_sizes:
                command = [
                    "ascend-dmi", "-l", "-t", direction, "-s", str(size),
                    "-d", str(device), "-q", "--fmt", "json",
                ]
                record = command_record(
                    self.root, directory / f"device-{device}" / f"size-{size}",
                    command, timeout=1800, label="microbenchmark", required=False,
                )
                commands.append(self._public_command(record))
                if record["returncode"] != 0:
                    errors.append(f"Device {device}, {size} bytes returned {record['returncode']}")
                    continue
                try:
                    metrics.extend(parse_latency(
                        record["stdout_text"], direction=direction,
                        size_bytes=size, device=device,
                    ))
                except EvidenceError as exc:
                    errors.append(f"Device {device}, {size} bytes: {exc}")
        expected = {(str(device), size) for device in self.devices for size in self.latency_sizes}
        observed = {(str(item.get("device")), item["size_bytes"]) for item in metrics}
        missing = sorted(expected - observed)
        return metrics, commands, {
            "sweep_scope": {"direction": direction, "expected_points": len(expected),
                            "completed_points": len(observed), "missing_points": missing},
            "coverage_error": "; ".join(errors) or (
                f"latency sweep covers {len(observed)}/{len(expected)} points" if missing else None
            ),
        }

    def run_p2p_latency_sweep(
        self, directory: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        commands: list[dict[str, Any]] = []
        errors: list[str] = []
        if not self.explicit_selection:
            size = self.latency_sizes[0]
            command = ["ascend-dmi", "-l", "-t", "p2p", "-s", str(size), "-q"]
            record = command_record(self.root, directory / f"matrix-size-{size}", command,
                                    timeout=1800, label="microbenchmark", required=False)
            commands.append(self._public_command(record))
            if record["returncode"] == 0:
                try:
                    metrics = parse_p2p_latency_matrix(record["stdout_text"], size)
                except EvidenceError as exc:
                    errors.append(str(exc))
            else:
                errors.append(f"P2P latency matrix returned {record['returncode']}")
            return metrics, commands, {
                "sweep_scope": {"mode": "full-device-512B-matrix", "sizes_bytes": [size],
                                "reverse_pairs_inferred": False},
                "coverage_error": "; ".join(errors) or None,
            }
        pairs = list(permutations(self.devices, 2))
        if not pairs:
            return [], [], {"coverage_error": "P2P latency requires at least two selected Devices"}
        for source, destination in pairs:
            for size in self.latency_sizes:
                command = ["ascend-dmi", "-l", "-t", "p2p", "-s", str(size),
                           "--ds", str(source), "--dd", str(destination),
                           "-q", "--fmt", "json"]
                record = command_record(
                    self.root, directory / f"device-{source}-to-{destination}" / f"size-{size}",
                    command, timeout=1800, label="microbenchmark", required=False,
                )
                commands.append(self._public_command(record))
                if record["returncode"] != 0:
                    errors.append(f"{source}->{destination}, {size} bytes returned {record['returncode']}")
                    continue
                try:
                    metrics.extend(parse_latency(
                        record["stdout_text"], direction="p2p", size_bytes=size,
                        source=source, destination=destination,
                    ))
                except EvidenceError as exc:
                    errors.append(f"{source}->{destination}, {size} bytes: {exc}")
        expected = {(str(a), str(b), size) for a, b in pairs for size in self.latency_sizes}
        observed = {(item["source_device"], item["destination_device"], item["size_bytes"])
                    for item in metrics}
        missing = sorted(expected - observed)
        return metrics, commands, {
            "sweep_scope": {"mode": "selected-directed-pairs", "expected_points": len(expected),
                            "completed_points": len(observed), "missing_points": missing,
                            "reverse_pairs_inferred": False},
            "coverage_error": "; ".join(errors) or (
                f"P2P latency covers {len(observed)}/{len(expected)} points" if missing else None
            ),
        }

    def run_hccl_allreduce(
        self, directory: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        if len(self.devices) < 2:
            return [], [], {"coverage_error": "HCCL AllReduce requires at least two Devices"}, {
                "schema_version": 2, "status": "partial",
                "reasons": ["HCCL AllReduce requires at least two Devices"],
            }
        executable = "/usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test/bin/all_reduce_test"
        ranks = len(self.devices)
        buffer_mib = max(128, math.ceil((2 * self.hccl_max_bytes) / (1 << 20)))
        command = [
            "env", f"HCCL_TEST_USE_DEVS={','.join(map(str, self.devices))}",
            f"HCCL_BUFFSIZE={buffer_mib}", "mpirun", "-n", str(ranks), executable,
            "-p", str(ranks), "-b", str(self.hccl_min_bytes), "-e", str(self.hccl_max_bytes),
            "-f", "2", "-d", "fp32", "-o", "sum", "-w", "10", "-n", "20", "-c", "1",
        ]
        metrics, record, monitor_unit = self.run_data_movement_unit(
            "interconnect-MPI_intraserver", directory,
            directory / "monitor" / "all-reduce", command, list(self.devices),
            parser=parse_hccl_allreduce, label="all-reduce",
        )
        monitor = self.aggregate_data_movement_units(
            directory, [monitor_unit], list(self.devices)
        )
        commands = [self._public_command(record)]
        extra: dict[str, Any] = {
            "hccl_scope": {
                "collective": "all_reduce", "datatype": "fp32", "op": "sum",
                "rank_count": ranks, "selected_device_ids": self.devices,
                "rank_to_device": {str(rank): device for rank, device in enumerate(self.devices)},
                "min_bytes": self.hccl_min_bytes, "max_bytes": self.hccl_max_bytes,
                "factor": 2, "warmup": 10, "iterations": 20,
                "verification": True, "hccl_buffsize_mib": buffer_mib,
            }
        }
        if record["returncode"] != 0:
            extra["coverage_error"] = f"HCCL AllReduce returned {record['returncode']}"
            return [], commands, extra, monitor
        for metric in metrics:
            metric.update(rank_count=ranks, selected_device_ids=self.devices,
                          warmup=10, iterations=20)
        failed = [item["message_size_bytes"] for item in metrics
                  if not item["verification_passed"]]
        if failed:
            extra["correctness_error"] = f"HCCL verification failed for sizes {failed}"
            return metrics, commands, extra, monitor
        expected: list[int] = []
        size = self.hccl_min_bytes
        while size <= self.hccl_max_bytes:
            expected.append(size)
            size *= 2
        observed = {item["message_size_bytes"] for item in metrics}
        missing = sorted(set(expected) - observed)
        if missing:
            extra["coverage_error"] = f"HCCL output missing message sizes {missing}"
        return metrics, commands, extra, monitor

    def parse_case(self, case: str, text: str) -> list[dict[str, Any]]:
        if case in COMPUTE_DTYPES:
            dtype, unit = COMPUTE_DTYPES[case]
            return parse_compute(text, dtype, unit)
        if case == "interconnect-P2P_intraserver":
            return parse_p2p(text)
        metrics = parse_tabular_bandwidth(text)
        if case in ("interconnect-h2d", "interconnect-d2h"):
            direction = "h2d" if case == "interconnect-h2d" else "d2h"
            for metric in metrics:
                metric.update(
                    direction=direction,
                    transfer_size_bytes=536870912,
                    execute_times=50,
                )
        return metrics

    def validate_coverage(self, case: str, metrics: list[dict[str, Any]]) -> None:
        if not self.devices:
            raise PartialEvidence("no Ascend device IDs discovered from /dev")
        if case in COMPUTE_DTYPES:
            if len(metrics) == 1 and str(metrics[0].get("device", "")).lower() == "all":
                return
            observed = {str(metric.get("device")) for metric in metrics if metric.get("device") is not None}
            if not observed:
                raise PartialEvidence("compute output does not identify an all-device aggregate or devices")
        elif case == "interconnect-P2P_intraserver":
            participants = {
                str(metric[key]) for metric in metrics
                for key in ("source_device", "destination_device") if key in metric
            }
            expected_cards = max(2, (len(self.devices) + 1) // 2)
            expected_participants = {str(index) for index in range(expected_cards)}
            if participants != expected_participants:
                raise PartialEvidence(
                    f"P2P output participants {sorted(participants)} do not match "
                    f"expected cards {sorted(expected_participants)}"
                )
            expected_pairs = {
                (source, destination)
                for source in expected_participants
                for destination in expected_participants
                if source != destination
            }
            for direction in ("unidirectional", "bidirectional"):
                observed_pairs = {
                    (str(metric["source_device"]), str(metric["destination_device"]))
                    for metric in metrics if metric.get("direction") == direction
                }
                if observed_pairs != expected_pairs:
                    raise PartialEvidence(
                        f"P2P {direction} matrix covers {len(observed_pairs)}/"
                        f"{len(expected_pairs)} directed card pairs"
                    )

    def bandwidth_covers_devices(self, metrics: list[dict[str, Any]]) -> bool:
        labels = {str(metric.get("device", "")).lower() for metric in metrics}
        if "all" in labels:
            return True
        labels.discard("")
        return len(labels) >= len(self.devices)

    def run_bandwidth_with_fallback(
        self, case: str, directory: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        base = self.optimized_command(case)
        metrics, record, monitor = self.run_data_movement_unit(
            case, directory / "all-devices", directory / "monitor" / "all-devices",
            base, list(self.devices),
        )
        monitor_units = [monitor]
        commands = [{key: value for key, value in record.items()
                     if not key.endswith("_text")}]
        if self.bandwidth_covers_devices(metrics):
            return metrics, commands, self.aggregate_data_movement_units(
                directory, monitor_units, []
            )

        per_device = list(metrics)
        observed = {str(metric.get("device")) for metric in metrics}
        for device in self.devices:
            if str(device) in observed:
                continue
            device_command = [*base]
            insert_at = device_command.index("-q")
            device_command[insert_at:insert_at] = ["-d", str(device)]
            device_metrics, device_record, device_monitor = self.run_data_movement_unit(
                case, directory / f"device-{device}",
                directory / "monitor" / f"device-{device}",
                device_command, [device],
            )
            monitor_units.append(device_monitor)
            commands.append({key: value for key, value in device_record.items()
                             if not key.endswith("_text")})
            if device_record["returncode"] != 0:
                raise PartialEvidence(
                    f"{case} all-device output was incomplete and explicit "
                    f"device {device} failed"
                )
            for metric in device_metrics:
                if metric.get("device") in (None, ""):
                    metric["device"] = str(device)
            per_device.extend(device_metrics)
        if not self.bandwidth_covers_devices(per_device):
            raise PartialEvidence(f"{case} metrics do not cover every discovered device")
        return per_device, commands, self.aggregate_data_movement_units(
            directory, monitor_units, []
        )

    def emit_results(self, case: str, metrics: list[dict[str, Any]], extra: dict[str, Any]) -> None:
        for metric in metrics:
            identity = metric.get("device")
            if "source_device" in metric:
                identity = f"{metric['source_device']}-to-{metric['destination_device']}-{metric['direction']}"
            suffix = f"[{identity}]" if identity is not None else ""
            displayed_value = metric.get("value_raw", metric["value"])
            print(f"[FlagPerf Result] {case}{suffix}={displayed_value} {metric['unit']}", flush=True)

    def run_legacy_probe(self, case: str) -> None:
        directory = self.root / "legacy-comparison" / case
        record = command_record(
            self.root, directory, legacy_command(case), timeout=1800,
            label="legacy", required=False,
        )
        fixed = legacy_fixed_value(case, record["stdout_text"])
        try:
            if case == "main_memory-capacity":
                legacy_metrics = [parse_capacity(record["stdout_text"], 0, 0)]
            else:
                legacy_metrics = self.parse_case(case, record["stdout_text"])
            semantic_values = [metric["value"] for metric in legacy_metrics]
            semantic_error = None
        except EvidenceError as exc:
            semantic_values = []
            semantic_error = str(exc)
        comparison = {
            "legacy_command": record["command"],
            "legacy_returncode": record["returncode"],
            "fixed_position_value": fixed,
            "semantic_values": semantic_values,
            "semantic_parse_error": semantic_error,
            "position_dependency_present": case != "main_memory-capacity",
            "fixed_value_matches_any_semantic_value": (
                fixed is not None and any(math.isclose(float(fixed), float(value), rel_tol=1e-6)
                                          for value in semantic_values)
            ) if fixed and _finite_positive(fixed) is not None else False,
            "unit_issue": "legacy INT8 labelled TFLOPS; DMI reports TOPS"
            if case == "computation-INT8" else None,
        }
        write_json(directory / "comparison.json", comparison)

    def run_case(self, case: str) -> None:
        directory = self.root / "cases" / case
        started = time.perf_counter()
        result: dict[str, Any] = {
            "status": "running",
            "diagnosis": DIAGNOSIS_FOR_CASE[case],
        }
        self.manifest["cases"][case] = result
        try:
            extra: dict[str, Any] = {}
            if case == "main_memory-capacity":
                metrics, extra = self.run_capacity(directory)
                command_public = extra.pop("commands")
                result["commands"] = command_public
                result["monitor"] = extra.pop("monitor")
                result["monitoring_status"] = extra.pop("monitoring_status")
            elif self.explicit_selection and case in COMPUTE_DTYPES:
                metrics, command_public, monitor = self.run_compute_groups(case, directory)
                result["commands"] = command_public
                result["monitor"] = monitor
                result["monitoring_status"] = monitor["status"]
            elif case in COMPUTE_DTYPES:
                metrics, command_public, monitor = self.run_compute_all(case, directory)
                result["command"] = command_public
                result["monitor"] = monitor
                result["monitoring_status"] = monitor["status"]
            elif self.explicit_selection and case in (
                "main_memory-bandwidth", "interconnect-h2d", "interconnect-d2h",
            ):
                metrics, command_public, monitor = self.run_per_device(case, directory)
                result["commands"] = command_public
                result["monitor"] = monitor
                result["monitoring_status"] = monitor["status"]
            elif self.explicit_selection and case == "interconnect-P2P_intraserver":
                metrics, command_public, p2p_scope, monitor = self.run_p2p_combinations(directory)
                result["commands"] = command_public
                result["p2p_scope"] = p2p_scope
                result["monitor"] = monitor
                result["monitoring_status"] = monitor["status"]
            elif case in ("interconnect-h2d-latency", "interconnect-d2h-latency"):
                metrics, command_public, sweep_extra = self.run_device_latency_sweep(case, directory)
                result["commands"] = command_public
                extra.update(sweep_extra)
            elif case == "interconnect-P2P_intraserver-latency":
                metrics, command_public, sweep_extra = self.run_p2p_latency_sweep(directory)
                result["commands"] = command_public
                extra.update(sweep_extra)
            elif case == "interconnect-MPI_intraserver":
                metrics, command_public, hccl_extra, monitor = self.run_hccl_allreduce(directory)
                result["commands"] = command_public
                result["monitor"] = monitor
                result["monitoring_status"] = monitor["status"]
                extra.update(hccl_extra)
            elif case in ("main_memory-bandwidth", "interconnect-h2d", "interconnect-d2h"):
                metrics, command_public, monitor = self.run_bandwidth_with_fallback(
                    case, directory
                )
                result["commands"] = command_public
                result["monitor"] = monitor
                result["monitoring_status"] = monitor["status"]
            elif case == "interconnect-P2P_intraserver":
                metrics, record, monitor_unit = self.run_data_movement_unit(
                    case, directory, directory / "monitor" / "all-devices",
                    self.optimized_command(case), list(self.devices),
                )
                result["command"] = self.public_command(record)
                monitor = self.aggregate_data_movement_units(
                    directory, [monitor_unit], list(self.devices)
                )
                result["monitor"] = monitor
                result["monitoring_status"] = monitor["status"]
                self.validate_coverage(case, metrics)
            else:
                record = command_record(
                    self.root, directory, self.optimized_command(case),
                    timeout=3600, label="microbenchmark",
                )
                result["command"] = {
                    key: value for key, value in record.items() if not key.endswith("_text")
                }
                metrics = self.parse_case(case, record["stdout_text"])
                self.validate_coverage(case, metrics)
            if case == "main_memory-bandwidth":
                for metric in metrics:
                    metric["scope"] = "logical-device"
                extra["bandwidth_scope"] = {
                    "metric_scope": "logical-device",
                    "aggregation": "none",
                    "unit": "GB/s",
                    "source": "ascend-dmi",
                    "selected_device_ids": list(self.devices),
                }
            result["metrics"] = metrics
            result.update(extra)
            if self.legacy_probe and case in {
                *COMPUTE_DTYPES, "main_memory-bandwidth", "main_memory-capacity",
                "interconnect-h2d", "interconnect-P2P_intraserver",
            }:
                self.run_legacy_probe(case)
                result["legacy_comparison"] = f"legacy-comparison/{case}/comparison.json"
            self.emit_results(case, metrics, extra)
            if result.get("correctness_error"):
                raise EvidenceError(result["correctness_error"])
            if result.get("coverage_error"):
                raise PartialEvidence(result["coverage_error"])
            result["measurement_status"] = "passed"
            result["status"] = "passed"
            if result.get("monitoring_status") == "partial":
                result["status"] = "partial"
                result["monitoring_error"] = "; ".join(
                    result.get("monitor", {}).get("reasons", [])
                ) or "monitoring evidence is incomplete"
        except PartialEvidence as exc:
            result.update(status="partial", error=str(exc))
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        finally:
            # Keep the performance command/coverage result independent from the
            # vendor diagnosis applied later.  Human-facing reports must not
            # infer that an unsupported threshold means the measurement failed.
            if "measurement_status" not in result:
                result["measurement_status"] = result["status"]
            if result.get("error") and "measurement_error" not in result:
                result["measurement_error"] = result["error"]
            result["duration_s"] = round(time.perf_counter() - started, 4)
            write_json(directory / "metrics.json", result)
            self.save()

    def run(self) -> int:
        self.root.mkdir(parents=True, exist_ok=False)
        self.save()
        fatal: Exception | None = None
        fatal_partial = False
        post_health_error: Exception | None = None
        try:
            if not self.inventory_devices:
                raise EvidenceError("no /dev/davinciN devices found")
            if not self.devices:
                raise EvidenceError("selected logical Device set is empty")
            if not set(self.devices).issubset(set(self.inventory_devices)):
                raise EvidenceError(
                    f"selected logical Devices {self.devices} are not a subset of "
                    f"container inventory {self.inventory_devices}"
                )
            check_idle_devices(self.devices)
            self.collect_environment()
            self.collect_topology()
            self.collect_health("pre")
            for case in self.cases:
                self.run_case(case)
            self.collect_diagnostics()
            for case, result in self.manifest["cases"].items():
                if result["diagnosis"] is None:
                    result["diagnosis_status"] = "not-run"
                    result["diagnosis_reason"] = "no direct vendor threshold is defined for this Case"
                else:
                    diagnosis = self.manifest["diagnostics"].get(result["diagnosis"], {})
                    result["diagnosis_status"] = diagnosis.get("status", "failed")

                measurement = result.get("measurement_status", "failed")
                monitoring = result.get("monitoring_status", "not-run")
                diagnosis_status_value = result["diagnosis_status"]
                reasons: list[str] = []
                if result.get("measurement_error"):
                    reasons.append(str(result["measurement_error"]))
                if result.get("monitoring_error"):
                    reasons.append(str(result["monitoring_error"]))
                if diagnosis_status_value == "unsupported":
                    reasons.append(
                        f"vendor diagnosis {result['diagnosis']} is unsupported "
                        "or lacks a threshold for this device"
                    )
                elif diagnosis_status_value not in ("passed", "not-run"):
                    reasons.append(f"vendor diagnosis {result['diagnosis']} failed")

                if measurement == "failed" or diagnosis_status_value == "failed":
                    result["status"] = "failed"
                elif (
                    measurement == "partial" or monitoring == "partial"
                    or diagnosis_status_value == "unsupported"
                ):
                    result["status"] = "partial"
                else:
                    result["status"] = "passed"
                if reasons:
                    result["error"] = "; ".join(dict.fromkeys(reasons))
                # Keep the case-level artifact aligned with the final manifest
                # after the corresponding vendor diagnosis is applied.
                write_json(self.root / "cases" / case / "metrics.json", result)
        except Exception as exc:
            fatal = exc
            fatal_partial = isinstance(exc, PartialEvidence)
            self.manifest["fatal_error"] = str(exc)
        finally:
            if "pre" in self.manifest["health"]:
                try:
                    self.collect_health("post")
                except Exception as exc:
                    post_health_error = exc
                    self.manifest["post_health_error"] = str(exc)

        statuses = [result["status"] for result in self.manifest["cases"].values()]
        if (fatal and not fatal_partial) or post_health_error or "failed" in statuses:
            self.manifest["status"] = "failed"
            returncode = 1
        elif fatal_partial or "partial" in statuses or len(statuses) != len(self.cases):
            self.manifest["status"] = "partial"
            returncode = 2
        else:
            self.manifest["status"] = "passed"
            returncode = 0
        self.manifest["finished_at"] = utc_now()
        self.save()
        print(json.dumps({
            "status": self.manifest["status"],
            "manifest": str(self.root / "manifest.json"),
        }, indent=2), flush=True)
        return returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=SUPPORTED_CASES, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--legacy-probe", action="store_true")
    parser.add_argument("--allow-disruptive-dmi", action="store_true")
    parser.add_argument(
        "--compute-monitor", choices=("on", "off"), default="on",
        help="collect concurrent npu-smi usages evidence for compute Cases",
    )
    parser.add_argument(
        "--data-movement-monitor", choices=("on", "off"), default="on",
        help="collect raw usage/link/static evidence for data movement Cases",
    )
    parser.add_argument("--device-ids")
    parser.add_argument(
        "--latency-sizes", type=parse_latency_sizes,
        default=DEFAULT_LATENCY_SIZES,
        help="comma-separated byte sizes; K/M/G use binary powers",
    )
    parser.add_argument("--hccl-min-bytes", type=parse_byte_size, default=8 << 10)
    parser.add_argument("--hccl-max-bytes", type=parse_byte_size, default=1 << 30)
    parser.add_argument(
        "--selection-source",
        choices=("default-all", "npu-ids", "device-ids"),
        default="default-all",
    )
    return parser.parse_args()


def default_output() -> Path:
    configured = os.environ.get("FLAGPERF_TOOLKIT_ARTIFACT_DIR")
    if configured:
        base = Path(configured)
    else:
        base = Path.cwd() / "toolkit-evidence"
    return base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    args = parse_args()
    allowed = args.allow_disruptive_dmi or os.environ.get(
        "FLAGPERF_ALLOW_DISRUPTIVE_DMI"
    ) == "1"
    if not allowed:
        raise EvidenceError(
            "DMI performance tests can affect the selected devices (all inventory "
            "devices by default); pass "
            "--allow-disruptive-dmi or set FLAGPERF_ALLOW_DISRUPTIVE_DMI=1"
        )
    selected_devices = (
        parse_id_spec(args.device_ids, "logical Device IDs")
        if args.device_ids else None
    )
    if args.legacy_probe and selected_devices is not None:
        raise EvidenceError("explicit device selection cannot be combined with legacy probe")
    if selected_devices is None and args.selection_source != "default-all":
        raise EvidenceError("non-default selection source requires --device-ids")
    if args.hccl_min_bytes > args.hccl_max_bytes:
        raise EvidenceError("--hccl-min-bytes must not exceed --hccl-max-bytes")
    if "interconnect-MPI_intraserver" not in args.cases and (
        args.hccl_min_bytes != 8 << 10 or args.hccl_max_bytes != 1 << 30
    ):
        raise EvidenceError("HCCL size overrides require interconnect-MPI_intraserver")
    root = (args.output or default_output()).resolve()
    return Runner(
        root, list(dict.fromkeys(args.cases)), args.legacy_probe,
        selected_devices=selected_devices,
        selection_source=args.selection_source,
        latency_sizes=args.latency_sizes,
        hccl_min_bytes=args.hccl_min_bytes,
        hccl_max_bytes=args.hccl_max_bytes,
        compute_monitor=args.compute_monitor == "on",
        data_movement_monitor=args.data_movement_monitor == "on",
    ).run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
