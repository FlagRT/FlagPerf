# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Benchmark-specific lifecycle and vendor-neutral evidence contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any, Sequence

from executors.common import write_json, sha256_file
from base.vendors.registry import get_provider
from base.vendors.protocol import DeviceBinding

UsageMonitor = Any


def artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def selected_provider(provider=None):
    # Compatibility default for pre-provider Ascend monitor callers.
    return provider if provider is not None else get_provider("ascend")

MONITOR_SCHEMA_VERSION = 2
EVENT_DIRECTORY_MODE = 0o1777
EVENT_FILE_MODE = 0o644


def prepare_benchmark_event_exchange(result_dir: Path) -> dict[str, Any]:
    """Create one UID-independent, per-run event exchange on the host."""
    path = result_dir / "benchmark-events"
    path.mkdir(mode=EVENT_DIRECTORY_MODE, parents=False, exist_ok=False)
    path.chmod(EVENT_DIRECTORY_MODE)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"Benchmark event exchange is not a directory: {path}")
    actual_mode = stat.S_IMODE(metadata.st_mode)
    if actual_mode != EVENT_DIRECTORY_MODE:
        raise RuntimeError(
            "Benchmark event exchange mode mismatch: "
            f"expected={oct(EVENT_DIRECTORY_MODE)}, actual={oct(actual_mode)}"
        )
    return {
        "path": path.relative_to(result_dir).as_posix(),
        "directory_mode": "01777",
        "event_file_mode": "0644",
        "ownership_strategy": "host-created per-run sticky exchange",
    }


def _write_container_window(
    result_dir: Path, container_record: dict[str, Any],
) -> Path:
    path = result_dir / "benchmark-events" / "container-window.json"
    write_json(path, container_record)
    os.chmod(path, EVENT_FILE_MODE)
    return path


def monitor_policy(enabled: bool, *, provider=None) -> dict[str, Any]:
    return selected_provider(provider).monitor_policy(enabled)


def resolve_monitor_targets(preflight, selected_device_ids, *, provider=None):
    return selected_provider(provider).monitor_targets(preflight, selected_device_ids)


def create_usage_monitor(targets, *, provider=None):
    return selected_provider(provider).create_monitor(targets)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _relative_window(
    record: dict[str, Any], monitor: UsageMonitor, role: str,
    **extra: Any,
) -> dict[str, Any] | None:
    started = record.get("started_monotonic_ns")
    finished = record.get("finished_monotonic_ns")
    if (
        isinstance(started, bool) or not isinstance(started, int)
        or isinstance(finished, bool) or not isinstance(finished, int)
        or started >= finished
    ):
        return None
    return {
        "role": role,
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "started_offset_s": round(
            started / 1e9 - monitor.origin_monotonic_s, 9
        ),
        "finished_offset_s": round(
            finished / 1e9 - monitor.origin_monotonic_s, 9
        ),
        **extra,
    }


def _contains(
    outer: dict[str, Any], inner: dict[str, Any], tolerance_s: float = 0.05,
) -> bool:
    return (
        float(outer["started_offset_s"]) - tolerance_s
        <= float(inner["started_offset_s"])
        and float(inner["finished_offset_s"])
        <= float(outer["finished_offset_s"]) + tolerance_s
    )


def _event_ref(path: Path, root: Path) -> dict[str, Any] | None:
    return artifact_ref(path, root) if path.is_file() else None


def finalize_benchmark_monitor(
    monitor: UsageMonitor,
    result_dir: Path,
    container_record: dict[str, Any],
    selected_device_ids: Sequence[int],
    expected_ranks: int,
    *, provider=None, bindings: Sequence[DeviceBinding] | None = None,
) -> dict[str, Any]:
    provider = selected_provider(provider)
    policy = provider.monitor_policy(True)
    events_dir = result_dir / "benchmark-events"
    monitor_dir = result_dir / "benchmark-monitor"
    reasons: list[str] = []
    event_artifacts: dict[str, Any] = {}

    container_path = events_dir / "container-window.json"
    _write_container_window(result_dir, container_record)
    event_artifacts["container"] = artifact_ref(container_path, result_dir)
    container_window = _relative_window(
        container_record, monitor, "container"
    )
    lifecycle_windows: list[dict[str, Any]] = []
    if container_window is None:
        reasons.append("container lifecycle event is invalid")
    else:
        lifecycle_windows.append(container_window)

    torchrun_path = events_dir / "torchrun-window.json"
    torchrun_record = _read_json(torchrun_path)
    torchrun_window = None
    if torchrun_record is None:
        reasons.append("torchrun lifecycle event is missing or invalid")
    else:
        event_artifacts["torchrun"] = artifact_ref(torchrun_path, result_dir)
        torchrun_window = _relative_window(
            torchrun_record, monitor, "torchrun"
        )
        if torchrun_window is None:
            reasons.append("torchrun lifecycle event has an invalid clock window")
        else:
            lifecycle_windows.append(torchrun_window)
            if container_window is not None and not _contains(
                container_window, torchrun_window
            ):
                reasons.append("torchrun window is not nested in the container window")

    by_rank = {binding.framework_local_rank: binding for binding in bindings or []}
    if bindings is not None and (len(by_rank) != expected_ranks or set(by_rank) != set(range(expected_ranks))):
        raise RuntimeError("device binding local ranks are not unique and complete")
    observed_local_ranks = set()
    rank_device_map: list[dict[str, int]] = []
    measurement_windows: list[dict[str, Any]] = []
    measurement_refs: list[dict[str, Any]] = []
    for rank in range(expected_ranks):
        event_path = events_dir / f"measurement-rank-{rank}.json"
        event = _read_json(event_path)
        if event is None:
            reasons.append(f"measurement event for rank {rank} is missing or invalid")
            continue
        measurement_refs.append(artifact_ref(event_path, result_dir))
        if event.get("schema_version") != 1 or event.get("kind") != "measurement-window":
            reasons.append(f"measurement event for rank {rank} has an unsupported schema")
            continue
        if event.get("rank") != rank or event.get("world_size") != expected_ranks:
            reasons.append(f"measurement event identity mismatch for rank {rank}")
            continue
        local_rank = event.get("local_rank")
        if (
            isinstance(local_rank, bool) or not isinstance(local_rank, int)
            or not 0 <= local_rank < len(selected_device_ids)
        ):
            reasons.append(f"measurement event local_rank is invalid for rank {rank}")
            continue
        if local_rank in observed_local_ranks:
            reasons.append(f"duplicate measurement local_rank {local_rank}")
            continue
        observed_local_ranks.add(local_rank)
        binding = by_rank.get(local_rank)
        target = provider.rank_target(monitor.targets, selected_device_ids, local_rank, binding)
        if target is None:
            reasons.append(f"measurement event rank {rank} maps to an unknown device")
            continue
        identity = provider.measurement_identity(target, binding)
        window = _relative_window(
            event,
            monitor,
            "measurement",
            rank=rank,
            local_rank=local_rank,
            **identity,
        )
        if window is None:
            reasons.append(f"measurement event clock window is invalid for rank {rank}")
            continue
        if torchrun_window is not None and not _contains(torchrun_window, window):
            reasons.append(f"rank {rank} measurement is not nested in torchrun")
        if container_window is not None and not _contains(container_window, window):
            reasons.append(f"rank {rank} measurement is not nested in container")
        measurement_windows.append(window)
        rank_device_map.append({
            "rank": rank,
            "local_rank": local_rank,
            **identity,
        })
    event_artifacts["measurements"] = measurement_refs

    result = monitor.finish(
        result_dir,
        monitor_dir,
        measurement_windows,
        [],
        min_samples_per_target=policy["required_samples_per_target"],
        primary_role="measurement",
        window_semantics=(
            "collector command intervals overlapping each rank's exact original "
            "Benchmark timer window; vendor statistic integration semantics are "
            "not inferred"
        ),
        target_resource=policy["target_resource"],
    )
    result.update({
        "schema_version": MONITOR_SCHEMA_VERSION,
        "vendor": provider.name,
        "policy": policy,
        "required_samples_per_target": policy["required_samples_per_target"],
        "clock_domain": "same-host Linux monotonic clock, validated by nesting",
        "lifecycle_windows": lifecycle_windows,
        "rank_device_map": rank_device_map,
        "event_artifacts": event_artifacts,
        "automatic_workload_extension": False,
    })
    result["reasons"].extend(reasons)
    result["reasons"] = list(dict.fromkeys(result["reasons"]))
    if reasons:
        result["status"] = "partial"
    summary_path = monitor_dir / "summary.json"
    public = dict(result)
    public.pop("summary", None)
    write_json(summary_path, public)
    result["summary"] = artifact_ref(summary_path, result_dir)
    return result


def finalize_benchmark_monitor_safely(
    monitor: UsageMonitor | None,
    result_dir: Path,
    container_record: dict[str, Any],
    selected_device_ids: Sequence[int],
    expected_ranks: int,
    targets: Sequence[dict[str, int]],
    initial_result: dict[str, Any] | None,
    *, provider=None, bindings: Sequence[DeviceBinding] | None = None,
) -> dict[str, Any]:
    """Finalize monitoring without allowing observer failures to escape."""
    if monitor is None:
        result = initial_result or write_monitor_terminal_summary(
            result_dir,
            provider=provider,
            status="failed",
            enabled=True,
            targets=targets,
            reason="Benchmark monitor was unavailable at finalization",
        )
        try:
            _write_container_window(result_dir, container_record)
        except Exception as exc:
            reasons = list(result.get("reasons") or [])
            reasons.append(
                "container lifecycle event write failed: "
                f"{type(exc).__name__}: {exc}"
            )
            result = write_monitor_terminal_summary(
                result_dir,
                provider=provider,
                status="failed",
                enabled=True,
                targets=targets,
                reason="; ".join(reasons),
            )
        return result

    try:
        return finalize_benchmark_monitor(
            monitor,
            result_dir,
            container_record,
            selected_device_ids,
            expected_ranks,
            provider=provider, bindings=bindings,
        )
    except Exception as exc:
        reasons = [
            "monitor finalization failed: "
            f"{type(exc).__name__}: {exc}"
        ]
        try:
            monitor.stop()
        except Exception as stop_exc:
            reasons.append(
                "monitor stop failed: "
                f"{type(stop_exc).__name__}: {stop_exc}"
            )
        return write_monitor_terminal_summary(
            result_dir,
            provider=provider,
            status="failed",
            enabled=True,
            targets=targets,
            reason="; ".join(reasons),
        )


def write_monitor_terminal_summary(
    result_dir: Path,
    *,
    status: str,
    enabled: bool,
    targets: Sequence[dict[str, int]],
    reason: str,
    provider=None,
) -> dict[str, Any]:
    result = {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "status": status,
        "vendor": selected_provider(provider).name,
        "policy": monitor_policy(enabled, provider=provider),
        "collector": monitor_policy(enabled, provider=provider)["collector"],
        "targets": list(targets),
        "workload_windows": [],
        "lifecycle_windows": [],
        "rank_device_map": [],
        "reasons": [reason],
        "automatic_workload_extension": False,
    }
    path = result_dir / "benchmark-monitor" / "summary.json"
    try:
        write_json(path, result)
        result["summary"] = artifact_ref(path, result_dir)
    except Exception as exc:
        result["reasons"].append(
            "monitor terminal summary write failed: "
            f"{type(exc).__name__}: {exc}"
        )
        result["summary"] = None
    return result
