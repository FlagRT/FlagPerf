# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Best-effort Benchmark phase events that never alter measurement status."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


EVENTS_DIR_ENV = "FLAGPERF_BENCHMARK_EVENTS_DIR"
CASE_ENV = "FLAGPERF_BENCHMARK_CASE"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _integer_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class MeasurementToken:
    started_at: str
    started_monotonic_ns: int
    rank: int
    local_rank: int
    world_size: int
    case: str


def benchmark_measurement_start() -> MeasurementToken | None:
    """Capture a rank-local start immediately before the existing timer."""
    if not os.environ.get(EVENTS_DIR_ENV):
        return None
    return MeasurementToken(
        started_at=_utc_now(),
        started_monotonic_ns=time.monotonic_ns(),
        rank=_integer_env("RANK", 0),
        local_rank=_integer_env("LOCAL_RANK", 0),
        world_size=_integer_env("WORLD_SIZE", 1),
        case=os.environ.get(CASE_ENV, "unknown"),
    )


def benchmark_measurement_finish(token: MeasurementToken | None) -> None:
    """Persist one atomic rank window after the existing timer has stopped.

    Monitoring is an observer.  An evidence-write failure is surfaced in the
    Benchmark log but is never allowed to replace the workload result.
    """
    if token is None:
        return
    directory_text = os.environ.get(EVENTS_DIR_ENV)
    if not directory_text:
        return
    finished_monotonic_ns = time.monotonic_ns()
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "measurement-window",
        "case": token.case,
        "rank": token.rank,
        "local_rank": token.local_rank,
        "world_size": token.world_size,
        "pid": os.getpid(),
        "started_at": token.started_at,
        "finished_at": _utc_now(),
        "started_monotonic_ns": token.started_monotonic_ns,
        "finished_monotonic_ns": finished_monotonic_ns,
        "duration_s": round(
            (finished_monotonic_ns - token.started_monotonic_ns) / 1e9, 9
        ),
    }
    try:
        directory = Path(directory_text)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"measurement-rank-{token.rank}.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            temporary = Path(stream.name)
        os.chmod(temporary, 0o644)
        temporary.replace(destination)
    except Exception as exc:  # evidence failure must not fail the workload
        print(
            "[FlagPerf Monitor Warning] measurement event write failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
