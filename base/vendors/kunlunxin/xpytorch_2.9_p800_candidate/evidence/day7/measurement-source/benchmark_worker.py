#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Container-internal worker for original FlagPerf Base Benchmark Cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import runpy
from pathlib import Path
import shlex
import subprocess
import sys
import time


if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarks.case_assets import resolve_case_assets, verify_worker_assets


LOGGER = logging.getLogger("flagperf.base.benchmark_worker")


def run_bound_case(context, context_path, output):
    """Run the bounded case in the already gated child (one or two ranks).

    Two-rank Day 6 communication cases spawn torchrun, which sets
    RANK/LOCAL_RANK/WORLD_SIZE per child; all FLAGPERF_* evidence variables
    are inherited so each rank re-verifies its own device binding.
    """
    nproc = context.get('nproc_per_node') or 1
    if nproc not in (1, 2) or context.get('kind') != 'benchmark':
        raise RuntimeError('bound case worker requires a one- or two-rank benchmark context')
    base = Path(__file__).resolve().parent
    contract = context_path.parent / 'case-assets.json'
    assets = verify_worker_assets(base, contract, context['case'], 'kunlunxin')
    environment = {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(context['master_port']),
                   'FLAGPERF_CASE_ASSETS': str(contract),
                   'FLAGPERF_HOST_CONTEXT': str(context_path),
                   'FLAGPERF_RUNTIME_BINDINGS': str(output / 'runtime-bindings.json'),
                   'FLAGPERF_BENCHMARK_OUTPUT': str(output),
                   'FLAGPERF_BENCHMARK_EVENTS_DIR': str(output / 'benchmark-events'),
                   'FLAGPERF_BENCHMARK_CASE': context['case']}
    if nproc == 1:
        environment.update({'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1'})
    os.environ.update(environment)
    os.chdir(base / 'benchmarks' / assets['case_name'])
    sys.path.insert(0, str(base / 'benchmarks'))
    if nproc > 1:
        command = [sys.executable, '-m', 'torch.distributed.run',
                   f'--nproc-per-node={nproc}', "--master-port=" + str(context['master_port']),
                   assets['entrypoint']['path'], '--vendor=' + assets['selector'],
                   f'--node_size={nproc}']
        child = dict(os.environ)
        # torchrun children need the full import surface the in-process rank
        # gets: repo root for `base.*` provider bindings, base/ for the
        # `benchmarks.day6_contract` package, base/benchmarks for `drivers`
        # and `case_assets`.
        parts = [str(base.parent), str(base), str(base / 'benchmarks')]
        existing = child.get('PYTHONPATH')
        if existing:
            parts.append(existing)
        child['PYTHONPATH'] = os.pathsep.join(parts)
        completed = subprocess.run(command, check=False, env=child)
        return completed.returncode
    sys.argv = [assets['entrypoint']['path'], '--vendor=' + assets['selector'], '--node_size=1']
    try:
        runpy.run_path(assets['entrypoint']['path'], run_name='__main__')
    finally:
        import torch.distributed as distributed
        if distributed.is_initialized():
            distributed.destroy_process_group()
    return 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o644)
    temporary.replace(path)


def resolve_benchmark_entrypoint(
    perf_path: str, case_spec: str, vendor: str,
) -> tuple[str, str, str, str]:
    assets = resolve_case_assets(Path(perf_path), case_spec, vendor, require_contract=False)
    return (assets["case_name"], str(Path(perf_path) / "benchmarks" / assets["case_name"]),
            assets["entrypoint"]["path"], assets["selector"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Base Benchmark inside the prepared container",
    )
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--case-assets", type=Path, help="host-resolved immutable asset contract")
    parser.add_argument("--case-env-ready", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--log_level", required=True)
    parser.add_argument("--master_port", type=int, required=True)
    parser.add_argument("--master_addr", required=True)
    parser.add_argument("--host_addr", required=True)
    parser.add_argument("--node_rank", type=int, required=True)
    parser.add_argument("--perf_path", required=True)
    parser.add_argument(
        "--bench_or_tool", choices=("BENCHMARK",), default="BENCHMARK",
        help="compatibility selector; this worker is Benchmark-only",
    )
    parser.add_argument(
        "--allow_disruptive_toolkit", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--monitor-events", action="store_true",
        help="record rank-local Benchmark measurement windows",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def write_pid_file(log_dir: Path) -> Path:
    path = log_dir / "start_base_task.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    return path


def build_torchrun_command(
    config: argparse.Namespace,
) -> tuple[list[str], Path, Path]:
    contract = getattr(config, "case_assets", None)
    if contract is not None:
        assets = verify_worker_assets(Path(config.perf_path), Path(contract), config.case_name, config.vendor)
        if assets["requirements"].get("supported") is False:
            raise RuntimeError("unsupported case cannot execute")
        case_name = assets["case_name"]
        case_dir = str(Path(config.perf_path) / "benchmarks" / case_name)
        entrypoint, vendor_selector = assets["entrypoint"]["path"], assets["selector"]
    else:
        case_name, case_dir, entrypoint, vendor_selector = resolve_benchmark_entrypoint(
            config.perf_path, config.case_name, config.vendor)

    case_path = Path(case_dir)
    entrypoint_path = Path(entrypoint)
    if not case_path.is_dir():
        raise RuntimeError(f"Benchmark Case directory does not exist: {case_path}")
    if not entrypoint_path.is_file():
        raise RuntimeError(f"Benchmark entrypoint does not exist: {entrypoint_path}")
    command = [
        "torchrun",
        f"--nproc-per-node={config.nproc_per_node}",
        f"--nnodes={config.nnodes}",
        f"--node-rank={config.node_rank}",
        f"--master-addr={config.master_addr}",
        f"--master-port={config.master_port}",
        str(entrypoint_path),
        f"--vendor={vendor_selector}",
        f"--node_size={config.nproc_per_node}",
    ]
    log_dir = (
        Path(config.log_dir)
        / case_name
        / f"{config.host_addr}_noderank{config.node_rank}"
    )
    return command, case_path, log_dir


def configure_logging(path: Path, level: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(level.upper())
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    command, case_dir, log_dir = build_torchrun_command(config)
    if config.case_assets is not None and not config.case_env_ready:
        assets = verify_worker_assets(Path(config.perf_path), config.case_assets, config.case_name, config.vendor)
        setup = "".join("source " + shlex.quote(item["path"]) + " && " for item in assets["environments"])
        if setup:
            # Validate before executing setup, then re-enter with its exported
            # environment. Setup is outside the torchrun lifecycle timer.
            reentry = [sys.executable, str(Path(__file__).resolve()),
                       *(sys.argv[1:] if argv is None else argv), "--case-env-ready"]
            os.execvpe("/bin/bash", ["/bin/bash", "-c", setup + "exec " + shlex.join(reentry)], os.environ.copy())
    # Keep the historical artifact name for one schema/compatibility cycle.
    configure_logging(log_dir / "container_main.log.txt", config.log_level)
    pid_path = write_pid_file(Path(config.log_dir))
    LOGGER.info("Benchmark worker PID file: %s", pid_path)
    LOGGER.info("Benchmark Case directory: %s", case_dir)
    LOGGER.info("Benchmark command: %s", shlex.join(command))

    benchmark_log = log_dir / "benchmark.log.txt"
    benchmark_log.parent.mkdir(parents=True, exist_ok=True)
    child_environment = os.environ.copy()
    if config.case_assets is not None:
        verify_worker_assets(Path(config.perf_path), config.case_assets, config.case_name, config.vendor)
        child_environment["FLAGPERF_CASE_ASSETS"] = str(config.case_assets.resolve())
    events_dir = Path(config.log_dir) / "benchmark-events"
    monitor_events = bool(getattr(config, "monitor_events", False))
    if monitor_events:
        child_environment.update({
            "FLAGPERF_BENCHMARK_EVENTS_DIR": str(events_dir),
            "FLAGPERF_BENCHMARK_CASE": config.case_name,
        })
    started_at = utc_now()
    started_monotonic_ns = time.monotonic_ns()
    with benchmark_log.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(
            command,
            cwd=case_dir,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            env=child_environment,
        )
    finished_monotonic_ns = time.monotonic_ns()
    if monitor_events:
        write_json_atomic(events_dir / "torchrun-window.json", {
            "schema_version": 1,
            "kind": "torchrun-window",
            "case": config.case_name,
            "started_at": started_at,
            "finished_at": utc_now(),
            "started_monotonic_ns": started_monotonic_ns,
            "finished_monotonic_ns": finished_monotonic_ns,
            "duration_s": round(
                (finished_monotonic_ns - started_monotonic_ns) / 1e9, 9
            ),
            "returncode": proc.returncode,
        })
    LOGGER.info("Benchmark task finished with return code %s", proc.returncode)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
