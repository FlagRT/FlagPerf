# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Host-side planner and executor for original FlagPerf Base Benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

from benchmark_monitor import (
    create_usage_monitor,
    finalize_benchmark_monitor_safely,
    monitor_policy,
    prepare_benchmark_event_exchange,
    resolve_monitor_targets,
    write_monitor_terminal_summary,
)
from generate_benchmark_report import (
    REPORT_SCHEMA_VERSION as BENCHMARK_REPORT_SCHEMA_VERSION,
    generate_and_record as generate_and_record_benchmark_report,
)
from executors.common import (
    BASE_DIR,
    BaseRunContext,
    ConfigurationError,
    DeviceLease,
    context_record,
    load_host_config,
    runtime_lock_record,
    run_timestamp,
    utc_now,
    validate_runtime_identity,
    write_json,
)
from executors.toolkit import docker_inspect, run_host_preflight


RESULT_RE = re.compile(
    r"^\[FlagPerf Result\]Rank\s+(?P<rank>\d+)(?:'s)?\s+"
    r"(?P<metric>[^=]+)=(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?P<unit>[A-Za-z/]+)\s*$"
)
SAFE_CASE_RE = re.compile(r"[A-Za-z0-9_.:-]+")
HIGH_RISK_CASES = {"main_memory-capacity"}


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path,
        default=BASE_DIR / "configs" / "ascend910_cann9_local.yaml",
        help="Ascend host/runtime profile",
    )
    parser.add_argument("--case", required=True, help="Base Benchmark Case name")
    parser.add_argument(
        "--case-config", type=Path,
        help="read-only Ascend Case YAML override mounted into the container",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--npu-ids", help="physical NPU IDs/ranges, for example 7 or 1-3",
    )
    selection.add_argument(
        "--device-ids", help="logical Device IDs/ranges, for example 14,15",
    )
    parser.add_argument(
        "--nproc-per-node", type=int,
        help="torchrun local ranks; defaults to the resolved logical Device count",
    )
    parser.add_argument("--master-port", type=int, default=29721)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--monitor", choices=("on", "off"), default="on",
        help=(
            "collect selected-device npu-smi usage evidence; enabled by "
            "default and independently statused"
        ),
    )
    parser.add_argument(
        "--allow-privileged-root", action="store_true",
        help="allow the locked Benchmark container to run privileged as root",
    )
    parser.add_argument(
        "--allow-high-risk-case", action="store_true",
        help="allow capacity/OOM/long-running Cases classified as high risk",
    )
    parser.add_argument(
        "--allow-candidate-runtime", action="store_true",
        help=(
            "allow an unvalidated runtime image for bounded candidate testing; "
            "does not promote the image"
        ),
    )


@dataclass(frozen=True)
class BenchmarkRunRequest:
    context: BaseRunContext
    case: str
    case_config: Path | None
    nproc_per_node: int | None
    master_port: int
    log_level: str
    allow_privileged_root: bool
    allow_high_risk_case: bool
    monitor: str = "on"
    allow_candidate_runtime: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "BenchmarkRunRequest":
        return cls(
            context=BaseRunContext(
                config=args.config,
                npu_ids=args.npu_ids,
                device_ids=args.device_ids,
                result_root=args.result_root,
                timeout=args.timeout,
                dry_run=args.dry_run,
            ),
            case=args.case,
            case_config=args.case_config,
            nproc_per_node=args.nproc_per_node,
            master_port=args.master_port,
            log_level=args.log_level,
            allow_privileged_root=args.allow_privileged_root,
            allow_high_risk_case=args.allow_high_risk_case,
            monitor=args.monitor,
            allow_candidate_runtime=args.allow_candidate_runtime,
        )

    def validate(self) -> None:
        self.context.validate(require_selection=True)
        if not SAFE_CASE_RE.fullmatch(self.case):
            raise ConfigurationError(f"invalid Benchmark Case name: {self.case!r}")
        case_name = self.case.split(":", 1)[0]
        case_dir = BASE_DIR / "benchmarks" / case_name
        if not case_dir.is_dir():
            raise ConfigurationError(f"Benchmark Case does not exist: {case_name}")
        if self.case_config is not None and not self.case_config.expanduser().is_file():
            raise ConfigurationError(
                f"Benchmark Case override does not exist: {self.case_config}"
            )
        if self.nproc_per_node is not None and self.nproc_per_node <= 0:
            raise ConfigurationError("--nproc-per-node must be positive")
        if not 1 <= self.master_port <= 65535:
            raise ConfigurationError("--master-port must be between 1 and 65535")
        if self.monitor not in ("on", "off"):
            raise ConfigurationError("--monitor must be on or off")
        if self.case in HIGH_RISK_CASES and not self.allow_high_risk_case:
            raise ConfigurationError(
                f"Benchmark Case {self.case} is high risk; explicitly pass "
                "--allow-high-risk-case after reserving an isolated device"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def case_runtime_requirement_record(
    request: BenchmarkRunRequest,
) -> dict[str, Any] | None:
    """Load an optional, repository-owned Case/runtime compatibility contract."""
    case_name = request.case.split(":", 1)[0]
    path = (
        BASE_DIR / "benchmarks" / case_name / "ascend" /
        "runtime_requirements.json"
    )
    if not path.is_file():
        return None
    try:
        requirements = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Benchmark runtime requirements are invalid JSON: {path}: {exc}"
        ) from exc
    if requirements.get("schema_version") != 1:
        raise ConfigurationError(
            "unsupported Benchmark runtime requirement schema: "
            f"{requirements.get('schema_version')}"
        )
    return {
        **file_record(path.resolve()),
        "requirements": requirements,
    }


def validate_case_runtime_requirements(
    request: BenchmarkRunRequest,
    config: dict[str, Any],
    record: dict[str, Any] | None,
) -> None:
    if record is None:
        return
    requirements = record["requirements"]
    required_profile = requirements.get("runtime_profile")
    actual_profile = config.get("runtime_profile", "torch_fl_2.10")
    if required_profile and actual_profile != required_profile:
        raise ConfigurationError(
            f"Benchmark Case {request.case} requires runtime profile "
            f"{required_profile!r}, got {actual_profile!r}; use the dedicated "
            "communication host config"
        )
    required_role = requirements.get("runtime_role")
    actual_role = config.get("runtime_role", "operator")
    if required_role and actual_role != required_role:
        raise ConfigurationError(
            f"Benchmark Case {request.case} requires runtime role "
            f"{required_role!r}, got {actual_role!r}"
        )
    process_scope = requirements.get("process_scope", {})
    exact_nproc = process_scope.get("nproc_per_node")
    if request.nproc_per_node is not None and exact_nproc is not None:
        if request.nproc_per_node != exact_nproc:
            raise ConfigurationError(
                f"Benchmark Case {request.case} requires exactly {exact_nproc} "
                f"local ranks, got {request.nproc_per_node}"
            )
    if (
        requirements.get("candidate_requires_case_override") is True
        and request.case_config is None
    ):
        raise ConfigurationError(
            f"Benchmark Case {request.case} requires an explicit bounded "
            "--case-config while its communication runtime is a candidate"
        )
    allowed_configs = requirements.get("candidate_allowed_case_configs", [])
    if request.case_config is not None and allowed_configs:
        actual_sha256 = sha256(request.case_config.expanduser().resolve())
        allowed_sha256 = {
            item.get("sha256")
            for item in allowed_configs
            if isinstance(item, dict) and isinstance(item.get("sha256"), str)
        }
        if actual_sha256 not in allowed_sha256:
            raise ConfigurationError(
                f"Benchmark Case {request.case} candidate override is not in "
                "the repository-owned bounded configuration allowlist"
            )


def validate_resolved_process_scope(
    case: str, record: dict[str, Any] | None, *, nnodes: int, nproc: int,
) -> None:
    if record is None:
        return
    process_scope = record["requirements"].get("process_scope", {})
    for field, actual in (("nnodes", nnodes), ("nproc_per_node", nproc)):
        expected = process_scope.get(field)
        if expected is not None and actual != expected:
            raise ConfigurationError(
                f"Benchmark Case {case} requires {field}={expected}, got {actual}"
            )


def case_configuration_records(request: BenchmarkRunRequest) -> dict[str, Any]:
    case_name = request.case.split(":", 1)[0]
    case_dir = BASE_DIR / "benchmarks" / case_name
    records: dict[str, Any] = {}
    for label, path in (
        ("generic", case_dir / "case_config.yaml"),
        ("ascend", case_dir / "ascend" / "case_config.yaml"),
    ):
        if path.is_file():
            records[label] = file_record(path.resolve())
    if request.case_config is not None:
        records["override"] = file_record(request.case_config.expanduser().resolve())
    return records


def snapshot_case_configuration(
    request: BenchmarkRunRequest, result_dir: Path,
) -> dict[str, Any]:
    """Preserve exact Case YAML inputs inside the immutable run evidence."""
    records = case_configuration_records(request)
    snapshot_dir = result_dir / "case-config"
    for label, record in records.items():
        source = Path(record["path"])
        destination = snapshot_dir / f"{label}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(destination)
        if sha256(destination) != record["sha256"]:
            raise RuntimeError(f"Case configuration snapshot hash changed: {source}")
        record["artifact_path"] = destination.relative_to(result_dir).as_posix()
    return records


def parse_benchmark_results(text: str, expected_ranks: int) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = RESULT_RE.match(line.strip())
        if match:
            metrics.append({
                "rank": int(match.group("rank")),
                "metric": match.group("metric"),
                "value": float(match.group("value")),
                "unit": match.group("unit"),
                "raw": line,
            })
    observed = sorted({item["rank"] for item in metrics})
    expected = list(range(expected_ranks))
    if observed == expected:
        status = "passed"
    elif observed:
        status = "partial"
    else:
        status = "failed"
    return {
        "schema_version": 1,
        "status": status,
        "expected_ranks": expected,
        "observed_ranks": observed,
        "missing_ranks": sorted(set(expected) - set(observed)),
        "metrics": metrics,
    }


def aggregate_benchmark_status(
    execution_status: str,
    measurement_status: str,
    monitoring_status: str,
    *,
    monitor_enabled: bool,
) -> str:
    """Keep measurement failure fatal and requested monitor gaps partial."""
    if execution_status == "failed" or measurement_status == "failed":
        return "failed"
    if (
        measurement_status == "partial"
        or (monitor_enabled and monitoring_status != "passed")
    ):
        return "partial"
    return "passed"


def generate_benchmark_report(result_dir: Path) -> dict[str, Any]:
    """Compatibility import for existing callers of the executor module."""
    return generate_and_record_benchmark_report(result_dir)


class BenchmarkExecutor:
    def plan(self, request: BenchmarkRunRequest) -> dict[str, Any]:
        request.validate()
        config_path, config = load_host_config(request.context.config)
        runtime_requirements = case_runtime_requirement_record(request)
        validate_case_runtime_requirements(request, config, runtime_requirements)
        return {
            "schema_version": 1,
            "kind": "benchmark",
            "mode": "static-dry-run" if request.context.dry_run else "execution",
            "host_config": str(config_path),
            "image": config["image"],
            "runtime_identity": {"image_id": "deferred-until-image-inspection"},
            "runtime_lock": runtime_lock_record(config.get("runtime_profile")),
            "runtime_requirements": runtime_requirements,
            "selection_request": request.context.selection_request(),
            "selection_note": (
                "physical-to-logical mapping and idle state are resolved only "
                "by the side-effecting host preflight"
            ),
            "case": request.case,
            "case_config": case_configuration_records(request),
            "nproc_per_node": (
                request.nproc_per_node
                if request.nproc_per_node is not None
                else "derive-from-resolved-logical-devices"
            ),
            "permissions": {
                "privileged_root": request.allow_privileged_root,
                "network": "host",
                "ipc_namespace": "host",
                "pid_namespace": "private",
                "active_dmi": False,
                "candidate_runtime": request.allow_candidate_runtime,
            },
            "monitoring": monitor_policy(request.monitor == "on"),
            "worker": "benchmark_worker.py",
            "context": context_record(request.context),
        }

    def execute(self, request: BenchmarkRunRequest) -> int:
        plan = self.plan(request)
        if request.context.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        return self._execute(request, plan)

    def _execute(
        self, request: BenchmarkRunRequest, static_plan: dict[str, Any],
    ) -> int:
        config_path, config = load_host_config(request.context.config)
        wall_started = time.perf_counter()
        run_id = "benchmark-" + run_timestamp()
        result_dir = request.context.resolved_result_root(config) / run_id
        result_dir.mkdir(parents=True, exist_ok=False)
        summary: dict[str, Any] = {
            "schema_version": 2,
            "run_id": run_id,
            "kind": "benchmark",
            "case": request.case,
            "started_at": utc_now(),
            "status": "running",
            "execution_status": "running",
            "measurement_status": "not_started",
            "monitoring_status": (
                "running" if request.monitor == "on" else "not-run"
            ),
            "host_config": str(config_path),
            "static_plan": static_plan,
            "report_generation": {"status": "not_started"},
        }
        write_json(result_dir / "summary.json", summary)
        stage = "case-configuration"
        try:
            stored_case_config = snapshot_case_configuration(request, result_dir)
            summary["case_config"] = stored_case_config
            write_json(result_dir / "summary.json", summary)

            stage = "authorization"
            if config.get("vendor") != "ascend":
                raise ConfigurationError(
                    f"BenchmarkExecutor currently supports ascend, got {config.get('vendor')}"
                )
            if config.get("requires_privileged_root") and not request.allow_privileged_root:
                raise ConfigurationError(
                    "this host profile requires a privileged root container; "
                    "review the security impact and pass --allow-privileged-root"
                )

            stage = "image-inspection"
            image_info = docker_inspect(config["image"])
            runtime_lock = validate_runtime_identity(
                config, image_info,
                allow_candidate=request.allow_candidate_runtime,
            )
            summary["runtime"] = {
                "image": config["image"],
                "image_id": image_info["Id"],
                "image_labels": image_info.get("Config", {}).get("Labels") or {},
                "lock": runtime_lock,
            }

            stage = "host-environment"
            missing = [
                path for path in config["required_devices"]
                if not Path(path).exists()
            ]
            if missing:
                raise RuntimeError(f"required Ascend devices are missing: {missing}")

            stage = "host-preflight"
            preflight_path = run_host_preflight(
                result_dir,
                config["expected_device_ids"],
                npu_ids=request.context.npu_ids,
                device_ids=request.context.device_ids,
            )
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            selection = preflight.get("selection")
            if not isinstance(selection, dict):
                raise RuntimeError("host preflight did not return a selection")
            selected_ids = selection.get("selected_device_ids")
            if not isinstance(selected_ids, list) or not selected_ids:
                raise RuntimeError("host preflight returned an empty selection")
            nproc = request.nproc_per_node or len(selected_ids)
            if nproc != len(selected_ids):
                raise ConfigurationError(
                    "--nproc-per-node must equal the resolved logical Device count: "
                    f"nproc={nproc}, selected={selected_ids}"
                )
            validate_resolved_process_scope(
                request.case, static_plan.get("runtime_requirements"),
                nnodes=1, nproc=nproc,
            )
            selected_nodes = [Path(f"/dev/davinci{item}") for item in selected_ids]
            missing_selected = [str(path) for path in selected_nodes if not path.exists()]
            if missing_selected:
                raise RuntimeError(
                    f"selected Ascend device nodes are missing: {missing_selected}"
                )
            summary["selection"] = selection
            summary["host_preflight"] = {
                "path": str(preflight_path.relative_to(result_dir)),
                "result": preflight,
            }
            monitor_target_error = None
            try:
                monitor_targets = resolve_monitor_targets(preflight, selected_ids)
            except Exception as exc:
                monitor_targets = []
                monitor_target_error = (
                    "monitor target resolution failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            summary["monitoring"] = {
                "status": summary["monitoring_status"],
                "policy": monitor_policy(request.monitor == "on"),
                "targets": monitor_targets,
            }
            monitor_exchange_error = None
            if request.monitor == "on":
                try:
                    summary["monitoring"]["event_exchange"] = (
                        prepare_benchmark_event_exchange(result_dir)
                    )
                except Exception as exc:
                    monitor_exchange_error = (
                        "monitor event exchange preparation failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            stage = "container-plan"
            case_name = request.case.split(":", 1)[0]
            host_addr = "127.0.0.1"
            case_log_dir = result_dir / case_name / f"{host_addr}_noderank0"
            case_log_dir.mkdir(parents=True, exist_ok=True)
            container_name = re.sub(
                r"[^a-zA-Z0-9_.-]+", "-", f"flagperf-{case_name}-{run_id}"
            ).lower()
            command = [
                "docker", "run", "--rm", "--name", container_name,
                "--network=host", "--ipc=host",
                f"--shm-size={config['shm_size']}",
                "-e", f"ASCEND_RT_VISIBLE_DEVICES={','.join(map(str, selected_ids))}",
                "-e", "GEMS_VENDOR=ascend",
                "-e", "TRITON_ENABLE_TASKQUEUE=false",
                "-e", "DO_NOT_TRACK=1",
                "-e", "FLAGOS_LOG_FALLBACK=1",
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-v", f"{BASE_DIR}:/workspace/FlagPerf/base:ro",
                "-v", f"{result_dir}:/workspace/FlagPerf/results:rw",
                "-w", "/workspace/FlagPerf/base",
            ]
            for name, value in sorted(
                config.get("runtime_environment", {}).items()
            ):
                command.extend(["-e", f"{name}={value}"])
            if request.allow_privileged_root:
                command.append("--privileged")
            for device in [*config["required_devices"], *map(str, selected_nodes)]:
                command.append(f"--device={device}:{device}:rwm")
            for mount in config["host_mounts"]:
                path = Path(mount)
                if path.exists():
                    command.extend(["-v", f"{path}:{path}:ro"])
            if request.case_config is not None:
                override = request.case_config.expanduser().resolve()
                destination = (
                    f"/workspace/FlagPerf/base/benchmarks/{case_name}/"
                    "ascend/case_config.yaml"
                )
                command.extend(["-v", f"{override}:{destination}:ro"])

            env_script = BASE_DIR / "benchmarks" / case_name / "ascend" / "env.sh"
            setup = ""
            if env_script.is_file():
                setup = (
                    "source "
                    + shlex.quote(
                        f"/workspace/FlagPerf/base/benchmarks/{case_name}/ascend/env.sh"
                    )
                    + " && "
                )
            worker_args = [
                "python3", "/workspace/FlagPerf/base/benchmark_worker.py",
                "--case_name", request.case,
                "--nnodes", "1",
                "--nproc_per_node", str(nproc),
                "--node_rank", "0",
                "--master_addr", "127.0.0.1",
                "--master_port", str(request.master_port),
                "--host_addr", host_addr,
                "--vendor", "ascend",
                "--bench_or_tool", "BENCHMARK",
                "--perf_path", "/workspace/FlagPerf/base",
                "--log_dir", "/workspace/FlagPerf/results",
                "--log_level", request.log_level,
            ]
            if request.monitor == "on" and monitor_exchange_error is None:
                worker_args.append("--monitor-events")
            inner = setup + "exec " + shlex.join(worker_args)
            command.extend([
                config["image"], "/bin/bash", "-lc", inner,
            ])
            summary["resolved_plan"] = {
                **static_plan,
                "case_config": stored_case_config,
                "runtime_identity": {"image_id": image_info["Id"]},
                "selection": selection,
                "monitor_event_exchange": summary["monitoring"].get(
                    "event_exchange"
                ),
                "nproc_per_node": nproc,
                "container_name": container_name,
                "command": command,
            }
            write_json(result_dir / "resolved-plan.json", summary["resolved_plan"])
            write_json(result_dir / "summary.json", summary)

            stage = "container-run"
            lease = DeviceLease(selected_ids, run_id=run_id, kind="benchmark")
            lease.acquire()
            summary["device_lease"] = lease.record()
            timed_out = False
            usage_monitor = None
            if request.monitor == "on":
                monitor_setup_error = monitor_target_error or monitor_exchange_error
                if monitor_setup_error is not None:
                    monitor_result = write_monitor_terminal_summary(
                        result_dir,
                        status="failed",
                        enabled=True,
                        targets=monitor_targets,
                        reason=monitor_setup_error,
                    )
                else:
                    try:
                        usage_monitor = create_usage_monitor(monitor_targets)
                        usage_monitor.start()
                    except Exception as exc:
                        reasons = [
                            "monitor startup failed: "
                            f"{type(exc).__name__}: {exc}"
                        ]
                        if usage_monitor is not None:
                            try:
                                usage_monitor.stop()
                            except Exception as stop_exc:
                                reasons.append(
                                    "monitor cleanup after startup failure failed: "
                                    f"{type(stop_exc).__name__}: {stop_exc}"
                                )
                        usage_monitor = None
                        monitor_result = write_monitor_terminal_summary(
                            result_dir,
                            status="failed",
                            enabled=True,
                            targets=monitor_targets,
                            reason="; ".join(reasons),
                        )
            else:
                monitor_result = write_monitor_terminal_summary(
                    result_dir,
                    status="not-run",
                    enabled=False,
                    targets=monitor_targets,
                    reason="Benchmark monitoring disabled by --monitor off",
                )
            container_started_at = utc_now()
            container_started_monotonic_ns = time.monotonic_ns()
            try:
                try:
                    proc = subprocess.run(
                        command,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=request.context.timeout,
                    )
                    runner_log = proc.stdout or ""
                    returncode = proc.returncode
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    runner_log = exc.stdout or ""
                    if isinstance(runner_log, bytes):
                        runner_log = runner_log.decode(errors="replace")
                    runner_log += (
                        f"\nFlagPerf Benchmark timeout after "
                        f"{request.context.timeout}s\n"
                    )
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    returncode = 124
                except KeyboardInterrupt:
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    runner_log = "FlagPerf Benchmark interrupted by operator\n"
                    returncode = 130
            finally:
                container_finished_monotonic_ns = time.monotonic_ns()
                container_record = {
                    "schema_version": 1,
                    "kind": "container-window",
                    "case": request.case,
                    "container_name": container_name,
                    "started_at": container_started_at,
                    "finished_at": utc_now(),
                    "started_monotonic_ns": container_started_monotonic_ns,
                    "finished_monotonic_ns": container_finished_monotonic_ns,
                    "duration_s": round(
                        (
                            container_finished_monotonic_ns
                            - container_started_monotonic_ns
                        ) / 1e9,
                        9,
                    ),
                }
                if request.monitor == "on":
                    monitor_result = finalize_benchmark_monitor_safely(
                        usage_monitor,
                        result_dir,
                        container_record,
                        selected_ids,
                        nproc,
                        monitor_targets,
                        monitor_result if usage_monitor is None else None,
                    )
                summary["monitoring"] = monitor_result
                summary["monitoring_status"] = monitor_result["status"]
                try:
                    lease.release()
                finally:
                    summary["device_lease"]["released_at"] = utc_now()
            (result_dir / "runner.log").write_text(
                runner_log, encoding="utf-8", errors="replace"
            )

            stage = "result-parsing"
            benchmark_log_path = case_log_dir / "benchmark.log.txt"
            benchmark_log = (
                benchmark_log_path.read_text(encoding="utf-8", errors="replace")
                if benchmark_log_path.is_file()
                else ""
            )
            parsed = parse_benchmark_results(benchmark_log, nproc)
            parsed.update({
                "case": request.case,
                "container_returncode": returncode,
                "timed_out": timed_out,
                "fallback_count": (
                    runner_log.count("[flagos cpu_fallback]")
                    + benchmark_log.count("[flagos cpu_fallback]")
                ),
                "benchmark_log": (
                    str(benchmark_log_path.relative_to(result_dir))
                    if benchmark_log_path.is_file() else None
                ),
            })
            if parsed["fallback_count"]:
                parsed["status"] = "failed"
                parsed["failure_reason"] = "cpu-fallback-detected"
            write_json(result_dir / "benchmark-result.json", parsed)

            execution_status = "passed" if returncode == 0 else "failed"
            measurement_status = parsed["status"]
            status = aggregate_benchmark_status(
                execution_status,
                measurement_status,
                str(summary.get("monitoring_status")),
                monitor_enabled=request.monitor == "on",
            )

            stage = "host-postflight"
            try:
                postflight_path = run_host_preflight(
                    result_dir,
                    config["expected_device_ids"],
                    npu_ids=request.context.npu_ids,
                    device_ids=request.context.device_ids,
                    label="host-postflight",
                )
                postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
                summary["host_postflight"] = {
                    "status": "passed",
                    "path": str(postflight_path.relative_to(result_dir)),
                    "result": postflight,
                }
            except Exception as exc:
                summary["host_postflight"] = {
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                if status == "passed":
                    status = "partial"

            summary.update({
                "status": status,
                "execution_status": execution_status,
                "measurement_status": measurement_status,
                "container_returncode": returncode,
                "timed_out": timed_out,
                "benchmark_result": "benchmark-result.json",
                "finished_at": utc_now(),
                "wall_clock_duration_s": round(
                    time.perf_counter() - wall_started, 4
                ),
            })
            write_json(result_dir / "summary.json", summary)
        except Exception as exc:
            monitor_summary_path = (
                result_dir / "benchmark-monitor" / "summary.json"
            )
            if not monitor_summary_path.is_file():
                embedded_monitor = summary.get("monitoring")
                targets = (
                    embedded_monitor.get("targets", [])
                    if isinstance(embedded_monitor, dict) else []
                )
                monitor_result = write_monitor_terminal_summary(
                    result_dir,
                    status="not-run",
                    enabled=request.monitor == "on",
                    targets=targets,
                    reason=(
                        f"Benchmark failed at stage {stage} before a complete "
                        "monitor lifecycle was available"
                        if request.monitor == "on"
                        else "Benchmark monitoring disabled by --monitor off"
                    ),
                )
                summary["monitoring"] = monitor_result
                summary["monitoring_status"] = monitor_result["status"]
            summary.update({
                "status": "failed",
                "execution_status": "failed",
                "measurement_status": "not_available",
                "failure_stage": stage,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "finished_at": utc_now(),
                "wall_clock_duration_s": round(
                    time.perf_counter() - wall_started, 4
                ),
            })
            result_path = result_dir / "benchmark-result.json"
            if not result_path.is_file():
                write_json(result_path, {
                    "schema_version": 1,
                    "status": "failed",
                    "case": request.case,
                    "failure_stage": stage,
                    "error": str(exc),
                    "metrics": [],
                })
            write_json(result_dir / "summary.json", summary)
            try:
                generate_benchmark_report(result_dir)
            except Exception as report_exc:
                summary["report_generation"] = {
                    "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
                    "status": "failed",
                    "error": str(report_exc),
                    "error_type": type(report_exc).__name__,
                }
                write_json(result_dir / "summary.json", summary)
            print(f"ERROR: {exc}", file=sys.stderr)
            print(f"Result directory: {result_dir}")
            return 1

        try:
            generate_benchmark_report(result_dir)
        except Exception as exc:
            current = json.loads(
                (result_dir / "summary.json").read_text(encoding="utf-8")
            )
            current["report_generation"] = {
                "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            write_json(result_dir / "summary.json", current)
            print(f"WARNING: Benchmark report generation failed: {exc}", file=sys.stderr)

        print(f"Result directory: {result_dir}")
        if summary["status"] == "passed":
            return 0
        if summary["status"] == "partial":
            return 2
        return 1
