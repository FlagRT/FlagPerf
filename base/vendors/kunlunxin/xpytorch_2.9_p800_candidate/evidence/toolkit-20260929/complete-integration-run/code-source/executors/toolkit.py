#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Vendor-dispatched Base Toolkit request, planner, and host executor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import grp
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

from executors.progress import RunProgress
from executors.common import (
    BaseRunContext,
    DeviceLease,
    context_record,
    load_host_config,
    runtime_lock_record,
    validate_runtime_identity,
)


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 6
P2P_LATENCY_CASE = "interconnect-P2P_intraserver-latency"
MIN_TOOLBOX_VERSION = (26, 1, 0)


def validate_toolbox_version(install_info: str) -> str:
    """Require an identifiable ToolBox release at least 26.1.0."""
    versions = re.findall(r"^version\s*=\s*(\S+)\s*$", install_info, re.MULTILINE)
    if len(versions) != 1:
        fail("ToolBox version is missing or ambiguous; required >= 26.1.0")
    version = versions[0]
    # Vendor RC numbering replaces the patch component, e.g. 7.2.RC1.
    match = re.fullmatch(r"(\d+)\.(\d+)\.(?:(\d+)|RC(\d+))", version)
    if match is None:
        fail(f"Unrecognized ToolBox version {version!r}; required >= 26.1.0")
    major, minor, patch, rc = match.groups()
    actual = (int(major), int(minor), int(patch or 0), 0 if rc else 1)
    if actual < (*MIN_TOOLBOX_VERSION, 1):
        fail(f"ToolBox version {version} is below the required minimum 26.1.0")
    return version


def container_namespace_args(cases: list[str] | None) -> list[str]:
    """Share only the namespaces required by the selected Toolkit Cases."""
    arguments = ["--ipc=host"]
    if cases is None or P2P_LATENCY_CASE in cases:
        # DMI P2P latency exchanges ACL IPC memory between child processes.
        # The driver validates host PIDs; a private PID namespace makes
        # aclrtIpcMemImportByKey fail with ACL_ERROR_RT_DRV_INTERNAL_ERROR.
        arguments.append("--pid=host")
    return arguments


def add_cli_arguments(
    parser: argparse.ArgumentParser, *, include_compat_suite: bool = False,
) -> None:
    parser.add_argument(
        "--config", type=Path,
        default=BASE_DIR / "configs" / "ascend910_cann9_local.yaml",
    )
    parser.add_argument("--case", action="append", dest="cases")
    if include_compat_suite:
        parser.add_argument(
            "--suite", choices=("ascend-toolkit",), default="ascend-toolkit",
            help=("compatibility selector; the command is Toolkit-only and "
                  "defaults to ascend-toolkit"),
        )
    parser.add_argument(
        "--legacy-probe", action="store_true",
        help="also run the old toolkit commands and save fixed-position comparisons",
    )
    parser.add_argument(
        "--compute-monitor", choices=("on", "off"), default="on",
        help="collect concurrent npu-smi evidence for Ascend computation Cases",
    )
    parser.add_argument(
        "--data-movement-monitor", choices=("on", "off"), default="on",
        help="collect concurrent/static npu-smi evidence for data movement Cases",
    )
    parser.add_argument(
        "--allow-disruptive-dmi", action="store_true",
        help=("explicitly allow active Ascend performance and diagnosis commands on the "
              "selected devices, or all inventory devices when no selector is given"),
    )
    parser.add_argument(
        "--latency-sizes", default=None,
        help="Override latency sizes; default P2P: 64K, H2D/D2H: 512,4K,64K,1M",
    )
    parser.add_argument("--hccl-min-bytes", default="8K")
    parser.add_argument("--hccl-max-bytes", default="1G")
    parser.add_argument(
        "--allow-privileged-root", action="store_true",
        help=("explicitly allow the container to run as root with --privileged; "
              "required by this host's legacy Ascend driver path; P2P latency "
              "also shares the host PID namespace for ACL IPC"),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--npu-ids",
        help=("physical NPU IDs/ranges for ascend-toolkit, for example 1-7; "
              "resolved through the current npu-smi map"),
    )
    selection.add_argument(
        "--device-ids",
        help=("logical Device IDs/ranges for ascend-toolkit, for example "
              "2,3,6-9"),
    )
    selection.add_argument("--physical-device-ids", help="P800 physical card IDs/ranges; card 1 is excluded")
    from executors.p800_toolkit import add_cli_arguments as add_p800_arguments
    add_p800_arguments(parser)
    parser.add_argument(
        "--result-root", type=Path,
        help="override the host profile result root for this run",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="hard timeout for the Toolkit container in seconds",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print a static plan without inspecting devices or starting Docker",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the evidence-first Ascend Base Toolkit",
    )
    add_cli_arguments(parser, include_compat_suite=True)
    return parser.parse_args(argv)


@dataclass(frozen=True)
class ToolkitRunRequest:
    context: BaseRunContext
    cases: tuple[str, ...] | None
    legacy_probe: bool
    compute_monitor: str
    data_movement_monitor: str
    allow_disruptive_dmi: bool
    latency_sizes: str | None
    hccl_min_bytes: str
    hccl_max_bytes: str
    allow_privileged_root: bool
    p800_options: dict[str, Any] | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "ToolkitRunRequest":
        return cls(
            context=BaseRunContext(
                config=args.config,
                npu_ids=args.npu_ids,
                device_ids=args.device_ids,
                physical_device_ids=getattr(args, "physical_device_ids", None),
                result_root=args.result_root,
                timeout=args.timeout,
                dry_run=args.dry_run,
            ),
            cases=tuple(args.cases) if args.cases else None,
            legacy_probe=args.legacy_probe,
            compute_monitor=args.compute_monitor,
            data_movement_monitor=args.data_movement_monitor,
            allow_disruptive_dmi=args.allow_disruptive_dmi,
            latency_sizes=args.latency_sizes,
            hccl_min_bytes=args.hccl_min_bytes,
            hccl_max_bytes=args.hccl_max_bytes,
            allow_privileged_root=args.allow_privileged_root,
            p800_options={key: getattr(args, key, default) for key, default in p800_defaults().items()},
        )

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            config=self.context.config,
            cases=list(self.cases) if self.cases else None,
            suite="ascend-toolkit",
            legacy_probe=self.legacy_probe,
            compute_monitor=self.compute_monitor,
            data_movement_monitor=self.data_movement_monitor,
            allow_disruptive_dmi=self.allow_disruptive_dmi,
            latency_sizes=self.latency_sizes,
            hccl_min_bytes=self.hccl_min_bytes,
            hccl_max_bytes=self.hccl_max_bytes,
            allow_privileged_root=self.allow_privileged_root,
            npu_ids=self.context.npu_ids,
            device_ids=self.context.device_ids,
            physical_device_ids=self.context.physical_device_ids,
            result_root=self.context.result_root,
            timeout=self.context.timeout,
            dry_run=self.context.dry_run,
            **(self.p800_options or p800_defaults()),
        )

    def validate(self, *, require_selection: bool) -> None:
        self.context.validate(require_selection=require_selection)
        validate_static_args(self.to_namespace())


class ToolkitExecutor:
    """Plan or execute the existing evidence-first Toolkit protocol."""

    def __init__(self, *, require_selection: bool = True) -> None:
        self.require_selection = require_selection

    def plan(self, request: ToolkitRunRequest) -> dict[str, Any]:
        request.validate(require_selection=self.require_selection)
        config_path, config = load_host_config(request.context.config)
        if config["vendor"] == "kunlunxin":
            from executors.p800_toolkit import plan as p800_plan
            return p800_plan(request, config_path, config)
        if request.context.physical_device_ids is not None:
            fail("Ascend Toolkit uses --npu-ids or --device-ids")
        namespace_args = container_namespace_args(
            list(request.cases) if request.cases else None
        )
        return {
            "schema_version": 1,
            "kind": "toolkit",
            "mode": "static-dry-run" if request.context.dry_run else "execution",
            "host_config": str(config_path),
            "image": config["image"],
            "runtime_identity": {
                "image_id": "deferred-until-image-inspection",
            },
            "runtime_lock": runtime_lock_record(config.get("runtime_profile")),
            "selection_request": request.context.selection_request(),
            "selection_note": (
                "physical-to-logical mapping and idle state are resolved only "
                "by the side-effecting host preflight"
            ),
            "cases": list(request.cases) if request.cases else "default-toolkit-set",
            "permissions": {
                "privileged_root": request.allow_privileged_root,
                "active_dmi": request.allow_disruptive_dmi,
                "ipc_namespace": "host",
                "pid_namespace": (
                    "host" if "--pid=host" in namespace_args else "private"
                ),
            },
            "monitoring": {
                "compute": request.compute_monitor,
                "data_movement": request.data_movement_monitor,
            },
            "context": context_record(request.context),
        }

    def execute(self, request: ToolkitRunRequest) -> int:
        plan = self.plan(request)
        if request.context.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if plan.get("vendor") == "kunlunxin":
            from executors.p800_toolkit import execute as p800_execute
            return p800_execute(request, plan)
        return execute_toolkit(request.to_namespace(), plan=plan)


def fail(message: str) -> None:
    raise RuntimeError(message)


def p800_defaults():
    from executors.p800_toolkit import DEFAULTS
    return DEFAULTS.copy()


def validate_static_args(args: argparse.Namespace) -> None:
    """Reject selector combinations that cannot form a valid execution plan."""
    selected = bool(args.npu_ids or args.device_ids)
    if args.legacy_probe and selected:
        fail("explicit device selection cannot be combined with --legacy-probe")
    cases = getattr(args, "cases", None) or []
    custom_hccl = (
        getattr(args, "hccl_min_bytes", "8K") != "8K"
        or getattr(args, "hccl_max_bytes", "1G") != "1G"
    )
    if custom_hccl and "interconnect-MPI_intraserver" not in cases:
        fail("HCCL size overrides require --case interconnect-MPI_intraserver")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_summary(result_dir: Path, summary: dict[str, Any]) -> None:
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_report_safely(result_dir: Path) -> dict[str, Any]:
    """Generate the presentation layer without changing experiment status."""
    summary_path = result_dir / "summary.json"
    original = json.loads(summary_path.read_text(encoding="utf-8"))
    original_status = original.get("status")
    try:
        from generate_toolkit_report import generate_and_record
        return generate_and_record(result_dir)
    except Exception as exc:
        # generate_and_record normally persists this itself.  The fallback also
        # covers import/syntax failures in the optional presentation layer.
        current = json.loads(summary_path.read_text(encoding="utf-8"))
        current["report_generation"] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        current["status"] = original_status
        write_summary(result_dir, current)
        print(f"WARNING: toolkit Markdown report generation failed: {exc}",
              file=sys.stderr)
        return current["report_generation"]


# Compatibility names retained for Toolkit callers and tests.
from executors.host import docker_inspect
from base.vendors.ascend.provider import run_host_preflight


def execute_toolkit(
    args: argparse.Namespace, *, plan: dict[str, Any] | None = None,
) -> int:
    wall_started = time.perf_counter()
    validate_static_args(args)
    namespace_args = container_namespace_args(args.cases)
    config_path, config = load_host_config(args.config)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.result_root is not None:
        result_root = args.result_root.expanduser().resolve()
    else:
        configured_root = Path(config["result_root"])
        result_root = (
            configured_root.resolve()
            if configured_root.is_absolute()
            else (BASE_DIR / configured_root).resolve()
        )
    result_dir = result_root / timestamp
    result_dir.mkdir(parents=True, exist_ok=False)
    final_summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": timestamp,
        "started_at": utc_now(),
        "status": "running",
        "kind": "toolkit",
        "suite": "ascend-toolkit",
        "host_config": str(config_path),
        "image": config.get("image"),
        "privileged_root": args.allow_privileged_root,
        "compute_monitor": args.compute_monitor,
        "data_movement_monitor": args.data_movement_monitor,
        "container_namespaces": {
            "ipc": "host",
            "pid": "host" if "--pid=host" in namespace_args else "private",
        },
        "resolved_plan": plan,
    }
    stage = "authorization"

    try:
        if config.get("requires_privileged_root") and not args.allow_privileged_root:
            fail(
                "this host profile requires a privileged root container for Ascend "
                "driver access; review the security impact, then explicitly pass "
                "--allow-privileged-root"
            )
        if not args.allow_disruptive_dmi:
            fail(
                "the Ascend toolkit suite runs active tests on the selected devices "
                "(all inventory devices by default) and may affect workloads; "
                "explicitly pass --allow-disruptive-dmi"
            )
        stage = "image-inspection"
        image = config["image"]
        image_info = docker_inspect(image)
        runtime_lock = validate_runtime_identity(config, image_info)
        final_summary.update(
            image=image,
            image_id=image_info["Id"],
            image_labels=image_info.get("Config", {}).get("Labels") or {},
            runtime_lock=runtime_lock,
        )

        stage = "host-environment"
        toolbox_host_path = Path(config["toolbox_host_path"]).resolve()
        cann_runtime_gid = grp.getgrnam("HwHiAiUser").gr_gid
        toolbox_version_file = toolbox_host_path / "latest" / "ascend_toolbox_install.info"
        if not toolbox_version_file.is_file():
            fail(f"ToolBox installation is unavailable: {toolbox_host_path}")
        final_summary.update(
            toolbox_host_path=str(toolbox_host_path),
            toolbox_install_info=toolbox_version_file.read_text(encoding="utf-8"),
        )
        final_summary["toolbox_minimum_version"] = "26.1.0"
        final_summary["toolbox_version"] = validate_toolbox_version(
            final_summary["toolbox_install_info"]
        )

        stage = "device-inventory"
        missing = [path for path in config["required_devices"] if not Path(path).exists()]
        if missing:
            fail(f"required Ascend devices are missing: {missing}")
        davinci_devices = sorted(
            (path for path in Path("/dev").glob("davinci[0-9]*")
             if re.fullmatch(r"davinci\d+", path.name)),
            key=lambda path: int(path.name.removeprefix("davinci")),
        )
        if not davinci_devices:
            fail("no /dev/davinciN devices found")
        final_summary["device_nodes_exposed"] = [str(path) for path in davinci_devices]

        host_preflight_summary: Path | None = None
        host_preflight_result: dict[str, Any] | None = None
        selected_davinci_devices = list(davinci_devices)
        expected_device_ids = config.get("expected_device_ids")
        if (not isinstance(expected_device_ids, list) or not expected_device_ids
                or any(not isinstance(item, int) or item < 0
                       for item in expected_device_ids)
                or len(expected_device_ids) != len(set(expected_device_ids))):
            fail("ascend-toolkit requires a non-empty expected_device_ids inventory")
        stage = "host-preflight"
        host_preflight_summary = run_host_preflight(
            result_dir, expected_device_ids,
            npu_ids=args.npu_ids, device_ids=args.device_ids,
        )
        host_preflight_result = json.loads(
            host_preflight_summary.read_text(encoding="utf-8")
        )
        selection = host_preflight_result.get("selection")
        if not isinstance(selection, dict):
            fail("Ascend host preflight did not resolve a device selection")
        selected_ids = selection.get("selected_device_ids")
        if not isinstance(selected_ids, list) or not selected_ids:
            fail("Ascend host preflight resolved an empty device selection")
        selected_davinci_devices = [Path(f"/dev/davinci{item}") for item in selected_ids]
        missing_selected = [str(path) for path in selected_davinci_devices if not path.exists()]
        if missing_selected:
            fail(f"selected Ascend device nodes are missing: {missing_selected}")
        final_summary["selection"] = selection
        final_summary["device_nodes_requested"] = [
            str(path) for path in selected_davinci_devices
        ]
        final_summary["host_preflight"] = {
            "path": str(host_preflight_summary.relative_to(result_dir)),
            "result": host_preflight_result,
        }

        dmi_log_dir = result_dir / "ascend-dmi-log"
        dmi_log_dir.mkdir()
        container_name = f"flagperf-toolkit-{timestamp}".lower()
        command = [
            "docker", "run", "--rm", "--name", container_name,
            "--network=none", *namespace_args,
            f"--shm-size={config['shm_size']}",
            "-e", "FLAGOS_LOG_FALLBACK=1",
            "-e", "GEMS_VENDOR=ascend",
            "-e", "TRITON_ENABLE_TASKQUEUE=false",
            "-e", "DO_NOT_TRACK=1",
            "-v", f"{BASE_DIR}:/workspace/FlagPerf/base:ro",
            "-v", f"{result_dir}:/workspace/FlagPerf/results:rw",
            "-v", f"{dmi_log_dir}:/var/log/ascend-dmi:rw",
            "-v", "/etc/passwd:/etc/passwd:ro",
            "-v", "/etc/group:/etc/group:ro",
        ]
        toolbox_setup = ""
        if args.allow_privileged_root:
            command.append("--privileged")
            # Ascend-DMI refuses latency model objects that are not owned by
            # root.  Keep the host ToolBox immutable and materialize a
            # root-owned, container-lifetime-only copy in tmpfs instead.
            command.extend([
                "-e", "HOME=/root",
                "-v", f"{toolbox_host_path}:/opt/flagperf-toolbox-host:ro",
                "--tmpfs", "/usr/local/Ascend/toolbox:rw,exec,nosuid,nodev,mode=755",
            ])
            toolbox_setup = (
                "cp -a /opt/flagperf-toolbox-host/. /usr/local/Ascend/toolbox/ && "
                "chown -R 0:0 /usr/local/Ascend/toolbox && "
            )
            final_summary["toolbox_container_materialization"] = {
                "source": "/opt/flagperf-toolbox-host",
                "destination": "/usr/local/Ascend/toolbox",
                "storage": "tmpfs",
                "owner": "root:root",
                "host_source_read_only": True,
            }
        else:
            username = os.environ.get("USER", str(os.getuid()))
            command.extend([
                "-v", f"{toolbox_host_path}:/usr/local/Ascend/toolbox:ro",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--group-add", str(cann_runtime_gid),
                "--tmpfs",
                f"/home/{username}:rw,nosuid,nodev,uid={os.getuid()},gid={os.getgid()},mode=700",
                "-e", f"HOME=/home/{username}",
            ])
        for device in [*config["required_devices"], *map(str, selected_davinci_devices)]:
            command.append(f"--device={device}:{device}:rwm")
        for mount in config["host_mounts"]:
            path = Path(mount)
            if path.exists():
                command.extend(["-v", f"{path}:{path}:ro"])

        selected = args.cases or [
            "computation-BF16", "computation-FP16", "computation-FP32",
            "computation-INT8", "main_memory-bandwidth",
            "main_memory-capacity", "interconnect-h2d",
            "interconnect-d2h", "interconnect-h2d-latency",
            "interconnect-d2h-latency", "interconnect-P2P_intraserver-latency",
            "interconnect-P2P_intraserver",
        ]
        runner = (
            "/workspace/FlagPerf/base/toolkits/_common/ascend/A3/"
            "evidence_runner.py"
        )
        legacy = " --legacy-probe" if args.legacy_probe else ""
        device_selection = ""
        if args.npu_ids or args.device_ids:
            assert host_preflight_result is not None
            resolved = host_preflight_result["selection"]["selected_device_ids"]
            source = host_preflight_result["selection"]["source"]
            device_selection = (
                " --device-ids " + ",".join(map(str, resolved))
                + " --selection-source " + source
            )
        latency_args = f"--latency-sizes {shlex.quote(args.latency_sizes)} " if args.latency_sizes else ""
        inner = (
            toolbox_setup
            + "source /usr/local/Ascend/toolbox/set_env.sh && "
            f"exec python3 {runner} "
            "--output /workspace/FlagPerf/results/toolkit-evidence "
            f"--allow-disruptive-dmi{legacy}{device_selection} "
            f"{latency_args}"
            f"--hccl-min-bytes {shlex.quote(args.hccl_min_bytes)} "
            f"--hccl-max-bytes {shlex.quote(args.hccl_max_bytes)} "
            f"--compute-monitor {args.compute_monitor} "
            f"--data-movement-monitor {args.data_movement_monitor} --cases \"$@\""
        )
        command.extend([image, "/bin/bash", "-lc", inner, "flagperf-local", *selected])
        final_summary["container_name"] = container_name
        final_summary["resolved_plan"] = {
            **(plan or {}),
            "mode": "execution",
            "runtime_identity": {"image_id": image_info["Id"]},
            "selection": selection,
            "device_nodes": [str(path) for path in selected_davinci_devices],
            "container_name": container_name,
            "command": command,
        }
        write_summary(result_dir, final_summary)
        (result_dir / "resolved-plan.json").write_text(
            json.dumps(final_summary["resolved_plan"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        stage = "container-run"
        lease = DeviceLease(
            selected_ids, run_id=timestamp, kind="toolkit",
        )
        lease.acquire()
        final_summary["device_lease"] = lease.record()
        try:
            try:
                with RunProgress(
                    "Toolkit",
                    events=result_dir / "toolkit-evidence" / "progress.jsonl",
                ) as progress:
                    proc = subprocess.run(
                        command,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=args.timeout,
                    )
                    progress.finished(proc.returncode)
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                output += f"\nFlagPerf Toolkit timeout after {args.timeout}s\n"
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                )
                proc = subprocess.CompletedProcess(command, 124, output)
            except KeyboardInterrupt:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                )
                proc = subprocess.CompletedProcess(
                    command, 130, "FlagPerf Toolkit interrupted by operator\n"
                )
        finally:
            lease.release()
            final_summary["device_lease"]["released_at"] = utc_now()
        log = proc.stdout or ""
        print(log, end="" if log.endswith("\n") else "\n")
        (result_dir / "runner.log").write_text(log, encoding="utf-8")
        fallback_count = log.count("[flagos cpu_fallback]")

        stage = "result-finalization"
        container_summary_path = result_dir / "summary.container.json"
        if container_summary_path.is_file():
            final_summary.update(json.loads(container_summary_path.read_text(encoding="utf-8")))
        final_summary.update(
            image=image,
            image_id=image_info["Id"],
            image_labels=image_info.get("Config", {}).get("Labels") or {},
            host_config=str(config_path),
            toolbox_host_path=str(toolbox_host_path),
            toolbox_install_info=toolbox_version_file.read_text(encoding="utf-8"),
            device_nodes_exposed=[str(path) for path in davinci_devices],
            device_nodes_requested=[str(path) for path in selected_davinci_devices],
            privileged_root=args.allow_privileged_root,
            fallback_count=fallback_count,
            container_returncode=proc.returncode,
            suite="ascend-toolkit",
            kind="toolkit",
            run_id=timestamp,
        )
        if host_preflight_summary is not None:
            assert host_preflight_result is not None
            final_summary["selection"] = host_preflight_result["selection"]
            final_summary["host_preflight"] = {
                "path": str(host_preflight_summary.relative_to(result_dir)),
                "result": host_preflight_result,
            }
        toolkit_manifest = result_dir / "toolkit-evidence" / "manifest.json"
        if toolkit_manifest.is_file():
            evidence = json.loads(toolkit_manifest.read_text(encoding="utf-8"))
            final_summary["toolkit_evidence"] = {
                "path": str(toolkit_manifest.relative_to(result_dir)),
                "schema_version": evidence.get("schema_version"),
                "status": evidence.get("status"),
            }
            final_summary["status"] = evidence.get("status", "failed")
        else:
            final_summary["status"] = "failed"
            final_summary["error"] = "toolkit manifest was not produced"
        if fallback_count:
            final_summary["status"] = "failed"
            final_summary.setdefault("failed_cases", []).append("cpu-fallback-detected")
        if proc.returncode != 0 and not (
            proc.returncode == 2
            and final_summary.get("status") == "partial"
        ):
            final_summary["status"] = "failed"
        final_summary["finished_at"] = utc_now()
        final_summary["wall_clock_duration_s"] = round(
            time.perf_counter() - wall_started, 4
        )
        write_summary(result_dir, final_summary)
    except Exception as exc:
        final_summary.update(
            status="failed",
            failure_stage=stage,
            error=str(exc),
            error_type=type(exc).__name__,
            finished_at=utc_now(),
            wall_clock_duration_s=round(
                time.perf_counter() - wall_started, 4
            ),
        )
        summaries = list((result_dir / "host-preflight").glob("*/summary.json"))
        if len(summaries) == 1:
            final_summary["host_preflight"] = {
                "path": str(summaries[0].relative_to(result_dir)),
                "result": json.loads(summaries[0].read_text(encoding="utf-8")),
            }
        write_summary(result_dir, final_summary)
        generate_report_safely(result_dir)
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Result directory: {result_dir}")
        return 2

    generate_report_safely(result_dir)
    print(f"Result directory: {result_dir}")
    if final_summary.get("status") == "passed":
        return 0
    if final_summary.get("status") == "partial":
        return 2
    return 1


def main(argv: list[str] | None = None) -> int:
    request = ToolkitRunRequest.from_namespace(parse_args(argv))
    # The compatibility commands retain the historical default-all selection.
    # The unified facade instantiates ToolkitExecutor(require_selection=True).
    return ToolkitExecutor(require_selection=False).execute(request)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
