#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Evidence-first host runner for bounded Ascend FlagCX candidate gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


PROFILE_DIR = Path(__file__).resolve().parent
BASE_DIR = PROFILE_DIR.parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from executors.common import (  # noqa: E402
    ConfigurationError,
    DeviceLease,
    load_host_config,
    parse_id_spec,
    run_timestamp,
    sha256_file,
    utc_now,
    validate_runtime_identity,
    write_json,
)
from executors.toolkit import docker_inspect, run_host_preflight  # noqa: E402


DEFAULT_CONFIG = BASE_DIR / "configs" / "ascend910_cann9_p2p_candidate.yaml"
SAFE_RUN_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")


def artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def parse_json_objects(text: str) -> list[dict[str, Any]]:
    """Parse JSON objects from diagnostics or concatenated process output."""
    records = []
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + consumed
        if isinstance(value, dict):
            records.append(value)
    return records


def parse_rank_record_lines(text: str) -> list[dict[str, Any]]:
    """Select rank records from verifier output."""
    return [
        value for value in parse_json_objects(text)
        if value.get("schema_version") == 1 and "rank" in value
    ]


def validate_sentinel_records(
    records: list[dict[str, Any]], *, expected_stream_mode: str = "default",
) -> None:
    ranks = sorted(record.get("rank") for record in records)
    if ranks != [0, 1]:
        raise RuntimeError(f"sentinel verifier must report ranks [0, 1], got {ranks}")
    for record in records:
        if record.get("status") != "passed":
            raise RuntimeError(f"sentinel rank did not pass: {record}")
        if record.get("world_size") != 2:
            raise RuntimeError(f"sentinel world size drifted: {record}")
        if record.get("public_backend") != "flagos":
            raise RuntimeError(f"sentinel public backend drifted: {record}")
        if record.get("stream_mode") != expected_stream_mode:
            raise RuntimeError(f"sentinel stream mode drifted: {record}")
        if "flagcx" not in str(record.get("inner_backend", "")).lower():
            raise RuntimeError(f"sentinel did not select FlagCX: {record}")
        payloads = record.get("payloads")
        if not isinstance(payloads, list) or [
            item.get("elements") for item in payloads
        ] != [1, 4, 257, 65536]:
            raise RuntimeError(f"sentinel payload coverage drifted: {record}")


def validate_expected_timeout_record(record: dict[str, Any]) -> None:
    """Require an observed deadline and proof that its exact container is gone."""
    if record.get("returncode") != 124 or record.get("timed_out") is not True:
        raise RuntimeError(
            "timeout cleanup gate must observe returncode 124 and timed_out=true"
        )
    cleanup = record.get("timeout_cleanup")
    if not isinstance(cleanup, dict):
        raise RuntimeError("timeout cleanup gate did not record container cleanup")
    if cleanup.get("remove_returncode") != 0:
        raise RuntimeError(f"timed-out container removal failed: {cleanup}")
    if cleanup.get("inspect_returncode") == 0:
        raise RuntimeError(f"timed-out container still exists: {cleanup}")


def validate_acl_event_records(records: list[dict[str, Any]]) -> None:
    probes = [record for record in records if record.get("kind") == "acl-event-flag-ab"]
    if len(probes) != 1:
        raise RuntimeError(f"ACL event gate must report one probe, got {len(probes)}")
    probe = probes[0]
    if probe.get("schema_version") != 1 or probe.get("device") != 0:
        raise RuntimeError(f"ACL event probe identity drifted: {probe}")
    if probe.get("create_stream_rc") != 0 or probe.get("destroy_stream_rc") != 0:
        raise RuntimeError(f"ACL event probe stream lifecycle failed: {probe}")
    by_name = {
        result.get("name"): result for result in probe.get("results", [])
        if isinstance(result, dict)
    }
    if set(by_name) != {"timeline", "sync"}:
        raise RuntimeError(f"ACL event probe coverage drifted: {probe}")
    timeline = by_name["timeline"]
    if timeline.get("flag") != 0x8:
        raise RuntimeError(f"timeline event flag drifted: {timeline}")
    if timeline.get("create_rc") != 0 or timeline.get("record_rc") != 0:
        raise RuntimeError(f"timeline event setup failed before wait: {timeline}")
    if timeline.get("wait_rc") in (None, 0):
        raise RuntimeError(f"timeline event unexpectedly supported cross-stream wait: {timeline}")
    if timeline.get("destroy_rc") != 0:
        raise RuntimeError(f"timeline event cleanup failed: {timeline}")
    sync = by_name["sync"]
    expected_sync = {
        "flag": 0x1,
        "create_rc": 0,
        "record_rc": 0,
        "wait_rc": 0,
        "synchronize_rc": 0,
        "destroy_rc": 0,
    }
    for key, expected in expected_sync.items():
        if sync.get(key) != expected:
            raise RuntimeError(f"sync event {key} failed: {sync}")


def build_container_command(
    config: dict[str, Any], device_ids: tuple[int, int], *,
    container_name: str, master_port: int, gate: str,
    acl_probe_path: Path | None = None,
) -> list[str]:
    selected_nodes = [Path(f"/dev/davinci{device}") for device in device_ids]
    command = [
        "docker", "run", "--rm", "--name", container_name,
        "--stop-timeout=5", "--network=host", "--ipc=host",
        f"--shm-size={config['shm_size']}",
        "-e", f"ASCEND_RT_VISIBLE_DEVICES={','.join(map(str, device_ids))}",
        "-e", "GEMS_VENDOR=ascend",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
    ]
    for name, value in sorted(config.get("runtime_environment", {}).items()):
        command.extend(["-e", f"{name}={value}"])
    command.append("--privileged")
    for device in [*config["required_devices"], *map(str, selected_nodes)]:
        command.append(f"--device={device}:{device}:rwm")
    for mount in config["host_mounts"]:
        path = Path(mount)
        if path.exists():
            command.extend(["-v", f"{path}:{path}:ro"])
    if gate == "c3-acl-event-ab":
        if acl_probe_path is None:
            raise ConfigurationError("c3-acl-event-ab requires a probe snapshot")
        command.extend([
            "-v", f"{acl_probe_path}:/opt/flagrt/verify_acl_event_flags.py:ro",
        ])
        verifier_args = ["exec", "python3", "/opt/flagrt/verify_acl_event_flags.py"]
    elif gate == "c3-timeout-cleanup":
        # A deterministic stalled process exercises the host executor deadline,
        # exact-name container removal, lease release and postflight without
        # pretending to be a failed collective/peer diagnosis.
        verifier_args = ["exec", "sleep", "600"]
    else:
        verifier_args = [
            "exec", "torchrun", "--nproc-per-node=2", "--nnodes=1",
            "--node-rank=0", "--master-addr=127.0.0.1",
            f"--master-port={master_port}",
            "/opt/flagrt/verify_flagcx_p2p.py",
        ]
        if gate == "c3-nondefault-stream":
            verifier_args.extend(["--stream-mode", "nondefault"])
    inner = shlex.join(verifier_args)
    command.extend([config["image"], "/bin/bash", "-lc", inner])
    return command


def run_command(
    command: list[str], root: Path, label: str, *, timeout: int,
    container_name: str,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.perf_counter()
    timed_out = False
    timeout_cleanup = None
    try:
        proc = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stderr += f"\nFlagCX qualification timeout after {timeout}s\n"
        cleanup_started_at = utc_now()
        try:
            remove = subprocess.run(
                ["docker", "rm", "-f", container_name], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                timeout=30,
            )
            remove_returncode = remove.returncode
            remove_stdout = remove.stdout or ""
            remove_stderr = remove.stderr or ""
        except subprocess.TimeoutExpired as cleanup_exc:
            remove_returncode = 124
            remove_stdout = cleanup_exc.stdout or ""
            remove_stderr = cleanup_exc.stderr or ""
            if isinstance(remove_stdout, bytes):
                remove_stdout = remove_stdout.decode(errors="replace")
            if isinstance(remove_stderr, bytes):
                remove_stderr = remove_stderr.decode(errors="replace")
        inspect = subprocess.run(
            ["docker", "container", "inspect", container_name], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=30,
        )
        cleanup_stdout_path = root / f"{label}.cleanup.stdout"
        cleanup_stderr_path = root / f"{label}.cleanup.stderr"
        cleanup_stdout_path.write_text(
            remove_stdout + (inspect.stdout or ""), encoding="utf-8",
            errors="replace",
        )
        cleanup_stderr_path.write_text(
            remove_stderr + (inspect.stderr or ""), encoding="utf-8",
            errors="replace",
        )
        timeout_cleanup = {
            "container_name": container_name,
            "started_at": cleanup_started_at,
            "finished_at": utc_now(),
            "remove_command": ["docker", "rm", "-f", container_name],
            "remove_returncode": remove_returncode,
            "inspect_command": ["docker", "container", "inspect", container_name],
            "inspect_returncode": inspect.returncode,
            "stdout": artifact_ref(cleanup_stdout_path, root),
            "stderr": artifact_ref(cleanup_stderr_path, root),
        }
        returncode = 124
    stdout_path = root / f"{label}.stdout"
    stderr_path = root / f"{label}.stderr"
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    record = {
        "command": command,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_s": round(time.perf_counter() - started, 4),
        "timeout_s": timeout,
        "timed_out": timed_out,
        "returncode": returncode,
        "stdout": artifact_ref(stdout_path, root),
        "stderr": artifact_ref(stderr_path, root),
    }
    if timeout_cleanup is not None:
        record["timeout_cleanup"] = timeout_cleanup
    write_json(root / f"{label}.command.json", record)
    return record | {"stdout_text": stdout, "stderr_text": stderr}


def render_report(summary: dict[str, Any], result_dir: Path) -> None:
    selection = summary.get("selection", {})
    runtime = summary.get("runtime", {})
    lines = [
        "# Ascend FlagCX 候选资格验证",
        "",
        f"- 状态：`{summary.get('status')}`",
        f"- Gate：`{summary.get('gate')}`",
        f"- 逻辑 Device：`{selection.get('selected_device_ids')}`",
        f"- 物理 NPU：`{selection.get('selected_npu_ids')}`",
        f"- 镜像：`{runtime.get('image')}`",
        f"- 镜像 ID：`{runtime.get('image_id')}`",
        "",
        "## 重复执行",
        "",
        "| 序号 | rc | 语义状态 | 时长(s) | inner backend | 原始证据 |",
        "|---:|---:|---|---:|---|---|",
    ]
    for item in summary.get("attempts", []):
        records = item.get("semantic_records", item.get("rank_records", []))
        inner = ", ".join(sorted({
            str(value.get("inner_backend") or value.get("kind"))
            for value in records
        }))
        evidence = item.get("command", {})
        stdout = evidence.get("stdout", {}).get("path", "")
        stderr = evidence.get("stderr", {}).get("path", "")
        links = f"[stdout]({stdout}) / [stderr]({stderr})"
        lines.append(
            f"| {item.get('attempt')} | {evidence.get('returncode')} | "
            f"{item.get('status')} | {evidence.get('duration_s')} | "
            f"{inner} | {links} |"
        )
    lines.extend([
        "",
        "## 证据边界",
        "",
        "sentinel gate 只证明所选双 Device、当前镜像和当前小消息集合的通信正确性；"
        "timeout-cleanup gate 只证明 executor deadline、精确容器清理、租约释放与 postflight。"
        "ACL event A/B gate 只证明所选本地 Device 上两个 event flag 的直接 API 行为。"
        "它们都不是性能基线，也不代表其他拓扑、多节点、peer failure 或长稳能力。",
        "preflight、postflight、命令、stdout/stderr 和 SHA-256 均保存在本目录。",
        "",
    ])
    report = result_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    summary["report_generation"] = {
        "status": "passed",
        "path": report.relative_to(result_dir).as_posix(),
        "sha256": sha256_file(report),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device-ids", required=True)
    parser.add_argument(
        "--gate", choices=(
            "c2-sentinel", "c3-nondefault-stream", "c3-timeout-cleanup",
            "c3-acl-event-ab",
        ),
        default="c2-sentinel",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--master-port", type=int, default=29821)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--allow-privileged-root", action="store_true")
    parser.add_argument("--allow-candidate-runtime", action="store_true")
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> int:
    if not args.allow_privileged_root:
        raise ConfigurationError("qualification requires --allow-privileged-root")
    if not args.allow_candidate_runtime:
        raise ConfigurationError("qualification requires --allow-candidate-runtime")
    if args.repeats < 1 or args.repeats > 10:
        raise ConfigurationError("--repeats must be between 1 and 10")
    if args.gate in {"c3-timeout-cleanup", "c3-acl-event-ab"} and args.repeats != 1:
        raise ConfigurationError(f"{args.gate} requires exactly one repeat")
    if args.timeout < 10 or args.timeout > 600:
        raise ConfigurationError("--timeout must be between 10 and 600 seconds")
    if not 1 <= args.master_port <= 65525 - args.repeats:
        raise ConfigurationError("--master-port cannot allocate the requested repeats")
    parsed = parse_id_spec(args.device_ids, "logical Device IDs")
    if len(parsed) != 2:
        raise ConfigurationError("FlagCX qualification requires exactly two Devices")
    device_ids = (parsed[0], parsed[1])

    config_path, config = load_host_config(args.config)
    if config.get("runtime_role") != "communication":
        raise ConfigurationError("FlagCX qualification requires communication runtime")
    if not set(device_ids).issubset(set(config["expected_device_ids"])):
        raise ConfigurationError("selected Devices are outside the configured inventory")

    run_id = "flagcx-qualification-" + run_timestamp()
    if not SAFE_RUN_ID_RE.fullmatch(run_id):
        raise ConfigurationError("generated qualification run ID is unsafe")
    configured_root = Path(config["result_root"])
    result_root = (
        args.result_root.expanduser().resolve()
        if args.result_root is not None
        else (BASE_DIR / configured_root).resolve()
    )
    result_dir = result_root / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    wall_started = time.perf_counter()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "flagcx-qualification",
        "run_id": run_id,
        "gate": args.gate,
        "started_at": utc_now(),
        "status": "running",
        "host_config": str(config_path),
        "authorization": {
            "candidate_runtime": True,
            "privileged_root": True,
            "requested_device_ids": list(device_ids),
        },
        "attempts": [],
    }
    write_json(result_dir / "summary.json", summary)
    lease = None
    failure: Exception | None = None
    try:
        image_info = docker_inspect(config["image"])
        runtime_lock = validate_runtime_identity(
            config, image_info, allow_candidate=True,
        )
        summary["runtime"] = {
            "image": config["image"],
            "image_id": image_info["Id"],
            "image_labels": image_info.get("Config", {}).get("Labels") or {},
            "lock": runtime_lock,
        }
        preflight_path = run_host_preflight(
            result_dir, config["expected_device_ids"],
            device_ids=",".join(map(str, device_ids)),
        )
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        selection = preflight.get("selection", {})
        if selection.get("selected_device_ids") != list(device_ids):
            raise RuntimeError(f"preflight selection drifted: {selection}")
        summary["selection"] = selection
        summary["host_preflight"] = {
            "path": preflight_path.relative_to(result_dir).as_posix(),
            "sha256": sha256_file(preflight_path),
        }
        lease = DeviceLease(device_ids, run_id=run_id, kind="flagcx-qualification")
        lease.acquire()
        summary["device_lease"] = lease.record()
        acl_probe_path = None
        if args.gate == "c3-acl-event-ab":
            acl_probe_path = result_dir / "verify_acl_event_flags.py"
            shutil.copyfile(PROFILE_DIR / acl_probe_path.name, acl_probe_path)
            summary["probe_source"] = artifact_ref(acl_probe_path, result_dir)
        for attempt in range(1, args.repeats + 1):
            gate_token = args.gate.replace("-", "")
            name = f"flagcx-{gate_token}-{run_timestamp().lower()}-{attempt}"
            command = build_container_command(
                config, device_ids, container_name=name,
                master_port=args.master_port + attempt - 1, gate=args.gate,
                acl_probe_path=acl_probe_path,
            )
            command_record = run_command(
                command, result_dir, f"attempt-{attempt}", timeout=args.timeout,
                container_name=name,
            )
            rank_records = parse_rank_record_lines(command_record["stdout_text"])
            semantic_records = parse_json_objects(command_record["stdout_text"])
            attempt_status = "passed"
            try:
                if args.gate == "c3-timeout-cleanup":
                    validate_expected_timeout_record(command_record)
                elif args.gate == "c3-acl-event-ab":
                    if command_record["returncode"] != 0:
                        raise RuntimeError(
                            "ACL event probe container returned "
                            f"{command_record['returncode']}"
                        )
                    validate_acl_event_records(semantic_records)
                else:
                    if command_record["returncode"] != 0:
                        raise RuntimeError(
                            "sentinel container returned "
                            f"{command_record['returncode']}"
                        )
                    expected_stream = (
                        "nondefault"
                        if args.gate == "c3-nondefault-stream" else "default"
                    )
                    validate_sentinel_records(
                        rank_records, expected_stream_mode=expected_stream,
                    )
            except Exception:
                attempt_status = "failed"
                raise
            finally:
                summary["attempts"].append({
                    "attempt": attempt,
                    "status": attempt_status,
                    "rank_records": rank_records,
                    "semantic_records": semantic_records,
                    "command": {
                        key: value for key, value in command_record.items()
                        if not key.endswith("_text")
                    },
                })
                write_json(result_dir / "summary.json", summary)
    except Exception as exc:
        failure = exc
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if lease is not None:
            lease.release()
            summary["device_lease"]["released_at"] = utc_now()
        try:
            postflight_path = run_host_preflight(
                result_dir, config["expected_device_ids"],
                device_ids=",".join(map(str, device_ids)), label="host-postflight",
            )
            summary["host_postflight"] = {
                "path": postflight_path.relative_to(result_dir).as_posix(),
                "sha256": sha256_file(postflight_path),
                "status": "passed",
            }
        except Exception as exc:
            summary["host_postflight"] = {
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            }
            failure = failure or exc

    summary["status"] = "passed" if failure is None else "failed"
    summary["finished_at"] = utc_now()
    summary["wall_clock_duration_s"] = round(time.perf_counter() - wall_started, 4)
    render_report(summary, result_dir)
    write_json(result_dir / "summary.json", summary)
    print(f"Result directory: {result_dir}")
    if failure is not None:
        print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return execute(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
