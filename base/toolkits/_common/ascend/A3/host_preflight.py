#!/usr/bin/env python3
"""Fail-closed Ascend host inventory and occupancy preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any


DEVICE_NODE_RE = re.compile(r"davinci(\d+)$")
MAP_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s*$"
)


class PreflightError(RuntimeError):
    """The host cannot prove the requested Ascend devices are idle and present."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def command_record(
    root: Path, command: list[str], label: str, timeout: int = 120
) -> dict[str, Any]:
    stdout_path = root / f"{label}.stdout"
    stderr_path = root / f"{label}.stderr"
    started_at = utc_now()
    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout,
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
        stderr += f"\nFlagPerf host preflight timeout after {timeout}s\n"
        returncode = 124
        timed_out = True
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
    write_json(root / f"{label}.command.json", record)
    return record | {"stdout_text": stdout, "stderr_text": stderr}


def parse_expected_device_ids(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise PreflightError("expected device IDs must be comma-separated integers") from exc
    if not devices or len(devices) != len(set(devices)) or any(item < 0 for item in devices):
        raise PreflightError("expected device IDs must be a non-empty unique integer list")
    return sorted(devices)


def parse_id_spec(value: str, label: str = "device IDs") -> list[int]:
    """Parse comma-separated IDs and inclusive ranges without hiding mistakes."""
    if not isinstance(value, str) or not value.strip():
        raise PreflightError(f"{label} must be a non-empty ID/range expression")
    resolved: list[int] = []
    seen: set[int] = set()
    for raw_token in value.split(","):
        token = raw_token.strip()
        if re.fullmatch(r"\d+", token):
            values = [int(token)]
        elif match := re.fullmatch(r"(\d+)-(\d+)", token):
            start, end = map(int, match.groups())
            if start > end:
                raise PreflightError(f"{label} range is descending: {token}")
            values = list(range(start, end + 1))
        else:
            raise PreflightError(f"{label} contains an invalid token: {token or '<empty>'}")
        duplicate = next((item for item in values if item in seen), None)
        if duplicate is not None:
            raise PreflightError(f"{label} contains duplicate ID {duplicate}")
        resolved.extend(values)
        seen.update(values)
    return sorted(resolved)


def discover_device_ids(device_root: Path = Path("/dev")) -> list[int]:
    devices = []
    for path in device_root.glob("davinci[0-9]*"):
        match = DEVICE_NODE_RE.fullmatch(path.name)
        if match:
            devices.append(int(match.group(1)))
    return sorted(set(devices))


def parse_device_map(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        match = MAP_ROW_RE.match(line)
        if not match:
            continue
        npu_id, chip_id, logic_id, phy_id = map(int, match.groups()[:4])
        rows.append({
            "npu_id": npu_id,
            "chip_id": chip_id,
            "logic_id": logic_id,
            "phy_id": phy_id,
            "chip_name": match.group(5),
        })
    return sorted(rows, key=lambda item: item["logic_id"])


def parse_map_logic_ids(text: str) -> list[int]:
    return [item["logic_id"] for item in parse_device_map(text)]


def resolve_selection(
    expected: list[int], device_map: list[dict[str, Any]],
    *, requested_npu_ids: list[int] | None = None,
    requested_device_ids: list[int] | None = None,
) -> dict[str, Any]:
    if requested_npu_ids is not None and requested_device_ids is not None:
        raise PreflightError("NPU IDs and logical Device IDs are mutually exclusive")
    expected_set = set(expected)
    mapped_devices = {item["logic_id"] for item in device_map}
    available_npus = {item["npu_id"] for item in device_map}
    if requested_npu_ids is not None:
        missing_npus = sorted(set(requested_npu_ids) - available_npus)
        if missing_npus:
            raise PreflightError(f"requested physical NPU IDs are absent: {missing_npus}")
        selected = sorted(
            item["logic_id"] for item in device_map
            if item["npu_id"] in set(requested_npu_ids)
        )
        source = "npu-ids"
        requested = requested_npu_ids
    elif requested_device_ids is not None:
        missing_devices = sorted(set(requested_device_ids) - expected_set)
        if missing_devices:
            raise PreflightError(f"requested logical Device IDs are absent: {missing_devices}")
        selected = sorted(requested_device_ids)
        source = "device-ids"
        requested = requested_device_ids
    else:
        selected = list(expected)
        source = "default-all"
        requested = list(expected)
    if not selected or not set(selected).issubset(expected_set & mapped_devices):
        raise PreflightError(
            f"resolved logical Device IDs are not a non-empty inventory subset: {selected}"
        )
    participating_npus = sorted({
        item["npu_id"] for item in device_map if item["logic_id"] in set(selected)
    })
    return {
        "source": source,
        "requested_ids": requested,
        "selected_npu_ids": participating_npus,
        "selected_device_ids": selected,
        "excluded_device_ids": sorted(expected_set - set(selected)),
    }


def validate_inventory(
    expected: list[int], device_nodes: list[int], map_logic_ids: list[int]
) -> None:
    if device_nodes != expected:
        raise PreflightError(
            f"Ascend device nodes do not match expected inventory: "
            f"expected={expected}, actual={device_nodes}"
        )
    if map_logic_ids != expected:
        raise PreflightError(
            f"npu-smi map does not match expected logical devices: "
            f"expected={expected}, actual={map_logic_ids}"
        )


def run_preflight(
    output: Path, expected: list[int], *,
    requested_npu_ids: list[int] | None = None,
    requested_device_ids: list[int] | None = None,
) -> dict[str, Any]:
    hostname = socket.gethostname()
    root = output.resolve() / hostname
    root.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "host": hostname,
        "execution_context": "host",
        "started_at": utc_now(),
        "status": "running",
        "expected_device_ids": expected,
        "actual_device_node_ids": discover_device_ids(),
        "commands": {},
    }
    write_json(root / "summary.json", summary)
    try:
        command_stdout: dict[str, str] = {}
        for label, command in {
            "npu-smi-list": ["npu-smi", "info", "-l"],
            "npu-smi-map": ["npu-smi", "info", "-m"],
            "npu-smi-topology": ["npu-smi", "info", "-t", "topo"],
        }.items():
            record = command_record(root, command, label)
            summary["commands"][label] = {
                key: value for key, value in record.items()
                if not key.endswith("_text")
            }
            command_stdout[label] = record["stdout_text"]
            if record["returncode"] != 0:
                raise PreflightError(
                    f"host inventory probe failed ({record['returncode']}): {' '.join(command)}"
                )

        device_map = parse_device_map(command_stdout["npu-smi-map"])
        summary["actual_device_map"] = device_map
        summary["actual_map_logic_ids"] = [item["logic_id"] for item in device_map]
        validate_inventory(
            expected,
            summary["actual_device_node_ids"],
            summary["actual_map_logic_ids"],
        )
        selection = resolve_selection(
            expected, device_map,
            requested_npu_ids=requested_npu_ids,
            requested_device_ids=requested_device_ids,
        )
        summary["selection"] = selection

        fuser = shutil.which("fuser")
        if not fuser:
            raise PreflightError("fuser is unavailable; device occupancy is unknown")
        selected_devices = selection["selected_device_ids"]
        device_paths = [f"/dev/davinci{device}" for device in selected_devices]
        occupancy = command_record(root, [fuser, *device_paths], "occupancy")
        summary["commands"]["occupancy"] = {
            key: value for key, value in occupancy.items()
            if not key.endswith("_text")
        }
        occupied_text = (occupancy["stdout_text"] + occupancy["stderr_text"]).strip()
        if occupancy["returncode"] == 0 and occupied_text:
            raise PreflightError(f"Ascend devices are in use: {occupied_text}")
        if occupancy["returncode"] != 1:
            raise PreflightError(
                f"could not determine Ascend device occupancy: rc={occupancy['returncode']}"
            )
        summary["occupancy"] = {
            "status": "idle", "checked_device_ids": selected_devices,
        }
        summary["status"] = "passed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        summary["finished_at"] = utc_now()
        write_json(root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-device-ids", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--npu-ids")
    selection.add_argument("--device-ids")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = parse_expected_device_ids(args.expected_device_ids)
    requested_npus = parse_id_spec(args.npu_ids, "NPU IDs") if args.npu_ids else None
    requested_devices = (
        parse_id_spec(args.device_ids, "logical Device IDs") if args.device_ids else None
    )
    summary = run_preflight(
        args.output, expected,
        requested_npu_ids=requested_npus,
        requested_device_ids=requested_devices,
    )
    print(json.dumps({"status": summary["status"], "host": summary["host"]}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
