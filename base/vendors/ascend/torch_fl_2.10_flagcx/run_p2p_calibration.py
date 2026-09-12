#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Plan or execute the repository-pinned FlagCX P2P calibration matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


PROFILE_DIR = Path(__file__).resolve().parent
BASE_DIR = PROFILE_DIR.parents[2]
REPOSITORY_ROOT = BASE_DIR.parent
DEFAULT_PLAN = (
    BASE_DIR / "benchmarks" / "interconnect-P2P_intraserver" /
    "ascend" / "p2p-calibration-plan.json"
)
RUN_PATH = BASE_DIR / "run.py"
RESULT_LINE_RE = re.compile(r"^Result directory: (?P<path>.+)$", re.MULTILINE)
SAFE_PAIR_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")


class CalibrationError(RuntimeError):
    """Raised when a P2P calibration plan or child run is not trustworthy."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def repository_path(value: str) -> Path:
    path = (REPOSITORY_ROOT / value).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise CalibrationError(f"plan path escapes the repository: {value!r}")
    return path


def load_and_validate_plan(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        plan = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot load calibration plan {resolved}: {exc}") from exc
    if plan.get("schema_version") != 1:
        raise CalibrationError("unsupported P2P calibration plan schema")
    if plan.get("stage") != "p2p-calibration-only":
        raise CalibrationError("plan is not explicitly calibration-only")
    if plan.get("formal_baseline_eligible") is not False:
        raise CalibrationError("calibration plan must forbid formal baseline claims")

    contract = plan.get("measurement_contract")
    if not isinstance(contract, dict):
        raise CalibrationError("measurement_contract is missing")
    if contract.get("monitor") != "off" or contract.get("repeats") != 1:
        raise CalibrationError("calibration must use monitor=off and one repeat")
    expected_volume = contract.get("one_way_payload_bytes_per_point")
    if not isinstance(expected_volume, int) or expected_volume <= 0:
        raise CalibrationError("invalid calibration payload volume")

    pairs = plan.get("representative_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise CalibrationError("representative_pairs must not be empty")
    pair_ids: set[str] = set()
    paths: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict) or not SAFE_PAIR_ID_RE.fullmatch(
            str(pair.get("id", ""))
        ):
            raise CalibrationError(f"invalid pair record: {pair!r}")
        pair_id = pair["id"]
        if pair_id in pair_ids:
            raise CalibrationError(f"duplicate pair id: {pair_id}")
        pair_ids.add(pair_id)
        devices = pair.get("device_ids")
        if (
            not isinstance(devices, list) or len(devices) != 2
            or any(not isinstance(item, int) or item < 0 for item in devices)
            or devices[0] == devices[1]
        ):
            raise CalibrationError(f"invalid Device pair: {devices!r}")
        expected_path = pair.get("expected_path")
        if expected_path not in {"SIO", "HCCS", "HCCS_SW"}:
            raise CalibrationError(f"unsupported topology path: {expected_path!r}")
        paths.add(expected_path)
    if not {"SIO", "HCCS_SW"}.issubset(paths):
        raise CalibrationError("plan must cover observed SIO and HCCS_SW paths")

    sizes = plan.get("size_points")
    if not isinstance(sizes, list) or len(sizes) < 3:
        raise CalibrationError("size curve requires at least three points")
    observed_sizes: list[int] = []
    for point in sizes:
        if not isinstance(point, dict):
            raise CalibrationError(f"invalid size point: {point!r}")
        message_bytes = point.get("message_bytes")
        melements = point.get("melements")
        iterations = point.get("iterations")
        if not all(isinstance(value, int) and value > 0 for value in (
            message_bytes, melements, iterations,
        )):
            raise CalibrationError(f"invalid size point values: {point!r}")
        if message_bytes != melements * 1024 * 1024 * 4:
            raise CalibrationError("message size drifted from the original FP32 Case")
        if message_bytes * iterations != expected_volume:
            raise CalibrationError("calibration points must transfer equal payload volume")
        config = repository_path(str(point.get("config", "")))
        if not config.is_file():
            raise CalibrationError(f"calibration config is missing: {config}")
        if sha256_file(config) != point.get("sha256"):
            raise CalibrationError(f"calibration config hash drifted: {config}")
        observed_sizes.append(message_bytes)
    if observed_sizes != sorted(set(observed_sizes)):
        raise CalibrationError("size points must be unique and ascending")
    return {**plan, "_path": str(resolved), "_sha256": sha256_file(resolved)}


def build_matrix(
    plan: dict[str, Any], pair_ids: set[str] | None, *,
    master_port: int, timeout: int,
) -> list[dict[str, Any]]:
    pairs = [
        pair for pair in plan["representative_pairs"]
        if pair_ids is None or pair["id"] in pair_ids
    ]
    if pair_ids is not None:
        missing = pair_ids - {pair["id"] for pair in pairs}
        if missing:
            raise CalibrationError(f"unknown pair ids: {sorted(missing)}")
    matrix: list[dict[str, Any]] = []
    for pair in pairs:
        for point in plan["size_points"]:
            port = master_port + len(matrix)
            if port > 65535:
                raise CalibrationError("master-port range exceeds 65535")
            devices = ",".join(str(item) for item in pair["device_ids"])
            command = [
                sys.executable, str(RUN_PATH), "benchmark", "run",
                "--config", str(repository_path(plan["runtime_config"])),
                "--case", plan["case"],
                "--device-ids", devices,
                "--nproc-per-node", "2",
                "--case-config", str(repository_path(point["config"])),
                "--monitor", "off",
                "--master-port", str(port),
                "--timeout", str(timeout),
                "--allow-privileged-root",
            ]
            matrix.append({
                "pair_id": pair["id"],
                "expected_path": pair["expected_path"],
                "device_ids": pair["device_ids"],
                "message_bytes": point["message_bytes"],
                "config_sha256": point["sha256"],
                "command": command,
            })
    return matrix


def topology_path(text: str, source: int, destination: int) -> str:
    for line in text.splitlines():
        match = re.match(r"^Phy-ID(\d+)\s+(.*)$", line.strip())
        if not match or int(match.group(1)) != source:
            continue
        relations = match.group(2).split()
        if destination >= len(relations):
            break
        return relations[destination]
    raise CalibrationError(
        f"topology matrix does not contain Phy-ID{source} -> Phy-ID{destination}"
    )


def validate_child_result(
    plan: dict[str, Any], item: dict[str, Any], child_root: Path,
) -> dict[str, Any]:
    summary_path = child_root / "summary.json"
    result_path = child_root / "benchmark-result.json"
    if not summary_path.is_file() or not result_path.is_file():
        raise CalibrationError(f"child evidence is incomplete: {child_root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    required_statuses = {
        "status": "passed", "execution_status": "passed",
        "measurement_status": "passed", "monitoring_status": "not-run",
    }
    for field, expected in required_statuses.items():
        if summary.get(field) != expected:
            raise CalibrationError(
                f"child {field} is {summary.get(field)!r}, expected {expected!r}"
            )
    if summary.get("host_postflight", {}).get("result", {}).get("status") != "passed":
        raise CalibrationError("child postflight did not pass")
    if summary.get("selection", {}).get("selected_device_ids") != item["device_ids"]:
        raise CalibrationError("child Device selection drifted")
    override = summary.get("case_config", {}).get("override", {})
    if override.get("sha256") != item["config_sha256"]:
        raise CalibrationError("child Case configuration hash drifted")
    if result.get("status") != "passed" or result.get("fallback_count") != 0:
        raise CalibrationError("child semantic result or fallback gate failed")
    ranks = sorted({
        metric.get("rank") for metric in result.get("metrics", [])
        if metric.get("unit") == "GB/s"
    })
    rates = [
        float(metric["value"]) for metric in result.get("metrics", [])
        if metric.get("unit") == "GB/s"
    ]
    if ranks != [0, 1] or len(rates) != 2 or any(rate <= 0 for rate in rates):
        raise CalibrationError("child does not contain two positive rank GB/s values")
    topology_files = list(
        (child_root / "host-preflight").glob("*/npu-smi-topology.stdout")
    )
    if len(topology_files) != 1:
        raise CalibrationError("child topology evidence is missing or ambiguous")
    device_map = summary.get("host_preflight", {}).get("result", {}).get(
        "actual_device_map", []
    )
    logic_to_phy = {
        entry.get("logic_id"): entry.get("phy_id")
        for entry in device_map if isinstance(entry, dict)
    }
    try:
        phy_ids = [logic_to_phy[device] for device in item["device_ids"]]
    except KeyError as exc:
        raise CalibrationError("child Device-to-Phy-ID mapping is incomplete") from exc
    actual_path = topology_path(
        topology_files[0].read_text(encoding="utf-8"), *phy_ids,
    )
    if actual_path != item["expected_path"]:
        raise CalibrationError(
            f"child topology is {actual_path}, expected {item['expected_path']}"
        )
    point = next(
        point for point in plan["size_points"]
        if point["message_bytes"] == item["message_bytes"]
    )
    payload = plan["measurement_contract"]["one_way_payload_bytes_per_point"]
    elapsed = [2 * payload / (rate * 1_000_000_000) for rate in rates]
    target = plan["formal_followup"]["target_measurement_seconds_per_run"]
    recommended = math.ceil(point["iterations"] * target / min(elapsed))
    recommended = math.ceil(recommended / 1000) * 1000
    return {
        "runtime_image_id": summary.get("runtime", {}).get("image_id"),
        "phy_ids": phy_ids,
        "topology_path": actual_path,
        "rank_gb_s": rates,
        "estimated_rank_elapsed_s": [round(value, 6) for value in elapsed],
        "recommended_formal_iterations": recommended,
        "formal_target_measurement_s": target,
        "derivation": (
            "ceil(calibration_iterations * target_seconds / minimum_derived_"
            "rank_elapsed_seconds), rounded up to 1000 iterations"
        ),
        "child_summary_sha256": sha256_file(summary_path),
        "benchmark_result_sha256": sha256_file(result_path),
        "topology_sha256": sha256_file(topology_files[0]),
    }


def audit_session(session_dir: Path) -> int:
    root = session_dir.expanduser().resolve()
    summary_path = root / "summary.json"
    snapshot = root / "p2p-calibration-plan.json"
    if not summary_path.is_file() or not snapshot.is_file():
        raise CalibrationError(f"not a P2P calibration session: {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    plan = load_and_validate_plan(snapshot)
    if summary.get("plan", {}).get("sha256") != sha256_file(snapshot):
        raise CalibrationError("session plan snapshot hash drifted")
    if summary.get("status") != "passed":
        raise CalibrationError("only a completed passed session can be audited")
    for item in summary.get("runs", []):
        child = item.get("child_result")
        if not isinstance(child, str):
            raise CalibrationError("session child_result is missing")
        item["calibration_audit"] = validate_child_result(
            plan, item, Path(child).resolve(),
        )
    summary["audit"] = {
        "status": "passed", "audited_at": utc_now(),
        "run_count": len(summary.get("runs", [])),
        "formal_baseline_eligible": False,
    }
    write_json(summary_path, summary)
    print(f"Audited calibration session: {root}")
    return 0


def execute_matrix(
    plan: dict[str, Any], matrix: list[dict[str, Any]], result_root: Path,
) -> int:
    session_dir = result_root.expanduser().resolve() / (
        "flagcx-p2p-calibration-" + run_timestamp()
    )
    session_dir.mkdir(parents=True, exist_ok=False)
    source_plan = Path(plan["_path"])
    snapshot = session_dir / "p2p-calibration-plan.json"
    shutil.copyfile(source_plan, snapshot)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "flagcx-p2p-calibration",
        "stage": "p2p-calibration-only",
        "formal_baseline_eligible": False,
        "status": "running",
        "plan": {"path": snapshot.name, "sha256": sha256_file(snapshot)},
        "runs": [],
    }
    write_json(session_dir / "summary.json", summary)
    for index, item in enumerate(matrix, start=1):
        proc = subprocess.run(
            item["command"], cwd=REPOSITORY_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        output = proc.stdout or ""
        log_path = session_dir / f"run-{index:02d}.log"
        log_path.write_text(output, encoding="utf-8")
        match = RESULT_LINE_RE.search(output)
        child_path = Path(match.group("path")).resolve() if match else None
        child_summary = child_path / "summary.json" if child_path else None
        record = {
            **{key: value for key, value in item.items() if key != "command"},
            "command": item["command"],
            "returncode": proc.returncode,
            "log": {"path": log_path.name, "sha256": sha256_file(log_path)},
            "child_result": str(child_path) if child_path else None,
            "child_summary_sha256": (
                sha256_file(child_summary)
                if child_summary is not None and child_summary.is_file() else None
            ),
        }
        summary["runs"].append(record)
        write_json(session_dir / "summary.json", summary)
        if proc.returncode != 0 or child_summary is None or not child_summary.is_file():
            summary["status"] = "failed"
            summary["error"] = "child Benchmark failed or did not expose its summary"
            write_json(session_dir / "summary.json", summary)
            print(f"Calibration session: {session_dir}")
            return 1
        try:
            record["calibration_audit"] = validate_child_result(
                plan, record, child_path,
            )
        except CalibrationError as exc:
            summary["status"] = "failed"
            summary["error"] = f"child calibration audit failed: {exc}"
            write_json(session_dir / "summary.json", summary)
            print(f"Calibration session: {session_dir}")
            return 1
        write_json(session_dir / "summary.json", summary)
    summary["status"] = "passed"
    summary["claim_boundary"] = plan["claim_boundary"]
    write_json(session_dir / "summary.json", summary)
    print(f"Calibration session: {session_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--pair-id", action="append", dest="pair_ids")
    parser.add_argument("--master-port", type=int, default=29921)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--result-root", type=Path, default=BASE_DIR / "result")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--audit-session", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.audit_session is not None:
            if args.execute or args.pair_ids:
                raise CalibrationError(
                    "--audit-session cannot be combined with execution selection"
                )
            return audit_session(args.audit_session)
        if not 1 <= args.master_port <= 65535:
            raise CalibrationError("--master-port must be between 1 and 65535")
        if args.timeout < 60 or args.timeout > 1800:
            raise CalibrationError("--timeout must be between 60 and 1800 seconds")
        plan = load_and_validate_plan(args.plan)
        matrix = build_matrix(
            plan, set(args.pair_ids) if args.pair_ids else None,
            master_port=args.master_port, timeout=args.timeout,
        )
        if not args.execute:
            print(json.dumps({
                "schema_version": 1,
                "mode": "plan-only",
                "formal_baseline_eligible": False,
                "plan_sha256": plan["_sha256"],
                "run_count": len(matrix),
                "matrix": matrix,
            }, indent=2, sort_keys=True))
            return 0
        return execute_matrix(plan, matrix, args.result_root)
    except (CalibrationError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
