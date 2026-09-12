#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Plan or execute and audit the pinned single-node FlagCX P2P matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any


PROFILE_DIR = Path(__file__).resolve().parent
if str(PROFILE_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILE_DIR))
import run_p2p_calibration as common  # noqa: E402


BASE_DIR = common.BASE_DIR
REPOSITORY_ROOT = common.REPOSITORY_ROOT
RUN_PATH = common.RUN_PATH
DEFAULT_PLAN = (
    BASE_DIR / "benchmarks" / "interconnect-P2P_intraserver" /
    "ascend" / "p2p-qualification-plan.json"
)
PROTOCOLS = {
    "p2p-single-node-v1": {
        "stage": "p2p-single-node-qualification",
        "plan_status": "qualified",
        "minimum_measurement_seconds_per_run": 45,
        "target_measurement_seconds_per_run": 60,
        "session_prefix": "flagcx-p2p-qualification-",
        "summary_kind": "flagcx-p2p-qualification",
        "production_eligible": True,
    },
}


def yaml_integer(text: str, name: str) -> int:
    match = re.search(rf"(?m)^\s*{re.escape(name)}:\s*(\d+)\s*$", text)
    if not match:
        raise common.CalibrationError(f"formal config has no integer {name}")
    return int(match.group(1))


def load_and_validate_plan(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        plan = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise common.CalibrationError(f"cannot load formal plan {resolved}: {exc}") from exc
    protocol_id = plan.get("protocol_id")
    protocol = PROTOCOLS.get(protocol_id)
    if protocol is None:
        raise common.CalibrationError(
            f"unsupported P2P qualification protocol: {protocol_id!r}"
        )
    if (
        plan.get("schema_version") != 1
        or plan.get("stage") != protocol["stage"]
        or plan.get("case") != "interconnect-P2P_intraserver"
        or plan.get("runtime_profile") != PROFILE_DIR.name
    ):
        raise common.CalibrationError("unsupported P2P qualification plan")
    if plan.get("status") != protocol["plan_status"]:
        raise common.CalibrationError("source formal plan is not qualified")
    qualification = plan.get("qualification_evidence", {})
    qualification_path = common.repository_path(
        str(qualification.get("path", ""))
    )
    try:
        qualification_attestation = json.loads(
            qualification_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise common.CalibrationError(
            f"formal qualification evidence is unavailable: {exc}"
        ) from exc
    if (
        common.sha256_file(qualification_path) != qualification.get("sha256")
        or qualification_attestation.get("schema_version") != 1
        or qualification_attestation.get("status") != "passed"
        or qualification_attestation.get("protocol_id") != protocol_id
        or qualification_attestation.get("evidence", {}).get("run_count") != 30
        or qualification_attestation.get("decision", {}).get(
            "formal_execution_eligible"
        ) is not True
        or qualification_attestation.get("decision", {}).get(
            "production_eligible_for_declared_scope"
        ) is not True
    ):
        raise common.CalibrationError("formal qualification evidence is invalid")
    raw_evidence = qualification_attestation.get("evidence", {})
    raw_path = common.repository_path(str(raw_evidence.get("path", "")))
    if (
        raw_path.is_file()
        and common.sha256_file(raw_path) != raw_evidence.get("sha256")
    ):
        raise common.CalibrationError("local raw formal evidence hash drifted")
    calibration = plan.get("calibration_evidence", {})
    calibration_path = common.repository_path(str(calibration.get("path", "")))
    if (
        not calibration_path.is_file()
        or common.sha256_file(calibration_path) != calibration.get("sha256")
        or calibration.get("status") != "passed"
        or calibration.get("formal_baseline_eligible") is not False
    ):
        raise common.CalibrationError("formal plan calibration evidence is invalid")
    calibration_record = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        calibration_record.get("schema_version") != 1
        or calibration_record.get("record_type") != "maintainer-attestation"
        or calibration_record.get("status") != "passed"
        or calibration_record.get("formal_baseline_eligible") is not False
        or calibration_record.get("runtime_image_id")
        != qualification_attestation.get("runtime_image_id")
    ):
        raise common.CalibrationError("calibration provenance record is invalid")
    # Bind execution fields to the distributable qualification record. Renaming
    # the plan must not extend the measured scope or weaken its acceptance.
    published = qualification_attestation.get("published_contract", {})
    for field in (
        "measurement_contract", "matrix", "monitor_ab", "acceptance", "runtime_config",
    ):
        if plan.get(field) != published.get(field):
            raise common.CalibrationError(f"formal {field.replace('_', ' ')} drifted")
    if calibration.get("sha256") != published.get("calibration_record_sha256"):
        raise common.CalibrationError("calibration record differs from qualification")
    runtime_config_path = common.repository_path(plan["runtime_config"])
    if common.sha256_file(runtime_config_path) != published.get("runtime_config_sha256"):
        raise common.CalibrationError("qualified runtime configuration drifted")
    runtime = json.loads((PROFILE_DIR / "image-manifest.json").read_text(encoding="utf-8"))
    if (
        runtime["image_id"] != qualification_attestation["runtime_image_id"]
        or runtime["image"] != qualification_attestation["runtime_image"]
    ):
        raise common.CalibrationError("qualified runtime image identity drifted")
    contract = plan.get("measurement_contract", {})
    if (
        contract.get("minimum_measurement_seconds_per_run")
        != protocol["minimum_measurement_seconds_per_run"]
        or contract.get("target_measurement_seconds_per_run")
        != protocol["target_measurement_seconds_per_run"]
        or contract.get("size_curve_repeats") != 3
        or contract.get("size_curve_monitor") != "off"
    ):
        raise common.CalibrationError("formal measurement contract drifted")
    matrix = plan.get("matrix")
    if not isinstance(matrix, list) or len(matrix) < 6:
        raise common.CalibrationError("formal size matrix is incomplete")
    keys: set[tuple[str, int]] = set()
    paths: set[str] = set()
    for item in matrix:
        key = (str(item.get("pair_id")), item.get("message_bytes"))
        if key in keys:
            raise common.CalibrationError(f"duplicate formal matrix cell: {key}")
        keys.add(key)
        config = common.repository_path(str(item.get("config", "")))
        if not config.is_file() or common.sha256_file(config) != item.get("sha256"):
            raise common.CalibrationError(f"formal config hash drifted: {config}")
        text = config.read_text(encoding="utf-8")
        melements = yaml_integer(text, "Melements")
        if item.get("message_bytes") != melements * 1024 * 1024 * 4:
            raise common.CalibrationError("formal message size drifted from FP32 Case")
        if item.get("iterations") != yaml_integer(text, "ITERS"):
            raise common.CalibrationError("formal iteration count drifted")
        if yaml_integer(text, "WARMUP") != contract.get("warmup_iterations"):
            raise common.CalibrationError("formal warmup count drifted")
        if item.get("expected_path") not in {"SIO", "HCCS_SW"}:
            raise common.CalibrationError("formal topology path is unsupported")
        paths.add(item["expected_path"])
    if paths != {"SIO", "HCCS_SW"}:
        raise common.CalibrationError("formal plan must cover SIO and HCCS_SW")
    monitor_ab = plan.get("monitor_ab", {})
    ab_key = (monitor_ab.get("pair_id"), monitor_ab.get("message_bytes"))
    if ab_key not in keys or monitor_ab.get("order") != [
        "off", "on", "on", "off", "off", "on",
    ]:
        raise common.CalibrationError("formal monitor A/B contract drifted")
    return {
        **plan,
        "_path": str(resolved),
        "_protocol_id": protocol_id,
        "_protocol": protocol,
        "_sha256": common.sha256_file(resolved),
    }


def command_for(
    plan: dict[str, Any], item: dict[str, Any], *, monitor: str,
    port: int, result_root: Path,
) -> list[str]:
    return [
        sys.executable, str(RUN_PATH), "benchmark", "run",
        "--config", str(common.repository_path(plan["runtime_config"])),
        "--case", plan["case"],
        "--device-ids", ",".join(map(str, item["device_ids"])),
        "--nproc-per-node", "2",
        "--case-config", str(common.repository_path(item["config"])),
        "--monitor", monitor,
        "--master-port", str(port),
        "--timeout", str(plan["measurement_contract"]["timeout_seconds"]),
        "--result-root", str(result_root),
        "--allow-privileged-root",
    ]


def build_tasks(
    plan: dict[str, Any], *, master_port: int, result_root: Path,
) -> list[dict[str, Any]]:
    tasks = []
    repeats = plan["measurement_contract"]["size_curve_repeats"]
    for item in plan["matrix"]:
        for repeat in range(1, repeats + 1):
            tasks.append({
                "phase": "size-curve", "repeat": repeat, "monitor": "off",
                "cell": item,
            })
    ab = plan["monitor_ab"]
    ab_item = next(
        item for item in plan["matrix"]
        if item["pair_id"] == ab["pair_id"]
        and item["message_bytes"] == ab["message_bytes"]
    )
    for sequence, monitor in enumerate(ab["order"], start=1):
        tasks.append({
            "phase": "monitor-ab", "sequence": sequence,
            "monitor": monitor, "cell": ab_item,
        })
    if master_port + len(tasks) - 1 > 65535:
        raise common.CalibrationError("master-port range exceeds 65535")
    for index, task in enumerate(tasks):
        task["command"] = command_for(
            plan, task["cell"], monitor=task["monitor"],
            port=master_port + index, result_root=result_root,
        )
    return tasks


def audit_child(
    plan: dict[str, Any], task: dict[str, Any], child_root: Path,
) -> dict[str, Any]:
    summary_path = child_root / "summary.json"
    result_path = child_root / "benchmark-result.json"
    if not summary_path.is_file() or not result_path.is_file():
        raise common.CalibrationError("formal child evidence is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_monitoring = "passed" if task["monitor"] == "on" else "not-run"
    for field, expected in {
        "status": "passed", "execution_status": "passed",
        "measurement_status": "passed", "monitoring_status": expected_monitoring,
    }.items():
        if summary.get(field) != expected:
            raise common.CalibrationError(
                f"formal child {field}={summary.get(field)!r}, expected {expected!r}"
            )
    if summary.get("host_postflight", {}).get("result", {}).get("status") != "passed":
        raise common.CalibrationError("formal child postflight did not pass")
    cell = task["cell"]
    if summary.get("selection", {}).get("selected_device_ids") != cell["device_ids"]:
        raise common.CalibrationError("formal child Device selection drifted")
    if summary.get("case_config", {}).get("override", {}).get("sha256") != cell["sha256"]:
        raise common.CalibrationError("formal child config hash drifted")
    if result.get("status") != "passed" or result.get("fallback_count") != 0:
        raise common.CalibrationError("formal child result/fallback gate failed")
    rates = [
        float(metric["value"]) for metric in result.get("metrics", [])
        if metric.get("unit") == "GB/s"
    ]
    ranks = sorted({
        metric.get("rank") for metric in result.get("metrics", [])
        if metric.get("unit") == "GB/s"
    })
    if ranks != [0, 1] or len(rates) != 2 or any(rate <= 0 for rate in rates):
        raise common.CalibrationError("formal child rank metrics are incomplete")
    device_map = summary.get("host_preflight", {}).get("result", {}).get(
        "actual_device_map", []
    )
    logic_to_phy = {
        entry.get("logic_id"): entry.get("phy_id")
        for entry in device_map if isinstance(entry, dict)
    }
    try:
        phy_ids = [logic_to_phy[device] for device in cell["device_ids"]]
    except KeyError as exc:
        raise common.CalibrationError("formal child Device map is incomplete") from exc
    topology_files = list(
        (child_root / "host-preflight").glob("*/npu-smi-topology.stdout")
    )
    if len(topology_files) != 1:
        raise common.CalibrationError("formal child topology evidence is ambiguous")
    actual_path = common.topology_path(
        topology_files[0].read_text(encoding="utf-8"), *phy_ids,
    )
    if actual_path != cell["expected_path"]:
        raise common.CalibrationError("formal child topology path drifted")
    payload = cell["message_bytes"] * cell["iterations"]
    elapsed = [2 * payload / (rate * 1_000_000_000) for rate in rates]
    minimum = plan["measurement_contract"]["minimum_measurement_seconds_per_run"]
    if min(elapsed) < minimum:
        raise common.CalibrationError(
            f"formal derived measurement window is below {minimum}s: {elapsed}"
        )
    return {
        "runtime_image_id": summary.get("runtime", {}).get("image_id"),
        "topology_path": actual_path,
        "rank_gb_s": rates,
        "rank_mean_gb_s": statistics.fmean(rates),
        "derived_rank_elapsed_s": [round(value, 3) for value in elapsed],
        "child_summary_sha256": common.sha256_file(summary_path),
        "benchmark_result_sha256": common.sha256_file(result_path),
    }


def aggregate(plan: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    cv_limit = plan["acceptance"]["maximum_cv_pct_per_topology_size_cell"]
    cells = []
    for item in plan["matrix"]:
        values = [
            run["audit"]["rank_mean_gb_s"] for run in runs
            if run["phase"] == "size-curve"
            and run["pair_id"] == item["pair_id"]
            and run["message_bytes"] == item["message_bytes"]
        ]
        if len(values) != plan["measurement_contract"]["size_curve_repeats"]:
            raise common.CalibrationError("formal size-curve repeat count is incomplete")
        mean = statistics.fmean(values)
        cv = statistics.pstdev(values) / abs(mean) * 100
        cells.append({
            "pair_id": item["pair_id"], "message_bytes": item["message_bytes"],
            "run_rank_means_gb_s": values, "mean_gb_s": mean, "cv_pct": cv,
            "status": "passed" if cv <= cv_limit else "failed",
        })
    ab_runs = [run for run in runs if run["phase"] == "monitor-ab"]
    medians = {
        state: statistics.median([
            run["audit"]["rank_mean_gb_s"] for run in ab_runs
            if run["monitor"] == state
        ]) for state in ("off", "on")
    }
    delta = (medians["on"] - medians["off"]) / medians["off"] * 100
    delta_limit = plan["acceptance"]["maximum_absolute_monitor_median_delta_pct"]
    return {
        "size_curve": cells,
        "monitor_ab": {
            "median_off_gb_s": medians["off"],
            "median_on_gb_s": medians["on"],
            "median_delta_pct": delta,
            "status": "passed" if abs(delta) <= delta_limit else "failed",
        },
        "status": "passed" if (
            all(cell["status"] == "passed" for cell in cells)
            and abs(delta) <= delta_limit
        ) else "failed",
    }


def prepare_resumed_session(
    plan: dict[str, Any], tasks: list[dict[str, Any]], result_root: Path,
    resume_session: Path,
) -> tuple[Path, dict[str, Any], int, str | None]:
    session = resume_session.expanduser().resolve()
    if not session.is_relative_to(result_root.resolve()):
        raise common.CalibrationError("resume session escapes the result root")
    summary_path = session / "summary.json"
    if not summary_path.is_file():
        raise common.CalibrationError("resume session has no summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "failed":
        raise common.CalibrationError("only a failed formal session can be resumed")
    if summary.get("protocol_id") != plan["_protocol_id"]:
        raise common.CalibrationError("resume session protocol drifted")
    plan_record = summary.get("plan", {})
    snapshot = session / str(plan_record.get("path", ""))
    expected_hash = plan["_sha256"]
    if (
        plan_record.get("sha256") != expected_hash
        or not snapshot.is_file()
        or common.sha256_file(snapshot) != expected_hash
    ):
        raise common.CalibrationError("resume session plan hash drifted")

    records = summary.get("runs")
    if not isinstance(records, list) or not records:
        raise common.CalibrationError("resume session has no failed task record")
    completed: list[dict[str, Any]] = []
    failed_attempts = summary.setdefault("failed_attempts", [])
    for index, record in enumerate(records):
        if index >= len(tasks):
            raise common.CalibrationError("resume session has excess task records")
        task = tasks[index]
        expected = {
            "phase": task["phase"],
            "monitor": task["monitor"],
            "pair_id": task["cell"]["pair_id"],
            "message_bytes": task["cell"]["message_bytes"],
            "repeat": task.get("repeat"),
            "sequence": task.get("sequence"),
            "command": task["command"],
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise common.CalibrationError(
                f"resume task {index + 1} metadata drifted"
            )
        if record.get("returncode") == 0 and isinstance(record.get("audit"), dict):
            child = record.get("child_result")
            if not isinstance(child, str):
                raise common.CalibrationError(
                    f"resume task {index + 1} child evidence is missing"
                )
            audited = audit_child(plan, task, Path(child).resolve())
            if audited != record["audit"]:
                raise common.CalibrationError(
                    f"resume task {index + 1} audit drifted"
                )
            completed.append(record)
            continue
        if index != len(records) - 1:
            raise common.CalibrationError(
                "resume session has a failed task before later records"
            )
        failed_attempts.append({
            **record,
            "task_index": index + 1,
            "preserved_at": common.utc_now(),
        })

    if len(completed) == len(records):
        raise common.CalibrationError("resume session has no trailing failed task")
    runtime_ids = {
        record["audit"].get("runtime_image_id") for record in completed
    }
    if len(runtime_ids) > 1:
        raise common.CalibrationError("runtime image changed before resume")
    runtime_id = next(iter(runtime_ids), None)
    summary["runs"] = completed
    summary["status"] = "running"
    summary.pop("error", None)
    summary["resume_count"] = int(summary.get("resume_count", 0)) + 1
    summary["resumed_at"] = common.utc_now()
    common.write_json(summary_path, summary)
    return session, summary, len(completed), runtime_id


def execute(
    plan: dict[str, Any], tasks: list[dict[str, Any]], result_root: Path,
    resume_session: Path | None = None,
) -> int:
    protocol = plan["_protocol"]
    if resume_session is None:
        session = result_root / (protocol["session_prefix"] + common.run_timestamp())
        session.mkdir(parents=True, exist_ok=False)
        snapshot = session / Path(plan["_path"]).name
        shutil.copyfile(Path(plan["_path"]), snapshot)
        summary: dict[str, Any] = {
            "schema_version": 1, "kind": protocol["summary_kind"],
            "protocol_id": plan["_protocol_id"],
            "status": "running",
            "formal_execution_eligible": True,
            "production_eligible": protocol["production_eligible"],
            "plan": {"path": snapshot.name, "sha256": common.sha256_file(snapshot)},
            "runs": [],
        }
        common.write_json(session / "summary.json", summary)
        start_index = 0
        runtime_id = None
    else:
        session, summary, start_index, runtime_id = prepare_resumed_session(
            plan, tasks, result_root, resume_session,
        )
    for index, task in enumerate(tasks[start_index:], start=start_index + 1):
        proc = subprocess.run(
            task["command"], cwd=REPOSITORY_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        output = proc.stdout or ""
        log = session / f"run-{index:02d}.log"
        log.write_text(output, encoding="utf-8")
        match = common.RESULT_LINE_RE.search(output)
        child = Path(match.group("path")).resolve() if match else None
        record = {
            "phase": task["phase"], "monitor": task["monitor"],
            "pair_id": task["cell"]["pair_id"],
            "message_bytes": task["cell"]["message_bytes"],
            "repeat": task.get("repeat"), "sequence": task.get("sequence"),
            "command": task["command"], "returncode": proc.returncode,
            "log": {"path": log.name, "sha256": common.sha256_file(log)},
            "child_result": str(child) if child else None,
        }
        summary["runs"].append(record)
        if proc.returncode != 0 or child is None:
            summary.update(status="failed", error="formal child execution failed")
            common.write_json(session / "summary.json", summary)
            print(f"Formal session: {session}")
            return 1
        try:
            record["audit"] = audit_child(plan, task, child)
            observed_runtime = record["audit"]["runtime_image_id"]
            runtime_id = runtime_id or observed_runtime
            if observed_runtime != runtime_id:
                raise common.CalibrationError("runtime image changed within formal session")
        except common.CalibrationError as exc:
            summary.update(status="failed", error=f"formal child audit failed: {exc}")
            common.write_json(session / "summary.json", summary)
            print(f"Formal session: {session}")
            return 1
        common.write_json(session / "summary.json", summary)
    summary["analysis"] = aggregate(plan, summary["runs"])
    summary["status"] = summary["analysis"]["status"]
    summary["formal_execution_eligible"] = summary["status"] == "passed"
    summary["production_eligible"] = (
        summary["status"] == "passed" and protocol["production_eligible"]
    )
    summary["claim_boundary"] = plan["claim_boundary"]
    common.write_json(session / "summary.json", summary)
    print(f"Formal session: {session}")
    return 0 if summary["status"] == "passed" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--master-port", type=int, default=30021)
    parser.add_argument("--result-root", type=Path, default=BASE_DIR / "result")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-session", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        plan = load_and_validate_plan(args.plan)
        result_root = args.result_root.expanduser().resolve()
        tasks = build_tasks(plan, master_port=args.master_port, result_root=result_root)
        if not args.execute:
            if args.resume_session is not None:
                raise common.CalibrationError("--resume-session requires --execute")
            print(json.dumps({
                "schema_version": 1, "mode": "plan-only",
                "protocol_id": plan["_protocol_id"],
                "plan_sha256": plan["_sha256"], "run_count": len(tasks),
                "estimated_total_measurement_minutes":
                    plan["estimated_total_measurement_minutes"],
                "tasks": tasks,
            }, indent=2, sort_keys=True))
            return 0
        return execute(plan, tasks, result_root, args.resume_session)
    except (common.CalibrationError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
