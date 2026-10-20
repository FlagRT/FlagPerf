#!/usr/bin/env python3
"""Container-side PPU Toolkit execution with the image's own official tools.

No PyTorch and no Base workload: transfer, bandwidth and host-device latency
points run the locked image's sailbandwidth, P2P points run
p2pBandwidthLatencyTest (single-process binary plus `mpirun
--allow-run-as-root -np 2` for bidirectional; invocation forms pinned by
probes/day9/toolkit/t9b_p2p_confirmed.txt), computation points run the
isolated acblas child process, and capacity points parse a full in-container
`ppu-smi -q` with the same grammar the host monitor uses. CUDA_VISIBLE_DEVICES
addresses cards by their local index among the selected set, and every point
re-verifies tool-reported BDF/UUID identity against the host bindings before
its record is accepted; any failed gate records the point as failed and it
never contributes a number.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[3]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))
from toolkits._common.thead.PPU.contract import (CASES, SAIL_TESTCASES, SAIL_EXPECTED_NAMES,
    SAIL_ROUTED_MODES,
    case_points, contract_record, layer_status, metric_from_record, target_name,
    validate_compute_record, validate_selection, validate_tool_record,
    MATRIX_SIZE_RANGE, COMPUTE_SAMPLES_RANGE, TRANSFER_SAMPLES_RANGE, P2P_SAMPLES_RANGE,
    SAIL_BUFFER_MIB_RANGE, P2P_BYTES_RANGE)
from toolkits._common.thead.PPU.evidence import command, now, reference, sha256, write_json
from base.vendors.thead.provider import normalize_bdf
from monitoring.thead_usage import parse_card, parse_q_sections

SDK = Path("/usr/local/PPU_SDK")
SAIL_SP = SDK / "comm_tools/single_process/sailbandwidth"
P2P_SP = SDK / "comm_tools/single_process/p2pBandwidthLatencyTest"
P2P_MP = SDK / "comm_tools/multi_process/p2pBandwidthLatencyTest"
SAIL_MP = SDK / "comm_tools/multi_process/sailbandwidth"
MPIRUN = Path("/usr/bin/mpirun")
LIBACBLAS = SDK / "lib/libacblas.so"
LIBACBLASLT = SDK / "lib/libacblasLt.so"
CUDA_RUNTIME_CANDIDATES = (SDK / "lib/libcudart.so", Path("/usr/local/cuda/lib64/libcudart.so"))
COMPUTE_CHILD = HERE / "compute_child.py"


def validate_settings(settings):
    checks = (("payload_mib", SAIL_BUFFER_MIB_RANGE), ("matrix_size", MATRIX_SIZE_RANGE),
              ("compute_samples", COMPUTE_SAMPLES_RANGE), ("transfer_samples", TRANSFER_SAMPLES_RANGE),
              ("p2p_samples", P2P_SAMPLES_RANGE))
    for key, (low, high) in checks:
        value = settings.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError("setting %s=%r outside contract range %s" % (key, value, (low, high)))
    p2p_bytes = settings.get("p2p_bytes")
    if type(p2p_bytes) is not int or not P2P_BYTES_RANGE[0] <= p2p_bytes <= P2P_BYTES_RANGE[1] or p2p_bytes % 4:
        raise ValueError("setting p2p_bytes must be an int multiple of 4 inside the contract range")
    for key in ("repeat", "command_timeout", "compute_timeout"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError("setting %s missing or invalid" % key)
    if type(settings.get("warmup")) is not int or settings["warmup"] < 0:
        raise ValueError("setting warmup missing or invalid")
    return settings


def device_env(slot_expression):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = slot_expression
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    return env


def parse_json_stdout(text, label):
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON document in " + label + " stdout")
    document, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(document, dict) or "sailbandwidth" not in document:
        raise ValueError("unrecognized " + label + " JSON document")
    return document["sailbandwidth"]


_TOOL_BDF = re.compile(r"([0-9A-Fa-f]{4,8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})(?:\.(\d))?")


def tool_bdf(value):
    """Join tool-reported BDFs (sail `00000000:a3:00`, p2p `0000:A3:00.0`) to the
    canonical host form the provider and monitor key on (`0000:a3:00.0`)."""
    match = _TOOL_BDF.fullmatch(value.strip())
    if not match:
        raise ValueError("unrecognized tool BDF: %r" % (value,))
    domain, bus, device = match[1], match[2], match[3]
    return normalize_bdf("%08x:%s:%s.%s" % (int(domain, 16), bus, device, match[4] or "0"))


def parse_dev_bdfs(text):
    """p2p header lines `Dev<i>, pciBus: <bdf>, ...` -> {visible index: canonical bdf}."""
    found = {}
    for match in re.finditer(r"Dev(\d+), pciBus: ([0-9A-Fa-f:.]+)", text):
        index, bdf = int(match[1]), tool_bdf(match[2])
        if index in found and found[index] != bdf:
            raise ValueError("conflicting p2p header identity for Dev%d" % index)
        found[index] = bdf
    if not found:
        raise ValueError("p2p header carries no device pciBus identity")
    return found


def parse_matrix_sections(text, keyword, unit_token, expected_side):
    """All `<keyword> ... (<unit_token>)` matrix blocks as {header, values[row][col]}.

    Cell tokens are floats; the diagonal prints `x` and is stored as None.
    Multiple blocks occur when mpirun prints one per rank; every block is kept.
    """
    matrices = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if keyword not in line or unit_token not in line:
            continue
        rows = []
        for candidate in lines[index + 1:]:
            row = re.match(r"DEV(\d+)\s+(.+?)\s*$", candidate)
            if row:
                rows.append((int(row[1]), [None if token == "x" else float(token) for token in row[2].split()]))
                continue
            if not rows and re.fullmatch(r"\s*(?:DEV\d+\s+)+", candidate):
                continue  # indented column-header row above the data rows
            break
        if len(rows) != expected_side or [r for r, _ in rows] != list(range(expected_side)) or \
                any(len(cells) != expected_side for _, cells in rows) or \
                any(rows[i][1][i] not in (None, 0.0) for i in range(expected_side)):
            raise ValueError("malformed %s matrix block after: %s" % (keyword, line.strip()))
        for i in range(len(rows)):
            rows[i][1][i] = None  # bandwidth prints `x`, latency prints 0.00 on the diagonal
        matrices.append({"header": line.strip(), "values": [cells for _, cells in rows]})
    if not matrices:
        raise ValueError("missing %s (%s) matrix in p2p output" % (keyword, unit_token))
    return matrices


def expected_bytes_token(header, p2p_bytes, bidirectional):
    match = re.search(r"\((\d+) ?(\*2 )?Bytes\)", header)
    if not match:
        raise ValueError("bandwidth matrix header lacks a byte annotation: " + header)
    if int(match[1]) != p2p_bytes or bool(match[2]) != bidirectional:
        raise ValueError("bandwidth matrix byte annotation differs from the planned payload: " + header)


def sail_argv(testcase, settings):
    return [str(SAIL_SP), "-t", str(testcase), "-b", str(settings["payload_mib"]),
            "-i", str(settings["transfer_samples"]), "-j"]


def p2p_argv(binary, perf, direction, settings):
    args = [str(binary), "--perf", perf, "--p2p_type", "CE",
            "--num_elems", str(settings["p2p_bytes"] // 4), "--repeat", str(settings["p2p_samples"])]
    if perf == "uniBw":
        args += ["--num_gpus", "2", "--dev_list", "0,1", "--p2p_direction", direction, "--verify", "1"]
    elif perf == "biBw":
        args += ["--verify", "1"]
    return args


def window_ns(record):
    return {"started_monotonic_ns": int(round(record["started_monotonic_s"] * 1e9)),
            "finished_monotonic_ns": int(round(record["finished_monotonic_s"] * 1e9))}


def run_sail_point(root, directory, point, slot, settings, case, repetition):
    """One sailbandwidth point; with minimum_seconds > 0 the closed tool is
    re-invoked inside the same point window until the floor is reached and the
    metric is the median across invocations (each archived under its own stem).
    The returned command record is the last invocation with its start moved to
    the first invocation's start, so the declared window spans the whole floor.
    """
    testcase, expected_name, unit = SAIL_EXPECTED_NAMES[(case, point["mode"])]
    floor_s = float(settings.get("minimum_seconds", 0.0))
    values, invocations, first_start = [], [], None
    record = payload = None
    k = 0
    while True:
        k += 1
        stem = "sailbandwidth" if k == 1 else "sailbandwidth-%d" % k
        record = command(sail_argv(testcase, settings), directory, root, stem=stem,
                         timeout=settings["command_timeout"], env=device_env(str(slot["local_index"])))
        if record["returncode"]:
            raise RuntimeError("sailbandwidth exited " + str(record["returncode"]))
        payload = parse_json_stdout((root / record["stdout"]["path"]).read_text(errors="replace"), "sailbandwidth")
        observed = [tool_bdf(re.search(r"\(([0-9A-Fa-f:.]+)\)", entry)[1])
                    for entry in payload.get("GPU Device list", [])]
        if observed != [slot["pci_bdf"]]:
            raise ValueError("sail device list %r differs from the selected card %r" % (observed, slot["pci_bdf"]))
        testcases = payload.get("testcases", [])
        if len(testcases) != 1 or testcases[0].get("name") != expected_name:
            raise ValueError("sail testcase mismatch: %r" % [t.get("name") for t in testcases])
        if testcases[0].get("status") != "Passed":
            raise ValueError("sail reported status %r" % testcases[0].get("status"))
        matrix = testcases[0]["bandwidth_matrix"]
        if len(matrix) != 1 or len(matrix[0]) != 1:
            raise ValueError("sail matrix is not 1x1 for a single-card point")
        values.append(float(matrix[0][0]))
        invocations.append({"invocation": k, "value": values[-1],
                            "stdout": record["stdout"]["path"], "duration_s": record["duration_s"]})
        first_start = record["started_monotonic_s"] if first_start is None else first_start
        if time.monotonic() - first_start >= floor_s:
            break
    value = values[0] if len(values) == 1 else statistics.median(values)
    window_record = dict(record)
    window_record["started_monotonic_s"] = first_start
    return {"schema_version": 1, "point": point, "repetition": repetition,
            "tool": "sailbandwidth", "tool_status": testcases[0]["status"],
            "tool_testcase": expected_name, "tool_value": value, "tool_unit": unit,
            "value": value, "unit": unit,
            "verification": "not-offered" if unit == "ns" else "tool-verified",
            "identity": {"bdf_matches": True, "expected_bdfs": [slot["pci_bdf"]], "observed_bdfs": observed},
            "invocations": invocations, "minimum_seconds_floor": floor_s,
            "stdout": record["stdout"]["path"], **window_ns(window_record)}, window_record


def run_p2p_point(root, directory, point, source, destination, settings, perf, direction, repetition):
    """One p2pBandwidthLatencyTest point; same minimum_seconds floor semantics
    as run_sail_point (re-invoked closed tool, median across invocations)."""
    floor_s = float(settings.get("minimum_seconds", 0.0))
    values, invocations, first_start = [], [], None
    record = merged = None
    k = 0
    while True:
        k += 1
        stem = "p2p-" + perf if k == 1 else "p2p-%s-%d" % (perf, k)
        record, value, merged = p2p_single_invocation(root, directory, point, source, destination,
                                                      settings, perf, direction, repetition, stem)
        values.append(value)
        invocations.append({"invocation": k, "value": value,
                            "stdout": record["stdout"]["path"], "duration_s": record["duration_s"]})
        first_start = record["started_monotonic_s"] if first_start is None else first_start
        if time.monotonic() - first_start >= floor_s:
            break
    value = values[0] if len(values) == 1 else statistics.median(values)
    merged["tool_value"] = value
    merged["value"] = value * (1000.0 if merged["unit"] == "ns" else 1.0)
    merged["invocations"] = invocations
    merged["minimum_seconds_floor"] = floor_s
    merged["stdout"] = record["stdout"]["path"]
    window_record = dict(record)
    window_record["started_monotonic_s"] = first_start
    merged.update(window_ns(window_record))
    return merged, window_record


def p2p_single_invocation(root, directory, point, source, destination, settings, perf, direction, repetition, stem):
    argv = p2p_argv(P2P_MP if perf == "biBw" else P2P_SP, perf, direction, settings)
    env = device_env(",".join(str(s) for s in sorted((source["local_index"], destination["local_index"]))))
    if perf == "biBw":
        argv = [str(MPIRUN), "--allow-run-as-root", "-np", "2"] + argv
    record = command(argv, directory, root, stem=stem,
                     timeout=settings["command_timeout"], env=env)
    if record["returncode"]:
        raise RuntimeError("p2pBandwidthLatencyTest exited " + str(record["returncode"]))
    text = (root / record["stdout"]["path"]).read_text(errors="replace")
    if "P2P test finished" not in text:
        raise ValueError("p2p output does not reach a finished marker")
    verified = perf in ("uniBw", "biBw")
    if verified and "data matched" not in text:
        raise ValueError("p2p verification output missing the data-matched marker")
    dev_bdfs = parse_dev_bdfs(text)
    expected = {source["pci_bdf"], destination["pci_bdf"]}
    if set(dev_bdfs.values()) != expected or len(dev_bdfs) != 2:
        raise ValueError("p2p header devices %r differ from the pair %r" % (dev_bdfs, expected))
    by_bdf = {bdf: index for index, bdf in dev_bdfs.items()}
    if perf == "latency":
        matrices = parse_matrix_sections(text, "GPU P2P Transmission Latency Matrix", "(US)", 2)
        unit, scale = "us", 1000.0
        keyword = None
    else:
        bidirectional = perf == "biBw"
        keyword = ("Bidirectional) P2P Transport Bandwidth Matrix" if bidirectional
                   else "Unidirectional) P2P Transport Bandwidth Matrix")
        matrices = parse_matrix_sections(text, keyword, "(GB/s)", 2)
        unit, scale = "GB/s", 1.0
        for matrix in matrices:
            expected_bytes_token(matrix["header"], settings["p2p_bytes"], bidirectional)
        wanted = "WRITE" if (direction == "write" or bidirectional) else "READ"
        if wanted not in matrices[0]["header"]:
            raise ValueError("p2p matrix header lacks the planned direction %s: %s" % (wanted, matrices[0]["header"]))
    primary = matrices[0]["values"]
    row, col = by_bdf[source["pci_bdf"]], by_bdf[destination["pci_bdf"]]
    value = primary[row][col]
    if value is None or not value > 0:
        raise ValueError("p2p matrix cell [%d][%d] is not a positive measurement" % (row, col))
    tool_record = {"schema_version": 1, "point": point, "repetition": repetition,
                   "tool": "p2pBandwidthLatencyTest", "perf": perf, "direction": direction,
                   "tool_status": "Passed", "tool_value": value, "tool_unit": unit,
                   "value": value * scale, "unit": "ns" if unit == "us" else "GB/s",
                   "verification": "tool-verified" if verified else "not-offered",
                   "matrix_keyword": keyword, "matrix_count": len(matrices), "matrices": matrices,
                   "identity": {"bdf_matches": True, "expected_bdfs": sorted(expected),
                                "observed_bdfs": [dev_bdfs[i] for i in sorted(dev_bdfs)]}}
    return record, value, tool_record


def run_capacity_point(root, directory, point, slot, paths, repetition):
    record = command([str(paths["ppu-smi"]), "-q"], directory, root, stem="ppu-smi",
                     timeout=30, env=device_env(""))
    if record["returncode"]:
        raise RuntimeError("ppu-smi exited " + str(record["returncode"]))
    sections = parse_q_sections((root / record["stdout"]["path"]).read_text(errors="replace"))
    canonical = slot["pci_bdf"]
    if canonical not in sections:
        raise ValueError("in-container ppu-smi does not expose the selected card " + canonical)
    card = parse_card(sections[canonical], canonical)
    if card["uuid"] != slot["uuid"]:
        raise ValueError("capacity query UUID differs from the selected card")
    return {"schema_version": 1, "point": point, "repetition": repetition,
            "tool": "ppu-smi", "tool_status": "Passed", "verification": "not-offered",
            "capacity_mib": {key: card[key] for key in ("total_memory_mib", "used_memory_mib", "free_memory_mib")},
            "identity": {"bdf_matches": True, "expected_bdfs": [canonical], "observed_bdfs": [canonical]},
            "uuid": card["uuid"], "stdout": record["stdout"]["path"], **window_ns(record)}, record


def run_compute_point(root, directory, point, slot, settings, repetition):
    spec = {"schema_version": 1, "dtype": point["dtype"], "matrix_size": settings["matrix_size"],
            "samples": settings["compute_samples"], "warmup": settings["warmup"],
            "minimum_seconds": float(settings.get("minimum_seconds", 0.0)),
            "expected_uuid": slot["uuid"]}
    spec_path, out_path = directory / "child-spec.json", directory / "child-record.json"
    write_json(spec_path, spec)
    record = command([sys.executable, str(COMPUTE_CHILD), "--spec", str(spec_path), "--output", str(out_path)],
                     directory, root, stem="acblas", timeout=settings["compute_timeout"],
                     env=device_env(str(slot["local_index"])))
    if record["returncode"]:
        raise RuntimeError("acblas child exited " + str(record["returncode"]))
    child = json.loads(out_path.read_text(encoding="utf-8"))
    validate_compute_record(child, point, settings["matrix_size"], settings["compute_samples"])
    if child.get("uuid") != slot["uuid"]:
        raise ValueError("compute child UUID %r differs from the selected card %r"
                         % (child.get("uuid"), slot["uuid"]))
    return child, record


def run_point(root, case, point, repetition, prepared, settings, bindings):
    name = target_name(point, repetition)
    directory = root / "cases" / case / name
    directory.mkdir(parents=True, exist_ok=False)
    involved = [bindings[point[k]] for k in ("source", "destination") if k in point]
    result = {"schema_version": 1, "target": name, "point": point, "repetition": repetition,
              "commands": [], "metrics": [], "status": "failed", "measurement_status": "failed",
              "monitoring_status": "pending-host-finalization", "diagnosis_status": "not-run",
              "sweep_scope": "explicit physical cards/mode/payload; vendor official tools",
              "duration_s": 0, "bindings": involved}
    try:
        if point["mode"] == "gemm":
            tool_record, record = run_compute_point(root, directory, point, bindings[point["source"]], settings, repetition)
            result["measurement_window"] = {"started_monotonic_s": tool_record["started_monotonic_ns"] / 1e9,
                                            "finished_monotonic_s": tool_record["finished_monotonic_ns"] / 1e9,
                                            "role": "measurement"}
        elif point["mode"] == "capacity-query":
            tool_record, record = run_capacity_point(root, directory, point, bindings[point["source"]], prepared, repetition)
            result["measurement_window"] = {"started_monotonic_s": record["started_monotonic_s"],
                                            "finished_monotonic_s": record["finished_monotonic_s"],
                                            "role": "capacity-query"}
            result["metrics"] = [{"field": "Memory Usage/" + key, "source": tool_record["stdout"],
                                  "source_kind": "ppu-smi", "value": value, "unit": "MiB", "point": point,
                                  "scope": "reported capacity query; not held allocation or OOM stress"}
                                 for key, value in tool_record["capacity_mib"].items()]
        else:
            if point["mode"] in SAIL_ROUTED_MODES:
                tool_record, record = run_sail_point(root, directory, point, bindings[point["source"]],
                                                     settings, case, repetition)
            elif point["mode"] in ("uni-write", "uni-read"):
                tool_record, record = run_p2p_point(root, directory, point, bindings[point["source"]],
                                                    bindings[point["destination"]], settings, "uniBw",
                                                    "write" if point["mode"] == "uni-write" else "read", repetition)
            elif point["mode"] == "bi-write":
                tool_record, record = run_p2p_point(root, directory, point, bindings[point["source"]],
                                                    bindings[point["destination"]], settings, "biBw",
                                                    "write", repetition)
            elif point["mode"] == "latency":
                tool_record, record = run_p2p_point(root, directory, point, bindings[point["source"]],
                                                    bindings[point["destination"]], settings, "latency",
                                                    "write", repetition)
            else:
                raise ValueError("unroutable point mode " + point["mode"])
            result["measurement_window"] = {"started_monotonic_s": record["started_monotonic_s"],
                                            "finished_monotonic_s": record["finished_monotonic_s"],
                                            "role": "measurement"}
        result["commands"].append(record)
        result["duration_s"] = record["duration_s"]
        if point["mode"] == "capacity-query":
            write_json(directory / "tool-record.json", tool_record)
            result["tool_record"] = tool_record
        elif point["mode"] == "gemm":
            result["tool_record"] = {k: v for k, v in tool_record.items() if k != "samples_ns"}
            metric = metric_from_record(tool_record, case, settings["matrix_size"])
            # Metric rows carry the full point so per-card and per-direction
            # grouping in the report can never lump distinct measurements.
            metric["point"] = point
            metric["repetition"] = repetition
            metric["source"] = directory.relative_to(root).as_posix() + "/samples.json"
            result["metrics"] = [metric]
        else:
            write_json(directory / "tool-record.json", tool_record)
            metric = metric_from_record(tool_record, case, settings["matrix_size"])
            metric["point"] = point
            metric["repetition"] = repetition
            metric["source"] = directory.relative_to(root).as_posix() + "/tool-record.json"
            result["metrics"] = [metric]
            result["tool_record"] = {k: v for k, v in tool_record.items() if k != "matrices"}
        result.update(status="partial", measurement_status="passed", diagnosis_status="passed")
    except Exception as exc:
        result["error"] = str(exc)
    write_json(directory / "metrics.json", result)
    return result


def prepare(root, context):
    """Provenance snapshot and static tool checks; no device is touched here."""
    paths = {"sailbandwidth-single": SAIL_SP, "p2pBandwidthLatencyTest-single": P2P_SP,
             "p2pBandwidthLatencyTest-multi": P2P_MP, "sailbandwidth-multi": SAIL_MP,
             "mpirun": MPIRUN, "libacblas": LIBACBLAS, "libacblasLt": LIBACBLASLT}
    smi = shutil.which("ppu-smi")
    if smi:
        paths["ppu-smi"] = Path(smi)
    for candidate in CUDA_RUNTIME_CANDIDATES:
        if candidate.is_file():
            paths["libcudart"] = candidate
            break
    provenance = {"tools": {}, "environment": {k: os.environ.get(k, "") for k in ("PATH", "LD_LIBRARY_PATH")},
                  "backend": "official-comm-tools+acblas", "no_base_workload": True,
                  "runner_source": reference(HERE / "evidence_runner.py", BASE),
                  "compute_child_source": reference(COMPUTE_CHILD, BASE),
                  "contract_source": reference(HERE / "contract.py", BASE),
                  "runtime_lock": context["runtime_lock"], "context_sha256": context["context_sha256"]}
    missing = sorted(label for label, path in paths.items() if not path.is_file())
    for label, path in sorted(paths.items()):
        provenance["tools"][label] = {"path": str(path), "present": path.is_file()}
        if path.is_file():
            provenance["tools"][label].update(sha256=sha256(path), bytes=path.stat().st_size,
                                              resolved_path=str(path.resolve()))
    if missing:
        raise FileNotFoundError("official toolkit dependencies missing in image: " + ", ".join(missing))
    if not COMPUTE_CHILD.is_file():
        raise FileNotFoundError("compute_child.py missing next to the runner")
    write_json(root / "provenance.json", provenance)
    write_json(root / "toolkit-contract.json", contract_record(context["settings"]))
    write_json(root / "runtime-bindings.json", {"schema_version": 1,
                "method": "host ppu-smi UUID+BDF joined to per-point tool self-identity and child UUID readback",
                "context_sha256": context["context_sha256"], "devices": context["host"]["devices"]})
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw_context = args.context.read_bytes()
    context = json.loads(raw_context)
    context["context_sha256"] = hashlib.sha256(raw_context).hexdigest()
    ids = [d["host_physical_id"] for d in context["host"]["devices"]]
    validate_selection(ids)
    settings = validate_settings(context["settings"])
    order = sorted(ids)
    for device in context["host"]["devices"]:
        device["local_index"] = order.index(device["host_physical_id"])
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError("result evidence already exists")
    cases = context["cases"]
    for case in cases:
        if case not in CASES:
            raise ValueError("unregistered case in context: " + case)
    manifest = {"schema_version": 1, "kind": "toolkit", "suite": "thead-ppu-toolkit", "vendor": "thead",
                "chip": "PPU", "run_id": context["run_id"], "started_at": now(),
                "context_sha256": context["context_sha256"], "selected_physical_device_ids": ids,
                "settings": settings, "cases": {}, "status": "running",
                "monitoring_status": "pending-host-finalization", "diagnosis_status": "not-run"}
    write_json(root / "environment.json", {"host": context["host"], "settings": settings,
                                           "image_identity": context["image_identity"]})
    try:
        prepared = prepare(root, context)
        bindings = {d["host_physical_id"]: d for d in context["host"]["devices"]}
    except Exception as exc:
        manifest["setup_error"] = str(exc)
        prepared = None
    for case in cases:
        points = case_points(case, ids, settings["payload_mib"], settings["p2p_bytes"])
        targets = []
        if not points or prepared is None:
            reason = manifest.get("setup_error", "case resolves to no points")
            directory = root / "cases" / case / "unavailable"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "unavailable.stderr").write_text(reason + "\n")
            targets.append({"target": "unavailable", "commands": [], "metrics": [],
                            "status": "failed" if prepared is None else "resource-blocked",
                            "measurement_status": "failed" if prepared is None else "not-run",
                            "monitoring_status": "not-run", "diagnosis_status": "not-run",
                            "error": reason, "sweep_scope": "requested case not measured", "duration_s": 0})
            write_json(directory / "metrics.json", targets[-1])
        else:
            for point in points:
                for repetition in range(1, settings["repeat"] + 1):
                    targets.append(run_point(root, case, point, repetition, prepared, settings, bindings))
                    print(case, targets[-1]["target"], targets[-1]["measurement_status"], flush=True)
        result = {"targets": targets, "commands": [cmd for t in targets for cmd in t["commands"]],
                  "metrics": [m for t in targets for m in t["metrics"]],
                  "duration_s": sum(t["duration_s"] for t in targets),
                  "measurement_status": layer_status(t["measurement_status"] for t in targets),
                  "monitoring_status": "pending-host-finalization",
                  "diagnosis_status": layer_status(t["diagnosis_status"] for t in targets),
                  "status": "partial",
                  "sweep_scope": {"physical_ids": ids, "point_count": len(points), "repeat": settings["repeat"]}}
        if result["measurement_status"] == "failed":
            result["status"] = "failed"
        manifest["cases"][case] = result
        write_json(root / "cases" / case / "metrics.json", result)
        write_json(root / "manifest.json", manifest)
    manifest.update(finished_at=now(),
                    measurement_status=layer_status(c["measurement_status"] for c in manifest["cases"].values()),
                    diagnosis_status=layer_status(c["diagnosis_status"] for c in manifest["cases"].values()))
    manifest["status"] = "failed" if manifest["measurement_status"] == "failed" else "partial"
    write_json(root / "manifest.json", manifest)
    return 1 if manifest["status"] == "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
