"""PPU Toolkit host lifecycle: locked image, selected nodes, leases and evidence."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import time
import uuid

from executors.common import (BASE_DIR, ConfigurationError, DeviceLease, load_host_config,
                              runtime_lock_record, validate_runtime_identity)
from executors.lifecycle import HostCommands, ManagedContainer
from base.vendors.registry import get_provider
from base.vendors.thead import preflight
from toolkits._common.thead.PPU.contract import (CASES, SMOKE, case_points, contract_record,
                                                 layer_status, validate_selection,
                                                 SAIL_BUFFER_MIB_RANGE, MATRIX_SIZE_RANGE,
                                                 COMPUTE_SAMPLES_RANGE, TRANSFER_SAMPLES_RANGE,
                                                 P2P_SAMPLES_RANGE, P2P_BYTES_RANGE)
from toolkits._common.thead.PPU.evidence import index, now, sha256, write_json

# Shared flag names (--allow-candidate-runtime, --privilege-command, --repeat,
# --warmup, --matrix-size, --command-timeout, --smoke) are registered once by the
# P800 executor on the same parser, so their P800-registered defaults are the
# single source of truth; PPU defaults for those keys stay aligned with them.
DEFAULTS = {"allow_candidate_runtime": False, "privilege_command": "", "repeat": 5,
            "payload_mib": 512, "p2p_bytes": 40000000, "matrix_size": 8192,
            "compute_samples": 50, "transfer_samples": 25, "p2p_samples": 15,
            "warmup": 5, "command_timeout": 600, "compute_timeout": 900, "smoke": False}

# P800-only options: inert for the PPU runtime, so an explicitly changed value
# must be rejected instead of silently ignored (e.g. --allow-busy-devices has
# no PPU equivalent; foreign handles are never permitted).
P800_ONLY_OPTIONS = ("allow_busy_devices", "samples", "payload_bytes", "p2p_payload_bytes",
                     "pinned_api", "async_chunk_bytes")


def add_cli_arguments(parser):
    parser.add_argument("--payload-mib", type=int, default=512, help="sailbandwidth buffer size in MiB")
    parser.add_argument("--p2p-bytes", type=int, default=40000000, help="P2P transmission bytes; multiple of 4")
    parser.add_argument("--compute-samples", type=int, default=50, help="acblas timed calls per point; window floor comes from --minimum-seconds")
    parser.add_argument("--transfer-samples", type=int, default=25, help="sailbandwidth -i iterations per point")
    parser.add_argument("--p2p-samples", type=int, default=15, help="p2pBandwidthLatencyTest --repeat per point")
    parser.add_argument("--compute-timeout", type=int, default=900, help="bounds one acblas child including vendor JIT warmup")


def settings_for(request):
    options = {**DEFAULTS, **(request.thead_options or {})}
    # --minimum-seconds stays registered on the shared P800 flag set; the PPU
    # runner applies the same floor semantics (window held open until reached).
    options["minimum_seconds"] = float((getattr(request, "p800_options", None) or {}).get("minimum_seconds") or 0.0)
    if options["smoke"]:
        options.update({**SMOKE, "minimum_seconds": 0.0})
    return options


def plan(request, config_path, config):
    if request.context.physical_device_ids is None:
        raise ConfigurationError("PPU Toolkit requires explicit --physical-device-ids; legacy aliases are ambiguous")
    ids = request.context.selection_request()["requested_ids"]
    try:
        validate_selection(ids)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    if not set(ids).issubset(config["expected_device_ids"]):
        raise ConfigurationError("physical selection is outside the locked inventory")
    cases = list(request.cases or CASES)
    if len(cases) != len(set(cases)) or set(cases)-set(CASES):
        raise ConfigurationError("select unique registered PPU Toolkit cases")
    if request.legacy_probe or request.allow_privileged_root or request.allow_disruptive_dmi:
        raise ConfigurationError("Ascend legacy/privileged/DMI flags do not apply to PPU Toolkit")
    if request.latency_sizes is not None or request.hccl_min_bytes != "8K" or request.hccl_max_bytes != "1G":
        raise ConfigurationError("PPU uses the official tool latency semantics; HCCL options do not apply")
    from executors.p800_toolkit import DEFAULTS as P800_DEFAULTS
    changed_p800_only = [key for key in P800_ONLY_OPTIONS
                         if (request.p800_options or {}).get(key, P800_DEFAULTS[key]) != P800_DEFAULTS[key]]
    if changed_p800_only:
        raise ConfigurationError("P800-only options do not apply to PPU Toolkit: " + ", ".join(sorted(changed_p800_only)))
    options = settings_for(request)
    if not (1 <= options["repeat"] <= 20 and 0 <= options["warmup"] <= 100
            and SAIL_BUFFER_MIB_RANGE[0] <= options["payload_mib"] <= SAIL_BUFFER_MIB_RANGE[1]
            and MATRIX_SIZE_RANGE[0] <= options["matrix_size"] <= MATRIX_SIZE_RANGE[1] and options["matrix_size"] % 8 == 0
            and COMPUTE_SAMPLES_RANGE[0] <= options["compute_samples"] <= COMPUTE_SAMPLES_RANGE[1]
            and TRANSFER_SAMPLES_RANGE[0] <= options["transfer_samples"] <= TRANSFER_SAMPLES_RANGE[1]
            and P2P_SAMPLES_RANGE[0] <= options["p2p_samples"] <= P2P_SAMPLES_RANGE[1]
            and P2P_BYTES_RANGE[0] <= options["p2p_bytes"] <= P2P_BYTES_RANGE[1] and options["p2p_bytes"] % 4 == 0
            and 0 <= options["minimum_seconds"] <= 120 and options["command_timeout"] > options["minimum_seconds"]
            and 5 <= options["command_timeout"] <= 1800 and 30 <= options["compute_timeout"] <= 3600):
        raise ConfigurationError("invalid or unbounded PPU workload settings")
    try:
        HostCommands(options["privilege_command"])
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    return {"schema_version": 1, "kind": "toolkit", "suite": "thead-ppu-toolkit", "vendor": "thead", "chip": "PPU",
            "mode": "static-dry-run" if request.context.dry_run else "execution", "host_config": str(config_path),
            "cases": cases, "settings": options, "selection_request": request.context.selection_request(),
            "excluded_capabilities": sorted(contract_record(options)["excluded_capabilities"]),
            "runtime_lock": runtime_lock_record(config.get("runtime_profile"), vendor="thead"),
            "permissions": {"network": "none", "privileged_root": False, "ipc_namespace": "private", "pid_namespace": "private",
                            "read_only_root": True, "cap_drop": ["ALL"],
                            "foreign_handles": "not permitted; idle cards only (THEAD_ALLOW_FOREIGN_HANDLES never set by the toolkit)"},
            "monitoring": {"compute": request.compute_monitor, "data_movement": request.data_movement_monitor, "required_samples": 10},
            "logical_point_counts": {c: len(case_points(c, ids, options["payload_mib"], options["p2p_bytes"])) for c in cases},
            "repeat_count": options["repeat"],
            "point_counts": {c: len(case_points(c, ids, options["payload_mib"], options["p2p_bytes"])) * options["repeat"] for c in cases},
            "source": "official image comm_tools (sailbandwidth/p2pBandwidthLatencyTest) + isolated acblas children; no Base workload reuse",
            "identity_gate": "host ppu-smi UUID/PCI/minor -> selected nodes only -> tool self-reported BDF + child cudaDeviceProp UUID",
            "stages": ["image-identity", "preflight", "lease", "locked-preflight", "container-inspect", "official-tool-run", "monitor-finalize", "cleanup", "postflight", "lease-release", "report", "hash-index"]}


def inspect(config, selected, commands, directory):
    """Provider host inventory plus the toolkit's degraded-card gate."""
    result = get_provider("thead").inspect_host(config, selected, commands, directory)
    degraded = [d["host_physical_id"] for d in result["devices"] if any(d.get("uncorrectable_ecc_counts") or [])]
    if degraded:
        raise RuntimeError("selected cards carry uncorrectable ECC evidence: " + str(degraded))
    return result


def container_spec(root, host, image):
    mounts = [(str(BASE_DIR), "/workspace/FlagPerf/base", False),
              (str(root / "control"), "/run/flagperf", False),
              (str(root / "toolkit-evidence"), "/workspace/FlagPerf/results/toolkit-evidence", True)]
    devices = [(d["host_device_node"], d["container_node"], "rwm") for d in host["devices"]]
    # /dev/alixpu_ctl is part of the t9b-confirmed invocation envelope for the
    # comm_tools (probes/day9/toolkit/t9b_p2p_confirmed.txt).
    devices += [("/dev/alixpu", "/dev/alixpu", "rwm"), ("/dev/alixpu_ctl", "/dev/alixpu_ctl", "rwm")]
    args = ["--network=none", "--ipc=private", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only",
            "--pids-limit=512", "--shm-size=256m", "--tmpfs", "/tmp:rw,nosuid,size=1g",
            # cap-drop ALL removes DAC_OVERRIDE, so container root needs the host
            # user's group to write the group-writable toolkit-evidence mount
            # (same contract as the bounded-probe container).
            "--group-add", str(os.getgid()),
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONNOUSERSITE=1"]
    for source, target, permissions in devices:
        args += ["--device", source+":"+target+":"+permissions]
    for source, target, writable in mounts:
        args += ["--mount", f"type=bind,src={source},dst={target}" + ("" if writable else ",readonly")]
    args += ["--entrypoint", "/bin/bash", image, "/workspace/FlagPerf/base/toolkits/_common/thead/PPU/bootstrap.sh"]
    return args, {"image_id": image, "devices": devices, "mounts": mounts, "network": "none"}


def finalize_monitor(root, manifest, monitor, request):
    windows = []
    for case in manifest["cases"].values():
        for target in case["targets"]:
            if "measurement_window" in target and monitor is not None:
                win = target["measurement_window"]
                for binding in target["bindings"]:
                    windows.append({"device_id": "thead/"+binding["uuid"], "role": "measurement",
                                    "started_offset_s": win["started_monotonic_s"]-monitor.origin_monotonic_s,
                                    "finished_offset_s": win["finished_monotonic_s"]-monitor.origin_monotonic_s})
    if monitor is not None:
        monitor.finish(root, root / "toolkit-evidence/monitor", windows, [], min_samples_per_target=10,
                       window_semantics="official-tool and acblas completed-operation intervals; startup and JIT warmup excluded")
    for name, case in manifest["cases"].items():
        enabled = (request.compute_monitor if name.startswith("computation-") else request.data_movement_monitor) == "on"
        for target in case["targets"]:
            status, counts = "not-run", {}
            if target.get("measurement_status") == "passed" and monitor is not None and enabled:
                win = target["measurement_window"]
                invalid = 0
                for binding in target["bindings"]:
                    matching = [s for s in monitor.samples if s.get("device_id") == "thead/"+binding["uuid"]
                                and s["started_offset_s"]+monitor.origin_monotonic_s >= win["started_monotonic_s"]
                                and s["finished_offset_s"]+monitor.origin_monotonic_s <= win["finished_monotonic_s"]]
                    counts[str(binding["host_physical_id"])] = sum(s["valid"] for s in matching)
                    invalid += sum(not s["valid"] for s in matching)
                if win["role"] == "capacity-query":
                    # Capacity is a static query; its own raw query is its observation.
                    status = "passed"
                    target["monitor_scope"] = "static capacity/health query, no concurrent-load requirement"
                else:
                    status = "passed" if counts and min(counts.values()) >= 10 and not invalid else "partial"
                target["monitor_sample_counts"] = counts
                target["invalid_monitor_samples"] = invalid
            target["monitoring_status"] = status
            if target.get("status") != "resource-blocked":
                target["status"] = compose_status(target["measurement_status"], status, target.get("diagnosis_status", "not-run"))
            write_json(root / "toolkit-evidence/cases" / name / target["target"] / "metrics.json", target)
        case["monitoring_status"] = layer_status(t["monitoring_status"] for t in case["targets"])
        case["status"] = compose_status(case.get("measurement_status", "not-run"), case["monitoring_status"], case.get("diagnosis_status", "not-run"))
        write_json(root / "toolkit-evidence/cases" / name / "metrics.json", case)
    manifest["monitoring_status"] = layer_status(c["monitoring_status"] for c in manifest["cases"].values())
    manifest["diagnosis_status"] = layer_status(c.get("diagnosis_status", "not-run") for c in manifest["cases"].values())
    manifest["status"] = compose_status(manifest.get("measurement_status", "not-run"), manifest["monitoring_status"], manifest["diagnosis_status"])


def compose_status(measurement, monitoring, diagnosis):
    if measurement == "failed":
        return "failed"
    if (measurement, monitoring, diagnosis) == ("passed", "passed", "passed"):
        return "passed"
    return "partial"


def execute(request, static):
    _, config = load_host_config(request.context.config)
    options = static["settings"]
    if static["runtime_lock"]["image_manifest"]["validated"] is not True and not options["allow_candidate_runtime"]:
        raise ConfigurationError("PPU candidate requires --allow-candidate-runtime")
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    root = request.context.resolved_result_root(config) / run_id
    root.mkdir(parents=True, exist_ok=False)
    for name in ("control", "toolkit-evidence"):
        (root / name).mkdir()
    # Container root (with the host group added) writes evidence here while the
    # non-privileged host executor keeps enriching it after the run.
    os.chmod(root / "toolkit-evidence", 0o770)
    summary = {"schema_version": 1, "kind": "toolkit", "suite": "thead-ppu-toolkit", "vendor": "thead", "chip": "PPU",
               "run_id": run_id, "status": "running", "measurement_status": "not-run", "monitoring_status": "not-run",
               "diagnosis_status": "not-run", "started_at": now(), "resolved_plan": static,
               "qualification_status": "not-qualified", "lease_released": False}
    write_json(root / "summary.json", summary)
    write_json(root / "resolved-plan.json", static)
    commands = HostCommands(options["privilege_command"], deadline=time.monotonic()+request.context.timeout+180)
    container = ManagedContainer(commands, root, "flagperf-toolkit-"+run_id.lower(), run_id)
    selected = static["selection_request"]["requested_ids"]
    provider = get_provider("thead")
    lease = monitor = host = None
    stage = "image-identity"
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("termination requested")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        image = static["runtime_lock"]["image_manifest"]["image_id"]
        image_result = commands.checked(["docker", "image", "inspect", image], privileged=True, evidence=root / "image-inspect.json")
        records = json.loads(image_result["stdout"])
        if len(records) != 1:
            raise RuntimeError("image identity is ambiguous")
        identity = {k: records[0].get(k) for k in ("Id", "RepoDigests", "Architecture", "Created")}
        validate_runtime_identity(config, identity, allow_candidate=options["allow_candidate_runtime"])
        provider.validate_image(json.loads(Path(static["runtime_lock"]["image_manifest"]["path"]).read_text()), identity)
        write_json(root / "image-identity.json", identity)
        stage = "host-preflight"
        host = inspect(config, selected, commands, root / "host-preflight" / socket.gethostname())
        summary["selection"] = host
        stage = "lease"
        lease = DeviceLease([], run_id=run_id, kind="toolkit", **provider.lease_spec(host))
        lease.acquire()
        write_json(root / "lease.json", lease.record())
        stage = "locked-preflight"
        locked = inspect(config, selected, commands, root / "locked-preflight")
        provider.check_identity(host, locked)
        context = {"schema_version": 1, "run_id": run_id, "host": locked, "image_identity": identity,
                   "runtime_lock": static["runtime_lock"], "cases": static["cases"], "settings": options}
        write_json(root / "control/host-context.json", context)
        source_paths = [BASE_DIR / f for f in ("run.py", "generate_toolkit_report.py", "vendors/registry.py")]
        source_paths += [BASE_DIR / "executors" / f for f in ("thead_toolkit.py", "toolkit.py", "lifecycle.py", "common.py")]
        source_paths += [p for directory in ("toolkits/_common/thead/PPU", "vendors/thead", "monitoring")
                         for p in (BASE_DIR / directory).rglob("*")
                         if p.suffix in (".py", ".sh", ".json", ".yaml") and "evidence" not in p.parts and "__pycache__" not in p.parts]
        source_paths += [BASE_DIR / "toolkits" / c / "thead/PPU/main.sh" for c in static["cases"]]
        source_paths = sorted(set(source_paths))
        config_snapshot = root / "code-source/host-config.json"
        config_snapshot.parent.mkdir(parents=True, exist_ok=True)
        config_snapshot.write_bytes(Path(static["host_config"]).read_bytes())
        for p in source_paths:
            if p.is_file():
                destination = root / "code-source" / p.relative_to(BASE_DIR)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(p.read_bytes())
        write_json(root / "code-identity.json", {"git_head": commands.checked(["git", "-C", str(BASE_DIR.parent), "rev-parse", "HEAD"])["stdout"].strip(),
                   "git_diff": commands.checked(["git", "-C", str(BASE_DIR.parent), "diff", "--", "base/executors", "base/toolkits", "base/generate_toolkit_report.py"])["stdout"],
                   "sources": {p.relative_to(BASE_DIR).as_posix(): sha256(p) for p in source_paths if p.is_file()},
                   "context_sha256": sha256(root / "control/host-context.json"), "host_config_sha256": sha256(config_snapshot)})
        command = commands.run(["ppu-smi", "topo"], evidence=root / "toolkit-evidence/topology/host-topology.json")
        write_json(root / "toolkit-evidence/health/pre.json", host)
        write_json(root / "toolkit-evidence/diagnostics/coverage.json", {
                   "status": "passed",
                   "semantics": ("PPU diagnosis layer = toolchain compliance validation: isolated acblas child "
                                 "correctness readback per computation point, official-tool verification markers "
                                 "(sailbandwidth data check, p2p --verify data matched), contract coverage validation, "
                                 "and ppu-smi pre/post health observation"),
                   "vendor_threshold_diagnosis": {"status": "not-available",
                                                  "reason": "no ascend-dmi equivalent threshold (aiflops/bandwidth/hbm/signalQuality) tool in this toolchain"},
                   "available_observations": ["ppu-smi pre/post UUID/PCI", "uncorrectable ECC", "temperature", "power", "capacity", "topology"]})
        stage = "container-inspect"
        args, expected = container_spec(root, locked, image)
        write_json(root / "container-plan.json", {"arguments": args, "expected": expected})
        container.create(args, expected)
        if request.compute_monitor == "on" or request.data_movement_monitor == "on":
            monitor = provider.create_monitor(provider.monitor_targets(host))
            monitor.interval_s = 0.5
            monitor.start()
        stage = "official-tool-run"
        container.start()
        summary["container_exit_code"] = container.wait(time.monotonic()+request.context.timeout)
        if summary["container_exit_code"] not in (0, 1, 2):
            raise RuntimeError("official-tool runner exited unexpectedly: " + str(summary["container_exit_code"]))
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="failed", failure_stage=stage, error=str(exc), error_type=type(exc).__name__)
    finally:
        commands.deadline = time.monotonic()+90
        if monitor is not None:
            try:
                monitor.stop()
            except Exception as exc:
                summary["monitor_error"] = str(exc)
                summary["status"] = "failed"
        cleanup = container.cleanup()
        write_json(root / "cleanup.json", cleanup)
        summary["cleanup_status"] = cleanup["status"]
        if cleanup["container_absent"] is not True:
            summary.update(status="failed", recovery_required=True)
        if host is not None and cleanup["container_absent"] is True:
            try:
                post = inspect(config, selected, commands, root / "host-postflight")
                provider.check_identity(host, post)
                write_json(root / "toolkit-evidence/health/post.json", post)
                summary["postflight_status"] = "passed"
            except Exception as exc:
                summary.update(status="failed", postflight_status="failed", postflight_error=str(exc))
        if lease is not None:
            lease.release()
            summary["lease_released"] = True
        signal.signal(signal.SIGTERM, previous)
    manifest_path = root / "toolkit-evidence/manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("run_id") != run_id or manifest.get("context_sha256") != sha256(root / "control/host-context.json"):
            summary.update(status="failed", error="container evidence identity mismatch")
        try:
            finalize_monitor(root, manifest, monitor, request)
        except Exception as exc:
            manifest["monitoring_status"] = "failed"
            summary["monitor_error"] = str(exc)
            summary["status"] = "failed"
        summary.update({k: manifest[k] for k in ("measurement_status", "monitoring_status", "diagnosis_status")})
        if summary["status"] != "failed":
            summary["status"] = manifest["status"]
        manifest["status"] = summary["status"]
        write_json(manifest_path, manifest)
        summary["toolkit_evidence"] = {"path": "toolkit-evidence/manifest.json", "status": manifest["status"]}
    else:
        summary.update(status="failed", measurement_status="not-run")
        write_json(manifest_path, {"schema_version": 1, "suite": "thead-ppu-toolkit", "run_id": run_id,
                   "cases": {c: {"status": "not-run", "measurement_status": "not-run", "monitoring_status": "not-run",
                                 "diagnosis_status": "not-run", "metrics": [], "targets": [], "commands": [],
                                 "error": summary.get("error", "container did not produce evidence")} for c in static["cases"]},
                   "status": "failed", "measurement_status": "not-run", "monitoring_status": "not-run", "diagnosis_status": "not-run"})
    summary.update(finished_at=now(),
                   qualification_status="exploratory" if host and host.get("foreign_occupancy_observed") else "not-qualified",
                   **{key: static["runtime_lock"]["image_manifest"].get(key) for key in
                      ("release_stage", "validated", "validation_status", "validation_scope")})
    write_json(root / "summary.json", summary)
    from generate_toolkit_report import generate_and_record
    try:
        generate_and_record(root)
    except Exception as exc:
        summary["report_generation"] = {"status": "failed", "error": str(exc)}
        write_json(root / "summary.json", summary)
    index(root)
    print("Result directory:", root)
    if summary["status"] == "passed":
        return 0
    return 1 if summary["status"] == "failed" else 2
