"""P800 Toolkit host lifecycle: locked image, selected nodes, leases and evidence."""
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
from base.vendors.kunlunxin import preflight
from base.vendors.kunlunxin.reuse import mapping
from toolkits._common.kunlunxin.P800.contract import CASES, case_points, layer_status, validate_selection
from toolkits._common.kunlunxin.P800.evidence import index, now, sha256, write_json

DEFAULTS = {"allow_candidate_runtime": False, "allow_busy_devices": False,
            "privilege_command": "", "samples": 50, "warmup": 5, "repeat": 1,
            "payload_bytes": 536870912, "matrix_size": 2048, "minimum_seconds": 0.0,
            "command_timeout": 180, "smoke": False, "pinned_api": "register", "async_chunk_bytes": 1048576}


def add_cli_arguments(parser):
    parser.add_argument("--allow-candidate-runtime", action="store_true", help="P800: run the locked candidate without promoting it")
    parser.add_argument("--allow-busy-devices", action="store_true", help="P800: explicitly permit exploratory use of occupied selected cards")
    parser.add_argument("--privilege-command", default="", help="P800 host Docker/fuser prefix: empty or 'sudo -n'")
    parser.add_argument("--samples", type=int, default=50, help="P800 native timing samples per point (D2D has at least 25)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1, help="P800 independent processes per point; use 5 for repeat groups")
    parser.add_argument("--payload-bytes", type=int, default=536870912)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--minimum-seconds", type=float, default=0.0, help="P800 explicit timed workload duration, recorded as changed scope")
    parser.add_argument("--command-timeout", type=int, default=180)
    parser.add_argument("--smoke", action="store_true", help="P800 bounded correctness smoke: 1 MiB bandwidth and GEMM 128; no formal qualification")
    parser.add_argument("--pinned-api", choices=("register", "alloc"), default="register", help="P800 pinned host memory API; recorded, never changed implicitly")
    parser.add_argument("--async-chunk-bytes", type=int, default=1048576, help="P800 pinned async copy submission chunk; 0 requests one whole-payload API call")


def settings_for(request):
    options = {**DEFAULTS, **(request.p800_options or {})}
    if options["smoke"]:
        options.update(payload_bytes=1048576, matrix_size=128, samples=25, warmup=2, minimum_seconds=0.0)
    return options


def plan(request, config_path, config):
    if request.context.physical_device_ids is None:
        raise ConfigurationError("P800 Toolkit requires explicit --physical-device-ids; legacy aliases are ambiguous")
    ids = request.context.selection_request()["requested_ids"]
    try:
        validate_selection(ids)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    if not set(ids).issubset(config["expected_device_ids"]):
        raise ConfigurationError("physical selection is outside the locked inventory")
    cases = list(request.cases or CASES)
    if len(cases) != len(set(cases)) or set(cases)-set(CASES):
        raise ConfigurationError("select unique registered P800 Toolkit cases")
    if request.legacy_probe or request.allow_privileged_root or request.allow_disruptive_dmi:
        raise ConfigurationError("Ascend legacy/privileged/DMI flags do not apply to P800 Toolkit")
    if request.latency_sizes is not None or request.hccl_min_bytes != "8K" or request.hccl_max_bytes != "1G":
        raise ConfigurationError("P800 uses the frozen four-point latency sweep; HCCL options do not apply")
    options = settings_for(request)
    if options["pinned_api"] not in ("register", "alloc"):
        raise ConfigurationError("P800 pinned API must be register or alloc")
    if not 0 <= options["async_chunk_bytes"] <= 1073741824:
        raise ConfigurationError("P800 async chunk size must be 0..1 GiB")
    if not (1 <= options["repeat"] <= 20 and 25 <= options["samples"] <= 100000 and 0 <= options["warmup"] <= 100
            and 1 <= options["payload_bytes"] <= 1073741824 and 8 <= options["matrix_size"] <= 8192
            and options["matrix_size"] % 8 == 0 and 0 <= options["minimum_seconds"] <= 120
            and 5 <= options["command_timeout"] <= 600 and options["command_timeout"] > options["minimum_seconds"]):
        raise ConfigurationError("invalid or unbounded P800 workload settings")
    try:
        HostCommands(options["privilege_command"])
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    return {"schema_version": 1, "kind": "toolkit", "suite": "kunlunxin-p800-toolkit", "vendor": "kunlunxin", "chip": "P800",
            "mode": "static-dry-run" if request.context.dry_run else "execution", "host_config": str(config_path),
            "cases": cases, "settings": options, "selection_request": request.context.selection_request(),
            "excluded_physical_ids": [1], "runtime_lock": runtime_lock_record(config.get("runtime_profile"), vendor="kunlunxin"),
            "permissions": {"network": "none", "privileged_root": False, "ipc_namespace": "private", "pid_namespace": "private",
                            "read_only_root": True, "cap_drop": ["ALL"], "allow_busy_devices": options["allow_busy_devices"]},
            "monitoring": {"compute": request.compute_monitor, "data_movement": request.data_movement_monitor, "required_samples": 10},
            "point_counts": {c: len(case_points(c, ids, options["payload_bytes"])) * options["repeat"] for c in cases},
            "source": "independent native XRE/XBLAS Toolkit microbenchmark; no Base workload reuse",
            "identity_gate": "host UUID/PCI/minor -> selected nodes -> native XRE PCI join before allocating",
            "stages": ["image-identity", "preflight", "lease", "locked-preflight", "container-inspect", "native-run", "monitor-finalize", "cleanup", "postflight", "lease-release", "report", "hash-index"]}


def inspect(config, selected, commands, directory, allow_busy):
    directory.mkdir(parents=True, exist_ok=False)
    machine = commands.checked(["xpu-smi", "-m"], evidence=directory / "machine.json")
    inventory = preflight.machine_inventory(machine["stdout"])
    if sorted(d["physical_id"] for d in inventory) != sorted(config["expected_device_ids"]):
        raise RuntimeError("physical inventory changed from locked host config")
    lock = json.loads((get_provider("kunlunxin").runtime_root(BASE_DIR, config["runtime_profile"]) / "stack.lock.yaml").read_text())
    devices = []
    for physical in selected:
        row = next(d for d in inventory if d["physical_id"] == physical)
        query = commands.checked(["xpu-smi", "-i", str(physical), "-q"], evidence=directory / f"query-{physical}.json")
        info = preflight.query_record(query["stdout"], row["pci_bdf"])
        if any(info[k] != lock["host_observed"][k] for k in ("driver", "xpu_smi_runtime")):
            raise RuntimeError("driver/runtime differs from locked profile")
        if row["total_memory_mib"] != info["total_memory_mib"]:
            raise RuntimeError("machine/query capacity units disagree")
        ecc = [int(v) for v in re.findall(r"DRAM Uncorrectable\s*:\s*(\d+)", query["stdout"])]
        if any(ecc):
            raise RuntimeError("selected card has uncorrectable ECC evidence: " + str(physical))
        handles = commands.run(["fuser", info["host_device_node"]], privileged=True,
                               evidence=directory / f"handles-{physical}.json")
        if handles["returncode"] not in (0, 1) or handles.get("timed_out"):
            raise RuntimeError("cannot inspect selected device handles")
        busy = bool(row["used_memory_mib"] or info["used_memory_mib"] or row["utilization_percent"] or info["utilization_percent"] or not info["processes_empty"])
        if handles["returncode"] == 0:
            if not handles["stdout"].split() or any(not p.isdigit() for p in handles["stdout"].split()):
                raise RuntimeError("unrecognized device handle owners")
            # For an idle run, retain the original verified-read-only-observer policy.
            if not allow_busy:
                preflight.inspect_handles({**info, "host_physical_id": physical}, commands, directory, True)
            else:
                busy = True
        elif handles["stdout"].strip() or handles["stderr"].strip():
            raise RuntimeError("device handle query returned unexpected error output")
        if busy and not allow_busy:
            raise RuntimeError("selected card is occupied; --allow-busy-devices is required: " + str(physical))
        devices.append({**info, "host_physical_id": physical, "pci_bdf": row["pci_bdf"], "serial": row["serial"],
                        "container_node": info["host_device_node"], "foreign_occupancy_observed": busy,
                        "uncorrectable_ecc_counts": ecc, "foreign_handles_authorized": allow_busy})
    mapping.validate_device_set(devices)
    preflight.check_nodes(devices)
    result = {"schema_version": 1, "status": "passed", "devices": devices,
              "selection": {"source": "physical-device-ids", "requested_ids": selected},
              "foreign_occupancy_observed": any(d["foreign_occupancy_observed"] for d in devices),
              "foreign_handles_authorized": allow_busy}
    write_json(directory / "summary.json", result)
    return result


def container_spec(root, host, image):
    mounts = [(str(BASE_DIR), "/workspace/FlagPerf/base", False),
              (str(root / "control"), "/run/flagperf", False),
              (str(root / "toolkit-evidence"), "/workspace/FlagPerf/results/toolkit-evidence", True)]
    devices = [(d["host_device_node"], d["container_node"], "rwm") for d in host["devices"]] + [("/dev/xpuctrl", "/dev/xpuctrl", "rwm")]
    args = ["--network=none", "--ipc=private", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only", "--pids-limit=256",
            "--shm-size=128m", "--tmpfs", "/tmp:rw,nosuid,size=1g", "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONNOUSERSITE=1"]
    for source, target, permissions in devices:
        args += ["--device", source+":"+target+":"+permissions]
    for source, target, writable in mounts:
        args += ["--mount", f"type=bind,src={source},dst={target}" + ("" if writable else ",readonly")]
    args += ["--entrypoint", "/bin/bash", image, "/workspace/FlagPerf/base/toolkits/_common/kunlunxin/P800/bootstrap.sh"]
    return args, {"image_id": image, "devices": devices, "mounts": mounts, "network": "none"}


def finalize_monitor(root, manifest, monitor, request):
    windows = []
    for case in manifest["cases"].values():
        for target in case["targets"]:
            if "measurement_window" in target and monitor is not None:
                win = target["measurement_window"]
                for binding in target["bindings"]:
                    windows.append({"device_id": "kunlunxin/"+binding["uuid"], "role": "measurement",
                                    "started_offset_s": win["started_monotonic_s"]-monitor.origin_monotonic_s,
                                    "finished_offset_s": win["finished_monotonic_s"]-monitor.origin_monotonic_s})
    if monitor is not None:
        monitor.finish(root, root / "toolkit-evidence/monitor", windows, [], min_samples_per_target=10,
                       window_semantics="native completed-operation intervals; initialization excluded")
    for name, case in manifest["cases"].items():
        enabled = (request.compute_monitor if name.startswith("computation-") else request.data_movement_monitor) == "on"
        for target in case["targets"]:
            status, counts = "not-run", {}
            if target.get("measurement_status") == "passed" and monitor is not None and enabled:
                win = target["measurement_window"]
                invalid = 0
                for binding in target["bindings"]:
                    matching = [s for s in monitor.samples if s.get("device_id") == "kunlunxin/"+binding["uuid"]
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
            target["status"] = "failed" if target["measurement_status"] == "failed" else "partial"
            write_json(root / "toolkit-evidence/cases" / name / target["target"] / "metrics.json", target)
        case["monitoring_status"] = layer_status(t["monitoring_status"] for t in case["targets"])
        write_json(root / "toolkit-evidence/cases" / name / "metrics.json", case)
    manifest["monitoring_status"] = layer_status(c["monitoring_status"] for c in manifest["cases"].values())


def execute(request, static):
    _, config = load_host_config(request.context.config)
    options = static["settings"]
    if static["runtime_lock"]["image_manifest"]["validated"] is not True and not options["allow_candidate_runtime"]:
        raise ConfigurationError("P800 candidate requires --allow-candidate-runtime")
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    root = request.context.resolved_result_root(config) / run_id
    root.mkdir(parents=True, exist_ok=False)
    for name in ("control", "toolkit-evidence"):
        (root / name).mkdir()
    summary = {"schema_version": 1, "kind": "toolkit", "suite": "kunlunxin-p800-toolkit", "vendor": "kunlunxin", "chip": "P800",
               "run_id": run_id, "status": "running", "measurement_status": "not-run", "monitoring_status": "not-run",
               "diagnosis_status": "not-supported", "started_at": now(), "resolved_plan": static,
               "qualification_status": "not-qualified", "lease_released": False}
    write_json(root / "summary.json", summary)
    write_json(root / "resolved-plan.json", static)
    commands = HostCommands(options["privilege_command"], deadline=time.monotonic()+request.context.timeout+180)
    container = ManagedContainer(commands, root, "flagperf-toolkit-"+run_id.lower(), run_id)
    selected = static["selection_request"]["requested_ids"]
    provider = get_provider("kunlunxin")
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
        host = inspect(config, selected, commands, root / "host-preflight" / socket.gethostname(), options["allow_busy_devices"])
        summary["selection"] = host
        stage = "lease"
        lease = DeviceLease([], run_id=run_id, kind="toolkit", **provider.lease_spec(host))
        lease.acquire()
        write_json(root / "lease.json", lease.record())
        stage = "locked-preflight"
        locked = inspect(config, selected, commands, root / "locked-preflight", options["allow_busy_devices"])
        provider.check_identity(host, locked)
        lock = json.loads(Path(static["runtime_lock"]["stack_lock"]["path"]).read_text())
        context = {"schema_version": 1, "run_id": run_id, "host": locked, "image_identity": identity,
                   "runtime_lock": static["runtime_lock"], "conda_prefix": lock["conda_prefix"], "cases": static["cases"], "settings": options}
        write_json(root / "control/host-context.json", context)
        source_paths = [BASE_DIR / f for f in ("run.py", "generate_toolkit_report.py", "vendors/registry.py")]
        source_paths += [BASE_DIR / "executors" / f for f in ("p800_toolkit.py", "toolkit.py", "lifecycle.py", "common.py")]
        source_paths += [p for directory in ("toolkits/_common/kunlunxin/P800", "vendors/kunlunxin", "monitoring")
                         for p in (BASE_DIR / directory).rglob("*")
                         if p.suffix in (".py", ".cpp", ".sh", ".json", ".yaml") and "evidence" not in p.parts and "__pycache__" not in p.parts]
        source_paths += [BASE_DIR / "toolkits" / c / "kunlunxin/P800/main.sh" for c in static["cases"]]
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
        command = commands.run(["xpu-smi", "topo", "-m"], evidence=root / "toolkit-evidence/topology/host-topology.json")
        write_json(root / "toolkit-evidence/health/pre.json", host)
        write_json(root / "toolkit-evidence/diagnostics/coverage.json", {"status": "not-supported",
                   "reason": "No equivalent vendor threshold aiflops/bandwidth/hbm/signalQuality diagnosis in this toolchain",
                   "available_observations": ["xpu-smi pre/post UUID/PCI", "uncorrectable ECC", "temperature", "power", "capacity", "topology"],
                   "xprofiler": "tool availability/help archived; kernel trace not claimed by this measurement mode"})
        stage = "container-inspect"
        args, expected = container_spec(root, locked, image)
        write_json(root / "container-plan.json", {"arguments": args, "expected": expected})
        container.create(args, expected)
        if request.compute_monitor == "on" or request.data_movement_monitor == "on":
            monitor = provider.create_monitor(provider.monitor_targets(host))
            monitor.interval_s = 0.5
            monitor.start()
        stage = "native-run"
        container.start()
        summary["container_exit_code"] = container.wait(time.monotonic()+request.context.timeout)
        if summary["container_exit_code"] not in (0, 1, 2):
            raise RuntimeError("native runner exited unexpectedly: " + str(summary["container_exit_code"]))
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
                post = inspect(config, selected, commands, root / "host-postflight", options["allow_busy_devices"])
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
        write_json(manifest_path, {"schema_version": 1, "suite": "kunlunxin-p800-toolkit", "run_id": run_id,
                   "cases": {c: {"status": "not-run", "measurement_status": "not-run", "monitoring_status": "not-run",
                                 "diagnosis_status": "not-supported", "metrics": [], "targets": [], "commands": [],
                                 "error": summary.get("error", "container did not produce evidence")} for c in static["cases"]},
                   "status": "failed", "measurement_status": "not-run", "monitoring_status": "not-run", "diagnosis_status": "not-supported"})
    summary.update(finished_at=now(), qualification_status="exploratory" if host and host["foreign_occupancy_observed"] else "not-qualified",
                   release_stage="candidate", validated=False)
    write_json(root / "summary.json", summary)
    from generate_toolkit_report import generate_and_record
    try:
        generate_and_record(root)
    except Exception as exc:
        summary["report_generation"] = {"status": "failed", "error": str(exc)}
        write_json(root / "summary.json", summary)
    index(root)
    print("Result directory:", root)
    return 1 if summary["status"] == "failed" else 2
