#!/usr/bin/env python3
"""Container-side native Toolkit execution. No imports of PyTorch or Base workloads."""
import argparse
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
from toolkits._common.kunlunxin.P800.contract import (CASES, case_points, contract_record,
    layer_status, metric_from_native, validate_selection)
from toolkits._common.kunlunxin.P800.evidence import command, now, reference, sha256, write_json
from base.vendors.kunlunxin.preflight import query_record
from base.vendors.kunlunxin.reuse import mapping


def parse_record(stdout, prefix):
    rows = [json.loads(line[len(prefix):]) for line in stdout.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError("expected exactly one " + prefix + " record")
    return rows[0]


def resolve_native(host_devices, observed):
    validate_selection([d["host_physical_id"] for d in host_devices])
    if not isinstance(observed, list) or len(observed) != len(host_devices):
        raise ValueError("native inventory does not match exposed device count")
    pci = [mapping.normalize_bdf(d["pci_bdf"]) for d in observed]
    if len(set(pci)) != len(pci) or len({d["native_id"] for d in observed}) != len(observed):
        raise ValueError("duplicate native device identity")
    if set(pci) != {d["pci_bdf"] for d in host_devices}:
        raise ValueError("native XRE PCI set differs from selected host PCI set")
    return {d["host_physical_id"]: {**d, "native_id": observed[pci.index(d["pci_bdf"])]["native_id"]}
            for d in host_devices}


def env_for(prefix):
    package = Path(prefix) / "lib/python3.10/site-packages/torch_xmlir"
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "XPU_VISIBLE_DEVICES", "XPU_EVENT_KL3_ENABLE"):
        env.pop(key, None)
    # This executable uses native XRE IDs resolved through PCI, not framework ordinals.
    env.update(USE_FLAGGEMS="0", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               XDG_CACHE_HOME="/tmp/cache", HOME="/tmp", OMP_NUM_THREADS="1",
               LD_LIBRARY_PATH=":".join(str(package / p) for p in ("xre/so", ".", "xhpc/xblas/dependency_so")))
    return package, env


def prepare(root, context):
    package, env = env_for(context["conda_prefix"])
    xre, xblas = package / "xre", package / "xhpc/xblas"
    paths = {"xre_library": xre / "so/libxpurt.so", "xblas_library": package / "libxpu_blas.so",
             "xre_header": xre / "include/xpu/runtime.h", "xblas_header": xblas / "include/cublas_api.h",
             "test_dma": xre / "tools/test_dma", "test_memcpy_peer": xre / "tools/test_memcpy_peer",
             "xprofiler": xre / "profiler/xprofiler", "xpu-smi": xre / "bin/xpu-smi",
             "fc_effciency": xblas / "script/fc_effciency/run_eff.py",
             "vendor_versions": package / "_versions.txt"}
    provenance = {"tools": {}, "environment": {k: env[k] for k in ("LD_LIBRARY_PATH", "USE_FLAGGEMS", "HOME")},
                  "backend": "independent-native-xre-xblas", "no_base_workload": True,
                  "official_fc_effciency_status": "not-run; matching XBLAS unittest not supplied; native XBLAS API adapter used",
                  "native_source": reference(HERE / "native_bench.cpp", BASE),
                  "runtime_lock": context["runtime_lock"], "context_sha256": context["context_sha256"]}
    for label, p in paths.items():
        provenance["tools"][label] = {"path": str(p), "present": p.is_file()}
        if p.is_file():
            provenance["tools"][label].update(sha256=sha256(p), bytes=p.stat().st_size, resolved_path=str(p.resolve()))
    write_json(root / "provenance.json", provenance)
    write_json(root / "toolkit-contract.json", contract_record())
    for name in ("test_dma", "test_memcpy_peer", "xprofiler"):
        if paths[name].is_file():
            command([str(paths[name]), "-h"], root / "environment" / name, root, stem="help", timeout=15, env=env)
    exe = root / "build/native_bench"
    compiler = shutil.which("g++")
    if not compiler or not all(paths[k].is_file() for k in ("xre_library", "xblas_library", "xre_header", "xblas_header")):
        raise FileNotFoundError("native compiler/XRE/XBLAS build dependency missing (see provenance.json)")
    argv = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", str(HERE / "native_bench.cpp"),
            "-I"+str(xblas / "include"), "-I"+str(xre / "include"), "-L"+str(package), "-lxpu_blas",
            "-L"+str(xre / "so"), "-lxpurt", "-lcudart", "-ldl", "-pthread", "-o", str(exe)]
    build = command(argv, exe.parent, root, stem="compile", timeout=120, env=env)
    if build["returncode"] != 0:
        raise RuntimeError("native Toolkit compile failed; see build/compile.stderr")
    command(["ldd", str(exe)], exe.parent, root, stem="ldd", timeout=15, env=env)
    provenance["native_binary"] = reference(exe, root)
    write_json(root / "provenance.json", provenance)
    inventory = command([str(exe), "--mode", "inventory"], root / "environment", root, stem="native-inventory", timeout=30, env=env)
    if inventory["returncode"]:
        raise RuntimeError("native XRE inventory failed")
    observed = parse_record((root / inventory["stdout"]["path"]).read_text(), "P800_INVENTORY ")
    bindings = resolve_native(context["host"]["devices"], observed)
    write_json(root / "runtime-bindings.json", {"schema_version": 1, "method": "host query UUID+PCI+minor to native XRE PCI join",
                                              "context_sha256": context["context_sha256"], "devices": list(bindings.values())})
    return exe, env, paths, bindings


def target_name(point, repetition):
    label = "physical-" + str(point["source"])
    if point["mode"] == "d2d-kernel":
        label += "-d2d-kernel"
    if "destination" in point:
        label += "-to-" + str(point["destination"])
        label += "-" + point["direction"]
    if "bytes" in point:
        label += "-" + str(point["bytes"]) + "B"
    if point["mode"] in ("h2d", "d2h"):
        label += "-" + ("pinned" if point["pinned"] else "pageable") + "-" + ("nonblocking" if point["async"] else "blocking")
    return label + "/repeat-" + str(repetition)


def native_argv(exe, point, bindings, settings, case):
    src = bindings[point["source"]]
    bdf = lambda d: "0000" + d["pci_bdf"]  # native header returns 8-digit domain
    args = [str(exe), "--mode", point["mode"], "--src", str(src["native_id"]), "--src-pci", bdf(src),
            "--samples", str(settings["samples"]), "--warmup", str(settings["warmup"]),
            "--minimum-seconds", str(settings["minimum_seconds"]), "--dimension", str(settings["matrix_size"])]
    if "destination" in point:
        dst = bindings[point["destination"]]
        args += ["--dst", str(dst["native_id"]), "--dst-pci", bdf(dst)]
    if "bytes" in point:
        args += ["--bytes", str(point["bytes"]), "--pinned", str(int(point["pinned"])), "--async", str(int(point["async"])),
                 "--pinned-api", settings["pinned_api"], "--async-chunk-bytes", str(settings["async_chunk_bytes"])]
    if "dtype" in point:
        args += ["--dtype", point["dtype"]]
    return args


def run_point(root, case, point, repetition, prepared, settings):
    exe, env, paths, bindings = prepared
    name = target_name(point, repetition)
    directory = root / "cases" / case / name
    directory.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "target": name, "point": point, "repetition": repetition,
              "commands": [], "metrics": [], "status": "failed", "measurement_status": "failed",
              "monitoring_status": "pending-host-finalization", "diagnosis_status": "not-supported",
              "sweep_scope": "explicit physical cards/mode/payload; native operation", "duration_s": 0,
              "bindings": [bindings[point[k]] for k in ("source", "destination") if k in point]}
    try:
        if point["mode"] == "capacity-query":
            smi = str(paths["xpu-smi"]) if paths["xpu-smi"].is_file() else "xpu-smi"
            argv = [smi, "-i", str(bindings[point["source"]]["native_id"]), "-q"]
        else:
            argv = native_argv(exe, point, bindings, settings, case)
        record = command(argv, directory, root, timeout=settings["command_timeout"], env=env)
        result["commands"].append(record)
        result["duration_s"] = record["duration_s"]
        if record["returncode"]:
            raise RuntimeError("native command exited " + str(record["returncode"]))
        stdout = (root / record["stdout"]["path"]).read_text(errors="replace")
        if point["mode"] == "capacity-query":
            identity = query_record(stdout, bindings[point["source"]]["pci_bdf"])
            if identity["uuid"] != bindings[point["source"]]["uuid"]:
                raise ValueError("capacity query UUID differs from selected card")
            result["metrics"] = [{"field": "Memory Usage/"+key, "source": record["stdout"]["path"],
                                  "source_kind": "xpu-smi", "value": identity[key], "unit": "MiB", "point": point,
                                  "scope": "reported capacity query; not held allocation or OOM stress"}
                                 for key in ("total_memory_mib", "used_memory_mib", "free_memory_mib")]
            result["measurement_window"] = {"started_monotonic_s": record["started_monotonic_s"],
                                             "finished_monotonic_s": record["finished_monotonic_s"], "role": "capacity-query"}
        else:
            native = parse_record(stdout, "P800_METRIC ")
            metric = metric_from_native(native, point, case, record["stdout"]["path"], settings["matrix_size"], settings["samples"])
            metric["repetition"] = repetition
            if point.get("pinned"):
                metric["host_pinned_api"] = settings["pinned_api"]
            result["metrics"] = [metric]
            result["measurement_window"] = {"started_monotonic_s": native["started_monotonic_ns"]/1e9,
                                             "finished_monotonic_s": native["finished_monotonic_ns"]/1e9,
                                             "role": "measurement"}
            write_json(directory / "samples.json", native)
        result.update(status="partial", measurement_status="passed")
    except Exception as exc:
        result["error"] = str(exc)
    write_json(directory / "metrics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw_context = args.context.read_bytes()
    context = json.loads(raw_context)
    import hashlib
    context["context_sha256"] = hashlib.sha256(raw_context).hexdigest()
    ids = [d["host_physical_id"] for d in context["host"]["devices"]]
    validate_selection(ids)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError("result evidence already exists")
    settings, cases = context["settings"], context["cases"]
    manifest = {"schema_version": 1, "kind": "toolkit", "suite": "kunlunxin-p800-toolkit", "vendor": "kunlunxin",
                "run_id": context["run_id"], "started_at": now(), "context_sha256": context["context_sha256"],
                "selected_physical_device_ids": ids, "settings": settings, "cases": {}, "status": "running",
                "monitoring_status": "pending-host-finalization", "diagnosis_status": "not-supported"}
    write_json(root / "environment.json", {"host": context["host"], "conda_prefix": context["conda_prefix"],
                                          "settings": settings, "image_identity": context["image_identity"]})
    try:
        prepared = prepare(root, context)
    except Exception as exc:
        manifest["setup_error"] = str(exc)
        prepared = None
    for case in cases:
        points = case_points(case, ids, settings["payload_bytes"], settings.get("p2p_payload_bytes", 33554432))
        targets = []
        if not points or prepared is None:
            reason = manifest.get("setup_error", "P2P requires at least two selected physical cards")
            directory = root / "cases" / case / "unavailable"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "microbenchmark.stdout").write_text("")
            (directory / "microbenchmark.stderr").write_text(reason + "\n")
            targets.append({"target": "unavailable", "commands": [], "metrics": [], "status": "failed" if prepared is None else "resource-blocked",
                            "measurement_status": "failed" if prepared is None else "not-run", "monitoring_status": "not-run", "diagnosis_status": "not-supported",
                            "error": reason, "sweep_scope": "requested case not measured", "duration_s": 0})
            write_json(directory / "metrics.json", targets[-1])
        else:
            for point in points:
                for repetition in range(1, settings["repeat"]+1):
                    targets.append(run_point(root, case, point, repetition, prepared, settings))
                    print(case, targets[-1]["target"], targets[-1]["measurement_status"], flush=True)
        result = {"targets": targets, "commands": [cmd for t in targets for cmd in t["commands"]],
                  "metrics": [m for t in targets for m in t["metrics"]], "duration_s": sum(t["duration_s"] for t in targets),
                  "measurement_status": layer_status(t["measurement_status"] for t in targets),
                  "monitoring_status": "pending-host-finalization", "diagnosis_status": "not-supported",
                  "status": "partial", "sweep_scope": {"physical_ids": ids, "point_count": len(points), "repeat": settings["repeat"]}}
        if result["measurement_status"] == "failed":
            result["status"] = "failed"
        manifest["cases"][case] = result
        write_json(root / "cases" / case / "metrics.json", result)
        write_json(root / "manifest.json", manifest)
    manifest.update(finished_at=now(), measurement_status=layer_status(c["measurement_status"] for c in manifest["cases"].values()))
    manifest["status"] = "failed" if manifest["measurement_status"] == "failed" else "partial"
    write_json(root / "manifest.json", manifest)
    return 1 if manifest["status"] == "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
