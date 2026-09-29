"""P800 Toolkit measurement contract; importing this module never touches devices."""
from itertools import combinations, product
import math
import statistics

CASES = tuple("computation-" + x for x in ("BF16", "FP16", "FP32", "INT8")) + (
    "main_memory-bandwidth", "main_memory-capacity", "interconnect-h2d",
    "interconnect-d2h", "interconnect-h2d-latency", "interconnect-d2h-latency",
    "interconnect-P2P_intraserver", "interconnect-P2P_intraserver-latency",
)
LATENCY_BYTES = (512, 4096, 65536, 1048576)
MODES = tuple(product((False, True), (False, True)))
EXCLUDED_PHYSICAL_IDS = (1,)


def validate_selection(ids):
    if not ids or any(type(i) is not int or i < 0 or i > 7 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("P800 requires unique physical IDs in 0..7")
    if set(ids).intersection(EXCLUDED_PHYSICAL_IDS):
        raise ValueError("physical card 1 is excluded by the operator; ranges containing 1 are also rejected")


def case_points(case, devices, payload_bytes=536870912):
    """Generate every requested physical card, mode, size and unordered pair."""
    validate_selection(devices)
    if case not in CASES:
        raise ValueError("unregistered P800 Toolkit case: " + case)
    if "P2P" in case:
        directions = ("single-direction",) if case.endswith("latency") else ("single-direction", "bidirectional")
        return [{"source": a, "destination": b, "bytes": 65536 if case.endswith("latency") else min(payload_bytes, 33554432),
                 "mode": "p2p" if direction == "single-direction" else "p2p-bidir", "direction": direction, "pinned": False, "async": False}
                for (a, b), direction in product(combinations(sorted(devices), 2), directions)]
    if case.startswith("computation-"):
        return [{"source": i, "mode": "gemm", "dtype": case.split("-")[1]} for i in devices]
    if case == "main_memory-capacity":
        return [{"source": i, "mode": "capacity-query"} for i in devices]
    if case == "main_memory-bandwidth":
        return [{"source": i, "mode": "d2d", "bytes": payload_bytes, "pinned": False, "async": False} for i in devices]
    direction = "h2d" if "h2d" in case else "d2h"
    sizes = LATENCY_BYTES if case.endswith("latency") else (payload_bytes,)
    return [{"source": i, "mode": direction, "bytes": size, "pinned": pin, "async": asynchronous}
            for i, size, (pin, asynchronous) in product(devices, sizes, MODES)]


def validate_native(record, point, dimension, samples):
    if record.get("schema_version") != 1 or record.get("mode") != point["mode"]:
        raise ValueError("native schema/mode mismatch")
    if record.get("correctness") is not True or record.get("checked_values", 0) <= 0:
        raise ValueError("native correctness evidence missing")
    values = record.get("samples_ns")
    if not isinstance(values, list) or len(values) < samples or any(
        type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in values
    ):
        raise ValueError("timings must contain the requested finite positive samples")
    start, end = record.get("started_monotonic_ns"), record.get("finished_monotonic_ns")
    if type(start) is not int or type(end) is not int or start >= end:
        raise ValueError("native measurement interval missing or invalid")
    if sum(values) > (end-start)*1.01:
        raise ValueError("sample times exceed the measurement interval")
    if point["mode"] == "gemm":
        if record.get("dimension") != dimension:
            raise ValueError("GEMM dimension mismatch")
        if record.get("dtype") != point["dtype"] or record.get("output_dtype") != "FP32":
            raise ValueError("GEMM dtype/output mismatch")
    elif record.get("payload_bytes") != point["bytes"]:
        raise ValueError("copy payload mismatch")
    expected_checked = (dimension**2 if dimension <= 128 else 257) if point["mode"] == "gemm" else point["bytes"] * (2 if point["mode"] == "p2p-bidir" else 1)
    if type(record.get("checked_values")) is not int or record["checked_values"] != expected_checked:
        raise ValueError("correctness coverage does not match the measured operation")
    chunk = record.get("async_chunk_bytes", 0)
    calls = record.get("api_calls_per_sample", 1)
    if type(chunk) is not int or chunk < 0 or type(calls) is not int or calls < 1:
        raise ValueError("invalid native submission metadata")
    if chunk:
        if point["mode"] not in ("h2d", "d2h") or not point.get("pinned") or not point.get("async") or chunk >= point["bytes"]:
            raise ValueError("chunking is only valid for pinned async copies larger than the chunk")
        if calls != (point["bytes"] + chunk - 1) // chunk:
            raise ValueError("native chunk submission count mismatch")
    elif calls != (2 if point["mode"] == "p2p-bidir" else 1):
        raise ValueError("native copy submission count does not match the operation")
    return values


def metric_from_native(record, point, case, source, dimension, samples):
    values = validate_native(record, point, dimension, samples)
    median_ns = statistics.median(values)
    if point["mode"] == "gemm":
        value, unit, formula = 2 * dimension**3 / median_ns / 1000, ("TOPS" if point["dtype"] == "INT8" else "TFLOPS"), "2*M*N*K / median_elapsed_ns / 1000"
    elif case.endswith("latency"):
        value, unit, formula = median_ns, "ns", "median(completed API copy + synchronization nanoseconds)"
    else:
        multiplier = 2 if point["mode"] == "p2p-bidir" else 1
        value, unit, formula = multiplier * point["bytes"] / median_ns, "GB/s", "completed payload bytes / median_elapsed_ns (decimal GB/s)"
        if multiplier == 2:
            formula = "2*payload_bytes / independently measured concurrent two-direction completion time; not twice the unidirectional result"
    cv = statistics.pstdev(values) / statistics.mean(values) if len(values) > 1 else None
    return {"field": "P800_METRIC/samples_ns", "source": source, "source_kind": "native-xblas" if point["mode"] == "gemm" else "native-xre",
            "value": value, "unit": unit, "formula": formula, "sample_count": len(values),
            "median_ns": median_ns, "min_ns": min(values), "max_ns": max(values), "sample_cv": cv,
            "stability_status": "unstable" if cv is not None and cv > .05 else "observed",
            "timer": record["timer"], "correctness": True, "checked_values": record["checked_values"],
            "point": point, "physical_device_id": point["source"],
            "scope": "completed native API operation; not theoretical peak or device-only kernel time",
            "compute_semantics": ("INT8 inputs/TGEMM via XBLAS fc_fusion; maxima=127; alpha=1 beta=0; LINEAR; no bias; FP32 output"
                                  if point.get("dtype") == "INT8" else "XBLAS GemmEx FP32 accumulation/output" if point["mode"] == "gemm" else None),
            "api_calls_per_sample": record.get("api_calls_per_sample", 1), "async_chunk_bytes": record.get("async_chunk_bytes", 0),
            "submission_scope": "chunked pinned async logical payload" if record.get("async_chunk_bytes", 0) else "single native operation"}


def layer_status(statuses):
    values = list(statuses)
    if not values:
        return "not-run"
    if any(s == "failed" for s in values):
        return "failed"
    if all(s == "passed" for s in values):
        return "passed"
    if len(set(values)) == 1:
        return values[0]
    return "partial"


def contract_record():
    return {"schema_version": 1, "vendor": "kunlunxin", "chip": "P800", "cases": list(CASES),
            "excluded_physical_ids": list(EXCLUDED_PHYSICAL_IDS), "bandwidth_payload_bytes": 536870912,
            "latency_payload_bytes": list(LATENCY_BYTES), "p2p_latency_payload_bytes": 65536, "p2p_bandwidth_payload_bytes": 33554432,
            "transfer_variants": ["pageable-blocking", "pageable-nonblocking", "pinned-blocking", "pinned-nonblocking"],
            "pinned_api": "posix_memalign + xpu_host_register; optional xpu_host_alloc",
            "pinned_async_default_chunk_bytes": 1048576,
            "pinned_async_scope": "logical payload split into explicit native submissions with one completion wait; --async-chunk-bytes 0 preserves failed whole-payload reproduction",
            "p2p_scope": "all unordered pairs: ascending unidirectional plus separately timed concurrent bidirectional; no reverse inference",
            "capacity_scope": "per-card xpu-smi reported HBM capacity, not allocatable capacity/OOM stress",
            "d2d_scope": "copy payload bytes, no read+write factor of two",
            "computation": {"backend": "independent Toolkit C++ XBLAS cublasGemmEx (floating point) and fc_fusion (INT8); no Base or PyTorch workload",
                            "output": {"FP32": "FP32", "FP16": "FP32", "BF16": "FP32", "INT8": "FP32"},
                            "int8_scope": "fc_fusion INT8 input and TGEMM=int8_t, maxima=127, alpha=1, beta=0, no bias, LINEAR activation; FP32 output; not INT8-to-INT32 GemmEx",
                            "official_fc_effciency": "separate tool; requires vendor unittest and profiling artifacts; never claimed as run by native adapter"},
            "diagnosis_scope": "xpu-smi ECC/temperature/power and pre/post identity; vendor threshold diagnostics not-supported",
            "formal_gate": "idle resources, five independent repetitions, sufficient concurrent monitor, correctness and cleanup; shared-card runs stay exploratory"}
