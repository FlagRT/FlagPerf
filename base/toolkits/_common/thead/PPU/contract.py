"""PPU Toolkit measurement contract: cases, points, and record validation.

Mirrors the P800 contract shape. Measurement sources are the locked image's
own tools (PPU_SDK comm_tools sailbandwidth / p2pBandwidthLatencyTest) and
acblas GEMM entries proven by the day-9 dtype probes (probes/day9/toolkit).
INT8 has no working native path on this SDK build: acblasGemmEx returns
success with wrong output and acblasLtMatmul crashes inside the vendor JIT,
so no INT8 case is registered. Host-device and P2P latency reuse the tools'
pointer-chase semantics instead of inventing a copy-latency sweep.
"""
import math

SUITE = "thead-ppu-toolkit"
VENDOR = "thead"
CHIP = "PPU"

CASES = (
    "computation-FP32",
    "computation-FP64",
    "computation-FP16",
    "computation-BF16",
    "computation-TF32",
    "interconnect-h2d",
    "interconnect-d2h",
    "main_memory-bandwidth",
    "main_memory-capacity",
    "interconnect-host_device_latency",
    "interconnect-P2P_intraserver",
    "interconnect-P2P_intraserver-latency",
)

COMPUTE_CASES = {
    # case -> (dtype label, ctypes entry chosen by the day-9 probe matrix)
    "computation-FP32": {"dtype": "FP32", "entry": "acblasSgemm_v2",
                         "io": "R_32F -> R_32F"},
    "computation-FP64": {"dtype": "FP64", "entry": "acblasDgemm_v2",
                         "io": "R_64F -> R_64F"},
    "computation-FP16": {"dtype": "FP16", "entry": "acblasHgemm",
                         "io": "R_16F -> R_16F"},
    "computation-BF16": {"dtype": "BF16", "entry": "acblasLtMatmul",
                         "io": "R_16BF -> R_16BF, compute ACBLAS_COMPUTE_32F"},
    "computation-TF32": {"dtype": "TF32", "entry": "acblasGemmEx",
                         "io": "R_32F -> R_32F, compute ACBLAS_COMPUTE_32F_FAST_TF32, algo ACBLAS_GEMM_DEFAULT"},
}

# sailbandwidth testcase indices locked to this SDK build (v2.1.0, -l evidence
# in probes/day9/toolkit/t9_tool_samples.txt).
SAIL_TESTCASES = {
    "interconnect-h2d": {"ce": 0, "sm": 16},
    "interconnect-d2h": {"ce": 1, "sm": 17},
    "main_memory-bandwidth": {"d2d-local": 34},
    "interconnect-host_device_latency": {"access": 32},
}

# (testcase index, expected tool testcase name, reported unit), locked to the
# same v2.1.0 build as SAIL_TESTCASES; an import-time assertion keeps the two
# tables from drifting apart.
SAIL_EXPECTED_NAMES = {
    ("interconnect-h2d", "ce"): (0, "host_to_device_memcpy_ce", "GB/s"),
    ("interconnect-h2d", "sm"): (16, "host_to_device_memcpy_sm", "GB/s"),
    ("interconnect-d2h", "ce"): (1, "device_to_host_memcpy_ce", "GB/s"),
    ("interconnect-d2h", "sm"): (17, "device_to_host_memcpy_sm", "GB/s"),
    ("main_memory-bandwidth", "d2d-local"): (34, "device_local_copy", "GB/s"),
    ("interconnect-host_device_latency", "access"): (32, "host_device_latency_sm", "ns"),
}
for _case, _modes in SAIL_TESTCASES.items():
    for _mode, _index in _modes.items():
        assert SAIL_EXPECTED_NAMES[(_case, _mode)][0] == _index, (_case, _mode)

# Point modes run_point routes to run_sail_point. The runner dispatch keys off
# this tuple, so a mode added to case_points without a registered route fails
# the routing-completeness guard test instead of surfacing on hardware.
SAIL_ROUTED_MODES = ("ce", "sm", "d2d-local", "access")

TRANSFER_MODES = ("ce", "sm")
BANDWIDTH_MULTIPLIER = {"interconnect-h2d": 1, "interconnect-d2h": 1,
                        "main_memory-bandwidth": 1}
SAIL_BUFFER_MIB_RANGE = (1, 32768)
P2P_BYTES_RANGE = (1024, 4294967296)
MATRIX_SIZE_RANGE = (64, 16384)
COMPUTE_SAMPLES_RANGE = (25, 100000)
TRANSFER_SAMPLES_RANGE = (5, 1000)
P2P_SAMPLES_RANGE = (5, 1000)
SMOKE = {"payload_mib": 64, "matrix_size": 256, "compute_samples": 25,
         "transfer_samples": 5, "p2p_samples": 5}

# Formal gate wording is shared with the acceptance language; registered here
# so plans, evidence and reports quote one source.
FORMAL_GATE = ("idle resources, five independent repetitions, sufficient concurrent "
               "monitor, correctness and cleanup; shared-card runs stay exploratory")
CAPACITY_SCOPE = ("ppu-smi reported capacity of each selected card; a static query, "
                  "not a held allocation and not an OOM stress")


def validate_selection(ids):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("selection must be nonempty and unique")
    for physical in ids:
        if type(physical) is not int or not 0 <= physical <= 15:
            raise ValueError("physical index must be a PPU slot 0..15")


def case_points(case, ids, payload_mib=512, p2p_bytes=40000000):
    """Logical measurement points; never inferred, never reordered.

    Per-card cases contribute one or two fixed points per card. P2P cases
    contribute directed points over ordered pairs (both directions of every
    unordered pair): the tools measure each direction explicitly (day-9
    matrices show real asymmetry, e.g. 42.83 vs 46.14 GB/s), so both
    directions are recorded and neither is derived from the other.
    """
    validate_selection(ids)
    points = []
    if case in COMPUTE_CASES:
        points.extend({"mode": "gemm", "dtype": COMPUTE_CASES[case]["dtype"], "source": i}
                      for i in ids)
    elif case in ("interconnect-h2d", "interconnect-d2h"):
        points.extend({"mode": mode, "bytes": payload_mib * 1048576, "source": i}
                      for i in ids for mode in TRANSFER_MODES)
    elif case == "main_memory-bandwidth":
        points.extend({"mode": "d2d-local", "bytes": payload_mib * 1048576, "source": i}
                      for i in ids)
    elif case == "main_memory-capacity":
        points.extend({"mode": "capacity-query", "source": i} for i in ids)
    elif case == "interconnect-host_device_latency":
        points.extend({"mode": "access", "source": i} for i in ids)
    elif case == "interconnect-P2P_intraserver":
        points.extend({"mode": variant, "bytes": p2p_bytes, "source": s, "destination": d}
                      for s in sorted(ids) for d in sorted(ids) if s != d
                      for variant in ("uni-write", "uni-read", "bi-write"))
    elif case == "interconnect-P2P_intraserver-latency":
        points.extend({"mode": "latency", "source": s, "destination": d}
                      for s in sorted(ids) for d in sorted(ids) if s != d)
    else:
        raise ValueError("unregistered case: " + case)
    return points


def target_name(point, repetition):
    """Stable evidence directory name; encode full point identity to stay unique."""
    if "destination" in point:
        shape = "%s-%d-to-%d" % (point["mode"], point["source"], point["destination"])
    else:
        shape = "%s-%d" % (point["mode"], point["source"])
    if "dtype" in point:
        shape += "-" + point["dtype"]
    return "point/%s/rep-%d" % (shape, repetition)


def _positive_finite(values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values)


def validate_tool_record(record, point, repeat_count):
    """Structural gate for a sailbandwidth/p2pBandwidthLatencyTest result row.

    The runner stores one row per (point, repetition) with the tool-reported
    value plus provenance fields; anything missing, non-finite, or outside
    the recorded window fails the measurement.
    """
    if record.get("schema_version") != 1:
        raise ValueError("tool record schema mismatch")
    if record.get("point") != point:
        raise ValueError("tool record does not belong to this point")
    if record.get("repetition") not in range(1, repeat_count + 1):
        raise ValueError("repetition outside the resolved plan")
    if not record.get("tool_status") == "Passed":
        raise ValueError("tool reported non-passing status")
    for field in ("started_monotonic_ns", "finished_monotonic_ns"):
        if type(record.get(field)) is not int or record[field] <= 0:
            raise ValueError("missing measurement window timestamp: " + field)
    if record["finished_monotonic_ns"] <= record["started_monotonic_ns"]:
        raise ValueError("measurement window is empty or reversed")
    value = record.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("tool value must be positive and finite")
    if not record.get("identity", {}).get("bdf_matches"):
        raise ValueError("tool BDF evidence does not join to host records")
    if record.get("verification") not in ("tool-verified", "not-offered"):
        raise ValueError("verification provenance must be recorded")


def validate_compute_record(record, point, matrix_size, sample_count):
    """Structural gate for one acblas child-process result."""
    if record.get("schema_version") != 1:
        raise ValueError("compute record schema mismatch")
    if record.get("mode") != "gemm" or record.get("dtype") != point["dtype"]:
        raise ValueError("compute record does not belong to this point")
    if record.get("matrix_size") != matrix_size:
        raise ValueError("compute record matrix differs from the resolved plan")
    if record.get("visible_devices") != 1 or not record.get("uuid"):
        raise ValueError("compute child must see exactly one device and report its UUID")
    if not record.get("correctness", {}).get("passed"):
        raise ValueError("compute child failed its correctness readback")
    samples = record.get("samples_ns")
    # minimum-seconds extends the window past the plan count (P800 native_bench
    # semantics), so the plan count is a floor, not an equality.
    if not isinstance(samples, list) or len(samples) < sample_count or not _positive_finite(samples):
        raise ValueError("compute samples missing, short of the plan count, or not positive-finite")
    for field in ("started_monotonic_ns", "finished_monotonic_ns"):
        if type(record.get(field)) is not int or record[field] <= 0:
            raise ValueError("missing measurement window timestamp: " + field)
    if record["finished_monotonic_ns"] <= record["started_monotonic_ns"]:
        raise ValueError("measurement window is empty or reversed")


def metric_from_record(record, case, matrix_size):
    """One metric row; unit and formula come from the registered case family.

    Metric rows stay small: raw per-call samples live in the point directory
    (samples.json), tool matrices in tool-record.json; rows carry summaries.
    """
    if case in COMPUTE_CASES:
        samples = sorted(record["samples_ns"])
        median_ns = samples[len(samples) // 2]
        mean = sum(samples) / len(samples)
        cv = (sum((s - mean) ** 2 for s in samples) / (len(samples) - 1)) ** 0.5 / mean if len(samples) > 1 else None
        return {"field": "throughput", "unit": "TFLOPS",
                "value": 2.0 * matrix_size ** 3 / median_ns / 1000.0,
                "source_kind": "acblas-" + COMPUTE_CASES[case]["entry"],
                "sample_count": len(samples), "sample_cv": cv}
    field, unit = ("latency", "ns") if "latency" in case else ("bandwidth", "GB/s")
    return {"field": field, "unit": unit, "value": float(record["value"]),
            "source_kind": "official-tool", "tool": record.get("tool"),
            "verification": record.get("verification")}


def layer_status(values):
    values = list(values)
    if not values:
        return "not-run"
    if "failed" in values:
        return "failed"
    if "partial" in values or "not-run" in values:
        return "partial"
    return "passed"


def contract_record(settings):
    """Machine-readable contract snapshot stored in every run's evidence."""
    return {
        "schema_version": 1,
        "suite": SUITE,
        "vendor": VENDOR,
        "chip": CHIP,
        "cases": list(CASES),
        "compute_entries": {case: COMPUTE_CASES[case]["entry"] for case in COMPUTE_CASES},
        "sail_testcases": {case: dict(indexes) for case, indexes in SAIL_TESTCASES.items()},
        "sail_expected_names": {"%s/%s" % key: {"index": value[0], "name": value[1], "unit": value[2]}
                                for key, value in SAIL_EXPECTED_NAMES.items()},
        "excluded_capabilities": {
            "computation-INT8": ("no working native path on this SDK build: acblasGemmEx "
                                 "returns success with wrong output and acblasLtMatmul "
                                 "crashes in the vendor JIT (probes/day9/toolkit T8 evidence)"),
            "interconnect-h2d-latency / interconnect-d2h-latency": (
                "the tools offer pointer-chase access latency, not a four-size copy-API "
                "sweep; registered instead as interconnect-host_device_latency"),
            "interconnect-MPI_intraserver": ("optional pccl-tests/all_reduce_perf case; not "
                                             "registered in this toolkit round"),
        },
        "capacity_scope": CAPACITY_SCOPE,
        "formal_gate": FORMAL_GATE,
        "settings": settings,
    }
