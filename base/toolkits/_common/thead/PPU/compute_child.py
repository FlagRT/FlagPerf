"""Per-point acblas GEMM runner, executed as one isolated child process.

Only the day-9-proven entry points are implemented (probes/day9/toolkit
t8b_lt_probe.py): Sgemm_v2, Dgemm_v2, Hgemm, GemmEx with
ACBLAS_COMPUTE_32F_FAST_TF32 and algo -1, and acblasLtMatmul for BF16.
The child must see exactly one device; its cudaDeviceProp UUID must match
the host-selected card before any allocation. Correctness uses a sampled
python reference at 256 before any timing; failures exit nonzero and never
report numbers. The vendor JIT compiles kernels on first call, so warmup
happens strictly before the measured window.
"""
import argparse
import ctypes as C
import json
import struct
import sys
import time

RT = C.CDLL("libcudart.so")
AC = C.CDLL("/usr/local/PPU_SDK/lib/libacblas.so")
LT = C.CDLL("/usr/local/PPU_SDK/lib/libacblasLt.so")

RT.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
RT.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
RT.cudaMemset.argtypes = [C.c_void_p, C.c_int, C.c_size_t]
RT.cudaFree.argtypes = [C.c_void_p]
RT.cudaGetDeviceCount.argtypes = [C.POINTER(C.c_int)]
RT.cudaGetDeviceCount.restype = C.c_int
RT.cudaDeviceSynchronize.restype = C.c_int
_props = getattr(RT, "cudaGetDeviceProperties_v2", None) or getattr(RT, "cudaGetDeviceProperties")
_props.argtypes = [C.c_void_p, C.c_int]
_props.restype = C.c_int

AC.acblasCreate_v2.argtypes = [C.POINTER(C.c_void_p)]
AC.acblasDestroy_v2.argtypes = [C.c_void_p]
AC.acblasSgemm_v2.argtypes = [C.c_void_p] + [C.c_int] * 5 + [C.POINTER(C.c_float), C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.POINTER(C.c_float), C.c_void_p, C.c_int]
AC.acblasDgemm_v2.argtypes = [C.c_void_p] + [C.c_int] * 5 + [C.POINTER(C.c_double), C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.POINTER(C.c_double), C.c_void_p, C.c_int]
AC.acblasHgemm.argtypes = [C.c_void_p] + [C.c_int] * 5 + [C.POINTER(C.c_uint16), C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.POINTER(C.c_uint16), C.c_void_p, C.c_int]
AC.acblasGemmEx.argtypes = [C.c_void_p] + [C.c_int] * 5 + [C.POINTER(C.c_float), C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_float), C.c_void_p, C.c_int, C.c_int] + [C.c_int] * 2

LT.acblasLtCreate.argtypes = [C.POINTER(C.c_void_p)]
LT.acblasLtDestroy.argtypes = [C.c_void_p]
LT.acblasLtMatmulDescCreate.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_int]
LT.acblasLtMatmulDescDestroy.argtypes = [C.c_void_p]
LT.acblasLtMatmulDescSetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t]
LT.acblasLtMatrixLayoutCreate.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_uint64, C.c_uint64, C.c_int64]
LT.acblasLtMatrixLayoutDestroy.argtypes = [C.c_void_p]
LT.acblasLtMatmulPreferenceCreate.argtypes = [C.POINTER(C.c_void_p)]
LT.acblasLtMatmulPreferenceDestroy.argtypes = [C.c_void_p]
LT.acblasLtMatmulPreferenceSetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t]
LT.acblasLtMatmulAlgoGetHeuristic.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.POINTER(C.c_int)]
LT.acblasLtMatmul.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p]


class Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class HeuristicResult(C.Structure):
    _fields_ = [("algo", Algo), ("workspaceSize", C.c_size_t), ("state", C.c_int),
                ("wavesCount", C.c_float), ("reserved", C.c_int * 4)]


OP_N = 0
GEMM_DEFAULT = -1
HGGC = {"R_32F": 0, "R_64F": 1, "R_16F": 2, "R_8I": 3, "R_32I": 10, "R_16BF": 14}
COMPUTE_32F = 68
COMPUTE_FAST_TF32 = 77
DESC_TRANSA, DESC_TRANSB = 3, 4
PREF_MAX_WORKSPACE_BYTES = 1
CHECK_N = 256
CHECK_PAIRS = ((0, 0), (0, 255), (1, 2), (17, 31), (128, 128), (255, 255), (254, 5),
               (33, 129), (64, 200), (100, 7), (2, 2), (7, 7), (200, 64), (31, 17),
               (129, 33), (255, 0))

ENTRIES = {
    "FP32": {"tolerance": 2e-3},
    "FP64": {"tolerance": 1e-4},
    "FP16": {"tolerance": 5e-2},
    "BF16": {"tolerance": 5e-2},
    "TF32": {"tolerance": 5e-2},
}


class LCG:
    def __init__(self, seed=20261019):
        self.state = seed

    def next(self):
        self.state = (self.state * 1103515245 + 12345) % (1 << 31)
        return self.state


def device_uuid():
    buffer = (C.c_char * 16384)()
    if _props(C.cast(buffer, C.c_void_p), 0):
        raise RuntimeError("cudaGetDeviceProperties failed")
    return bytes(bytearray(buffer))


def read_uuid_at(buffer, offset):
    """Canonical 8-4-4-4-12 grouping so records join to host ppu-smi strings."""
    b = buffer[offset:offset + 16]
    return "-".join((b[0:4].hex(), b[4:6].hex(), b[6:8].hex(), b[8:10].hex(), b[10:16].hex()))


def find_uuid(buffer, expected):
    """Scan cudaDeviceProp for the expected 16-byte UUID (day-9: offset 256)."""
    hits = [o for o in range(0, len(buffer) - 16) if buffer[o:o + 16] == expected]
    if len(hits) != 1:
        raise RuntimeError("device UUID not uniquely present in device properties: %d hits" % len(hits))
    return hits[0]


def encode(values, style):
    if style == "f32":
        return (C.c_float * len(values))(*values)
    if style == "f64":
        return (C.c_double * len(values))(*values)
    if style == "half":
        return (C.c_uint16 * len(values))(*[struct.unpack("<H", struct.pack("<e", v))[0] for v in values])
    if style == "bf16":
        return (C.c_uint16 * len(values))(*[struct.unpack("<I", struct.pack("<f", v))[0] >> 16 for v in values])
    raise ValueError(style)


def decode_one(buf, style):
    if style in ("f32", "f64"):
        return float(buf)
    if style == "half":
        return struct.unpack("<e", struct.pack("<H", buf))[0]
    if style == "bf16":
        return struct.unpack("<f", struct.pack("<I", buf << 16))[0]
    raise ValueError(style)


def device_alloc(count, element_size):
    ptr = C.c_void_p()
    if RT.cudaMalloc(C.byref(ptr), C.c_size_t(count * element_size)):
        raise RuntimeError("cudaMalloc failed")
    return ptr


def reference_dot(a_row, b_col, k):
    total = 0.0
    for t in range(k):
        total += a_row[t] * b_col[t]
    return total


def make_call(handle, spec):
    """Return (call(pa, pb, pc, n) -> rc, a_style, c_style) for the dtype."""
    dtype = spec["dtype"]
    n_ld = lambda n: n
    # acblas is column-major: gemm(N,N, B, A) makes C_col = B_col*A_col, whose
    # flat buffer read row-major equals A_row*B_row — the row-major upload idiom.
    if dtype == "FP32":
        alpha, beta = C.c_float(1.0), C.c_float(0.0)
        call = lambda pa, pb, pc, n: AC.acblasSgemm_v2(handle, OP_N, OP_N, n, n, n, C.byref(alpha), pb, n_ld(n), pa, n_ld(n), C.byref(beta), pc, n_ld(n))
        return call, "f32", "f32"
    if dtype == "FP64":
        alpha, beta = C.c_double(1.0), C.c_double(0.0)
        call = lambda pa, pb, pc, n: AC.acblasDgemm_v2(handle, OP_N, OP_N, n, n, n, C.byref(alpha), pb, n_ld(n), pa, n_ld(n), C.byref(beta), pc, n_ld(n))
        return call, "f64", "f64"
    if dtype == "FP16":
        alpha, beta = C.c_uint16(0x3C00), C.c_uint16(0x0000)
        call = lambda pa, pb, pc, n: AC.acblasHgemm(handle, OP_N, OP_N, n, n, n, C.byref(alpha), pb, n_ld(n), pa, n_ld(n), C.byref(beta), pc, n_ld(n))
        return call, "half", "half"
    if dtype == "TF32":
        alpha, beta = C.c_float(1.0), C.c_float(0.0)
        call = lambda pa, pb, pc, n: AC.acblasGemmEx(handle, OP_N, OP_N, n, n, n, C.byref(alpha), pb, HGGC["R_32F"], n_ld(n), pa, HGGC["R_32F"], n_ld(n), C.byref(beta), pc, HGGC["R_32F"], n_ld(n), COMPUTE_FAST_TF32, GEMM_DEFAULT)
        return call, "f32", "f32"
    raise ValueError("BF16 uses the Lt path, not a typed call")


def lt_call(handle, spec, c_style_hggc):
    """Build the BF16 Lt pipelines; returns (call, a_hggc, c_hggc, cleanup).

    One pipeline per problem size, each with its own desc/layouts/heuristic/
    workspace: the correctness precheck runs at CHECK_N, measurement at the
    spec size. Applying a measurement-size layout to check-sized buffers is
    out-of-bounds — the 20261020T090026Z BF16 failure signature (illegal
    memory access at hggcMemPoolDestroy, AssertionError with zero samples).
    Both pipelines keep the toolkit operand order (column-major row-major
    idiom: feed B as the left operand, see make_call).
    """
    alpha, beta = C.c_float(1.0), C.c_float(0.0)

    def build_pipeline(n):
        desc, layout_a, layout_b, layout_c, pref = (C.c_void_p() for _ in range(5))
        if LT.acblasLtMatmulDescCreate(C.byref(desc), COMPUTE_32F, HGGC["R_32F"]):
            raise RuntimeError("Lt desc create failed")
        zero = C.c_int32(0)
        for attr in (DESC_TRANSA, DESC_TRANSB):
            if LT.acblasLtMatmulDescSetAttribute(desc, attr, C.byref(zero), C.c_size_t(4)):
                raise RuntimeError("Lt desc attribute failed")
        for lay in (C.byref(layout_a), C.byref(layout_b), C.byref(layout_c)):
            if LT.acblasLtMatrixLayoutCreate(lay, c_style_hggc, C.c_uint64(n), C.c_uint64(n), C.c_int64(n)):
                raise RuntimeError("Lt layout create failed")
        if LT.acblasLtMatmulPreferenceCreate(C.byref(pref)):
            raise RuntimeError("Lt preference create failed")
        ws = C.c_size_t(32 * 1024 * 1024)
        LT.acblasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE_BYTES, C.byref(ws), C.c_size_t(C.sizeof(ws)))
        results = (HeuristicResult * 4)()
        returned = C.c_int(0)
        if LT.acblasLtMatmulAlgoGetHeuristic(handle, desc, layout_a, layout_b, layout_c, layout_c, pref, 4, results, C.byref(returned)):
            raise RuntimeError("Lt heuristic failed")
        if returned.value < 1 or results[0].state != 0:
            raise RuntimeError("Lt heuristic returned no usable algorithm")
        ws_ptr = device_alloc(max(1, results[0].workspaceSize), 1)

        def call(pa, pb, pc):
            return LT.acblasLtMatmul(handle, desc, C.byref(alpha), pb, layout_a, pa, layout_b, C.byref(beta), pc, layout_c, pc, layout_c, C.byref(results[0].algo), ws_ptr, C.c_size_t(max(1, results[0].workspaceSize)), C.c_void_p())

        def cleanup():
            RT.cudaFree(ws_ptr)
            LT.acblasLtMatmulPreferenceDestroy(pref)
            LT.acblasLtMatrixLayoutDestroy(layout_c)
            LT.acblasLtMatrixLayoutDestroy(layout_b)
            LT.acblasLtMatrixLayoutDestroy(layout_a)
            LT.acblasLtMatmulDescDestroy(desc)

        return call, cleanup

    measure_call, measure_cleanup = build_pipeline(spec["matrix_size"])
    check_call, check_cleanup = build_pipeline(CHECK_N)

    def dispatch(pa, pb, pc, n):
        return (check_call if n == CHECK_N else measure_call)(pa, pb, pc)

    def cleanup():
        measure_cleanup()
        check_cleanup()

    return dispatch, "bf16", "bf16", cleanup, None


def correctness(handle, call, a_style, c_style, tolerance):
    n = CHECK_N
    count = n * n
    gen = LCG()
    host_a = [gen.next() / (1 << 31) - 0.5 for _ in range(count)]
    host_b = [gen.next() / (1 << 31) - 0.5 for _ in range(count)]
    buf_a, buf_b = encode(host_a, a_style), encode(host_b, a_style)
    buf_c = encode([0.0] * count, c_style)
    c_elem = 8 if c_style == "f64" else (2 if c_style in ("half", "bf16") else 4)
    pa, pb, pc = device_alloc(count, 4 if a_style != "f64" else 8), device_alloc(count, 4 if a_style != "f64" else 8), device_alloc(count, c_elem)
    worst = None
    try:
        assert RT.cudaMemcpy(pa, C.cast(buf_a, C.c_void_p), C.c_size_t(count * C.sizeof(type(buf_a)._type_)), 1) == 0
        assert RT.cudaMemcpy(pb, C.cast(buf_b, C.c_void_p), C.c_size_t(count * C.sizeof(type(buf_b)._type_)), 1) == 0
        assert RT.cudaMemset(pc, 0, C.c_size_t(count * c_elem)) == 0
        rc = call(pa, pb, pc, n)
        if rc:
            raise RuntimeError("correctness call failed with rc %d" % rc)
        assert RT.cudaDeviceSynchronize() == 0
        assert RT.cudaMemcpy(C.cast(buf_c, C.c_void_p), pc, C.c_size_t(count * c_elem), 2) == 0
        worst = 0.0
        for i, j in CHECK_PAIRS:
            got = decode_one(buf_c[i * n + j], c_style)
            want = reference_dot(host_a[i * n:(i * n) + n], host_b[j::n], n)
            worst = max(worst, abs(got - want) / max(1.0, abs(want)))
    finally:
        RT.cudaFree(pa)
        RT.cudaFree(pb)
        RT.cudaFree(pc)
    if worst > tolerance:
        raise RuntimeError("correctness failed: max relative error %.4g > %.4g" % (worst, tolerance))
    return worst


def run(handle, call, a_style, c_style, spec):
    n = spec["matrix_size"]
    count = n * n
    a_bytes = {"f32": 4, "f64": 8, "half": 2, "bf16": 2}[a_style]
    c_bytes = {"f32": 4, "f64": 8, "half": 2, "bf16": 2}[c_style]
    pa, pb, pc = device_alloc(count, a_bytes), device_alloc(count, a_bytes), device_alloc(count, c_bytes)
    samples = []
    try:
        RT.cudaMemset(pa, 0x3C, C.c_size_t(count * a_bytes))
        RT.cudaMemset(pb, 0x3C, C.c_size_t(count * a_bytes))
        RT.cudaMemset(pc, 0, C.c_size_t(count * c_bytes))
        # P800 native_bench floor semantics: the timed loop keeps taking samples
        # until the count is met AND the window reaches minimum_seconds, so one
        # point window can hold >=10 same-window monitor samples at 0.5 s polling.
        floor_ns = float(spec.get("minimum_seconds", 0.0)) * 1e9
        started = time.monotonic_ns()
        for _ in range(spec["warmup"]):
            if call(pa, pb, pc, n):
                raise RuntimeError("warmup call failed")
            assert RT.cudaDeviceSynchronize() == 0
        while len(samples) < spec["samples"] or time.monotonic_ns() - started < floor_ns:
            t0 = time.perf_counter_ns()
            if call(pa, pb, pc, n):
                raise RuntimeError("timed call failed")
            assert RT.cudaDeviceSynchronize() == 0
            samples.append(time.perf_counter_ns() - t0)
        finished = time.monotonic_ns()
    finally:
        RT.cudaFree(pa)
        RT.cudaFree(pb)
        RT.cudaFree(pc)
    return samples, started, finished


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    record = {"schema_version": 1, "status": "unknown"}
    handle = C.c_void_p()
    try:
        with open(args.spec, encoding="utf-8") as stream:
            spec = json.load(stream)
        if spec.get("schema_version") != 1 or spec.get("dtype") not in ENTRIES:
            raise RuntimeError("unsupported compute spec")
        record["spec"] = spec
        record["mode"] = "gemm"
        record["dtype"] = spec["dtype"]
        record["matrix_size"] = spec["matrix_size"]
        count = C.c_int(-1)
        if RT.cudaGetDeviceCount(C.byref(count)) or count.value != 1:
            raise RuntimeError("child must see exactly one visible device, saw %d" % count.value)
        record["visible_devices"] = count.value
        buffer = device_uuid()
        expected = bytes.fromhex(spec["expected_uuid"].replace("-", ""))
        offset = find_uuid(buffer, expected)
        record["uuid"] = read_uuid_at(buffer, offset)
        record["uuid_prop_offset"] = offset
        if AC.acblasCreate_v2(C.byref(handle)):
            raise RuntimeError("acblas handle create failed")
        tolerance = ENTRIES[spec["dtype"]]["tolerance"]
        if spec["dtype"] == "BF16":
            call, a_style, c_style, cleanup, _build = lt_call(handle, spec, HGGC["R_16BF"])
        else:
            call, a_style, c_style = make_call(handle, spec)
            cleanup = lambda: None
        try:
            record["correctness"] = {"passed": True, "max_rel_err": correctness(handle, call, a_style, c_style, tolerance)}
            samples, started, finished = run(handle, call, a_style, c_style, spec)
        finally:
            cleanup()
        record.update(samples_ns=samples, started_monotonic_ns=started,
                      finished_monotonic_ns=finished, status="passed")
    except Exception as exc:  # noqa: BLE001 - child reports, parent judges
        record["status"] = "failed"
        record["error"] = repr(exc)
        record["samples_ns"] = []
    finally:
        if handle:
            AC.acblasDestroy_v2(handle)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
