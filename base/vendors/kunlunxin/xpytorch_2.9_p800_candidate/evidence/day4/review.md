# Day 4 / PR3 qualification review — 2026-09-21

PR3 completed the bounded single-card P800 native FP32 path on the M1 candidate
runtime. The tested executable commit is `7bf4c9bf5ca1c7e76dc46fd4ff05fb2b208791fd`.
The immutable image is `sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975`.

## Acceptance

Physical card 6 was selected after a fresh preflight. It resolved to PCI
`0000:b6:00.0`, `/dev/xpu5`, UUID
`9e24d168-db57-5cd9-8cfb-aea4ae898897`, and framework `cuda:0`. The same UUID,
context hash, container binding and native runtime were rechecked inside every
benchmark child. Gloo was the single-rank CPU control backend; no device
collective was initialized. Containers were nonprivileged, selected-node only,
and absent after every run.

The smoke run passed with monitor off. A 4096×4096×4096 calibration using 64
iterations took 0.077924342 seconds and was used only to freeze 20,000 iterations.
Five qualification runs then passed with monitor on:

| Run | TFLOPS | Measurement seconds | Valid target samples |
|---|---:|---:|---:|
| 1 | 113.13232647068246 | 24.297025927 | 24 |
| 2 | 112.93051037294873 | 24.340446708 | 24 |
| 3 | 113.05773712582776 | 24.313055783 | 25 |
| 4 | 113.12158131469140 | 24.299333845 | 24 |
| 5 | 112.92612352409549 | 24.341392263 | 25 |

All five use the same image, code, case contract, UUID, shape, seed and runtime.
The recomputed median is **113.05773712582776 TFLOPS**; min/max are
112.92612352409549/113.13232647068246, sample standard deviation is
0.10031061399857082, and CV is **0.08874402347040153%** (ddof=1).

Three small matrices use full CPU FP64 references. The measured shape checks fixed
rows/columns over the complete reduction dimension and full-output finiteness.
The timer is `perf_counter_ns` bracketed by full target synchronization. CUDA
Event elapsed time remains unusable. Returned-output allocation may be included.
The measurement event is a host monotonic/UTC boundary artifact and encloses the
timer; it is not a CUDA Event duration.

## Failure and recovery gate

`timeout-attempt01-card6` intentionally completed correctness, then waited on CPU
until the 120-second watchdog. It remains failed/exit 1 with `TimeoutError`.
Cleanup, postflight, container absence, lease release and later lock reacquisition
all passed. The failure is retained and is not counted as a performance repetition.

Report regeneration was deterministic and left stored experiment summaries
unchanged. The independent acceptance review reacquired the resource and PR0
compatibility locks, rechecked card 6 idle state, and confirmed all benchmark
containers absent.

## Scope

This qualifies the locked M1/native XPYTORCH single-card FP32 input/output path
for the stated shape and configuration. It does not promote the aggregate runtime
manifest, certify strict internal IEEE arithmetic, exclude every possible CPU
fallback, or qualify FlagGems, other dtypes, memory/transfer cases or collectives.
The runtime manifest remains `candidate` / `validated: false`.

Full raw results remain under `base/result/p800-pr3-20260921-1011/`. Portable Day 4
evidence in this directory contains source hashes and explicit target/process
projections; projected logs are not presented as byte-identical originals.
