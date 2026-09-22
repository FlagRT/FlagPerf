# Day 5 / PR4 + PR5-front review — 2026-09-22

Day 5 extended the bounded single-card execution chain to the remaining
computation precisions and to host/device host-copy cases. The tested code
identity is `ed731d876024f494d0774d4696dd206ab20f1b2f` with the day-5 changes
present in the worktree (per-run `code-identity.json` records the dirty state).
The immutable image is
`sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975`.

## Acceptance

Physical card 5 was selected after fresh preflights. It resolved to PCI
`0000:9c:00.0`, `/dev/xpu7`, UUID `f396486d-9850-50e4-81c4-50f2e9f6ca87`, and
framework `cuda:0`. Every benchmark child rechecked the same UUID, context hash,
container binding and native runtime. Gloo remained the single-rank CPU control
backend; no device collective was initialized. Containers were nonprivileged,
selected-node only, and absent after every run.

Card 6 and card 7 preflights were rejected as occupied during the morning card
search; card 4 was also rejected. Those rejection summaries are retained with
`cleanup_status: not-created` and no lease acquired.

## Capability matrix

| Case | Conclusion | Evidence |
|---|---|---|
| computation-FP32 | supported and qualified on day 4 | 113.057737 TFLOPS median, CV 0.088744% |
| computation-FP16 | supported and qualified | `torch.mm` FP16 in/out; five runs, 253.127 TFLOPS median, CV 0.0796% |
| computation-BF16 | supported and qualified | `torch.mm` BF16 in/out; five runs, 114.0087 TFLOPS median, CV 0.0068% |
| computation-INT8 | supported and qualified at 2048-cubed | `torch._int_mm` int32 output, zero max abs error; five runs, 4.1639 TOPS median, CV 1.6457% |
| computation-FP64 | unsupported for the locked stack | capability probe observed float32 output for float64 input |
| computation-FP8 | unsupported for the locked stack | E4M3/E5M2 `torch.mm` and `torch._scaled_mm` both returned NOT IMPLEMENTED |
| computation-TF32 | blocked / unresolved | `allow_tf32` True and False produced element-identical outputs; no controllable TF32 route was demonstrated |
| interconnect-h2d | pageable modes qualified; pinned modes unstable | see transfer table |
| interconnect-d2h | pageable modes qualified; pinned modes unstable | see transfer table |

The INT8 metric counts a multiply-add as two operations and is derived from
`torch._int_mm` with int32 output. The 4096-cubed INT8 frozen configurations were
retried with warmup 10 and warmup 100; both groups failed the CV gate (25.97% and
36.12%) with bimodal fast/slow rounds that share UUID, config hash, correctness
and monitor identity. Those groups are retained unchanged and are not trimmed.
The 2048-cubed configuration is the qualified INT8 scope; the bimodality remains
a documented open item rather than a claim that INT8 is unsupported.

## Computation qualification runs

All runs use 4096×4096×4096, seed 519, warmup 10, 100,000 iterations except INT8.

| Group | Median | CV | Per-run measurement |
|---|---:|---:|---:|
| FP16 (tol 1e-3) | 253.127 TFLOPS | 0.0796% | ~54.3 s |
| BF16 (tol 8e-3) | 114.0087 TFLOPS | 0.0068% | ~120.6 s |
| INT8 2048³, 4000 iters, warmup 100 (tol 0) | 4.1639 TOPS | 1.6457% | ~16.5 s |

Correctness references are constructed from the quantized inputs, compared on
CPU FP64, and stored per run. Every measured shape checks fixed rows/columns over
the complete reduction dimension plus full-output finiteness.

## Transfer qualification runs

Frozen configuration: 64 MiB payload, 10,000 iterations, warmup 10, float32
`Tensor.copy_` on preallocated source and destination, timer `perf_counter_ns`
bracketed by full target synchronization. Content is verified over the full
payload before and after the timed loop, including a sentinel and a
changed-input phase outside the timed region. GB/s and GiB/s are derived from
the same bytes and time; a single direction is not doubled.

| Group | Median GB/s | CV | Status |
|---|---:|---:|---|
| h2d pageable blocking | 10.7706 | 0.1937% | passed |
| h2d pageable nonblocking | 20.2045 | 0.7236% | passed |
| d2h pageable blocking | 11.6205 | 1.1461% | passed |
| d2h pageable nonblocking | 18.4105 | 2.4006% | passed |
| h2d pinned blocking | 17.9045 | 7.4488% | unstable |
| h2d pinned nonblocking | 16.0520 | 14.1804% | unstable |
| d2h pinned blocking | 19.4425 | 18.6162% | unstable |
| d2h pinned nonblocking | 23.2407 | 5.4358% | unstable |

The four pinned groups failed the CV gate on a shared host with concurrent
third-party workloads while their device identity, monitor sample counts and
correctness checks stayed consistent between fast and slow rounds. They are
retained as measured and are scheduled for re-execution in a quiet window; no
slow samples were removed. These numbers are effective single-direction API
bandwidth under full synchronization; `non_blocking` is a requested mode and is
not evidence of overlap or independent DMA.

## Regression and gates

- Base 195, PR0 32, Toolkit 80 offline tests pass; targeted suites: computation 9,
  transfer 7, fp32 22, preflight 37.
- Ascend 15-case static planning stays 10 applicable / 5 skipped. The P800
  planning matrix resolves 4 computation cases applicable, 3 skipped with
  capability-probe reasons, and both transfer cases applicable.
- Selector gates: Ascend aliases still accepted; P800 requires
  `--physical-device-ids` and rejects legacy aliases; preflight without explicit
  selection fails closed; high-risk capacity refuses without
  `--allow-high-risk-case`.
- Public distribution remains 6/7 with the preexisting private Ascend Toolbox
  path failure unchanged.
- FP32 regression: generalized smoke passed; one monitored round at the day-4
  configuration produced 113.1 TFLOPS, consistent with the day-4 median and not
  merged into day-4 statistics.

## Failure and recovery gate

`timeout-FP16-a01` and `timeout-h2d-a01` intentionally completed diagnostics,
then waited on CPU until the 60-second watchdog. Both remain failed with
`TimeoutError`, and cleanup, postflight, container absence, lease release and
lock reacquisition all passed. These failures are retained and are not counted as
qualifications.

Report regeneration was deterministic for every run directory and left stored
summaries unchanged; two early failed smokes needed one content normalization on
the first post-hoc regeneration and were byte-stable afterwards. The independent
acceptance review reacquired the resource lock on card 5, rechecked the line
idle state under lock, and confirmed container absence for every run.

## Scope

This qualifies the locked M1/native XPYTORCH single-card FP16, BF16 and
2048-cubed INT8 computation paths and the pageable-mode copy paths in both
directions for the stated configurations. It does not promote the aggregate
runtime manifest, certify pinned-mode bandwidth stability, resolve the
4096-cubed INT8 bimodality, qualify TF32 or FP64/FP8, or cover device memory,
capacity and multi-card communication. The runtime manifest remains `candidate`
/ `validated: false`.

## Correction: the first INT8 round measured the host CPU (2026-09-22)

The first INT8 qualification used `torch._int_mm`, which the locked stack executes on the
host CPU even though it accepts device tensors and returns int32 device tensors. Two
independent observations establish this:

- Measured process CPU time inside the "device" loop equalled 192 times the wall time,
  exactly the container thread count, while the same measurement for FP16, BF16 and FP32
  was 1.0. CPU-only `torch._int_mm` at 2048-cubed was faster than the "device" call, and
  the "device" per-iteration cost matched a PCIe round trip of the operands (3.77 ms)
  plus the CPU product (1.07 ms).
- Device telemetry during those runs stayed at idle: 0% utilization, 39 C, 92 W. Runs of
  the same period for FP16, BF16 and FP32 show 100% utilization, 61-63 C and 296-400 W.

The 4096-cubed bimodality was the same fallback alternating between CPU-bound and
copy-bound regimes. The earlier INT8 numbers (4.1639 TOPS median at 2048-cubed, and the
two 4096-cubed groups at 4.44 and 4.41 TOPS) are therefore **retracted**: they are host
measurements and are not device results. `torch.mm` rejects int8 outright and
`torch._scaled_mm` accepts only Float8, so this stack has no ATen int8 path.

The case now uses the vendor device kernel `xtorch_ops.gemm_I8_I8_bf16_nt`, sharing the
Ascend contract semantics (`SCALE_A`, `SCALE_B`, bfloat16 output) and now the Ascend
scale of 8192-cubed. The kernel takes int8 quantization maxima and divides by 127
internally, so passing `127 * SCALE` reproduces `(A @ B) * SCALE_A * SCALE_B` in bf16.

| Run | TOPS | Window | Utilization | Temperature | Power |
|---|---:|---:|---:|---:|---:|
| device q31 | 506.36 | 17.37 s | 100% | 61 C | 400 W |
| device q32 | 504.59 | 17.43 s | | | |
| device q33 | 505.58 | 17.40 s | | | |
| device q34 | 501.95 | 17.52 s | | | |
| device q35 | 504.30 | 17.44 s | | | |

Median **504.591 TOPS**, CV **0.3315%** (ddof=1), five identical runs at 8192-cubed with
8000 iterations, warmup 10 and seed 519. This is 121 times the retracted figure and is
consistent with the FP16 result at the same scale (253 TFLOPS device-measured, int8
typically twice fp16 on this class of hardware).

The retracted directories (`qualification-INT8-q01..q05`, `q11..q15`, `q21..q25`) and the
probe directories that established the fallback (`capability-INT8-routes-a01`,
`capability-matmul-routes-a01`, `capability-xtorch-int8-a01..a05`) remain in the raw
result directory unchanged.

## Scale alignment with the Ascend cases

The Ascend computation cases run 8192-cubed with warmup 100, and the transfer cases use
`Melements: 1024` (a 4 GiB payload) with pinned memory and 100 iterations. The first P800
round used 4096-cubed computation and a 64 MiB transfer payload, so it was not comparable.

Every case has since been re-specified and re-qualified at the Ascend scale. The results,
the iteration rule and the remaining alignment targets are in the sections below.

## Monitoring note

`xpu-smi` telemetry does record device activity for these workloads: 100% utilization,
61-63 C and 296-400 W during compute runs, with memory in the 200-500 MiB range. The
transfer runs show 0% utilization and idle temperature and power, which is expected for
DMA-only traffic that does not engage the compute engine.

## BF16 executes at fp32-equivalent throughput on this locked stack

A standalone Chinese analysis with the full measurement set, the rejected routes and the reproduction commands is archived next to this review as `bf16-throughput-analysis.zh.md`.

The BF16 measurements are genuine device work — process CPU time stays at wall time — but the
kernel is not bf16-accelerated. At 8192-cubed, with operands preallocated so that only the
matrix multiply is measured:

| API | BF16 | FP16 | ratio |
|---|---:|---:|---:|
| ATen `torch.mm` | 9.265 ms / 118.7 TFLOPS | 4.091 ms / 268.8 TFLOPS | 2.27x |
| vendor `xtorch_ops._gemm.matmul` | 9.454 ms / 116.3 | 4.177 ms / 263.3 | 2.26x |
| reference: fp32 `torch.mm` | 9.301 ms / 118.2 | — | — |

The vendor's own dense GEMM returns the same bf16 number as ATen, and that number matches
the fp32 rate to within 0.6 percent; three consecutive rounds of each API reproduce it. The
only faster bf16-output kernels present are int8-input or MOE-grouped ones, and the grouped
kernel rejects a plain dense call (`moe_fc_v3_block` error), so no dense bf16 tensor-core
path is reachable from the locked stack. An earlier probe that reported 899 ms per bf16
matmul was measuring host-side input generation (`randn` over 67M elements); with
preallocated operands the call is stable at 9.27 ms.

The BF16 case therefore reports a measured 118.970 TFLOPS at the Ascend scale with this
limitation attached, instead of being presented as a native bf16 rate.

## Ascend-scale alignment

The Ascend cases run 8192-cubed computation with warmup 100, and the transfer cases use
`Melements: 1024` (a 4 GiB payload) with pinned host memory and 100 iterations. Every P800
case now runs at that scale. Iteration counts follow the Ascend values wherever they satisfy
this project's 15-second measurement floor and are raised where they do not.

<!-- alignment-table-start -->
| Group | Median | CV | Window | Configuration |
|---|---:|---:|---:|---|
| FP16-8192 | 258.869 TFLOPS | 0.2294% | 21.22 s | warmup 100, 5000 iterations (Ascend values) |
| BF16-8192 | 118.970 TFLOPS | 0.0029% | 46.21 s | warmup 100, 5000 iterations (Ascend values) |
| FP32-8192 | 117.049 TFLOPS | 0.0860% | 18.80 s | warmup 100, 2000 iterations (Ascend uses 1000, which measures 9.3 s) |
| INT8-8192 | 504.591 TOPS | 0.3315% | 17.43 s | warmup 10, 8000 iterations (Ascend uses 100, which measures 0.22 s) |
| h2d-pageable-blocking-4G | 10.147 GB/s | 3.3107% | 42.97 s | Ascend payload, warmup 10, 100 iterations |
| h2d-pageable-nonblocking-4G | 16.665 GB/s | 3.4531% | 26.10 s | Ascend payload, warmup 10, 100 iterations |
| h2d-pinned-blocking-4G | 18.399 GB/s | 3.6521% | 22.83 s | Ascend payload, warmup 10, 100 iterations |
| h2d-pinned-nonblocking-4G | 20.864 GB/s | 6.3299% | 31.89 s | 2 GiB payload (largest this runtime accepts on that path), warmup 10, 300 iterations |
| d2h-pageable-blocking-4G | 9.940 GB/s | 3.5951% | 44.04 s | Ascend payload, warmup 10, 100 iterations |
| d2h-pageable-nonblocking-4G | 14.222 GB/s | 6.8655% | 29.08 s | Ascend payload, warmup 10, 100 iterations |
| d2h-pinned-blocking-4G | 28.791 GB/s | 2.3411% | 22.73 s | Ascend payload, warmup 10, 150 iterations (Ascend uses 100, which measured 14.8-17.0 s) |
| d2h-pinned-nonblocking-4G | 29.066 GB/s | 1.2001% | 22.30 s | 2 GiB payload (largest this runtime accepts on that path), warmup 10, 300 iterations |
<!-- alignment-table-end -->

The transfer groups at the Ascend payload show higher variance than the 64 MiB round
(CV 3.3 percent for h2d pageable blocking against 0.19 percent), consistent with a shared
host whose load average was above 20 during these runs. Groups that fail the stability gate
are recorded as unstable and kept, never trimmed.

### Why some iteration counts differ from Ascend

This project requires every qualification window to hold at least 15 seconds so that the
one-second device telemetry yields at least ten samples. At the measured P800 rates the
Ascend iteration counts for FP32 and INT8 would produce 9.3 s and 0.22 s windows, so they
are raised; FP16 and BF16 keep the Ascend values exactly. Shapes, payloads, warmups,
scales, output dtypes and seeds all match Ascend.

### Alignment targets for the remaining cases

Shapes, payloads and warmups are aligned for every case implemented so far. The cases whose
P800 implementation belongs to day six already have their Ascend targets recorded here so the
first implementation uses the comparable configuration rather than a convenient one:

- `main_memory-bandwidth`: Melements 1024 (4 GiB) with warmup 2 and 10 iterations; the clone() form counts 2 x payload per call
- `main_memory-capacity`: POST_TEST_WAIT_SECONDS 0 and BOUND_REQUEST_BY_FREE_MEMORY true; the search must hold real allocations, not read the free-memory hint as a result
- `interconnect-P2P_intraserver`: device collective backend (Ascend uses flagos); the P800 profile needs its XCCL backend resolved first
- `interconnect-MPI_intraserver`: Melements 1 with warmup 10 and 100 iterations on the device collective backend
- `interconnect-P2P_interserver / interconnect-MPI_interserver`: explicitly unsupported: the unified entrypoint fixes nnodes=1

### Transfer limitations at the Ascend payload

Two behaviours appear only at the 4 GiB payload and are recorded with their evidence.

**`non_blocking` on pinned memory fails above 2 GiB.** A pinned non-blocking copy raises
`[RUNTIME ERROR]: error code= 999, unknown error` at a 4 GiB payload, deterministically
(three consecutive attempts) and identically in both directions, while the same call succeeds
from 64 MiB through 2 GiB, and while pinned blocking, pageable blocking and pageable
non-blocking all succeed at 4 GiB. The probe `capability-transfer-pinned-a01` walks every
stage — pinned allocation, device allocation, fill, synchronise, blocking copy, non-blocking
copy — at six payloads and localises the failure to the non-blocking copy alone. The two
pinned non-blocking groups therefore run at 2 GiB, the largest payload this runtime accepts
on that path, with the iteration count raised to hold the window above 15 s.

**The d2h pinned blocking group reached 25-29 GB/s**, so the Ascend iteration count of 100
produced 14.77-16.97 s windows and two of five runs fell below the 15 s floor. The
qualification gate rejected the group and it was re-run at 150 iterations.

**One group remains unstable**: d2h pageable non-blocking at the Ascend payload reports CV
6.87 percent over 13.90-16.15 GB/s on the shared host. It is retained as measured, never
trimmed, and scheduled for a quiet-window re-run.

The Ascend-parity variant (pinned, blocking, 4 GiB) qualifies normally at 18.399 GB/s
(h2d) and at the corrected iteration count (d2h).

## Transfer payload realigned to the toolkit reference (512 MiB)

A per-case comparison of the toolkit reference bundle against the P800 state, with the exact vendor parameters and the remaining work, is archived next to this review as `toolkit-alignment-notes.zh.md`.

The governing reference for this alignment is the toolkit result bundle supplied by the user
(`参考资料/20260914T133423Z`), whose H2D and D2H cases run
`ascend-dmi --bw -t h2d|d2h -s 536870912 --et 50`: a **512 MiB payload copied 50 times**.
The earlier 4 GiB payload came from the Ascend **Base** case configuration
(`Melements: 1024`), a different suite with a different payload; the toolkit reference
supersedes it for these two cases.

The payload is now 512 MiB in all eight request modes. Iterations are raised from the
reference 50 because 50 copies measure 1-3 s at P800 rates, below this project's 15 s
measurement floor; per-variant counts are in the table. Warmup, dtype, layout and the
content checks are unchanged.

Two consequences worth recording:

- **pinned non-blocking runs at 512 MiB.** The vendor `error code= 999` seen earlier only
  appears at a 4 GiB payload on that path; at the reference payload all eight modes run, so
  the two groups previously limited to 2 GiB return to the common payload and the 4 GiB
  failure is recorded as payload-specific rather than a mode limitation.
- The d2h pinned blocking group no longer needs the raised iteration count that its 4 GiB
  window required; the 512 MiB windows satisfy the floor at the listed counts.

| Group | Median | CV | Window | Iterations |
|---|---:|---:|---:|---:|
| h2d-pageable-blocking | 9.059 GB/s | 4.3517% | 24.03 s | 400 |
| h2d-pageable-nonblocking | 17.235 GB/s | 4.3352% | 19.17 s | 600 |
| h2d-pinned-blocking | 20.496 GB/s | 1.5476% | 17.03 s | 650 |
| h2d-pinned-nonblocking | 20.742 GB/s | 1.6613% | 18.20 s | 700 |
| d2h-pageable-blocking | 10.162 GB/s | 1.3816% | 21.22 s | 400 |
| d2h-pageable-nonblocking | 15.089 GB/s | 0.8811% | 17.82 s | 500 |
| d2h-pinned-blocking | 27.297 GB/s | 2.7212% | 19.64 s | 1000 |
| d2h-pinned-nonblocking | 27.948 GB/s | 3.1904% | 19.33 s | 1000 |

The 4 GiB results from the previous alignment remain in the raw result directory as
`qualification-<direction>-<variant>-4g-*` and are not mixed into this set: payload is part
of the configuration identity.
