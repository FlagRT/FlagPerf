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
