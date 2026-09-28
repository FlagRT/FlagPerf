# P800 Day 7 review — 2026-09-24

## Verdict

**Scoped delivery, not full-week or eight-card acceptance.** Runtime remains candidate / validated:false. Per-case outcome is listed below; unstable groups are retained as unstable. CPU and static Ascend coverage is not Ascend hardware acceptance.

## Scope and provenance

- User-authorized Day 7 execution uses healthy idle card 5 for single-card work and a freshly preflighted 5+6 pair for two-rank work. No foreign-handle override, tenant termination, driver/firmware change or eight-card run.
- Historical Day 6 formal 4+7 and 3+4 runs were already completed by the user. Their 5811 indexed files and 16 curve summaries were rechecked without modification. Day 7 results do not replace those records.
- Measurement code: c1d900d8d89eadba48540b34ec7873fb26da4671. Release code: e373664aa5324a21c04196d1971f9ea013ab9f3c. Source/config/image/UUID identity is recorded per run; measurement-source/ retains the frozen measurement snapshot.
- Eight-card work remains blocked by health/resources and the existing exactly-two-rank implementation. Both implementation and hardware qualification are still required.
- Authorization and time budget: authorization.json. Raw files: base/result/p800-day7-20260924/. Portable evidence is a labelled sanitized projection, with original and archive SHA256 in provenance.json.

## Fixed defects

1. synchronize() and memory_info() used rank 0 instead of the current verified rank. c1d900d8 tracks the active rank after set_device succeeds. Reordered logical-device mappings are covered by regression. Fresh communication runs use the corrected code.
2. Unsupported cases with bounded CLI arguments and --result-dir incorrectly failed before producing skip evidence. The release fix routes supported bounded providers to the existing skip renderer before Docker, lease or host inspection. Five real skip cases and three regression methods cover reasons, no external calls, immutable existing output, and conflicting output paths.
3. A stale driver test fixture retained _bindings across cases; fixtures now isolate that state. System Python without torch is not used as the full regression environment.

## Five-run qualification

Computation uses 8192 cubed. Transfers use 512 MiB and mode-specific frozen iterations. Device memory bandwidth uses 4 GiB and read+write traffic. Every admitted sample has correctness, binding, synchronized timing >=15 seconds, monitoring and successful cleanup. CV is sample standard deviation / mean, ddof=1, limit 5%. Two-rank repetitions use the lower validated rank bandwidth, never a sum.

| Group | Outcome | Median | Unit | CV % | Runs |
|---|---|---:|---|---:|---:|
| FP32 | passed | 117.877805 | TFLOPS | 0.3662 | 5 |
| FP16 | passed | 258.784198 | TFLOPS | 0.0448 | 5 |
| BF16 | passed | 118.967960 | TFLOPS | 0.0018 | 5 |
| INT8 | passed | 503.903759 | TOPS | 0.3211 | 5 |
| h2d-pageable-blocking | passed | 10.014424 | GB/s | 1.1785 | 5 |
| h2d-pageable-nonblocking | passed | 18.844314 | GB/s | 2.2547 | 5 |
| h2d-pinned-blocking | unstable | 20.113560 | GB/s | 6.3207 | 5 |
| h2d-pinned-nonblocking | passed | 19.289542 | GB/s | 4.1652 | 5 |
| d2h-pageable-blocking | passed | 10.802936 | GB/s | 3.7892 | 5 |
| d2h-pageable-nonblocking | passed | 17.548727 | GB/s | 3.3230 | 5 |
| d2h-pinned-blocking | unstable | 24.727412 | GB/s | 5.5925 | 5 |
| d2h-pinned-nonblocking | unstable | 25.154119 | GB/s | 5.9324 | 5 |
| memory-bandwidth | passed | 2235.901857 | GB/s | 0.0008 | 5 |
| comm-MPI-m64 | incomplete | — | — | — | 0 |
| comm-P2P-m64 | incomplete | — | — | — | 0 |

No slow samples were removed and no configuration/UUID groups were pooled. A run can pass its correctness/lifecycle checks while the five-run group is unstable. See qualification/*.json for min/max/mean/std and every run path.

## Two-card curve and timeout regression

| Case | MiB | Outcome | Lower-rank GB/s | Per-rank windows (s) |
|---|---:|---|---:|---|
| MPI | 4 | not-qualified | — |  |
| MPI | 16 | not-qualified | — |  |
| MPI | 64 | not-qualified | — |  |
| MPI | 256 | not-qualified | — |  |
| P2P | 4 | not-qualified | — |  |
| P2P | 16 | not-qualified | — |  |
| P2P | 64 | not-qualified | — |  |
| P2P | 256 | not-qualified | — |  |

At world size 2, AllReduce busbw equals algbw (2*(N-1)/N = 1). P2P is one-way rank 0 to rank 1. The first three curve sizes are one qualified run each, not five-run stability claims; the 256 MiB point has a separate five-run group.
The XPULink bandwidth discrepancy remains open. New card-pair results do not establish a causal XPULink-vs-PCIe comparison. Ring counts alone do not prove an interconnect maximum.

Timeout evidence:


## Capacity and monitoring

Capacity search is not a bandwidth qualification. Full bounded search, held allocations, OOM classification and release evidence are retained. Search duration can produce partial monitoring; that state is not rewritten as passed.

Capacity outcome: {"status": "partial", "measurement_status": "passed", "monitoring_status": "partial", "correctness_status": "passed", "cleanup_status": "passed", "lease_released": true}.
Held allocation: 98132 MiB; search window 0.114685122 s. See runs/capacity/artifacts/.

Alternating monitor off/on: three pairs per configured case; positive numbers mean monitored measurement was faster. These are observed differences including host variation, not causal overhead estimates.

| Group | Complete pairs | Same identity | Median (on/off - 1), % | Outcome |
|---|---:|---|---:|---|
| FP32 | 0 | True | — | incomplete |
| FP16 | 0 | True | — | incomplete |
| h2d-pinned | 3 | True | -0.9591 | complete |
| comm-MPI | 0 | False | — | incomplete |
| comm-P2P | 0 | False | — | incomplete |

## Offline, reports and rejection gates

- base-release: 220 tests, exit 0.
- pr0-release: 32 tests, exit 0.
- toolkit-release: 80 tests, exit 0.
- day6-release: 18 tests, exit 0.
- P800 15/15 dry-runs and Ascend 15/15 dry-runs complete: ten applicable, five skipped each. Ascend two-rank profile and explicit P2P smoke config are required.
- Public distribution remains 6/7; the existing Ascend private host-path check fails. Retained in regression/public-distribution.log; no unrelated Ascend configuration is changed.
- Passed/partial/failed/skipped reports were regenerated twice in temporary copies. Output hashes deterministic; original files and summary bytes unchanged. See release-report-regeneration.json.
- All five unsupported P800 cases accept --result-dir plus bounded arguments and expired reservation, then skip before any external execution; see release-skips.json. Capacity without opt-in and eight ranks are rejected before hardware.
- The initial overlong standalone preflight timeout, wrong Ascend planning profile and pre-fix early-skip error remain in their original logs. They are not hardware failures.

## Cleanup and delivery boundary

Final resource check: final-resource-check.json. It independently inspects the task container IDs, reacquires both lock protocols and checks selected cards under lock. Shared internal network is retained. All attempt statuses are listed in run-inventory.json.

Observed states: {"passed": 78, "partial": 7}.

Remaining work: eight-rank implementation and healthy whole-host qualification; any unstable qualification groups listed above; vendor investigation of XPULink throughput and BF16 arithmetic/dispatch limits; known 4 GiB pinned-nonblocking runtime error. Toolkit feature expansion and interserver tests remain outside this Base delivery.

Support/runbook: ../../../../../docs/p800-day7.md (from this evidence directory), or base/docs/p800-day7.md from the repository root. No push or publication was performed.
