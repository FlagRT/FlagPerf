# FlagPerf Base Benchmark

## Run

| Field | Evidence |
|---|---|
| run_id | benchmark-5c44c1a48b7e4824bdfa31dc5a5eeefc |
| vendor | kunlunxin |
| vendor_display_name | Kunlunxin P800 |
| case | interconnect-MPI_intraserver:P800 |
| status | passed |
| execution_status | passed |
| measurement_status | passed |
| monitoring_status | not-run |
| failure_stage | not recorded |
| error | not recorded |
| skip_reason | not recorded |

## Runtime

| Field | Evidence |
|---|---|
| image | flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608 |
| image_id | sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975 |
| lock | {"image_manifest": {"image": "flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608", "image_id": "sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975", "path": "/home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/image-manifest.json", "release_stage": "candidate", "schema_version": 1, "sha256": "ab59dc1cd14b65a28ffaef7b1b21b1d0b51f5334619301659d4123a5c681acbf", "validated": false, "validation_scope": [], "validation_status": null}, "runtime_profile": "xpytorch_2.9_p800_candidate", "stack_lock": {"path": "/home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/stack.lock.yaml", "sha256": "8f200d0b955b7e60881bc13ed75c8784c2c38fe2c755d018c618f1dc7b46cded"}, "vendor": "kunlunxin"} |

## Device bindings

| Local rank | Framework ID | Physical ID | Resource | Host node | Container node | PCI | UUID/serial |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 4 | kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | /dev/xpu4 | /dev/xpu4 | 0000:84:00.0 | b7942319-362e-5a53-a8eb-a2d06fbe84f6 |
| 1 | 1 | 7 | kunlunxin/4922595e-feb9-5762-9082-d8fd7f11abf5 | /dev/xpu6 | /dev/xpu6 | 0000:bb:00.0 | 4922595e-feb9-5762-9082-d8fd7f11abf5 |

## Rank metrics

| Rank | Metric | Value | Unit |
|---|---|---|---|
| 0 | allreduce-algbw | 28.720779126552916 | GB/s |
| 1 | allreduce-algbw | 34.39258338418254 | GB/s |

| Rank | Metric | Value | Unit |
|---|---|---|---|
| 0 | allreduce-algbw | 26.748309961103757 | GiB/s |
| 1 | allreduce-algbw | 32.030589305034404 | GiB/s |

Effective one-way API copy bandwidth; both units use identical bytes and synchronized time.
Full payload content is checked outside measurement. non_blocking is a request, not proof of overlap.

Missing ranks: []

## Qualification

```json
{
  "expected_ranks": [
    0,
    1
  ],
  "max_cv_percent": 5,
  "min_measurement_seconds": 15,
  "mode": "smoke",
  "repetitions_required": 5,
  "scope": "two-card native XPYTORCH interconnect-MPI_intraserver:P800; candidate runtime"
}
```

Correctness: passed

Measurement evidence: passed

[Correctness artifact](artifacts/correctness-rank-0.json)

[Raw rank metric and timing](artifacts/metric-rank-0.json)

[Runtime UUID binding](artifacts/runtime-bindings.json)

[Container ownership and mapping](container-inspect.json)

Transfer correctness checks the complete payload and buffer reuse outside timing.
Native input/output dtype evidence does not certify internal arithmetic or universal CPU-fallback exclusion.

![Rank metrics](report-assets/benchmark-rank-allreduce-algbw-gb-s.svg)

## Case configuration

Precedence: generic < vendor < chip < override.

| Layer | Source | SHA256 | Snapshot |
|---|---|---|---|
| chip | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-MPI_intraserver/kunlunxin/P800/case_config.yaml | 8e7beae642ea37820e7b28f7d28aa905b2bf02241ab1487fad215ffdc9b17651 | not recorded |
| generic | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-MPI_intraserver/case_config.yaml | 4084480e3af08ab4c7a2c18d1256f9f951d58c5242ffb032657a65cdc68b9afd | not recorded |
| override | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-MPI_intraserver/kunlunxin/P800/case_config.smoke.yaml | d46a4d988ac19f4ddcf40dd0f05bb02f2295dcacc5f5f098774b00de68364099 | not recorded |

## Evidence

[Monitor report](report_monitor.md)

- [summary.json](summary.json)
- [resolved-plan.json](resolved-plan.json)
- [case-assets.json](case-assets.json)
- [benchmark-result.json](benchmark-result.json)
- [runner.log](runner.log)
- [container-preflight.json](container-preflight.json)
- [benchmark-monitor/summary.json](benchmark-monitor/summary.json)

Host preflight: host-preflight/summary.json

Host postflight: host-postflight/summary.json

This report preserves the experiment status. Missing evidence is not evidence of success.
