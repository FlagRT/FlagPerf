# FlagPerf Base Benchmark

## Run

| Field | Evidence |
|---|---|
| run_id | benchmark-3421105d8bc046f9bab1c6f7dc103a99 |
| vendor | kunlunxin |
| vendor_display_name | Kunlunxin P800 |
| case | interconnect-P2P_intraserver:P800 |
| status | failed |
| execution_status | failed |
| measurement_status | failed |
| monitoring_status | not-run |
| failure_stage | container-probe |
| error | bounded container execution timed out |
| skip_reason | not recorded |

## Runtime

| Field | Evidence |
|---|---|
| image | flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608 |
| image_id | not recorded |
| lock | {"image_manifest": {"image": "flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608", "image_id": "sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975", "path": "/home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/image-manifest.json", "release_stage": "candidate", "schema_version": 1, "sha256": "ab59dc1cd14b65a28ffaef7b1b21b1d0b51f5334619301659d4123a5c681acbf", "validated": false, "validation_scope": [], "validation_status": null}, "runtime_profile": "xpytorch_2.9_p800_candidate", "stack_lock": {"path": "/home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/stack.lock.yaml", "sha256": "8f200d0b955b7e60881bc13ed75c8784c2c38fe2c755d018c618f1dc7b46cded"}, "vendor": "kunlunxin"} |

## Device bindings

| Local rank | Framework ID | Physical ID | Resource | Host node | Container node | PCI | UUID/serial |
|---|---|---|---|---|---|---|---|

## Rank metrics

| Rank | Metric | Value | Unit |
|---|---|---|---|

Missing ranks: [0, 1]

## Qualification

```json
{
  "expected_ranks": [
    0,
    1
  ],
  "max_cv_percent": 5,
  "min_measurement_seconds": 15,
  "mode": "timeout",
  "repetitions_required": 5,
  "scope": "two-card native XPYTORCH interconnect-P2P_intraserver:P800; candidate runtime"
}
```

Correctness: not recorded

Measurement evidence: not_started

[Correctness artifact](artifacts/correctness-rank-0.json)

[Raw rank metric and timing](artifacts/metric-rank-0.json)

[Runtime UUID binding](artifacts/runtime-bindings.json)

[Container ownership and mapping](container-inspect.json)

Large-shape correctness checks fixed rows/columns with the full reduction dimension and full-output finiteness.
Native input/output dtype evidence does not certify internal arithmetic or universal CPU-fallback exclusion.

## Case configuration

Precedence: generic < vendor < chip < override.

| Layer | Source | SHA256 | Snapshot |
|---|---|---|---|
| chip | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-P2P_intraserver/kunlunxin/P800/case_config.yaml | b6344d5b922fee39f904586ab5370652f821b47a91a2ef90d7061b4acdcb9cea | not recorded |
| generic | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-P2P_intraserver/case_config.yaml | 10f598c059e893845851f892d1cd332a14d6c31b3a005ffb950f614fd66c3726 | not recorded |
| override | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-P2P_intraserver/kunlunxin/P800/case_config.timeout.yaml | 3829303e787d7e5c7bc893a4fc88d1acb74f3e4ae7df4871e4d4c8ef98b0f234 | not recorded |

## Evidence

[Monitor report](report_monitor.md)

- [summary.json](summary.json)
- [resolved-plan.json](resolved-plan.json)
- [case-assets.json](case-assets.json)
- [benchmark-result.json](benchmark-result.json)
- [runner.log](runner.log)
- [container-preflight.json](container-preflight.json)
- [benchmark-monitor/summary.json](benchmark-monitor/summary.json)

Host preflight: not recorded

Host postflight: not recorded

This report preserves the experiment status. Missing evidence is not evidence of success.
