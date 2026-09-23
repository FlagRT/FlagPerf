# FlagPerf Base Benchmark

## Run

| Field | Evidence |
|---|---|
| run_id | benchmark-062c845b6b454cb39c78da75a84df19a |
| vendor | kunlunxin |
| vendor_display_name | Kunlunxin P800 |
| case | main_memory-bandwidth:P800 |
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

Missing ranks: [0]

## Qualification

```json
{
  "expected_ranks": [
    0
  ],
  "max_cv_percent": 5,
  "min_measurement_seconds": 15,
  "mode": "timeout",
  "repetitions_required": 5,
  "scope": "single-card native XPYTORCH main_memory-bandwidth:P800; candidate runtime"
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
| chip | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/main_memory-bandwidth/kunlunxin/P800/case_config.yaml | 088b4db94c36f9fea751cf7e8d700149635e79ad4e3ab8d4574e6190d564da1b | not recorded |
| generic | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/main_memory-bandwidth/case_config.yaml | 10f598c059e893845851f892d1cd332a14d6c31b3a005ffb950f614fd66c3726 | not recorded |
| override | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/main_memory-bandwidth/kunlunxin/P800/case_config.timeout.yaml | b8c36c0410aab2a3a903d82ccb6a43ab8a0595eb4ec5e14f4b25ab53a2ced3eb | not recorded |

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
