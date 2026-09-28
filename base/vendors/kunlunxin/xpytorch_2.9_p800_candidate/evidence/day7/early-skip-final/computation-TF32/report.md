# FlagPerf Base Benchmark

## Run

启动前跳过：blocked/unresolved: current stack TF32 route is not qualified; allow_tf32 A/B alone cannot establish support or hardware non-support

| Field | Evidence |
|---|---|
| run_id | benchmark-20260928T070508Z |
| vendor | kunlunxin |
| vendor_display_name | Kunlunxin P800 |
| case | computation-TF32:P800 |
| status | skipped |
| execution_status | not-run |
| measurement_status | not-run |
| monitoring_status | not-run |
| failure_stage | not recorded |
| error | not recorded |
| skip_reason | blocked/unresolved: current stack TF32 route is not qualified; allow_tf32 A/B alone cannot establish support or hardware non-support |

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

Missing ranks: []

## Case configuration

Precedence: generic < vendor < chip < override.

| Layer | Source | SHA256 | Snapshot |
|---|---|---|---|
| generic | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/computation-TF32/case_config.yaml | 8bc63a581a2aa8d7a187998c7d61e77e9d7534779954fe3a911615178cb0fbfd | not recorded |

## Evidence

[Monitor report](report_monitor.md)

- [summary.json](summary.json)
- [resolved-plan.json](resolved-plan.json)
- [benchmark-result.json](benchmark-result.json)

Host preflight: not recorded

Host postflight: not recorded

This report preserves the experiment status. Missing evidence is not evidence of success.
