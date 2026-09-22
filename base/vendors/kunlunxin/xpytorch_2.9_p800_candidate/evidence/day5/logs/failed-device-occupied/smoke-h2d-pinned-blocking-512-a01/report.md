# FlagPerf Base Benchmark

## Run

| Field | Evidence |
|---|---|
| run_id | benchmark-d8f78c7cb33e4a47885f0585445582f2 |
| vendor | kunlunxin |
| vendor_display_name | Kunlunxin P800 |
| case | interconnect-h2d:P800 |
| status | failed |
| execution_status | failed |
| measurement_status | failed |
| monitoring_status | not-run |
| failure_stage | host-preflight |
| error | selected physical device 5 is occupied |
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
  "max_cv_percent": 5,
  "min_measurement_seconds": 15,
  "mode": "smoke",
  "repetitions_required": 5,
  "scope": "single-card native XPYTORCH interconnect-h2d:P800; candidate runtime"
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
| chip | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-h2d/kunlunxin/P800/case_config.yaml | 80d44b77ebbe4ca890da869683936bc1e763da8d3ac3943c76e244e8c5a49ab1 | not recorded |
| generic | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-h2d/case_config.yaml | 601f289d6cdbf0f1571d4fb8bd9f4a6e3b9fbc700374bd2051eb482a4f785191 | not recorded |
| override | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-h2d/kunlunxin/P800/case_config.smoke-pinned-blocking.yaml | a5fe4f2dd331437b7f2e478c57e259f6db24691895a153997dcbcbc642c87b79 | not recorded |

## Evidence

[Monitor report](report_monitor.md)

- [summary.json](summary.json)
- [case-assets.json](case-assets.json)
- [benchmark-result.json](benchmark-result.json)
- [runner.log](runner.log)
- [benchmark-monitor/summary.json](benchmark-monitor/summary.json)

Host preflight: not recorded

Host postflight: not recorded

This report preserves the experiment status. Missing evidence is not evidence of success.
