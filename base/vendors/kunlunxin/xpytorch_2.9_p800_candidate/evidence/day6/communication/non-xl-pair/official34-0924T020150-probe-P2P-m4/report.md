# FlagPerf Base Benchmark

## Run

| Field | Evidence |
|---|---|
| run_id | benchmark-bc9df92c088b4e97986da66ed7efe1ae |
| vendor | kunlunxin |
| vendor_display_name | Kunlunxin P800 |
| case | interconnect-P2P_intraserver:P800 |
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
| 0 | 0 | 3 | kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | /dev/xpu1 | /dev/xpu1 | 0000:2e:00.0 | 9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 |
| 1 | 1 | 4 | kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | /dev/xpu4 | /dev/xpu4 | 0000:84:00.0 | b7942319-362e-5a53-a8eb-a2d06fbe84f6 |

## Rank metrics

| Rank | Metric | Value | Unit |
|---|---|---|---|
| 0 | p2p-one-way-bandwidth | 30.623552370377183 | GB/s |
| 1 | p2p-one-way-bandwidth | 30.62112443778349 | GB/s |

| Rank | Metric | Value | Unit |
|---|---|---|---|
| 0 | p2p-one-way-bandwidth | 28.520405637451617 | GiB/s |
| 1 | p2p-one-way-bandwidth | 28.518144449017466 | GiB/s |

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
  "scope": "two-card native XPYTORCH interconnect-P2P_intraserver:P800; candidate runtime"
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

![Rank metrics](report-assets/benchmark-rank-p2p-one-way-bandwidth-gb-s.svg)

## Case configuration

Precedence: generic < vendor < chip < override.

| Layer | Source | SHA256 | Snapshot |
|---|---|---|---|
| chip | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-P2P_intraserver/kunlunxin/P800/case_config.yaml | 04957170062b5bb64493cb7828ad02d15581529f34ba3d88148c5b628597f21b | not recorded |
| generic | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-P2P_intraserver/case_config.yaml | 10f598c059e893845851f892d1cd332a14d6c31b3a005ffb950f614fd66c3726 | not recorded |
| override | /home/kzhang519/Zhiyu/runtime-team/FlagPerf/base/benchmarks/interconnect-P2P_intraserver/kunlunxin/P800/case_config.smoke.yaml | db28317a6cb6e30e1ec571c1569d9c8c63a8a0d3e2ece8fee1bcaee657b628ae | not recorded |

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
