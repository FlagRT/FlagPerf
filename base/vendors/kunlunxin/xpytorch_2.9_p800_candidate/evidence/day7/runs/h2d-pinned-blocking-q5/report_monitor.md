# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4097170906961118, "finished_offset_s": 25.00768722081557, "rank": 0, "role": "measurement", "started_monotonic_ns": 4097152913914383, "started_offset_s": 7.014640485867858}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 139077, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "f2d7aa31291b68b95cbe50d3b3fba09f42bb3b2a54d1d9b4beea17ff7b943791"} |
| parsed_samples | {"bytes": 8963, "path": "benchmark-monitor/samples.jsonl", "sha256": "7dc5f20d2bf0e3b4eca3bf74461bf4bda887f34187566a54d1fcf43475f7020a"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 19} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 19 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 19 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
