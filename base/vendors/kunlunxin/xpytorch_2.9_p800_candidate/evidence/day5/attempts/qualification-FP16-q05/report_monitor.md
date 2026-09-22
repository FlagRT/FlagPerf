# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3885778156995494, "finished_offset_s": 62.43978882301599, "rank": 0, "role": "measurement", "started_monotonic_ns": 3885723864135326, "started_offset_s": 8.14692865498364}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 270272, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "bab93ea3288ebaa1a92b036c3d3cfd6ae2c9d7f71ffcf1b58a15480af9b8fc5e"} |
| parsed_samples | {"bytes": 21394, "path": "benchmark-monitor/samples.jsonl", "sha256": "9ff13253406cf9812d22d093d93214b6033e0efe3d529799dcfdc63b777d71e3"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 54} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 54 | 100 | 100.0 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 54 | 294 | 294.0 | 294 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
