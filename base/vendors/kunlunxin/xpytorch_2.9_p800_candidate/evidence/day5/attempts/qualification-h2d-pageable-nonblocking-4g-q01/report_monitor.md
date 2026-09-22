# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3912117641974208, "finished_offset_s": 46.23193609714508, "rank": 0, "role": "measurement", "started_monotonic_ns": 3912091424330569, "started_offset_s": 20.014292458072305}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 219604, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "fdde0fdf1daf6efc2cadcf125be30250d0221711ca6bce8003b0c4244ff31376"} |
| parsed_samples | {"bytes": 17335, "path": "benchmark-monitor/samples.jsonl", "sha256": "e323bc3c2aec6cc5963354a3d83bf89ba20ecd4f84452a7154da3052aa28eed5"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 27} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 27 | 43 | 95 | 99 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 27 | 4260 | 4260 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
