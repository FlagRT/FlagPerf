# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3903665291157031, "finished_offset_s": 41.59437760384753, "rank": 0, "role": "measurement", "started_monotonic_ns": 3903628839810041, "started_offset_s": 5.143030613660812}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 181034, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "134d0e013410736d9402532a99bcf3b1bca11e133764492bab4285bb8aafa30d"} |
| parsed_samples | {"bytes": 14306, "path": "benchmark-monitor/samples.jsonl", "sha256": "ae96e0f06066a0528f0cf01647b731a657c292f47a80c47d991137e1772fa8c1"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 36} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 36 | 37 | 57.5 | 65 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 36 | 228 | 228.0 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
