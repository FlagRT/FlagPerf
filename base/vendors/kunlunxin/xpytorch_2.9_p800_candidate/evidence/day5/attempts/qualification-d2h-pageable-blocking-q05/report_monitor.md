# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3903384596645055, "finished_offset_s": 63.2454388840124, "rank": 0, "role": "measurement", "started_monotonic_ns": 3903326501819852, "started_offset_s": 5.150613680947572}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 274189, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "8c4de180e4a5d2bccd8faef3c0725dcb0b6bb8d8c341e606f2e4e84435139702"} |
| parsed_samples | {"bytes": 21572, "path": "benchmark-monitor/samples.jsonl", "sha256": "18a234d3f704276c0a3cc2f17d3f24830d27f17cdd1789410adb0622e428fbb8"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 58} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 58 | 0 | 0.0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 58 | 228 | 228.0 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
