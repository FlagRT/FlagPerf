# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3904182497595232, "finished_offset_s": 34.127803310751915, "rank": 0, "role": "measurement", "started_monotonic_ns": 3904153457173846, "started_offset_s": 5.087381924968213}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 151414, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "968fcf16abb74552e8db21920420a91969dbf33535f00603efefb5256be98d34"} |
| parsed_samples | {"bytes": 12007, "path": "benchmark-monitor/samples.jsonl", "sha256": "bed97bf1c7f13d2ce03ef4c0ec6dfd8cf2b02dd188a9b36c25680760c319324e"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 30} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 30 | 30 | 100.0 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 30 | 228 | 228.0 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
