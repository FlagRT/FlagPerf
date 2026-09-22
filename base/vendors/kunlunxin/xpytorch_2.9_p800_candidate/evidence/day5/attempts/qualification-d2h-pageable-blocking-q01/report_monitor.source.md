# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3903071763118093, "finished_offset_s": 62.40007733972743, "rank": 0, "role": "measurement", "started_monotonic_ns": 3903014531148920, "started_offset_s": 5.1681081666611135}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 331185, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "e049bfa45316682bfba5e60dcabd22dd2b4b6d317e8967ba7fdc53a410ca23f3"} |
| parsed_samples | {"bytes": 21235, "path": "benchmark-monitor/samples.jsonl", "sha256": "4baf8f8ab0027874cd773755e3723b07bd93fc07e178a47e2480abbc1fe7dcac"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 57} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 57 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 57 | 228 | 228 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
