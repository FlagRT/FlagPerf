# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3887390596190267, "finished_offset_s": 36.487900625914335, "rank": 0, "role": "measurement", "started_monotonic_ns": 3887359584055716, "started_offset_s": 5.475766075309366}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 195999, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "8a345bb0ba6944e3e8c780ce9017d2c8ef9fedf2f4e95aaa2cb84f128c16d77d"} |
| parsed_samples | {"bytes": 12608, "path": "benchmark-monitor/samples.jsonl", "sha256": "490a0b35ebe8797c331c5e50c6dc33b3f8ee3695c1eb5a6b66d594925d1c6c5f"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 31} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 31 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 31 | 326 | 326 | 326 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
