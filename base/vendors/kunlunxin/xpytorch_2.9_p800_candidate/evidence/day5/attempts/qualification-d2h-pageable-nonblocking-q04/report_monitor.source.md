# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3903608835088305, "finished_offset_s": 40.65762667031959, "rank": 0, "role": "measurement", "started_monotonic_ns": 3903573341307653, "started_offset_s": 5.163846018258482}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 216933, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "e06413d44f376f3ebd48076a4ec4e06248964eaf29fdae74af05653aeeede639"} |
| parsed_samples | {"bytes": 13973, "path": "benchmark-monitor/samples.jsonl", "sha256": "1106c5602751616471ca7414a5ba64ebe5bdf771a5a6a0242e1d31fd2b2f5275"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 35} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 35 | 49 | 61 | 68 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 35 | 228 | 228 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
