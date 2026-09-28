# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4096693158512071, "finished_offset_s": 26.241886370815337, "rank": 0, "role": "measurement", "started_monotonic_ns": 4096675746746550, "started_offset_s": 8.830120849888772}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 144384, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "0f830b5acf2a49d18fd4c8758b70f9a992528583fb77451a097e2e0426ba8b3a"} |
| parsed_samples | {"bytes": 9350, "path": "benchmark-monitor/samples.jsonl", "sha256": "fc4e618c927550f071173afb70a39532d535861c5c72d1bff080ac5f532486ac"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 18} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 18 | 100 | 100.0 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 18 | 506 | 506.0 | 506 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
