# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4097601488576799, "finished_offset_s": 21.435988530982286, "rank": 0, "role": "measurement", "started_monotonic_ns": 4097586191999766, "started_offset_s": 6.139411497861147}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 118320, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "ec8d2f78712e2dbadb9df10eb9141e68d5510a9ed3e8cfe1802d98f5edb5b229"} |
| parsed_samples | {"bytes": 7648, "path": "benchmark-monitor/samples.jsonl", "sha256": "2a304ea59f115ed5f1edbe4e10e4e8db3c306cd94d62dd6ff6386dbceb186bd6"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 15} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 15 | 45 | 56 | 64 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 15 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
