# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3887237892823170, "finished_offset_s": 70.14016948314384, "rank": 0, "role": "measurement", "started_monotonic_ns": 3887173470706431, "started_offset_s": 5.718052744399756}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 367643, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "fa284318ddcfaa73973402e29188635d8d59edc1fdf33682601173a8c9ebe9e7"} |
| parsed_samples | {"bytes": 23553, "path": "benchmark-monitor/samples.jsonl", "sha256": "a2f6f92d7cefa41002638cb42c5b504ff0f14932a9b5cd661ff2f6673b75b3ae"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 65} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 65 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 65 | 326 | 326 | 326 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
