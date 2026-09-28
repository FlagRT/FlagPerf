# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4096915809095224, "finished_offset_s": 24.70063541876152, "rank": 0, "role": "measurement", "started_monotonic_ns": 4096898057805434, "started_offset_s": 6.949345628730953}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 134234, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "f08ad7206e85c0ec2aafab123a0b74ae9f5c6f93229ec05871dfc3da122a0bb4"} |
| parsed_samples | {"bytes": 8645, "path": "benchmark-monitor/samples.jsonl", "sha256": "805f4f21fb8a1dc725e840823883afab423afa802f6ae972f2059e04881d5fa6"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 18} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 18 | 52 | 88.0 | 95 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 18 | 676 | 676.0 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
