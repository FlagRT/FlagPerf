# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3923157264758512, "finished_offset_s": 24.113299807999283, "rank": 0, "role": "measurement", "started_monotonic_ns": 3923139244662073, "started_offset_s": 6.093203369062394}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 108962, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "4c7c2985c9775ae469a6738f5ec65890752bb9a56245a10a1e63b682301bde6d"} |
| parsed_samples | {"bytes": 8653, "path": "benchmark-monitor/samples.jsonl", "sha256": "2663b4f1c126c63ae2a9f834355ef4258eb050ed0f5d101f9356c7b238740938"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 19} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 19 | 20 | 47 | 56 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 19 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
