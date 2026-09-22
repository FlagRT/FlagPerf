# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3912252810399348, "finished_offset_s": 48.166239278856665, "rank": 0, "role": "measurement", "started_monotonic_ns": 3912227037870047, "started_offset_s": 22.393709978088737}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 232025, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "25e7f10bd3678ed45b894370f48de6285f0a96c14b8df6dac53e87777f936966"} |
| parsed_samples | {"bytes": 18329, "path": "benchmark-monitor/samples.jsonl", "sha256": "7eed50f7984c7086fad22c2e019f8ddad20ee8637e99f0111ecc21311b733712"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 26} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 26 | 58 | 95.0 | 99 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 26 | 4260 | 4260.0 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
