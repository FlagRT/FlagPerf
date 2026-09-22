# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3923078493027067, "finished_offset_s": 23.7166893966496, "rank": 0, "role": "measurement", "started_monotonic_ns": 3923060849714168, "started_offset_s": 6.073376497719437}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 105033, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "d24abdafa9b0ffd4559b6f3987c176860ebdf5a9a83fe19d9915ddecff720f0c"} |
| parsed_samples | {"bytes": 8316, "path": "benchmark-monitor/samples.jsonl", "sha256": "a4830cc4160d269b59e849ef8d333ef1517d6680b4ee9e09d37efff3c9c86e15"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 18} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 18 | 39 | 47.5 | 54 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 18 | 676 | 676.0 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
