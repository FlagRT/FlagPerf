# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3901881558544284, "finished_offset_s": 67.57642089342698, "rank": 0, "role": "measurement", "started_monotonic_ns": 3901819251237367, "started_offset_s": 5.269113976042718}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 291131, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "eabd6622f3e816bd25144b0ae2ada4ffb91b1c1c48ea25b131f3cc4d55ac1227"} |
| parsed_samples | {"bytes": 22891, "path": "benchmark-monitor/samples.jsonl", "sha256": "9b9805271a755221d6d7f11a17c1db5ef93699c3851d54f819ca434a1f5928b2"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 62} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 62 | 0 | 0.0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 62 | 228 | 228.0 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
