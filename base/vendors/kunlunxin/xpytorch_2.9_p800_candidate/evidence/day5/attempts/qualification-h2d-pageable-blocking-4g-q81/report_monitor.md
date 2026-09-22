# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3911666167043361, "finished_offset_s": 63.46838954510167, "rank": 0, "role": "measurement", "started_monotonic_ns": 3911624182148852, "started_offset_s": 21.483495036140084}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 291640, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "945c06f28f961c924690fff1b1687e3d71e204fefa4770a5980d5e88da3bd172"} |
| parsed_samples | {"bytes": 22950, "path": "benchmark-monitor/samples.jsonl", "sha256": "f35499cb04a585826ad9f0f0e13e2338b92fe72725923cfc13fa7b9ea18dfd0b"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 42} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 42 | 0 | 0.0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 42 | 4260 | 4260.0 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
