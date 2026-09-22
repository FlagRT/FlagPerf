# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3902395636840533, "finished_offset_s": 38.45096692210063, "rank": 0, "role": "measurement", "started_monotonic_ns": 3902362421985919, "started_offset_s": 5.2361123082228005}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 168315, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "9603fff9845c2a07a31e4e02e641b1f2cae54dd1f76f5ec9ca6287c6a4f56334"} |
| parsed_samples | {"bytes": 13314, "path": "benchmark-monitor/samples.jsonl", "sha256": "047e7379ad2046f2aa7e7933d6eecf67794d22434b300bc2fcf6c7e8cd20c5f1"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 33} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 33 | 80 | 95 | 99 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 33 | 228 | 228 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
