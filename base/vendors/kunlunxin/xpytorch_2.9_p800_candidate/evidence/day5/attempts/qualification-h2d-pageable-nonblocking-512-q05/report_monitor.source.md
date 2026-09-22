# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3922343980503276, "finished_offset_s": 25.47190183820203, "rank": 0, "role": "measurement", "started_monotonic_ns": 3922325438930020, "started_offset_s": 6.930328582413495}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 144116, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "bc79c822e6083b0e58a73e2bb907976d4fdaec2fbdaeadf9ad03574d5976c513"} |
| parsed_samples | {"bytes": 9320, "path": "benchmark-monitor/samples.jsonl", "sha256": "1221aa093a5800ddccdc5852755bbeeeba41ae1c14d80d2babe2f65a84b2feef"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 19} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 19 | 88 | 94 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 19 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
