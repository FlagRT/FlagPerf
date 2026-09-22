# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3902568361301311, "finished_offset_s": 39.2040210198611, "rank": 0, "role": "measurement", "started_monotonic_ns": 3902534250933232, "started_offset_s": 5.093652940820903}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 172478, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "581a2e9cdc0dbade949d6d8499ecda3c90a60e808124b01a7bad16e0360ab0ec"} |
| parsed_samples | {"bytes": 13603, "path": "benchmark-monitor/samples.jsonl", "sha256": "77193951b5d191d0c4cd979b0d1184637a27de8991d876783f964faf880b8a2b"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 35} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 35 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 35 | 228 | 228 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
