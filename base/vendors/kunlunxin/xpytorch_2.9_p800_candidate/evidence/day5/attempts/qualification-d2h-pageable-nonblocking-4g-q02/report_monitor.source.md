# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3913512303695801, "finished_offset_s": 42.536905377171934, "rank": 0, "role": "measurement", "started_monotonic_ns": 3913485717132361, "started_offset_s": 15.95034193713218}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 238139, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "c59cd43b5dff81ed29656cb80cda8816df038a179cbac32ba210dd6fbb7fbb84"} |
| parsed_samples | {"bytes": 15334, "path": "benchmark-monitor/samples.jsonl", "sha256": "42617c61d2f2355fa7fb36a5a803f4434215f71c0240d5c216b634b32c1dca80"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 27} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 27 | 33 | 52 | 63 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 27 | 4260 | 4260 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
