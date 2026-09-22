# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3923118077041369, "finished_offset_s": 24.087808655109257, "rank": 0, "role": "measurement", "started_monotonic_ns": 3923100145765441, "started_offset_s": 6.1565327271819115}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 108959, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "97991f6d4d1f2f9acedb54021b8276bc82c9dd32e1a58d80dc44779059050402"} |
| parsed_samples | {"bytes": 8651, "path": "benchmark-monitor/samples.jsonl", "sha256": "86aca0cb6168903e5976ba54a72a42b0f940b84207988d7e2b4843ceedb0b439"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 18} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 18 | 29 | 44.5 | 55 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 18 | 676 | 676.0 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
