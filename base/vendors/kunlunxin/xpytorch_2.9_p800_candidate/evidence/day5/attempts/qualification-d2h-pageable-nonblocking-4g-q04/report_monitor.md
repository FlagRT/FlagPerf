# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3913635193017994, "finished_offset_s": 44.496254126075655, "rank": 0, "role": "measurement", "started_monotonic_ns": 3913604871480030, "started_offset_s": 14.174716162029654}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 202596, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "9196c8d6088b26208a5f62b287bd02c298074e72c2e47dc1dbc9154a92602ac5"} |
| parsed_samples | {"bytes": 15996, "path": "benchmark-monitor/samples.jsonl", "sha256": "bbd36e558f232965725a5240f183a60fe01d836c9ab0024aaed9a7d7d86a1ef5"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 30} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 30 | 17 | 48.0 | 63 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 30 | 4260 | 4260.0 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
