# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3910795139682334, "finished_offset_s": 30.312487825751305, "rank": 0, "role": "measurement", "started_monotonic_ns": 3910773874242481, "started_offset_s": 9.047047972679138}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 164831, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "512cbc4e4a5bfa48eebe4b90e0b9b356c28f029c25a620feb5366249a757cce6"} |
| parsed_samples | {"bytes": 10684, "path": "benchmark-monitor/samples.jsonl", "sha256": "91d819d87bc6900b75f01ad08dc8e110ee1e50ca0887bd3b01b3745dc8ccf916"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 22} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 22 | 100 | 100.0 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 22 | 678 | 678.0 | 678 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
