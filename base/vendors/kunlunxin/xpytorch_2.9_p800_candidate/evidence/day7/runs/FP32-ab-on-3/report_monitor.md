# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4098324121597638, "finished_offset_s": 28.67707199184224, "rank": 0, "role": "measurement", "started_monotonic_ns": 4098305350585866, "started_offset_s": 9.906060219742358}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 155169, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "72133b7e8ed55609c381d3e998d864e83c45c03b9a839a7ddb184caebe6dc580"} |
| parsed_samples | {"bytes": 10042, "path": "benchmark-monitor/samples.jsonl", "sha256": "5ca044e8549ae17a87180af7527a4cd1c49a574e24958fd7f62665bece55d6a8"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 19} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 19 | 100 | 100 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 19 | 1190 | 1190 | 1190 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
