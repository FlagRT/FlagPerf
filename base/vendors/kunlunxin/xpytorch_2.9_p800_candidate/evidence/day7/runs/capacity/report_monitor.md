# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | partial |
| collector | xpu-smi -m and selected -q |
| reasons | ["kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87: 1/10 valid samples"] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4098131131145908, "finished_offset_s": 5.168327406048775, "rank": 0, "role": "measurement", "started_monotonic_ns": 4098131016460786, "started_offset_s": 5.05364228412509}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 35117, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "17b31af2010d0faa35b9efc8470c551d26310833c10642e076f3b7ccbe733e44"} |
| parsed_samples | {"bytes": 2323, "path": "benchmark-monitor/samples.jsonl", "sha256": "60759ca273bd6183a0b001bf8b722c46b5e12bfca184551177a77b36d9573555"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 1} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 1 | 21 | 21 | 21 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 1 | 88106 | 88106 | 88106 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
