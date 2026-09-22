# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3885624585359553, "finished_offset_s": 62.31276185903698, "rank": 0, "role": "measurement", "started_monotonic_ns": 3885570304994007, "started_offset_s": 8.032396313268691}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 266359, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "b84dac2b13602a90e55dbed1bfaddadea8b31e0478b24d570ef46ebadabfc4ca"} |
| parsed_samples | {"bytes": 21045, "path": "benchmark-monitor/samples.jsonl", "sha256": "6875f09dd0daa9d0016e752cc701e9e983131cc33b36162adbffe5f221ffc5d2"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 55} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 55 | 37 | 100 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 55 | 294 | 294 | 294 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
