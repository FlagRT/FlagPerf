# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3904348527468789, "finished_offset_s": 32.36582542210817, "rank": 0, "role": "measurement", "started_monotonic_ns": 3904324223499320, "started_offset_s": 8.06185595318675}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 175455, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "648e702113a7e818000e22c0acf559145a7b291a9b2aca15ee45f9ccbcc8ab09"} |
| parsed_samples | {"bytes": 11354, "path": "benchmark-monitor/samples.jsonl", "sha256": "09f97040696ffb6c99a95759ca202667c7602cb4831bcca7e32f09cf244b3886"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 25} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 25 | 31 | 100 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 25 | 422 | 422 | 422 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
