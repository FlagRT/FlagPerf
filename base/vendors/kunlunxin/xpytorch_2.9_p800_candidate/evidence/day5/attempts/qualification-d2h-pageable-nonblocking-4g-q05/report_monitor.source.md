# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3913692331227155, "finished_offset_s": 40.61409354489297, "rank": 0, "role": "measurement", "started_monotonic_ns": 3913664924789003, "started_offset_s": 13.207655393052846}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 227429, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "45ce5bf4ae3993c6f21e10b32a888e7f1fb98ec322379d657e96d1fbb746a52e"} |
| parsed_samples | {"bytes": 14659, "path": "benchmark-monitor/samples.jsonl", "sha256": "24e8150dbd847727b9424320f35ff97dd9687c5add97d4c98c7cb916cae675c4"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 27} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 27 | 11 | 53 | 63 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 27 | 4260 | 4260 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
