# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4097527167744546, "finished_offset_s": 22.307405678089708, "rank": 0, "role": "measurement", "started_monotonic_ns": 4097511095684388, "started_offset_s": 6.235345520079136}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 123513, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "95482be7ba6536b9b989262a5460ae469fbacbf8a306fbcc80d76810a52ea29a"} |
| parsed_samples | {"bytes": 7981, "path": "benchmark-monitor/samples.jsonl", "sha256": "d05be99da621f1ca857e426e3232c7d19645835a47f422643b973f10823315c6"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 16} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 16 | 31 | 48.5 | 57 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 16 | 676 | 676.0 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
