# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3887792218366000, "finished_offset_s": 76.4301740177907, "rank": 0, "role": "measurement", "started_monotonic_ns": 3887727461865424, "started_offset_s": 11.673673442099243}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 403661, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "8d00bf94d5a4e0e7e017422339b753a879ff24e742cb021237d590e936818adf"} |
| parsed_samples | {"bytes": 25850, "path": "benchmark-monitor/samples.jsonl", "sha256": "4a04b684c1bd3e0da15e438c8152a8a28608ab0a8c5d91fa1bc8f9c34e3b63a7"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 65} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 65 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 65 | 326 | 326 | 326 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
