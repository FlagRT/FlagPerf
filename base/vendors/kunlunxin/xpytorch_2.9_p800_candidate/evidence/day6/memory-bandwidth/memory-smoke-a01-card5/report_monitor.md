# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | partial |
| collector | xpu-smi -m and selected -q |
| reasons | ["kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87: 0/10 valid samples"] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3977233727743593, "finished_offset_s": 4.996217937208712, "rank": 0, "role": "measurement", "started_monotonic_ns": 3977233727697564, "started_offset_s": 4.996171907987446}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 30194, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "41ff72db57d72281f9a2354a503837b6ecf408c06e84a1ae7fc563209cdd647d"} |
| parsed_samples | {"bytes": 1990, "path": "benchmark-monitor/samples.jsonl", "sha256": "461f86546bd86d181af197e6755c89009b1cba24ca015dd23b4a1c514922fca6"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 0} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
