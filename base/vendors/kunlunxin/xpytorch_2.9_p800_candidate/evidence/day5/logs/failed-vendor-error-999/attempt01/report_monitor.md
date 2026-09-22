# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | partial |
| collector | xpu-smi -m and selected -q |
| reasons | ["kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87: 0/10 valid samples"] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 40530, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "30e107d26ef68924c8a9465b4eab700917cc0d69ba6effd04e12d8e5735dd066"} |
| parsed_samples | {"bytes": 2654, "path": "benchmark-monitor/samples.jsonl", "sha256": "50f774dc6477a40c7522fcdc1d04bdb0ac30fb776dd404a800837ab791982525"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 0} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
