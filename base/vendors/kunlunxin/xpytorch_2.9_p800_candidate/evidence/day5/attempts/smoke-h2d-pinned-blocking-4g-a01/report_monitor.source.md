# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | not-run |
| collector | xpu-smi -m and selected -q |
| reasons | not recorded |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": false, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [] |
| lifecycle_windows | not recorded |
| raw_samples | not recorded |
| parsed_samples | not recorded |
| sample_counts | not recorded |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
