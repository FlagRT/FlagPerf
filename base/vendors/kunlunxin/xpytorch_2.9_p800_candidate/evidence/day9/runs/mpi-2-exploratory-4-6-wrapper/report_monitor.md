# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | partial |
| collector | xpu-smi -m and selected -q |
| reasons | ["kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6: 0/10 valid samples", "kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897: 0/10 valid samples"] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6", "finished_monotonic_ns": 4430434691169861, "finished_offset_s": 10.676629812456667, "rank": 0, "role": "measurement", "started_monotonic_ns": 4430434690324469, "started_offset_s": 10.675784420222044}, {"device_id": "kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897", "finished_monotonic_ns": 4430434691190520, "finished_offset_s": 10.67665047198534, "rank": 1, "role": "measurement", "started_monotonic_ns": 4430434690325079, "started_offset_s": 10.67578503023833}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 123269, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "003693ca7cbe6f7cb0c4f9c23bc6a11de1e2283883ffe820a6d425ac41822d1d"} |
| parsed_samples | {"bytes": 8676, "path": "benchmark-monitor/samples.jsonl", "sha256": "465169c0571d75f23944c25f324011940ecdb6b4585eff5c587aafc244f448c0"} |
| sample_counts | {"kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897": 0, "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6": 0} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
