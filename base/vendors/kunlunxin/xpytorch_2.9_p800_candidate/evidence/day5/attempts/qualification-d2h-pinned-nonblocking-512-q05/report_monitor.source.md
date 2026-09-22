# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3923683152908620, "finished_offset_s": 26.392405522055924, "rank": 0, "role": "measurement", "started_monotonic_ns": 3923662841947636, "started_offset_s": 6.0814445381984115}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 144425, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "a8c85fea4bd19427a4d94ccba9a7ca074fa80353abd09c2849c7ffd70f99ba5e"} |
| parsed_samples | {"bytes": 9340, "path": "benchmark-monitor/samples.jsonl", "sha256": "08e2065c6a7ffe4d8c4cd08c0524e8e89f58b1ec3f48152182ad8f93d5e0bb96"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 21} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 21 | 86 | 100 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 21 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
