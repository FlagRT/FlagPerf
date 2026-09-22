# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3915186818314165, "finished_offset_s": 36.18394321389496, "rank": 0, "role": "measurement", "started_monotonic_ns": 3915163587571711, "started_offset_s": 12.953200759831816}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 168351, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "5bef249f53a63e20183ec8d300aa91b7903ef6b3a0bf1c1c26ad210777cf5f88"} |
| parsed_samples | {"bytes": 13300, "path": "benchmark-monitor/samples.jsonl", "sha256": "6a76f57c6eb897b33150683fd8221d9be383176e8c9fb98f9bb000f5f3ea947d"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 24} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 24 | 0 | 0.0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 24 | 4260 | 4260.0 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
