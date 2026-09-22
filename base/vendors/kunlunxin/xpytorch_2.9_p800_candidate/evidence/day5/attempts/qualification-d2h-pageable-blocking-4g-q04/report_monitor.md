# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3913295043410145, "finished_offset_s": 60.430604509077966, "rank": 0, "role": "measurement", "started_monotonic_ns": 3913251834658514, "started_offset_s": 17.22185287764296}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 270370, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "c361f424e8133367365abdcf3b8f76ec06fda2ae33151f5328560e33fdcba6c9"} |
| parsed_samples | {"bytes": 21291, "path": "benchmark-monitor/samples.jsonl", "sha256": "f7921d27edbfffa85e71e0ef4bcc1c214bb3e91c2bb7d106c69597a4d922b675"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 43} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 43 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 43 | 4260 | 4260 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
