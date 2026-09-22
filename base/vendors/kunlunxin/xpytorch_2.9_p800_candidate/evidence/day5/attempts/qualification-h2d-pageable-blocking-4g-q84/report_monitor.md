# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3911921250168305, "finished_offset_s": 67.94812940387055, "rank": 0, "role": "measurement", "started_monotonic_ns": 3911877314452316, "started_offset_s": 24.012413415126503}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 312825, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "35f8f8bc7bd408f6b8a09ed55bd3e94ece53f83622e34b4b91be119a2b19963c"} |
| parsed_samples | {"bytes": 24620, "path": "benchmark-monitor/samples.jsonl", "sha256": "d5bb61fd606b280a6267af5aca670297d08c557cb93657fbd825ea9e383e7657"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 44} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 44 | 0 | 0.0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 44 | 4260 | 4260.0 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
