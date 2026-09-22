# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3914585191299006, "finished_offset_s": 43.73658338468522, "rank": 0, "role": "measurement", "started_monotonic_ns": 3914554084348383, "started_offset_s": 12.629632761701941}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 248131, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "4f72bac172b66566a265f67e5fe2f34fec0d1b83a7d2737bd4f333f87cc9757f"} |
| parsed_samples | {"bytes": 15990, "path": "benchmark-monitor/samples.jsonl", "sha256": "46dbbcd7970346cecc6eeaa3a3f6a0a3bc5e732d883c227b19c774db5977b71a"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 31} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 31 | 95 | 96 | 99 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 31 | 2212 | 2212 | 2212 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
