# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4096334805583076, "finished_offset_s": 55.97156448988244, "rank": 0, "role": "measurement", "started_monotonic_ns": 4096288595250398, "started_offset_s": 9.761231811717153}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 300645, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "b6f33862b5308c43c4e2c2c7d62a2f0ce5e3e89cfea15ddaea0ec1044e4516b1"} |
| parsed_samples | {"bytes": 19426, "path": "benchmark-monitor/samples.jsonl", "sha256": "3ca443e8707587871b73346aa6195e5d3da96fef04b30aeae338679fb8565117"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 46} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 46 | 100 | 100.0 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 46 | 1062 | 1062.0 | 1062 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
