# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3902235786584393, "finished_offset_s": 38.885016803164035, "rank": 0, "role": "measurement", "started_monotonic_ns": 3902202204620945, "started_offset_s": 5.303053354844451}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 206736, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "0d198ed1abccccded7ef60c8bf3371206f50de829e6efacd7af471775592a24b"} |
| parsed_samples | {"bytes": 13317, "path": "benchmark-monitor/samples.jsonl", "sha256": "cc46f4883be05afeb7e183a31bd754553a62974f5aa4152f9099f608e7529bbe"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 33} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 33 | 74 | 98 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 33 | 228 | 228 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
