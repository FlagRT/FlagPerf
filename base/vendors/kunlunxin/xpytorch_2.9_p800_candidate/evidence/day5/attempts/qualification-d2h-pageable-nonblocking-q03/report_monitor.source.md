# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3903554112186897, "finished_offset_s": 42.868114861194044, "rank": 0, "role": "measurement", "started_monotonic_ns": 3903516314634030, "started_offset_s": 5.07056199433282}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 227645, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "6e904f0dbcbaa8e324cc0a81593ac38b22faf68e7db7b417884d60a6c3dc06a9"} |
| parsed_samples | {"bytes": 14640, "path": "benchmark-monitor/samples.jsonl", "sha256": "9d00336ef7a99663733a4013a108dbf2f4bb11b27e84ccadf0f19d72ea0b9ac0"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 38} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 38 | 15 | 54.0 | 68 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 38 | 228 | 228.0 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
