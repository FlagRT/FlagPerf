# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3903496499131988, "finished_offset_s": 41.73623789101839, "rank": 0, "role": "measurement", "started_monotonic_ns": 3903459889010464, "started_offset_s": 5.126116367056966}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 181291, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "31a8121f5b3bb502fdc4d3bad7dc35e0eb754e94976796006b89a3d6ef43475b"} |
| parsed_samples | {"bytes": 14305, "path": "benchmark-monitor/samples.jsonl", "sha256": "d0df4a17f610bcdc8a33f03da081319f182955b49bff11f46382fbc600fad3c2"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 36} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 36 | 38 | 55.5 | 73 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 36 | 228 | 228.0 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
