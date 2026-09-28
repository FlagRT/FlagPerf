# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4097912543707047, "finished_offset_s": 28.73445857129991, "rank": 0, "role": "measurement", "started_monotonic_ns": 4097889885798339, "started_offset_s": 6.076549862977117}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 155078, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "fc3110a6fbde38e1fa1b9d37f8a7f3914e416668ae0dc49110d62d29e705a863"} |
| parsed_samples | {"bytes": 10006, "path": "benchmark-monitor/samples.jsonl", "sha256": "d66c9926c25c20d16e5b557fdd05517d3f208b81b62deb351a2ccd44b4b2644a"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 23} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 23 | 99 | 100 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 23 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
