# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3885980581956531, "finished_offset_s": 128.53620244702324, "rank": 0, "role": "measurement", "started_monotonic_ns": 3885860020133568, "started_offset_s": 7.974379484076053}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 550359, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "d31abd4a7425a337a7308bf6e2936d439cfb606e5330a3cc189ddff01a1838ad"} |
| parsed_samples | {"bytes": 43477, "path": "benchmark-monitor/samples.jsonl", "sha256": "f8e5b5b468c526ab2e6ed253ac4bd34b99eb53c41a898b313a10538bf00e939e"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 121} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 121 | 74 | 100 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 121 | 390 | 390 | 390 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
