# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3903439998503273, "finished_offset_s": 40.867787692695856, "rank": 0, "role": "measurement", "started_monotonic_ns": 3903404165221109, "started_offset_s": 5.034505528863519}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 177110, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "6a22332553bd2d28dea257421d3322cddd22f7c8124946cac0579df73dd83a5b"} |
| parsed_samples | {"bytes": 13966, "path": "benchmark-monitor/samples.jsonl", "sha256": "3e1a1d6a4b74bde8bef29a85028e62591367d73656f9e6aac1fe0d482163ad62"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 36} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 36 | 35 | 60.5 | 70 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 36 | 228 | 228.0 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
