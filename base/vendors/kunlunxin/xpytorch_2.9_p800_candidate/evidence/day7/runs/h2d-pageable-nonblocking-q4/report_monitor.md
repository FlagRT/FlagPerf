# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4096999037789270, "finished_offset_s": 23.788313348777592, "rank": 0, "role": "measurement", "started_monotonic_ns": 4096982181954869, "started_offset_s": 6.932478948030621}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 133913, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "3cf73d8d28cfc6d05f0ce36fe37a22943378ee4db9740e146b683e908a7f76c9"} |
| parsed_samples | {"bytes": 8646, "path": "benchmark-monitor/samples.jsonl", "sha256": "74b9ee9328d71af3c156b3db739c9f5c9fe805ee360979f3a8cb63419cc18383"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 17} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 17 | 82 | 90 | 98 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 17 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
