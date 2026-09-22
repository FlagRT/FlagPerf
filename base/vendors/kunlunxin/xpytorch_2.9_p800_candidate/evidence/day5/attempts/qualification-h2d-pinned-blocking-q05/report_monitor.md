# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3902684465369536, "finished_offset_s": 40.50958633609116, "rank": 0, "role": "measurement", "started_monotonic_ns": 3902649257232651, "started_offset_s": 5.3014494511298835}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 176724, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "ad5b2f9ec9e6ae48fdc989eb4b4d075525ddd65b5263d8298ffeef977ec4f667"} |
| parsed_samples | {"bytes": 13935, "path": "benchmark-monitor/samples.jsonl", "sha256": "0b9436a380c52c21dd8b08cfccbd516255ac6d180f3e54552bc15d5dd5302506"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 35} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 35 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 35 | 228 | 228 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
