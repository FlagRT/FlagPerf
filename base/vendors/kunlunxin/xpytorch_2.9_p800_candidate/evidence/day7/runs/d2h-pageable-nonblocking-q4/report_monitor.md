# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4097576326684607, "finished_offset_s": 21.22102865204215, "rank": 0, "role": "measurement", "started_monotonic_ns": 4097561211118353, "started_offset_s": 6.10546239791438}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 118312, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "e2809e20f7a44e96598ded014a690b37933b5fc75313a2d330a42e9ba52bc0d6"} |
| parsed_samples | {"bytes": 7646, "path": "benchmark-monitor/samples.jsonl", "sha256": "443aba8085569a124c7af1bb05ed73de55e17920d9b2b509843ea4bcea60a152"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 16} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 16 | 34 | 50.5 | 65 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 16 | 676 | 676.0 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
