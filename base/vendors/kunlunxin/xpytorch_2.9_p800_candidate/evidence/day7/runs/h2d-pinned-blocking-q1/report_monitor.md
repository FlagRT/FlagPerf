# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4097055195381865, "finished_offset_s": 24.327988144010305, "rank": 0, "role": "measurement", "started_monotonic_ns": 4097037845589043, "started_offset_s": 6.978195322211832}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 134196, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "2c518435fb496d0ffc099775439d6f7936b6696a2efbd440c943118da3b06fa0"} |
| parsed_samples | {"bytes": 8636, "path": "benchmark-monitor/samples.jsonl", "sha256": "b4c4aa376ab06eab70db504c80744e81bcfb31b1a18fa3dee3581e301e54bcc1"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 18} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 18 | 0 | 0.0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 18 | 676 | 676.0 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
