# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4097501293009271, "finished_offset_s": 22.218061638996005, "rank": 0, "role": "measurement", "started_monotonic_ns": 4097485188815371, "started_offset_s": 6.113867739215493}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 123516, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "e62fd951fab08e47ac0bf0c9f7fef7ecc49547eb7e08a54ce2e39b2789a7abc4"} |
| parsed_samples | {"bytes": 7977, "path": "benchmark-monitor/samples.jsonl", "sha256": "4305af4d4e9e881157e9ad8251b4db7b02cf2fc955d06ae9f8efc3cf5f700387"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 17} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 17 | 36 | 52 | 67 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 17 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
