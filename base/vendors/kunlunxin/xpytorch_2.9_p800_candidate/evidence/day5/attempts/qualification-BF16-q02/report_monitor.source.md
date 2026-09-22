# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3886123799223661, "finished_offset_s": 128.60610412294045, "rank": 0, "role": "measurement", "started_monotonic_ns": 3886003247961262, "started_offset_s": 8.054841723758727}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 674353, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "8e743b08b15316f550b0532e2643bc850ec7815d95cc5d1339d7665eb3c018d5"} |
| parsed_samples | {"bytes": 43466, "path": "benchmark-monitor/samples.jsonl", "sha256": "a2665eb35a16c01f676ada3dff3df04a92f8b8604dc136a28aecad577796dc95"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 121} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 121 | 32 | 100 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 121 | 390 | 390 | 390 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
