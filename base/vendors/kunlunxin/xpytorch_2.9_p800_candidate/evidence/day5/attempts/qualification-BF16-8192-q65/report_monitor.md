# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3911308408300185, "finished_offset_s": 56.01996169099584, "rank": 0, "role": "measurement", "started_monotonic_ns": 3911262198193055, "started_offset_s": 9.809854560997337}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 244943, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "cb936e27fbd4ce7606b9d2960f25a1ecdb1cf92e12883a1e7f9983390285c5ea"} |
| parsed_samples | {"bytes": 19429, "path": "benchmark-monitor/samples.jsonl", "sha256": "67fec7477eabe8ecb3dd33d7ed34814a3439ba10c23dc3fd443e33b11e3f8ae7"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 47} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 47 | 55 | 100 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 47 | 1062 | 1062 | 1062 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
