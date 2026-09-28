# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4098033492743592, "finished_offset_s": 33.64940597862005, "rank": 0, "role": "measurement", "started_monotonic_ns": 4098017356894433, "started_offset_s": 17.513556819874793}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 202068, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "a2e64ca69ebdfb53da90df19eba6f5b1d8e75c960ac12bfca7155b338a6e755d"} |
| parsed_samples | {"bytes": 13058, "path": "benchmark-monitor/samples.jsonl", "sha256": "1b1105932df7d2f91bd4f72c5aa826d23e8ab50dadd73b19092d38e9ee07edd0"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 16} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 16 | 100 | 100.0 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 16 | 12452 | 12452.0 | 12452 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
