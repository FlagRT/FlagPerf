# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3886552584491375, "finished_offset_s": 128.7188566988334, "rank": 0, "role": "measurement", "started_monotonic_ns": 3886432041985522, "started_offset_s": 8.176350845955312}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 550657, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "e6e3607ae096ef4833155f9311d19cfa7637de35b3af39e88d845432d3c67abc"} |
| parsed_samples | {"bytes": 43474, "path": "benchmark-monitor/samples.jsonl", "sha256": "f1e09402ecaeda16fa84c8633127ee6f722f39418656c83b75c99145807eede5"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 120} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 120 | 100 | 100.0 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 120 | 390 | 390.0 | 390 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
