# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3913937779533500, "finished_offset_s": 29.3208161406219, "rank": 0, "role": "measurement", "started_monotonic_ns": 3913922588741263, "started_offset_s": 14.130023903679103}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 170132, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "3372de21a5d87a4e8961487c7162d5e6e1b1af3045f0bfb2942ce3fb00ca5867"} |
| parsed_samples | {"bytes": 10983, "path": "benchmark-monitor/samples.jsonl", "sha256": "27bcc32fe0b1fa816a85a0da869c37ad23e75a5bc74309a55c5fa6ebc8dca2cb"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 16} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 16 | 0 | 0.0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 16 | 4260 | 4260.0 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
