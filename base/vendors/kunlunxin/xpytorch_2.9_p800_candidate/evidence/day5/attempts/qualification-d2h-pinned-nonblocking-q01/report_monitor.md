# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3903987685091934, "finished_offset_s": 33.97097734315321, "rank": 0, "role": "measurement", "started_monotonic_ns": 3903958809561925, "started_offset_s": 5.095447333995253}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 147493, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "8e46dfc17578affeff6b0e08e787d481b5a9fdf67b7928506e8edf0d2a662e96"} |
| parsed_samples | {"bytes": 11673, "path": "benchmark-monitor/samples.jsonl", "sha256": "0c6747734db92c8d02c167420bd4a50fd0ab400ab3a1d887e807e48dc4d52dd0"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 29} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 29 | 23 | 100 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 29 | 228 | 228 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
