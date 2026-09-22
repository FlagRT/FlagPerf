# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3911833965943448, "finished_offset_s": 64.74753209995106, "rank": 0, "role": "measurement", "started_monotonic_ns": 3911791638850764, "started_offset_s": 22.420439416076988}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 367261, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "d25a8c79be67d31a274789fb4b0bea931d28717dc12d4a5b62162a6824d9b05a"} |
| parsed_samples | {"bytes": 23600, "path": "benchmark-monitor/samples.jsonl", "sha256": "6f1355f85c1adca6c76b44cc48abb4f3bb3d8195ce74e4fa75be3323cf07c0dc"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 42} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 42 | 0 | 0.0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 42 | 4260 | 4260.0 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
