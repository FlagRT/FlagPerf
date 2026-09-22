# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3913574786220436, "finished_offset_s": 44.905990599188954, "rank": 0, "role": "measurement", "started_monotonic_ns": 3913543884142889, "started_offset_s": 14.003913052380085}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 248531, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "08f914e8966a0ddc619104b1406307990f42fa62d0626fc0dc40b51afe92f5fb"} |
| parsed_samples | {"bytes": 15999, "path": "benchmark-monitor/samples.jsonl", "sha256": "71e1cc00973e62e9d91dba6f3b1c553791307cc3e3b45722ec779c7dba326c47"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 31} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 31 | 23 | 43 | 59 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 31 | 4260 | 4260 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
