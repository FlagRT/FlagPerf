# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3912388588954605, "finished_offset_s": 47.02925694035366, "rank": 0, "role": "measurement", "started_monotonic_ns": 3912362892969043, "started_offset_s": 21.3332713781856}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 274143, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "f5197fc67fdbb63b106836df89e3a5aa141eedfd245bf9fd1c19e1e9f005dd02"} |
| parsed_samples | {"bytes": 17663, "path": "benchmark-monitor/samples.jsonl", "sha256": "c99e71cbcb8e0070a5fe0daf8f9b5ca4d18efd1484b625de1b75e87a2708024d"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 26} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 26 | 45 | 93.0 | 99 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 26 | 4260 | 4260.0 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
