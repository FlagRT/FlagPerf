# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3977472700516386, "finished_offset_s": 27.71218618704006, "rank": 0, "role": "measurement", "started_monotonic_ns": 3977463100154521, "started_offset_s": 18.111824322026223}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 175472, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "79d54cfd881de8adc67433c46f40a1b166b4d1c08d9d8e3fedb7fac3dbbdea92"} |
| parsed_samples | {"bytes": 11363, "path": "benchmark-monitor/samples.jsonl", "sha256": "f9596741b8b43dac70e372643cf055dccfc39ffa748d6502ce069c4f4ee70535"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 10} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 10 | 6 | 100.0 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 10 | 12452 | 12452.0 | 12452 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
