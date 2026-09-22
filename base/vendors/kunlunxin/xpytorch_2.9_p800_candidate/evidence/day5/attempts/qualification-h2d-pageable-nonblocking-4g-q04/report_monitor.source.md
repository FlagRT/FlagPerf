# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3912321617893117, "finished_offset_s": 48.441658440977335, "rank": 0, "role": "measurement", "started_monotonic_ns": 3912293993481406, "started_offset_s": 20.817246729973704}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 284515, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "2f14364ec37a064052362b86183a95d0ff5d37d15b33978a1238e6e786169ce6"} |
| parsed_samples | {"bytes": 18321, "path": "benchmark-monitor/samples.jsonl", "sha256": "842503721eeb15fd7b01964e20b507ba6926451b880770d12c9a395234e9a28f"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 28} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 28 | 18 | 93.5 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 28 | 4260 | 4260.0 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
