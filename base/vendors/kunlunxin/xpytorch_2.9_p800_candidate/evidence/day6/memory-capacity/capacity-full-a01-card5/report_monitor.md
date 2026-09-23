# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | partial |
| collector | xpu-smi -m and selected -q |
| reasons | ["kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87: 1/10 valid samples"] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3977605602311155, "finished_offset_s": 5.016381793655455, "rank": 0, "role": "measurement", "started_monotonic_ns": 3977605490359647, "started_offset_s": 4.904430286027491}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 35029, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "4caad5065a42c3038a4988ea46e2939a8e270ef9b823c10e1f4bd058c2b0380a"} |
| parsed_samples | {"bytes": 2320, "path": "benchmark-monitor/samples.jsonl", "sha256": "835f61e19525e0e3a48c261b6597028cd368f02ea0d322ba8bc10b2d9394d784"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 1} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 1 | 20 | 20 | 20 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 1 | 166 | 166 | 166 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
