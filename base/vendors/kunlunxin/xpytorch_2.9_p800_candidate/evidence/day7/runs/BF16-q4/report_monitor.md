# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4096513466269211, "finished_offset_s": 55.95486477483064, "rank": 0, "role": "measurement", "started_monotonic_ns": 4096467255675441, "started_offset_s": 9.744271005038172}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 295780, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "b4b2aa61c570a0c211fc13abc365047f4d0607ae285629ba94894bf68290b98e"} |
| parsed_samples | {"bytes": 19091, "path": "benchmark-monitor/samples.jsonl", "sha256": "3c2719477e1b9224a78411abe88dd007dadeda550d850de9c7da6fe602e1537a"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 46} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 46 | 100 | 100.0 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 46 | 1062 | 1062.0 | 1062 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
