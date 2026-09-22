# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3888372598031444, "finished_offset_s": 22.320196566171944, "rank": 0, "role": "measurement", "started_monotonic_ns": 3888355651505930, "started_offset_s": 5.37367105204612}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 100451, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "2befeefbecfb6fedee849d8abb5801edaf3c95e045f30ee092ce98c4c06a201f"} |
| parsed_samples | {"bytes": 7966, "path": "benchmark-monitor/samples.jsonl", "sha256": "81b2bf08cdf0ca1b382098fd874c9769679f45c9aed1bb94da71e87847d6b202"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 17} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 17 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 17 | 218 | 218 | 218 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
