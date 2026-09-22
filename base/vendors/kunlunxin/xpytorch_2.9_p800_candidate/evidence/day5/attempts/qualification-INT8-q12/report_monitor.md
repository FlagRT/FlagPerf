# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3887646111227548, "finished_offset_s": 39.036259381100535, "rank": 0, "role": "measurement", "started_monotonic_ns": 3887615416148739, "started_offset_s": 8.341180571820587}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 172494, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "fad0991b1f0ad12d39179728bb75d182d14a2915a26dc7d956e750445711a34c"} |
| parsed_samples | {"bytes": 13603, "path": "benchmark-monitor/samples.jsonl", "sha256": "22be2169f4be1bf6a5238cefd0a49070621cf5ae27e8899c5c6e5939bdd5625d"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 31} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 31 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 31 | 164 | 326 | 326 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
