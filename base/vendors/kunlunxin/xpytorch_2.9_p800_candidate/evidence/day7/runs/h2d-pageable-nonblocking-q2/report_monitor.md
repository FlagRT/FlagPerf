# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4096943431438498, "finished_offset_s": 23.990946582052857, "rank": 0, "role": "measurement", "started_monotonic_ns": 4096926337553063, "started_offset_s": 6.89706114679575}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 133916, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "9087fb684fbfc983f7672480bf5a0de0985b9f4bd97467c9f61227af1eccd878"} |
| parsed_samples | {"bytes": 8650, "path": "benchmark-monitor/samples.jsonl", "sha256": "99f18befcda2195ee6a512f9a5d7b941b0f24e2357687d563cafccee8fe00c32"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 17} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 17 | 61 | 91 | 98 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 17 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
