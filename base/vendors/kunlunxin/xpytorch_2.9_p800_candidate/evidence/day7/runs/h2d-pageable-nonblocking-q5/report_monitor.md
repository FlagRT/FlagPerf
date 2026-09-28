# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4097027038664293, "finished_offset_s": 24.13704305002466, "rank": 0, "role": "measurement", "started_monotonic_ns": 4097009710217154, "started_offset_s": 6.808595911134034}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 134179, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "cfce67add960e8fc3de65312ae2282d57a19c8b75141a6d75cce001fecc21dd1"} |
| parsed_samples | {"bytes": 8651, "path": "benchmark-monitor/samples.jsonl", "sha256": "bb33c0a319268e85576b9cfcb78dde1e636efacfa4f34a30cba5e3cb95da6934"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 18} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 18 | 40 | 89.0 | 96 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 18 | 676 | 676.0 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
