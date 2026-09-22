# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3913141567143619, "finished_offset_s": 61.59461441915482, "rank": 0, "role": "measurement", "started_monotonic_ns": 3913096729618220, "started_offset_s": 16.7570890202187}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 336683, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "6fc76679dfa6eb485eb9df1b7535c5873d788711aa45e34dd9e9e91b61ec3362"} |
| parsed_samples | {"bytes": 21628, "path": "benchmark-monitor/samples.jsonl", "sha256": "4e5032d8493ce834aa85b880fc1c874ab5a8d1d2cc87d81ecf3f97522afc5a96"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 45} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 45 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 45 | 4260 | 4260 | 4260 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
