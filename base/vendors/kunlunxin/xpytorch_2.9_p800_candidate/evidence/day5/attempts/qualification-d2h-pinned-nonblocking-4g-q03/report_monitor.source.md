# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3914975381632548, "finished_offset_s": 30.941846140660346, "rank": 0, "role": "measurement", "started_monotonic_ns": 3914953249042184, "started_offset_s": 8.809255776926875}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 170466, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "706ea57feb5d1effc445702f4ef8e77178b4abd5956ce7626870a34854705f85"} |
| parsed_samples | {"bytes": 11009, "path": "benchmark-monitor/samples.jsonl", "sha256": "2cb36d4d94715a2c655cd919d0415c4b21f78e5a92f09925309c37061c7c4b21"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 22} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 22 | 95 | 98.0 | 99 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 22 | 2212 | 2212.0 | 2212 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
