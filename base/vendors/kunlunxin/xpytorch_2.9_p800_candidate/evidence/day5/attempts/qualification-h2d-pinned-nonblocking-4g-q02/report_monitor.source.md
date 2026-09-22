# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3914651533214711, "finished_offset_s": 48.579639374278486, "rank": 0, "role": "measurement", "started_monotonic_ns": 3914615705396168, "started_offset_s": 12.751820831093937}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 269198, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "d3bfd5509b7d8898b95e4aba8e8ff5c2e3d46ab4e4874d78be66600cd54d88aa"} |
| parsed_samples | {"bytes": 17338, "path": "benchmark-monitor/samples.jsonl", "sha256": "e061ba7dcee91c620c6628c5ca9064f36af19d8238763e61a575df8693eb5e7b"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 36} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 36 | 93 | 96.0 | 98 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 36 | 2212 | 2212.0 | 2212 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
