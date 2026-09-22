# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3888261267287181, "finished_offset_s": 21.633339503314346, "rank": 0, "role": "measurement", "started_monotonic_ns": 3888245048942454, "started_offset_s": 5.414994776248932}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 96209, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "b83e189fa194e72417180fb7b24ab441a6e49c25fa146c8f96aaf9fd83bcadac"} |
| parsed_samples | {"bytes": 7638, "path": "benchmark-monitor/samples.jsonl", "sha256": "4ac43becd092cde1ba234a5aa7d019466b9eb9cd931df8cf9768f8bb240540a5"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 16} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 16 | 0 | 0.0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 16 | 218 | 218.0 | 218 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
