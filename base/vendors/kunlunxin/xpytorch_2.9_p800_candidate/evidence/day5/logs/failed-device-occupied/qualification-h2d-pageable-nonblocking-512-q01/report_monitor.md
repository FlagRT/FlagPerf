# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3922066760733100, "finished_offset_s": 25.58876504097134, "rank": 0, "role": "measurement", "started_monotonic_ns": 3922048069726444, "started_offset_s": 6.8977583847008646}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 151076, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "c10e0299a93042f3fa1a4df8ecefd9940c808edbeac0c6ef03d433ec9b64676f"} |
| parsed_samples | {"bytes": 8998, "path": "benchmark-monitor/samples.jsonl", "sha256": "f28bb6ccf6c2630911b50734f026472d53febdf049988915ca41c45dde36bdd1"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 19} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 19 | 83 | 97 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 19 | 708 | 708 | 16576 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
