# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897", "finished_monotonic_ns": 3799022755770508, "finished_offset_s": 32.531113918405026, "rank": 0, "role": "measurement", "started_monotonic_ns": 3798998456436663, "started_offset_s": 8.231780073139817}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 142960, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "fa67f2e590fe75eca524dfac592b57b9cb6569f4453b1d56b8fcea283f104535"} |
| parsed_samples | {"bytes": 11357, "path": "benchmark-monitor/samples.jsonl", "sha256": "58cef6873a7d0c8f9a4359d74443dbf82726c3d49ad0441d9e05b295f7ac9310"} |
| sample_counts | {"kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897": 24} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897 | Device utilization | % | 24 | 100 | 100.0 | 100 |
| kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897 | Allocated device memory | MiB | 24 | 422 | 422.0 | 422 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
