# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897", "finished_monotonic_ns": 3798933080996428, "finished_offset_s": 32.456522182095796, "rank": 0, "role": "measurement", "started_monotonic_ns": 3798908740549720, "started_offset_s": 8.116075473837554}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 142971, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "7f0f10f8e95717271eaa50f8292fae4c8accb26a4fff7e12cb6e1af749589400"} |
| parsed_samples | {"bytes": 11359, "path": "benchmark-monitor/samples.jsonl", "sha256": "d505d0a5b6b4e508c7606c19b827351fdef744bec5137af312df2dcddb0de951"} |
| sample_counts | {"kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897": 24} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897 | Device utilization | % | 24 | 100 | 100.0 | 100 |
| kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897 | Allocated device memory | MiB | 24 | 422 | 422.0 | 422 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
