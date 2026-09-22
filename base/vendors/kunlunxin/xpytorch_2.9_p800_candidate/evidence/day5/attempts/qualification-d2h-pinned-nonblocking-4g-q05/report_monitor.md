# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3915068988573733, "finished_offset_s": 30.873853141907603, "rank": 0, "role": "measurement", "started_monotonic_ns": 3915046961721819, "started_offset_s": 8.84700122801587}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 139014, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "a1353deee3cdef56460de15b4fee460e02b64a1a6f6d86d57b40c187c57b17da"} |
| parsed_samples | {"bytes": 11008, "path": "benchmark-monitor/samples.jsonl", "sha256": "76937e22435f3225b62d996eb6bf9dfc455defff4721e246b09bf2aa4578d1e0"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 22} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 22 | 96 | 98.0 | 99 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 22 | 2212 | 2212.0 | 2212 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
