# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4097551793782280, "finished_offset_s": 21.23475820897147, "rank": 0, "role": "measurement", "started_monotonic_ns": 4097536748835377, "started_offset_s": 6.189811306074262}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 117995, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "e50e796d3103cdd2e8fb32959dcb2cfe0ec442582ad4ad299365e066b719c219"} |
| parsed_samples | {"bytes": 7648, "path": "benchmark-monitor/samples.jsonl", "sha256": "0a2a76733644307c560f9402e93de5d6825448e89998bef03c04ae912bb17543"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 15} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 15 | 35 | 53 | 64 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 15 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
