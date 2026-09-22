# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3902993987755993, "finished_offset_s": 39.37223138008267, "rank": 0, "role": "measurement", "started_monotonic_ns": 3902959836989232, "started_offset_s": 5.221464619040489}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 211837, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "ccfb3542bf8b856aa90fc754f09a49ad56e4ecffd13e8e9acd52f24b0dd7b00c"} |
| parsed_samples | {"bytes": 13674, "path": "benchmark-monitor/samples.jsonl", "sha256": "e75ecf62fd099dc6e6bf13b0727e8680472ebaa1884458a1235b94debafded16"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 34} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 34 | 100 | 100.0 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 34 | 228 | 228.0 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
