# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4096453805715573, "finished_offset_s": 56.01675830222666, "rank": 0, "role": "measurement", "started_monotonic_ns": 4096407594030082, "started_offset_s": 9.805072811432183}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 300330, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "3490040c637350c5615dd3a7061d5045ec87bea30408acc50a52bbadc0662992"} |
| parsed_samples | {"bytes": 19424, "path": "benchmark-monitor/samples.jsonl", "sha256": "2ca896eb9f09537ae468f990650125fd520e88ebe1f85238a9d3c1aff06965b2"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 47} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 47 | 51 | 100 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 47 | 1062 | 1062 | 1062 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
