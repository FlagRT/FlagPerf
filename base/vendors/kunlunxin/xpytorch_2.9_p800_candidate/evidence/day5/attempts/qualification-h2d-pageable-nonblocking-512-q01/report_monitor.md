# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3922178816584040, "finished_offset_s": 27.70539095206186, "rank": 0, "role": "measurement", "started_monotonic_ns": 3922158300715212, "started_offset_s": 7.1895221238955855}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 125911, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "3871fcbf805019164c7557c106f58fe6b3dbb023ab72a2c7b3a338fb90d39252"} |
| parsed_samples | {"bytes": 9979, "path": "benchmark-monitor/samples.jsonl", "sha256": "47e0bce8f3e2e2dcb0db091d8edbc2d91b5c7ec48994d479f97575f100134b40"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 20} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 20 | 73 | 81.0 | 86 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 20 | 676 | 676.0 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
