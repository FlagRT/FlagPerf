# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3914832825969924, "finished_offset_s": 43.61966613121331, "rank": 0, "role": "measurement", "started_monotonic_ns": 3914801973271234, "started_offset_s": 12.76696744095534}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 243209, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "7a74a24926afc492670b218b30afa48b7015321a8ca65f984ac7359191101c0f"} |
| parsed_samples | {"bytes": 15665, "path": "benchmark-monitor/samples.jsonl", "sha256": "a72831d5902d343811b99cbe1d8643ae9efe2a2f7d414c1e154a91a42b84f01c"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 31} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 31 | 96 | 98 | 100 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 31 | 2212 | 2212 | 2212 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
