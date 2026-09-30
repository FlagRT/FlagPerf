# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6", "finished_monotonic_ns": 4055524112161805, "finished_offset_s": 28.04319338174537, "rank": 0, "role": "measurement", "started_monotonic_ns": 4055505225131626, "started_offset_s": 9.156163202598691}, {"device_id": "kunlunxin/4922595e-feb9-5762-9082-d8fd7f11abf5", "finished_monotonic_ns": 4055524111554537, "finished_offset_s": 28.042586113791913, "rank": 1, "role": "measurement", "started_monotonic_ns": 4055505225026708, "started_offset_s": 9.156058284919709}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 285108, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "58de7c546b71ab6b33e9a91ff9c1ecac6d464ac28ed6f83eadba0e6da541d4ce"} |
| parsed_samples | {"bytes": 20009, "path": "benchmark-monitor/samples.jsonl", "sha256": "003712ff8695995a217063e24b0e06e81e96afaaec3d1290aa3837b29a70fb45"} |
| sample_counts | {"kunlunxin/4922595e-feb9-5762-9082-d8fd7f11abf5": 20, "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6": 19} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/4922595e-feb9-5762-9082-d8fd7f11abf5 | Device utilization | % | 20 | 3 | 100.0 | 100 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Device utilization | % | 19 | 70 | 98 | 100 |
| kunlunxin/4922595e-feb9-5762-9082-d8fd7f11abf5 | Allocated device memory | MiB | 20 | 414 | 414.0 | 414 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Allocated device memory | MiB | 19 | 514 | 578 | 578 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
