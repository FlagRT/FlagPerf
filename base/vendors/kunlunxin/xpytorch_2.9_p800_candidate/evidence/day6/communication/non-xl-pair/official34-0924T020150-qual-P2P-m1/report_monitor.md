# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57", "finished_monotonic_ns": 4056712423723767, "finished_offset_s": 25.020936923101544, "rank": 0, "role": "measurement", "started_monotonic_ns": 4056696308808317, "started_offset_s": 8.906021472997963}, {"device_id": "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6", "finished_monotonic_ns": 4056712423925443, "finished_offset_s": 25.021138599142432, "rank": 1, "role": "measurement", "started_monotonic_ns": 4056696308758788, "started_offset_s": 8.905971943866462}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 255141, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "70932ee1e33a956e119a42b2dc60e6df97d2738b4c8267c4376a8ee521ab6d20"} |
| parsed_samples | {"bytes": 17968, "path": "benchmark-monitor/samples.jsonl", "sha256": "8e0163a3bab3daf35160b4cfc312201b0fb45cc3560aaaa6defb9e07bc3c42c6"} |
| sample_counts | {"kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57": 17, "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6": 17} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | Device utilization | % | 17 | 42 | 74 | 87 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Device utilization | % | 17 | 29 | 100 | 100 |
| kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | Allocated device memory | MiB | 17 | 508 | 534 | 534 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Allocated device memory | MiB | 17 | 344 | 370 | 370 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
