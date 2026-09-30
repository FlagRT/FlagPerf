# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57", "finished_monotonic_ns": 4056874130807144, "finished_offset_s": 30.264083270914853, "rank": 0, "role": "measurement", "started_monotonic_ns": 4056855132156705, "started_offset_s": 11.265432831831276}, {"device_id": "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6", "finished_monotonic_ns": 4056874124463666, "finished_offset_s": 30.257739793043584, "rank": 1, "role": "measurement", "started_monotonic_ns": 4056855132019198, "started_offset_s": 11.265295324847102}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 314155, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "ad9019a377954b0cf7a18cea6221d9cc7ca9014e641d180728c040b47ed87a7e"} |
| parsed_samples | {"bytes": 21994, "path": "benchmark-monitor/samples.jsonl", "sha256": "1e3f86b48aec2ecd9620b8dc94eee868dfdb3f5c00d4bcf835cdf5dc65310c7b"} |
| sample_counts | {"kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57": 19, "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6": 19} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | Device utilization | % | 19 | 93 | 99 | 100 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Device utilization | % | 19 | 100 | 100 | 100 |
| kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | Allocated device memory | MiB | 19 | 770 | 770 | 770 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Allocated device memory | MiB | 19 | 606 | 606 | 606 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
