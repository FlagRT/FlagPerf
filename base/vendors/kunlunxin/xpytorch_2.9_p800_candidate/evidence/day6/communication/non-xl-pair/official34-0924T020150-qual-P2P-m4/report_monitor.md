# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57", "finished_monotonic_ns": 4056759860662795, "finished_offset_s": 27.489383406937122, "rank": 0, "role": "measurement", "started_monotonic_ns": 4056741155970885, "started_offset_s": 8.784691496752203}, {"device_id": "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6", "finished_monotonic_ns": 4056759860575486, "finished_offset_s": 27.48929609777406, "rank": 1, "role": "measurement", "started_monotonic_ns": 4056741155961425, "started_offset_s": 8.78468203684315}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 275489, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "ae939f91d02c13c2f2e60813a8df83ad09a10bb862140f9f3cd36bfd966c9af1"} |
| parsed_samples | {"bytes": 19307, "path": "benchmark-monitor/samples.jsonl", "sha256": "5c40c5865163f0938609f05ce5c5f354bed672240d273d829cb490528dd3fa79"} |
| sample_counts | {"kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57": 19, "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6": 19} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | Device utilization | % | 19 | 81 | 92 | 98 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Device utilization | % | 19 | 100 | 100 | 100 |
| kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | Allocated device memory | MiB | 19 | 530 | 530 | 530 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Allocated device memory | MiB | 19 | 366 | 366 | 366 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
