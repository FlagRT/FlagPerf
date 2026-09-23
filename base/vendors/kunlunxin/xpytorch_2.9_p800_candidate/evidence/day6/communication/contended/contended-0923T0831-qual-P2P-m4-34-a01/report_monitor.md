# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57", "finished_monotonic_ns": 3993773749880184, "finished_offset_s": 25.76172657078132, "rank": 0, "role": "measurement", "started_monotonic_ns": 3993756855552103, "started_offset_s": 8.867398489732295}, {"device_id": "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6", "finished_monotonic_ns": 3993773750145839, "finished_offset_s": 25.761992225889117, "rank": 1, "role": "measurement", "started_monotonic_ns": 3993756855523023, "started_offset_s": 8.867369409650564}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 298122, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "2eb0039e33696e08a66d82b14c5aa7220565e8f490d803f1cbc6522e4c8dcb4e"} |
| parsed_samples | {"bytes": 18722, "path": "benchmark-monitor/samples.jsonl", "sha256": "418933f09afa340f78f00a48575f47f701194d37e7f82adf52ebe85872c069a6"} |
| sample_counts | {"kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57": 17, "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6": 17} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | Device utilization | % | 17 | 84 | 92 | 96 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Device utilization | % | 17 | 100 | 100 | 100 |
| kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | Allocated device memory | MiB | 17 | 2374 | 2374 | 2374 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Allocated device memory | MiB | 17 | 2856 | 2856 | 2856 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
