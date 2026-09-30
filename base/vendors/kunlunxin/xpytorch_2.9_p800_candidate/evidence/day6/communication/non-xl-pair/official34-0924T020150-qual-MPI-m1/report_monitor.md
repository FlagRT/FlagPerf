# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57", "finished_monotonic_ns": 4056510366657128, "finished_offset_s": 27.6068044770509, "rank": 0, "role": "measurement", "started_monotonic_ns": 4056491467739528, "started_offset_s": 8.707886877004057}, {"device_id": "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6", "finished_monotonic_ns": 4056510347518983, "finished_offset_s": 27.587666331790388, "rank": 1, "role": "measurement", "started_monotonic_ns": 4056491467746898, "started_offset_s": 8.707894247025251}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 284122, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "00dc7557fa1e4a3cad54fdc40bbc87b0dd9a7362f0577a58a9afeb3e481914a9"} |
| parsed_samples | {"bytes": 20016, "path": "benchmark-monitor/samples.jsonl", "sha256": "f5076b7c7f5a15065405a8ef630ba37adc7e3e4c06668a8eba12e4b7fd4a3950"} |
| sample_counts | {"kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57": 19, "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6": 19} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | Device utilization | % | 19 | 99 | 100 | 100 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Device utilization | % | 19 | 100 | 100 | 100 |
| kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57 | Allocated device memory | MiB | 19 | 532 | 532 | 532 |
| kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6 | Allocated device memory | MiB | 19 | 368 | 368 | 368 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
