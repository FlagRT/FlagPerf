# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | partial |
| collector | xpu-smi -m and selected -q |
| reasons | ["kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57: 7/10 valid samples", "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6: 7/10 valid samples"] |
| policy | {} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57", "finished_monotonic_ns": 3993101056821267, "finished_offset_s": 15.908568073064089, "rank": 0, "role": "measurement", "started_monotonic_ns": 3993094018054422, "started_offset_s": 8.869801227934659}, {"device_id": "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6", "finished_monotonic_ns": 3993101035919185, "finished_offset_s": 15.88766599074006, "rank": 1, "role": "measurement", "started_monotonic_ns": 3993094018054202, "started_offset_s": 8.869801008142531}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 185644, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "f9e9bf416402b6d0fd5a7c881db2f00e05cc81d0a2029c553b04351d2b2162e8"} |
| parsed_samples | {"bytes": 12033, "path": "benchmark-monitor/samples.jsonl", "sha256": "dea5d00dec89437a0ba520f29f2d29c84426cc3c858b79fb7eb6aca9e14f5203"} |
| sample_counts | {"kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57": 7, "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6": 7} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
