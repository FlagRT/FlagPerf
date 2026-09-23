# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | partial |
| collector | xpu-smi -m and selected -q |
| reasons | ["kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57: 7/10 valid samples", "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6: 7/10 valid samples"] |
| policy | {} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57", "finished_monotonic_ns": 3993076520728200, "finished_offset_s": 15.668849896173924, "rank": 0, "role": "measurement", "started_monotonic_ns": 3993069493986832, "started_offset_s": 8.642108527943492}, {"device_id": "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6", "finished_monotonic_ns": 3993076500185230, "finished_offset_s": 15.648306925781071, "rank": 1, "role": "measurement", "started_monotonic_ns": 3993069493986242, "started_offset_s": 8.642107937950641}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 184578, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "03db6accc4dee7ed1ddb9ee60c63e4f3adf257b100cda8bfb4bc5dc9068c0b64"} |
| parsed_samples | {"bytes": 12032, "path": "benchmark-monitor/samples.jsonl", "sha256": "e6cb0c154cac66ce710c7685a19a8c4510a6952b628b1ff890fe7bfb16c1fe24"} |
| sample_counts | {"kunlunxin/9429f5bd-7d13-5cc5-b1b4-c05ddc572f57": 7, "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6": 7} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
