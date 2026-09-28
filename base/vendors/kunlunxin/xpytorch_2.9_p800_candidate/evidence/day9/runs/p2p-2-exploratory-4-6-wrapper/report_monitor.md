# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | partial |
| collector | xpu-smi -m and selected -q |
| reasons | ["kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6: 0/10 valid samples", "kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897: 0/10 valid samples"] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6", "finished_monotonic_ns": 4430539508636780, "finished_offset_s": 9.72305519785732, "rank": 0, "role": "measurement", "started_monotonic_ns": 4430539507312969, "started_offset_s": 9.721731387078762}, {"device_id": "kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897", "finished_monotonic_ns": 4430539508597141, "finished_offset_s": 9.7230155589059, "rank": 1, "role": "measurement", "started_monotonic_ns": 4430539507118453, "started_offset_s": 9.721536871045828}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 113072, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "65d579ee947b5a06c577531a3b36a239b1f73ce922ff4be047f917f8632a2b74"} |
| parsed_samples | {"bytes": 8007, "path": "benchmark-monitor/samples.jsonl", "sha256": "5e1940a2b19d89123c21f26d78e1bc1c2be7ef7aceded9455ad335937279cc58"} |
| sample_counts | {"kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897": 0, "kunlunxin/b7942319-362e-5a53-a8eb-a2d06fbe84f6": 0} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
