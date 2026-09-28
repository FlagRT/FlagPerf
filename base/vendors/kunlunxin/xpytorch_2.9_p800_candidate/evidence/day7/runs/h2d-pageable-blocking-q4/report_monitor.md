# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 4096854406666048, "finished_offset_s": 28.35374629497528, "rank": 0, "role": "measurement", "started_monotonic_ns": 4096833336553574, "started_offset_s": 7.283633821178228}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 154986, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "00d56e2a89f0d807425f3b8bdb0bf1b5649fa5b6e0ddf5e88d05aa2cc12e2390"} |
| parsed_samples | {"bytes": 9959, "path": "benchmark-monitor/samples.jsonl", "sha256": "91c005e030d2fd9b62d8077d146dceb2a9891c95367e6507af701683fa85f674"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 21} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 21 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 21 | 676 | 676 | 676 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
