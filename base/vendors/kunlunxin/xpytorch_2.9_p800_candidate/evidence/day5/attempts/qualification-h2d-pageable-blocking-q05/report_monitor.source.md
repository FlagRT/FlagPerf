# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3902129725366082, "finished_offset_s": 67.5266127390787, "rank": 0, "role": "measurement", "started_monotonic_ns": 3902067454206233, "started_offset_s": 5.255452889949083}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 356872, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "f7bce032dd927da0d05b5fc73d2958b2ae84e9323f4ef552f37a5af16e4e3de0"} |
| parsed_samples | {"bytes": 22883, "path": "benchmark-monitor/samples.jsonl", "sha256": "a777da4fca473c443e88ae2532db6a1513f4d6e2ef10a09d3b4227a3d2b5d72f"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 62} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 62 | 0 | 0.0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 62 | 228 | 228.0 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
