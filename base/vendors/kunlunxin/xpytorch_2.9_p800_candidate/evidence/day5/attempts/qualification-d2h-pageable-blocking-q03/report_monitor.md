# Benchmark device telemetry

| Field | Evidence |
|---|---|
| vendor | kunlunxin |
| status | passed |
| collector | xpu-smi -m and selected -q |
| reasons | [] |
| policy | {"automatic_workload_extension": false, "collector": "xpu-smi -m and selected -q", "enabled": true, "metric_fields": [{"key": "utilization_percent", "label": "Device utilization", "unit": "%"}, {"key": "used_memory_mib", "label": "Allocated device memory", "unit": "MiB"}], "required_samples_per_target": 10, "target_interval_s": 1.0, "target_resource": "p800-device", "vendor": "kunlunxin"} |
| rank_device_map | not recorded |
| measurement_windows | [{"device_id": "kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87", "finished_monotonic_ns": 3903228482379043, "finished_offset_s": 62.88152626622468, "rank": 0, "role": "measurement", "started_monotonic_ns": 3903170731808915, "started_offset_s": 5.130956138018519}] |
| lifecycle_windows | not recorded |
| raw_samples | {"bytes": 270263, "path": "benchmark-monitor/samples.raw.jsonl", "sha256": "4429e2b7e677fcea435079174a57337b37703fdd21166d68687fd0d0c00796f6"} |
| parsed_samples | {"bytes": 21228, "path": "benchmark-monitor/samples.jsonl", "sha256": "ef136b6bffb6d29ac565a38ccddb56b57846690de00e16fcc64ad7a40dd5a499"} |
| sample_counts | {"kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87": 57} |
| events | not recorded |

| Device | Metric | Unit | Samples | Min | Median | Max |
|---|---|---|---|---|---|---|
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Device utilization | % | 57 | 0 | 0 | 0 |
| kunlunxin/f396486d-9850-50e4-81c4-50f2e9f6ca87 | Allocated device memory | MiB | 57 | 228 | 228 | 228 |

![Device telemetry](report-assets/benchmark-monitor-usage.svg)

Only command intervals overlapping the recorded rank measurement windows are summarized.
Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.
