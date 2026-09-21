# Device preflight qualification

Identity, bounded tensor readback and observation only. No performance measurement.

- run_id: preflight-c17285e49a2545f78f60b6ca9e075b76
- vendor: kunlunxin
- status: passed
- probe_mode: identity
- failure_stage: not recorded
- error: not recorded
- measurement_status: not-run
- monitoring_status: passed
- cleanup_status: passed
- postflight_status: passed
- lease_released: True

## Verified runtime bindings

```json
[
  {
    "container_device_node": "/dev/xpu5",
    "framework_device_name": "cuda:0",
    "framework_local_rank": 0,
    "framework_logical_id": 0,
    "host_device_node": "/dev/xpu5",
    "host_physical_id": 6,
    "legacy_logical_id": null,
    "pci_bdf": "0000:b6:00.0",
    "request_index": 0,
    "resource_key": "kunlunxin/9e24d168-db57-5cd9-8cfb-aea4ae898897",
    "serial_or_uuid": "9e24d168-db57-5cd9-8cfb-aea4ae898897",
    "vendor": "kunlunxin"
  }
]
```

## Evidence

- [artifacts/probe-observation.json](artifacts/probe-observation.json)
- [artifacts/probe.json](artifacts/probe.json)
- [artifacts/runtime-audit.json](artifacts/runtime-audit.json)
- [artifacts/runtime-bindings.json](artifacts/runtime-bindings.json)
- [cleanup.json](cleanup.json)
- [code-identity.json](code-identity.json)
- [container-create.json](container-create.json)
- [container-inspect.json](container-inspect.json)
- [container-logs.json](container-logs.json)
- [container-start.json](container-start.json)
- [control/host-context.json](control/host-context.json)
- [host-postflight/handles-6.json](host-postflight/handles-6.json)
- [host-postflight/machine.json](host-postflight/machine.json)
- [host-postflight/query-6.json](host-postflight/query-6.json)
- [host-postflight/summary.json](host-postflight/summary.json)
- [host-preflight/handles-6.json](host-preflight/handles-6.json)
- [host-preflight/machine.json](host-preflight/machine.json)
- [host-preflight/query-6.json](host-preflight/query-6.json)
- [host-preflight/summary.json](host-preflight/summary.json)
- [image-identity.json](image-identity.json)
- [image-inspect-command.json](image-inspect-command.json)
- [lease.json](lease.json)
- [locked-preflight/handles-6.json](locked-preflight/handles-6.json)
- [locked-preflight/machine.json](locked-preflight/machine.json)
- [locked-preflight/query-6.json](locked-preflight/query-6.json)
- [locked-preflight/summary.json](locked-preflight/summary.json)
- [monitor/samples.jsonl](monitor/samples.jsonl)
- [monitor/samples.raw.jsonl](monitor/samples.raw.jsonl)
- [monitor/summary.json](monitor/summary.json)
- [resolved-plan.json](resolved-plan.json)
- [static-plan.json](static-plan.json)
- [summary.json](summary.json)
