# Device preflight qualification

Identity, bounded tensor readback and observation only. No performance measurement.

- run_id: preflight-235155abbfde4798a355b9d9ebc3ccda
- vendor: kunlunxin
- status: failed
- probe_mode: identity
- failure_stage: container-probe
- error: bounded container execution timed out
- measurement_status: not-run
- monitoring_status: partial
- cleanup_status: passed
- postflight_status: failed
- lease_released: True

## Verified runtime bindings

```json
[]
```

## Evidence

- [artifacts/runtime-audit.json](artifacts/runtime-audit.json)
- [artifacts/runtime-bindings.json](artifacts/runtime-bindings.json)
- [cleanup.json](cleanup.json)
- [code-identity.json](code-identity.json)
- [container-create.json](container-create.json)
- [container-inspect.json](container-inspect.json)
- [container-logs.json](container-logs.json)
- [container-start.json](container-start.json)
- [control/host-context.json](control/host-context.json)
- [host-postflight/handles-1.json](host-postflight/handles-1.json)
- [host-postflight/machine.json](host-postflight/machine.json)
- [host-postflight/query-1.json](host-postflight/query-1.json)
- [host-preflight/handles-1.json](host-preflight/handles-1.json)
- [host-preflight/machine.json](host-preflight/machine.json)
- [host-preflight/query-1.json](host-preflight/query-1.json)
- [host-preflight/summary.json](host-preflight/summary.json)
- [image-identity.json](image-identity.json)
- [image-inspect-command.json](image-inspect-command.json)
- [lease.json](lease.json)
- [locked-preflight/handles-1.json](locked-preflight/handles-1.json)
- [locked-preflight/machine.json](locked-preflight/machine.json)
- [locked-preflight/query-1.json](locked-preflight/query-1.json)
- [locked-preflight/summary.json](locked-preflight/summary.json)
- [monitor/samples.jsonl](monitor/samples.jsonl)
- [monitor/samples.raw.jsonl](monitor/samples.raw.jsonl)
- [monitor/summary.json](monitor/summary.json)
- [resolved-plan.json](resolved-plan.json)
- [static-plan.json](static-plan.json)
- [summary.json](summary.json)
