# Device preflight qualification

Identity, bounded tensor readback and observation only. No performance measurement.

- run_id: preflight-f36a2c935c324037989833ed90203c69
- vendor: kunlunxin
- status: failed
- probe_mode: timeout-check
- failure_stage: container-probe
- error: bounded container execution timed out
- measurement_status: not-run
- monitoring_status: not-run
- cleanup_status: passed
- postflight_status: passed
- lease_released: True

## Verified runtime bindings

```json
[]
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
- [resolved-plan.json](resolved-plan.json)
- [static-plan.json](static-plan.json)
- [summary.json](summary.json)
