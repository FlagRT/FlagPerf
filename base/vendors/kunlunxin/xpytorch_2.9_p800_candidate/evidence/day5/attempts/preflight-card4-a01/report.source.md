# Device preflight qualification

Identity, bounded tensor readback and observation only. No performance measurement.

- run_id: preflight-6a03b5ba52e54ce69b235b082dea4d95
- vendor: kunlunxin
- status: failed
- probe_mode: identity
- failure_stage: locked-preflight
- error: device has open handles or handle check failed
- measurement_status: not-run
- monitoring_status: not-run
- cleanup_status: not-created
- postflight_status: failed
- lease_released: True

## Verified runtime bindings

```json
[]
```

## Evidence

- [cleanup.json](cleanup.json)
- [host-postflight/handles-4.json](host-postflight/handles-4.json)
- [host-postflight/machine.json](host-postflight/machine.json)
- [host-postflight/query-4.json](host-postflight/query-4.json)
- [host-preflight/handles-4.json](host-preflight/handles-4.json)
- [host-preflight/machine.json](host-preflight/machine.json)
- [host-preflight/query-4.json](host-preflight/query-4.json)
- [host-preflight/summary.json](host-preflight/summary.json)
- [image-identity.json](image-identity.json)
- [image-inspect-command.json](image-inspect-command.json)
- [lease.json](lease.json)
- [locked-preflight/handles-4.json](locked-preflight/handles-4.json)
- [locked-preflight/machine.json](locked-preflight/machine.json)
- [locked-preflight/query-4.json](locked-preflight/query-4.json)
- [static-plan.json](static-plan.json)
- [summary.json](summary.json)
