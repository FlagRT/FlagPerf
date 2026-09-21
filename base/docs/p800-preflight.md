# P800 bounded preflight

`benchmark preflight` checks a selected P800 through the unified Base facade.
It has no performance case, TFLOPS result, or benchmark measurement window.
The M1 profile remains candidate, `validated: false`, with empty validation scope.
See the [Day 3 review](../vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day3/review.md)
for tested code, hardware outcomes, failures and evidence provenance.

## Run

From the repository root, plan without Docker, device queries, runtime imports,
locks, result writes or telemetry:

```bash
python3 -B base/run.py benchmark preflight \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids 1 --monitor on --dry-run
```

The ID is an example, not a reservation. Runtime UUID, node/minor and framework
logical ID remain deferred. This profile only accepts explicit physical IDs and
exactly one card. Its `expected_device_ids` describes host inventory; requalify
that configuration on a different machine.

For an authorized, idle card, supply a fresh end time with timezone, an actual
authorization reference, and a new result directory. Run `sudo -v` interactively
in the same terminal when Docker/fuser require privilege:

```bash
python3 -B base/run.py benchmark preflight \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids PHYSICAL_ID --probe-mode identity \
  --allow-candidate-runtime --monitor on --timeout 120 \
  --privilege-command 'sudo -n' \
  --reservation-end 'END_TIME_WITH_TIMEZONE' \
  --reservation-reference 'ACTUAL_AUTHORIZATION_REFERENCE' \
  --result-dir base/result/UNIQUE_ATTEMPT
```

Execution timeout is 30..180 seconds. The reservation must still cover execution
plus 30 seconds cleanup after host checks and container creation. Privilege is
for host commands; the container is nonprivileged. No credentials are persisted.

For the controlled cleanup drill, use another new directory and replace
`--probe-mode identity --monitor on` with `--probe-mode timeout-check --monitor off`.
After the same tensor check, the worker waits on the CPU for a bounded period
longer than the host watchdog. Expected experiment outcome is exit 1/failed with
`TimeoutError`; a separate review accepts the drill only after verified deletion,
idle postflight and successful lock reacquisition. Never rewrite failed to passed.

## Lifecycle and identity

1. Validate locked immutable image ID, independent RepoDigests and architecture.
2. Parse the locked 32-field machine schema and selected query. Check driver,
   runtime, PCI BDF, UUID, minor, character nodes, memory/utilization and fuser.
3. Acquire resource-hash locks plus the original PR0 UUID lock paths, in stable
   order. Repeat identity and occupancy checks while holding both protocols.
4. Create from the immutable image ID, with only the selected same-name node and
   `/dev/xpuctrl`. Inspect image, ownership, devices, bind permissions, isolation
   and capabilities before starting. Code/context are read-only; artifacts use
   the host group. PID/cache limits match the bounded PR0 configuration.
5. Bootstrap the locked Conda Python with `-S`. A supervisor validates context,
   visible nodes, image and package inventory before spawning a fresh `-S` worker.
   The worker repeats identity gates, enables site hooks, imports the backend and
   joins the entire visible UUID set to the selected host identity.
6. Write immutable binding data tied to the context SHA256 and run ID before
   allocating a four-element tensor. Synchronize the actual target, read back to
   CPU, release the tensor, and keep a 16-second observation window.
7. Stop telemetry, collect logs, remove this run's exact CID, verify absence, and
   perform idle/identity postflight while holding locks. Release only owned locks.

`DeviceBinding` separates physical index, local rank, framework ordinal, node,
PCI BDF and resource key. Single-card visibility is the selected subset ordinal
`CUDA_VISIBLE_DEVICES=0`, not the host physical index. `XPU_VISIBLE_DEVICES` and
`XPU_EVENT_KL3_ENABLE` are unset; `USE_FLAGGEMS=0`. Generic framework name `GPU`
and CUDA-compatible API naming do not identify the hardware vendor.

`HostCommands`, `DeviceLease` and `ManagedContainer` are generic executor
components; the provider supplies image/device/lease policy. The new preflight
lifecycle uses lock-held postflight and bounded cleanup. Existing Ascend
performance execution retains its previously tested lifecycle; this qualification
does not claim its Docker failure paths were migrated or hardware-tested.

## Evidence schemas and failure handling

| File | Contract |
| --- | --- |
| `summary.json` | schema 1, `kind=benchmark-preflight`; status, failure stage, cleanup/postflight and lease release |
| `control/host-context.json` | schema 1; host selection, immutable image identity, reservation, run ID |
| `artifacts/runtime-bindings.json` | schema 1; run ID, context SHA256, observed UUIDs, final bindings |
| `artifacts/probe.json` | schema 1; tiny tensor readback, PID, versions, loaded library hashes; no performance claim |
| `artifacts/probe-observation.json` | Linux monotonic start/end of bounded observation; not measurement |
| `monitor/summary.json` | schema 2; explicit targets, sample hashes, overlapping-window counts |
| `cleanup.json` | exact CID, command outcomes, confirmed absence or unknown/failure |

Telemetry keeps command argv/times/output/returncode/timeout and parsed samples.
Utilization and memory come from selected `-q` fields with explicit `%`/`MiB`;
machine totals are cross-checked. Missing fields and failed commands are invalid,
not zero. At least ten valid samples per target in the observation window are
required; missing/invalid samples yield partial. Monitor off is `not-run`.

Exit 0 means passed, exit 2 means partial or a configuration rejection, and exit
1 means experiment failure. CLI output and durable summary distinguish these.
On cleanup uncertainty, perform at most one bounded retry, record recovery-required
and the exact CID, then exit failed. Process exit releases flock; that does not
prove the card is free or block future PR0 callers. Do not automatically retry,
change cards or reset devices before resolving unknown cleanup.

Rebuild a deterministic report entirely from stored evidence:

```bash
python3 -B base/run.py report --result-root base/result --run-id UNIQUE_ATTEMPT
```

Rendering leaves the experiment summary unchanged. Raw all-card inventories stay
in ignored operational directories. Portable evidence explicitly labels selected
record extractions and preserves source hashes; it must not imply filtered logs
are byte-identical originals.

## Targeted recheck before performance work (2026-09-20)

Use physical card 6 or 7 only after a fresh authorized-window preflight. Both
passed the normal tensor/16-sample observation and controlled timeout cleanup.
All eight cards were checked: 0/3/4/5 were occupied and did not run workloads.
After the user stopped their xpu-smi monitor, 2/6/7 normal and timeout postflight
passed without follow-up recovery; card 1 still timed out with our monitor off.
Card 1 reproduced synchronization timeouts with this run's monitor on and off;
its selected PCI kernel log contains `KL_XID_KERNEL_EXCEPTION` and task/NOC idle
timeouts. Root cause is not yet established. Card 2 completed tensor/observation
but retains historical uncorrectable ECC 8/8 and remap pending YES; avoid using
it as a performance baseline until assessed. Transient `xpu-smi` handles can
reject strict pre/postflight even with zero utilization. Pause user-controlled
refresh loops during checks; never ignore handles or stop other users' processes.
Preserve failed summaries; independent later idle/lock recovery is separate.

See the [recheck evidence review](../vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day3-recheck/review.md).

## Next performance integration

P800 `benchmark run --case computation-FP32:P800` still rejects its missing case
contract before hardware access. Add the FP32 requirements, driver bootstrap,
same-container binding verification and worker consumption separately. Use
`framework_logical_id` from the verified binding, never `local_rank` as an ordinal.
The future timer remains wall-clock bracketed by full target synchronization;
PR0 event elapsed time was zero. Correctness, measurement-window monitoring,
repeatability, multi-rank/collectives and broad CPU-fallback exclusion require
their own qualification.
