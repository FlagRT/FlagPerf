# Day 3 / PR2 qualification review — 2026-09-20

The unified Base preflight implements bounded P800 identity, a four-element
tensor readback, telemetry, PR0-compatible leases and cleanup. Execution took
place on September 20, although the plan initially proposed September 21.
This is development qualification, not performance qualification or candidate
promotion. `validated: false` and the empty validation scope remain unchanged.

## Evidence and acceptance

Tested executable commit: `e936db64` on `zhiyu/kunlunxin-p800`.
Source hashes and each attempt's exact HEAD/dirty status are retained. The final
hardware attempts had documentation edits in progress; those edits did not change
the tested Python/bootstrap assets. Baseline was `d286d8c1`.

The immutable M1 image is
`sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975`.
Driver `5.0.21.47`, xpu-smi runtime `10.2`, Python `3.10.18` and the locked package
inventory were checked. Image ID, independent RepoDigests and architecture were
verified before creating nonprivileged containers.

The accepted normal attempt is `normal-attempt12-card6`, exit 0/passed:

- Physical 6, PCI `0000:b6:00.0`, UUID `9e24d168-db57-5cd9-8cfb-aea4ae898897`.
- Exact `/dev/xpu5` plus `/dev/xpuctrl`, observed single UUID, logical `cuda:0`.
- Four-element tensor, full target synchronization, CPU readback `[1,1,1,1]`.
- Sixteen valid target samples in a sixteen-second probe observation window.
- Container absence, idle postflight and lease release all passed.

The accepted controlled timeout drill is `timeout-attempt02-card6`: its tiny
probe passed, then the host's 120-second watchdog interrupted the bounded wait.
The experiment correctly remains failed/exit 1 with `TimeoutError`; container
absence, idle postflight and lease release passed. An independent review verified
both accepted containers absent, reacquired both lock protocols, and regenerated
reports without changing experiment summaries or report bytes.
See `logs/acceptance-review.json` for the final machine-readable decision.

## Offline regression

| Gate | Result |
| --- | --- |
| Base | 154 passed: previous 120 plus 34 new tests |
| Original PR0, Python `-S` | 32 passed |
| Standalone Ascend Toolkit | 80 passed |
| Ascend case planning | 15 cases: 10 applicable, 5 skipped |
| Additional static gates | Three selectors × monitor on/off; capacity authorization rejection; 22 total entries |
| Public distribution / Operation planning | 6/7; only the known original private Ascend Toolbox path fails; six Operation vendor plans pass |

No Ascend hardware ran on this P800 host. New tests cover strict machine/query
formats, active process tables, driver/identity drift, nodes, selectors, candidate
and reservation gates, UUID reorder/mismatch, PR0 lock competition in both
directions, symlink/hardlink/permission/partial acquisition, owned-container
inspection and bounded cleanup, interruption, timeout, partial monitor, pure
dry-run, deterministic reports, and supervisor child exit propagation.

## Retained failures and their meaning

All attempts remain in the raw directory and portable projection. Failed and
partial experiment summaries are never rewritten to passed.

1. `normal-attempt01`: Python 3.10 rejected the host UTC `Z` suffix before runtime
   import. Explicit normalization fixed this compatibility error.
2. `normal-attempt02` through `normal-attempt04`: physical card 1 resolved to its
   correct UUID and `/dev/xpu2`, but device synchronization did not complete before
   the 120-second deadline. All three containers were removed and postflight
   passed. Reusing PR0 limits and a gated child process did not resolve this
   card's behavior. Stack traces localize the latter attempts to synchronize;
   the root cause remains unproven. Do not use card 1 for subsequent performance
   qualification until separately investigated. No reset or driver change occurred.
3. Handle-check rejections: attempts 05–09 and 11 include transient nonempty
   fuser results. No container was created in these attempts. Some rejected
   attempts also have failed postflight because handles were still observed;
   this is not evidence of a workload created by this run. No other process was
   stopped. The name `normal-attempt05-card2` was prepared before a helper update
   transferred successfully; its actual selection was card 1, as recorded in
   its reservation and command. Names are not device identity evidence.
4. `normal-attempt10-card6`: tensor, binding and cleanup passed; monitor was partial
   because the parser only recognized `Processes: None`. The real active process
   table was added to the parser and fixtures; attempt 12 passed all monitor gates.
5. Timeout attempts rejected by occupancy checks are retained separately from
   the accepted watchdog drill; they cannot establish timeout cleanup acceptance.

Card 6's successful path does not prove card 1 is healthy, all P800 cards are
equivalent, or broad CPU fallback is excluded. Generic framework device name
`GPU` and CUDA-compatible APIs are interpreted through the verified UUID binding.

## Provenance and operating boundaries

Full raw evidence: `base/result/p800-pr2-20260920-1843/` on the development host.
`provenance.json` maps portable files to source paths and hashes. Original bytes
are copied except explicitly labeled projections: all-card `-m` stdout is reduced
to the selected record, and process identifiers/names are redacted. Such logs
are not presented as byte-identical originals. Original monitor hashes are in
`monitor/summary.source.json`; the adjacent summary references projected bytes.
The SHA256 index covers the portable deliverable. Historical Day 1/2 evidence
was not edited or reformatted. No credentials, private keys, reset actions,
collectives, OOM allocations, performance workload, push or published PR.

The new lifecycle shares generic command, lease and container components. The
old Ascend performance executor was not migrated to this new container state
machine; its existing offline regression is preserved, with no new hardware claim.
Cleanup uncertainty remains recovery-required; process exit releases flock and
does not guarantee resource freedom. Every later run must repeat occupancy gates.

## Day 4 handoff

Use the [preflight runbook](../../../../../docs/p800-preflight.md) with a fresh
authorized window and an actually idle card. Card 6 is the qualified identity for
this session, not a continuing reservation. Historical run windows cannot be reused.

Implement P800 FP32 requirements and driver dispatch next. Until then
`benchmark run --case computation-FP32:P800` rejects before touching devices.
Consume and reverify the context-bound runtime binding in the same container;
use `framework_logical_id`, never infer it from local rank. Preserve locked
Conda/`-S` startup, selected-subset visibility and target synchronization.
Event elapsed time remains unqualified; use wall-clock with full target sync.
Correctness, real measurement windows, repeatability and broader capability
qualification remain separate gates. Keep the card 1 synchronization issue visible.
