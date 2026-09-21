# Day 3 targeted recheck — 2026-09-20

The user first requested checks on physical cards 1, 2, 6 and 7, then expanded the scope to all eight cards and confirmed stopping their xpu-smi monitor before next-day
performance development. All runs were sequential, bounded, single-card probes
with the existing code at `e936db64446d62c73b380317fd37b80fcacfee1a` (repository
HEAD `80dd3f98b69bcca914325f4e51ce46df81131a1f`). No executable implementation changed for this recheck.
The immutable M1 image and driver gates are unchanged. The initial 45-minute window and subsequent 35-minute all-card window apply only to this session, not to tomorrow's work.

## Decision for the next session

| Card | Finding | Recommendation |
| --- | --- | --- |
| 0, 3, 4, 5 | Selected-card preflight rejected occupied memory (31948 / 1844 / 28268 / 36760 MiB); no workload launched | Recheck after existing allocations are released by their owners |
| 1 | Monitor on and off both reached target synchronization and hit the 120-second deadline; selected PCI kernel logs contain task timeouts, NOC idle timeouts and `KL_XID_KERNEL_EXCEPTION` | Exclude from performance qualification pending separate device/driver/runtime investigation |
| 2 | Repeated four-element tensor/readback and 16-sample observation passed; normal preflight also passed end-to-end after the user stopped monitoring; uncorrectable ECC counts 8/8 and remap pending YES were already present in Day 1 | Avoid as the performance baseline until the health condition is assessed |
| 6 | Normal preflight passed with 16 valid observation samples; controlled timeout cleanup passed | Preferred for FP32 implementation and qualification after fresh preflight |
| 7 | Normal preflight passed with 16 valid observation samples; controlled timeout cleanup passed | Alternative to card 6 after fresh preflight |

Card 1 maps to UUID `b3509946-0bc6-5744-bd7c-a0b87dadef02`, PCI `0000:16:00.0`
and `/dev/xpu2`. Both `extended-card1-attempt01` and
`monitor-off-card1-attempt02` logged `checkpoint: synchronizing device` and
faulthandler stacks in `torch.cuda.synchronize`. The monitor-off contrast only
disables this run's collector; independent host samplers were not controlled.
It therefore shows that disabling our collector did not resolve the failure,
not that all sampler interactions have been excluded. The kernel evidence
narrows investigation to the selected device execution stack but does not
distinguish hardware, driver or runtime root cause, nor establish causality
for each message. After the user reported stopping the external monitor, `after-stop-card1-attempt01` also ran with our monitor off and again timed out at synchronization. The fresh preflight had passed. This reproduces the failure after the known external monitoring was stopped, but cannot prove that no other monitoring existed.

Card 2 maps to `/dev/xpu3`, card 6 to `/dev/xpu5`, and card 7 to `/dev/xpu6`.
UUID joins and context hashes were independently checked for all launched runs.
Card 1/6/7 selected health queries reported zero uncorrectable ECC and no pending
remap. That is not a full hardware health certification; card 1 still fails sync.

## Status and cleanup interpretation

All **33 new attempts** are retained in this separate archive;
the earlier 14 Day 3 attempts remain byte-for-byte unchanged. Preflight handle
rejections do not count as tensor tests or timeout drills. Some launched probes
have a passed tensor/monitor but a failed overall summary due to postflight.
Those original statuses have not been rewritten.

A selected `fuser -v` observation identified an independent transient `xpu-smi`
holder. Not every nonempty handle observation was attributed. The tool did not
kill samplers or ignore handles. Where postflight failed after confirmed own
container removal, a separate review reacquired both lock protocols and repeated
the full selected host identity/idle checks. It proceeded only after an empty
handle check passed. The final independent audit confirms every launched CID is
absent, both locks can be reacquired, binding run IDs/context hashes match, and
all four selected cards passed a fresh idle/identity check under locks.

Controlled timeout drills on cards 2, 6 and 7 completed the tiny tensor first,
then deliberately waited on CPU for the host 120-second watchdog. Their
experiments correctly remain **failed/exit 1**, while container cleanup and
subsequent idle/lock recovery passed. Card 1's actual synchronization timeouts
are execution failures, not successful controlled timeout drills.

| Attempt | Physical card | Experiment | Probe | Valid samples | Original postflight |
| --- | --- | --- | --- | --- | --- |
| `after-stop-card1-attempt01` | 1 | failed | not completed | 0 | passed |
| `allcards-normal-card0-attempt01` | 0 | failed | not completed | 0 | not-run |
| `allcards-normal-card1-attempt01` | 1 | failed | not completed | 0 | not-run |
| `allcards-normal-card1-attempt02` | 1 | failed | not completed | 0 | failed |
| `allcards-normal-card2-attempt01` | 2 | failed | not completed | 0 | passed |
| `allcards-normal-card2-attempt02` | 2 | passed | passed | 16 | passed |
| `allcards-normal-card3-attempt01` | 3 | failed | not completed | 0 | not-run |
| `allcards-normal-card4-attempt01` | 4 | failed | not completed | 0 | not-run |
| `allcards-normal-card5-attempt01` | 5 | failed | not completed | 0 | not-run |
| `allcards-normal-card6-attempt01` | 6 | passed | passed | 16 | passed |
| `allcards-normal-card7-attempt01` | 7 | passed | passed | 16 | passed |
| `allcards-timeout-card2-attempt01` | 2 | failed | passed | 0 | passed |
| `allcards-timeout-card6-attempt01` | 6 | failed | passed | 0 | passed |
| `allcards-timeout-card7-attempt01` | 7 | failed | passed | 0 | passed |
| `confirm-card2-attempt01` | 2 | failed | passed | 16 | failed |
| `extended-card1-attempt01` | 1 | failed | not completed | 0 | failed |
| `extended-card6-attempt01` | 6 | passed | passed | 16 | passed |
| `extended-card7-attempt01` | 7 | passed | passed | 16 | passed |
| `monitor-off-card1-attempt01` | 1 | failed | not completed | 0 | not-run |
| `monitor-off-card1-attempt02` | 1 | failed | not completed | 0 | failed |
| `normal-card1-attempt01` | 1 | failed | not completed | 0 | failed |
| `normal-card1-attempt02` | 1 | failed | not completed | 0 | passed |
| `normal-card1-attempt03` | 1 | failed | not completed | 0 | failed |
| `normal-card2-attempt01` | 2 | failed | passed | 16 | failed |
| `repeat-card1-attempt01` | 1 | failed | not completed | 0 | not-run |
| `repeat-card1-attempt02` | 1 | failed | not completed | 0 | not-run |
| `repeat-card1-attempt03` | 1 | failed | not completed | 0 | passed |
| `repeat-card2-attempt01` | 2 | failed | passed | 16 | failed |
| `timeout-card2-attempt01` | 2 | failed | passed | 0 | failed |
| `timeout-card6-attempt01` | 6 | failed | not completed | 0 | not-run |
| `timeout-card6-attempt02` | 6 | failed | passed | 0 | passed |
| `timeout-card7-attempt01` | 7 | failed | not completed | 0 | not-run |
| `timeout-card7-attempt02` | 7 | failed | passed | 0 | failed |

See `logs/acceptance-allcards.json` (and the earlier `logs/acceptance-recheck.json`) for exact original summaries and final checks,
`logs/final-selected-kernel*.json` for PCI-selected kernel records, and
`logs/health-summary.json` and `logs/final-allcards-health.json` for health snapshots. A final idle check describes
that instant, not a continuing reservation.

## Evidence and limits

Full operational evidence remains at `base/result/p800-recheck-20260920-1952`. Portable evidence
projects global machine inventories to the selected card and redacts external
process identifiers/names. Source paths/hashes are in `provenance.json`; original
monitor hashes are preserved in `summary.source.json`. Diagnostic runners are
included for review of the retry and recovery policy. The SHA256 index covers
this archive. Original Day 1/2/3 evidence, stack lock and manifest are unchanged.

No reset, driver or SSH service changes, other-user process termination, OOM,
collective, formal performance measurement, runtime promotion or push occurred.
No new offline regression is claimed because executable code did not change;
the prior Base154/PR032/Toolkit80 results remain historical at their recorded
commit. P800 FP32 requirements and driver still need Day 4 implementation,
correctness, full target synchronization, measurement-window telemetry and
repeatability. Use wall-clock timing bracketed by full synchronization; do not
reuse the zero-valued PR0 event elapsed time as a performance timer.
