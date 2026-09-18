# P800 M1 candidate runtime (PR0)

This standalone qualification profile does not yet connect P800 to the Base
executor. M1 is retained for hardware qualification, not approved for formal
benchmarks. See `qualification-record.json` for remaining gates. The lock is
JSON-compatible YAML, readable with the Python standard library.

Day 1 is **incomplete**. Static preparation is complete; real single-card
import/FP32/CPU-reference/sync, seed/memory/pinned/OOM API checks, physical mapping
and CPU-fallback exclusion have not been verified. Maintainer questions are
prepared but unsent. The version discrepancy is recorded, not resolved.
There is no hardware-based go/no-go decision yet.

## Completed on 2026-09-18

- Created `zhiyu/kunlunxin-p800` from `origin/dev-1.0` at
  `3e7c558b6f56e6ea5f9c8b318852d97d1517a14c`.
- Develop directly on that branch in
  `/home/kzhang519/Zhiyu/runtime-team/FlagPerf`. The former linked worktree was
  removed after preserving the PR0 commits and copying/verifying raw evidence.
  Use `git switch zhiyu/kunlunxin-p800`; no additional clone/worktree is needed.
- Inspected the immutable M1 image; collected package, editable installation,
  startup hook, vendor version, library hash and pre-import `ldd` evidence.
- Collected host OS/kernel, driver/header, device nodes and `xpu-smi -q/-m` in
  `base/result/p800-pr0-20260918/` (ignored raw operational artifacts).
- Hardware probes have NOT run: reserved card and time window are pending.

The no-device audit uses `python -S` to avoid executable `.pth` hooks. An audit
exit of zero means inventory/lock collection succeeded, not hardware approval.
`evidence/static-audit.json` retains unresolved `ldd` dependencies. Loader
behavior must be checked in the eventual workers' `loaded_libraries` output.
The recorded XRE/XHPC version mismatch also needs maintainer confirmation.

## Static audit

Run an owned, disposable container using the manifest's immutable image ID,
no network, no device mappings, read-only source and root filesystem, a small
`/tmp` tmpfs, and `/bin/bash` as entrypoint. Execute:

```bash
bash /workspace/FlagPerf/base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/container_bootstrap.sh \
  --static --inspect-json /workspace/FlagPerf/base/result/p800-pr0-20260918/image-inspect.json
```

The inspect JSON must come from a fresh host `docker image inspect`, and the
container must use that exact ID. Do not overwrite or modify existing team
containers. Do not mount host driver libraries without a verified requirement.

## Reserved single-card probes

Before launching: confirm reservation, save fresh target-card telemetry, verify
idle state and PCI BDF, and create a binding JSON with these keys:

```json
{
  "schema_version": 1,
  "host_physical_id": 3,
  "container_node": "/dev/xpu3",
  "logical_device": 0,
  "pci_bdf": "0000:83:00.0",
  "reservation_reference": "REPLACE_WITH_CONFIRMED_RESERVATION",
  "reservation_start": "2026-01-01T00:00:00+08:00",
  "reservation_end": "2026-01-01T00:10:00+08:00"
}
```

These deliberately expired example values are not a reservation. Replace every
value with the actual reservation/mapping. Use an owned nonprivileged container
with ONLY same-number `/dev/xpuN` and `/dev/xpuctrl`, `CUDA_VISIBLE_DEVICES=N`,
no network, read-only source, a writable output mount, and a bounded writable
cache/tmp location if the runtime requires it. Unset `XPU_EVENT_KL3_ENABLE` and
do not enable FlagGems. No all-card mapping or host configuration changes.

Inside that container, run with an external container lifetime limit of 360s
(and 10s kill grace), using a fresh output directory:

```bash
bash /workspace/FlagPerf/base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/container_bootstrap.sh \
  --allow-candidate --inspect-json /evidence/image-inspect.json \
  --binding-json /evidence/binding.json --output-dir /results/attempt-001 --timeout 45
```

An external timeout must stop and remove only this attempt's named container;
terminating the Docker CLI alone does not guarantee the container is stopped.
Save postflight telemetry and confirm no owned container or allocation remains.
Keep all failed attempt directories. Do not reset cards.

The host launcher now performs selected-card `xpu-smi` and open-handle preflight,
checks the immutable image, creates a nonprivileged one-card container, records
its actual Docker identity/device spec, samples target-card telemetry, enforces
the container timeout, and removes only that attempt's container. It uses a
per-card advisory lock; the lock does not reserve resources from other users.
The launcher has not yet been exercised with hardware. After confirming a card
and window, run from the repository root (replace all placeholders):

```bash
sudo -v
python3 base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/launch_qualification.py \
  --card PHYSICAL_CARD --reservation-end 'END_TIME_WITH_TIMEZONE' \
  --reservation-reference 'CONFIRMED_RESERVATION' \
  --result-dir base/result/UNIQUE_PR0_ATTEMPT
```

The result directory must not already exist. The window must cover the default
360-second container timeout plus 30 seconds for cleanup. Host binding evidence
still needs comparison with the framework's device identity; Docker mapping
alone cannot establish that the framework used the intended physical card.

The supervisor validates identity and binding before enabling Python site hooks.
Each phase uses a fresh worker process and includes import and teardown in its
timeout. Core import/device or FP32 failure stops further phases. The remaining
probes record seed, memory/OOM API (no OOM allocation), pinned copies and event
timing. Timing zero is recorded as unusable; use wall time plus full device
synchronization for subsequent performance work. No throughput claim is made.

Tensor device checks alone cannot exclude internal CPU fallback or prove the
host mapping. Review target-card telemetry, device properties and actual loaded
libraries before marking these gates complete. The summary deliberately keeps
`physical_mapping_verified` and `cpu_fallback_excluded` false. No probe promotes
this candidate automatically. FlagCX collectives and FlagGems are separate work.

## Offline checks

```bash
python3 -S -m unittest discover \
  -s base/vendors/kunlunxin/xpytorch_2.9_p800_candidate -p 'test_*.py' -v
bash -n base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/container_bootstrap.sh
```

The initial 12 tests cover image/lock checks, reservation/device input gates,
and worker timeout/exit behavior. Four added tests cover the host inventory
parser (quoted product, PCI domain, missing/duplicate devices, malformed output).
These checks do not exercise torch operators, P800 hardware, live Docker
lifecycle, the Base executor or the full Ascend regression suite.

Team source references are pinned at
`runtime-team@e740bf78c08e1463c3920959e47dad3ed348118c`: device-context P800
backend/conformance, memory `device-smoke_p800.py`, and isolated probe patterns.
This implementation uses explicit-device synchronization, assertions, small
allocations and subprocess timeouts; it does not copy the legacy timing,
swallowed-error or print-only correctness behavior.
