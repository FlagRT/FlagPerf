# Day 1 single-card evidence review — 2026-09-18

Decision: retain M1 for further candidate development. Technical smoke and the
image go/no-go are complete. Maintainer compatibility confirmation is pending;
formal validation stays false. No performance result is claimed.

## Final attempt and identity

- Raw directory: `base/result/p800-pr0-20260918-card2-attempt16`.
- Code: `f38d2684e359bebf8a91cc9df9a498cbded0fc78`, clean tree at launch;
  `code-identity.json` records exact script/manifest/lock hashes. Subsequent
  documentation changes do not change the tested executable code.
- Container runtime: 2026-09-18 20:02:22–20:03:21 +08:00.
- User explicitly authorized choosing an idle card. The launcher end time was a
  bounded execution budget, not a reservation made with another person.
- xpu-smi index 2, PCI `0000:1c:00.0`, serial `02K15K0263D00327`,
  UUID `09d75c76-1b13-5686-a3d2-a9647e48f619`.
- Query Minor Number 3, host/container `/dev/xpu3`, major 195/minor 3;
  framework `cuda:0`, CUDA visibility 0, count 1, matching UUID.
- `container-binding.json` maps only that node and `/dev/xpuctrl`; nonprivileged.
  Worker `opened_device_nodes` matches. No host driver libraries were mounted.
- `host-summary.json`: exit 0, stop/remove successful, container absent.
- Raw preflight and postflight both show this card at 0 MiB / 0% utilization.
  Other users' all-card inventory and process data are not included here.

## Results

| Probe | Result |
|---|---|
| import/device/sync | Success, tensor creation and CPU readback match |
| FP32 | 32×32, CPU FP64 reference; rtol/atol 1e-4; max error 6.6186313503191485e-06 |
| seed | Seed 519 produces identical repeated device values |
| memory | Free 103045660672, total 103079215104 bytes; `torch.OutOfMemoryError` exists |
| pinned copy | 1024-byte H2D/D2H round trip exact |
| Event | Completion works after device sync; elapsed_ms=0, timer unusable |
| route control | Same inputs/reference/H2D preparation, no device matmul |

Memory/OOM is an API query, not proof of full allocatable capacity or successful
OOM recovery. The wall time contains first-use work and is not a benchmark.

Native stderr API summaries show `cu_xpu_launch_async` count 4 in the control
and 8 with four matmuls. Both have exactly two H2D calls; the FP32 worker also has
two D2H readbacks for correctness. Matching identity, native launch delta and
correct results support device execution of this bounded FP32 path. Counters
are not named kernel attribution and do not exclude all mixed CPU/device work.
Do not promote the broad `cpu_fallback_excluded` flag. `summary.json` preserves
the supervisor's conservative flags; this review separately verifies physical
mapping. Native runtime libraries and SHA256 values are in every result JSON.

## Prior attempts retained

- Card 1 attempt01: preflight rejected open handles; no container ran.
- Card 2 attempts01–10: output permissions, visibility and synchronization
  diagnostics. The initial assumption index=minor incorrectly mapped `/dev/xpu2`
  (actual xpu-smi card 1, also observed idle), while sampling card 2. Their
  telemetry cannot establish selected-card execution. This mapping error is
  explicitly excluded from successful isolation evidence.
- Card 2 attempt11: querying actual minor and UUID fixed the mapping; six API
  stages passed with `/dev/xpu3`, CUDA visibility 0, capabilities dropped.
- Attempt12: PyTorch profiler required an additional tracing switch.
- Attempt13: PyTorch profiler emitted only CPU events, not device kernels.
- Attempt14: native tracing plus PyTorch profiler caused a subscriber conflict;
  not accepted as a successful route qualification.
- Attempt15: native counters with separate no-matmul control passed; no profiler.
- Attempt16: committed code with strict binding gates reproduced attempt15.

No reset, host driver change, all-card container mapping or interference with
other containers was used. Failed attempts remain in ignored raw directories.

## Remaining gates

- Resolve Driver/XPU-RT/header/XRE/XHPC version relationships with maintainers;
  questions are prepared, no messages have been sent.
- Verify registry distribution and reproducibility of the local digest.
- Obtain named-kernel or equivalent stronger route evidence where formal case
  qualification requires comprehensive fallback exclusion.
- Qualify other shapes/precisions, FlagGems and FlagCX separately. The loaded
  BKCL library does not prove collectives work.
- Implement later Base provider/driver/executor integration and performance
  protocol. PR0 does not claim those tasks or Ascend regression are complete.
