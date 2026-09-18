# P800 M1 candidate runtime (PR0)

M1 passes the bounded single-card PR0 smoke checks and is retained for further
Base development. This standalone profile is not connected to the Base executor
and is not approved for formal benchmarks. Day 1 technical smoke and the image
decision are complete; maintainer compatibility confirmation remains open.
`image-manifest.json` remains `validated: false`, with no formal validation scope.

## Development and evidence

Develop directly on `zhiyu/kunlunxin-p800` in
`/home/kzhang519/Zhiyu/runtime-team/FlagPerf`, based on `origin/dev-1.0` at
`3e7c558b6f56e6ea5f9c8b318852d97d1517a14c`. Do not create another worktree.
Use Conventional Commits without attribution trailers.

The immutable image is
`flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`, ID
`sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975`.
The lock is JSON-compatible YAML, readable with the Python standard library.

Static inventory is in `evidence/static-audit.json`; portable single-card
evidence and its review are in `evidence/day1/`. Full operational logs, including
failed attempts, remain under ignored `base/result/p800-pr0-20260918*` directories.
Read `qualification-record.json` for status and unsent maintainer questions.

## Observed API and mapping (2026-09-18)

| Item | Observed behavior |
|---|---|
| Host identity | xpu-smi index 2, PCI `0000:1c:00.0`, UUID `09d75c76-1b13-5686-a3d2-a9647e48f619` |
| Actual device node | `/dev/xpu3` (major 195, minor 3), plus `/dev/xpuctrl` |
| Container/runtime | Nonprivileged, capabilities dropped; `CUDA_VISIBLE_DEVICES=0`, device `cuda:0`, device count 1 |
| Initialization | Explicit Conda activation, site hooks, `torch` and `torch_xmlir` imports |
| FP32 | 32×32, CPU FP64 reference, rtol/atol 1e-4, maximum absolute error `6.6186313503191485e-06` |
| Seed | `torch.cuda.manual_seed_all(519)`, exact repeat |
| Memory | `torch.cuda.mem_get_info(0)`: total 103079215104 bytes; OOM exception API exists, allocation failure NOT exercised |
| Pinned copy | 1024-byte H2D/D2H round trip matches exactly with full synchronization |
| Event | Completes after device synchronization, but elapsed time is 0; unsuitable for Perf timing |
| Environment | `XPU_EVENT_KL3_ENABLE` and `XPU_VISIBLE_DEVICES` unset; `USE_FLAGGEMS=0` |

**Never equate xpu-smi index with node minor.** Query `xpu-smi -i INDEX -q`,
match its PCI BDF against the machine inventory, then use its Minor Number and
UUID. The framework reports a generic `GPU` name and zero PCI fields; its UUID
must match the selected host card before allocating tensors. One mapped node is
enumerated as logical zero. The launcher implements these checks.

Native tracing showed 4 `cu_xpu_launch_async` calls for the input-only control
and 8 for the same preparation plus 4 matmuls. Combined with matching UUID,
opened nodes, correct output and actual Kunlun libraries, this supports P800
execution for this FP32 smoke. It is not a named-kernel trace or proof that every
internal operation avoids CPU execution. The raw supervisor keeps its broad
`cpu_fallback_excluded` flag false; the scoped review is separate. PyTorch
profiler exposed CPU events only, and combining it with native tracing caused a
subscriber conflict. Do not enable both together.

The worker's loaded-library paths and hashes resolve the earlier pre-import
`ldd` uncertainty: runtime uses the Conda `xcudart/lib/*.kunlun` libraries and
`torch_xmlir/xccl/so/libbkcl.so`. This does not establish collective support or
resolve XRE 5.13 versus XHPC metadata requiring 5.18.

## Reproduce a bounded single-card probe

Obtain authorization for an idle card, then run from the repository root. The
launcher rechecks memory, utilization and open device handles. Its advisory lock
coordinates only instances of this tool; it is not a team-wide reservation.
Replace all placeholders; previous card numbers and windows are not reusable
reservations.

```bash
sudo -v
python3 base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/launch_qualification.py \
  --card SMI_INDEX --profile-route \
  --reservation-end 'END_TIME_WITH_TIMEZONE' \
  --reservation-reference 'AUTHORIZATION_OR_RESERVATION_REFERENCE' \
  --result-dir base/result/UNIQUE_PR0_ATTEMPT
```

The result directory must be new. The window must cover the default 360-second
container timeout plus 30 seconds for cleanup. The launcher records the commit,
working-tree status, script hashes, immutable image, host identity, actual Docker
device spec, per-stage output, selected-card telemetry and postflight state.
It removes only its own container. Keep failed attempt directories; never reset
cards or modify another user's container. No host driver library mount is needed.

The container has no network, a read-only source and root filesystem, bounded
writable cache/tmp, and a group-writable result directory. Only the selected
node and control node are mapped. Bootstrap explicitly unsets KL3 and native XPU
visibility. The supervisor starts with `python -S`, checks image/binding/time
before site hooks, and uses isolated workers with a 45-second stage timeout.
`--profile-route` adds a seventh, input-only control stage and native API counters.

Use `perf_counter` with synchronization before and after the measurement window
for subsequent performance work. PR0 numbers include initialization/JIT costs
and are not throughput results. No large allocations, FlagGems qualification,
collectives, other precisions, Base end-to-end case or Ascend regression were run.

## Static audit and offline checks

In an owned no-device, no-network container using the manifest image, read-only
source and /bin/bash entrypoint, the static audit command is:

```bash
bash /workspace/FlagPerf/base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/container_bootstrap.sh \
  --static --inspect-json /workspace/FlagPerf/base/result/p800-pr0-20260918/image-inspect.json
```

Obtain the inspect JSON from a fresh host `docker image inspect`. Static audit
uses `python -S` and does not import accelerator packages or imply hardware approval.

```bash
python3 -S -m unittest discover \
  -s base/vendors/kunlunxin/xpytorch_2.9_p800_candidate -p 'test_*.py' -v
bash -n base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/container_bootstrap.sh
```

24 offline tests cover image/package locks, binding/UUID/minor/visibility gates,
time windows, worker failure/timeout, and host identity parsing. They do not
simulate hardware success or verify the entire Docker lifecycle.

Team references are pinned at
`runtime-team@e740bf78c08e1463c3920959e47dad3ed348118c`: device-context P800 API and
conformance, memory device/copy probes, and isolated process patterns. Perf adds
explicit-device synchronization, assertions, bounded allocation and evidence;
legacy timing and swallowed-error behavior are not reused.
