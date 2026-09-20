# Benchmark vendor control plane (PR1)

PR1 prepares the host/worker/monitor/report contracts for additional vendors.
The production registry still contains only Ascend. Kunlunxin host requests fail
explicitly until PR2 implements and qualifies its provider. No P800 performance
support or new runtime validation is implied by this refactor.

## Provider responsibilities

`vendors/protocol.py` defines `VendorProvider` and immutable `DeviceBinding`;
`vendors/registry.py` uses explicit static registration. A provider owns runtime
defaults and validation, host preflight, device bindings, container permissions
and arguments, bootstrap and optional container probe, monitor collection and
rank targeting, and fallback markers. The executor owns lifecycle orchestration.
Docker image inspection is in `executors/host.py`. Ascend preflight lives in
`vendors/ascend/preflight.py`; the old Toolkit path remains a compatibility shim.

Planning must be pure: CLI dry-run reads repository configuration, produces JSON,
and must not inspect Docker, invoke device commands, acquire leases, start
collectors, import device runtimes, or create a run directory. The test-only fake
provider exercises this boundary and a mocked execution lifecycle.

## Device selection and leases

Benchmark accepts mutually exclusive `--physical-device-ids`, `--npu-ids`
(Ascend compatibility), or legacy logical `--device-ids`. The new physical
selector preserves request order. A binding separately records physical ID,
framework logical ID/name, local rank, request order, nodes, PCI/UUID when known,
and a stable vendor resource key. Never infer framework ID from rank: Day 1 P800
evidence includes physical request `[6,2]` mapping to logical `[1,0]`.

New resource leases hash vendor/resource identities into filenames and acquire
in deterministic order. Ascend also acquires legacy logical-device locks so
Benchmark and Toolkit/older callers continue to interlock. Partial acquisition
and exceptional execution release only the caller's acquired locks. This is
cooperative locking, not a replacement for resource reservation.

## Case contract and migration

`benchmarks/case_assets.py` is shared by host, worker and actual case processes.
Configuration merges `generic < vendor < chip < override`; entrypoint takes the
most specific existing `main.py`; environment scripts run generic to vendor to
chip before torchrun timing. Use `--case CASE:CHIP` for a chip layer.

**Compatibility change:** `--case-config` now overlays vendor defaults instead
of replacing the vendor YAML. Copying a full vendor YAML still works; omitted
keys now inherit earlier layers. Override is snapshotted into
`case-config/override.yaml`. Portable `case-assets.json` records relative paths,
content hashes and merged values. Worker and case consumers re-resolve and
verify it, rejecting changed assets. Files and layers cannot escape the case
directory via traversal/symlinks; an explicitly supplied override can be external
before snapshotting. A100 fallback is limited to the legacy NVIDIA path.

Host execution requires an explicit valid runtime requirements contract. Eight
previously implicit Ascend cases now carry contracts preserving prior behavior.
Child requirements may restrict supported status or minimum ranks but cannot
silently loosen parent constraints. Malformed/missing contracts are configuration
errors. Explicit unsupported cases skip before device/Docker/lease access and
produce a static plan and skipped evidence. Workload overrides cannot change
these capability gates. Runtime/profile, lock and manifest paths are confined
to their vendor directory; unknown vendor/environment fields fail closed.

## Evidence and report compatibility

| Artifact | New schema | Compatibility |
| --- | --- | --- |
| Benchmark summary | 3 | Offline renderer still reads Ascend 1/2 |
| Benchmark monitor summary | 2 | Offline renderer still reads monitor 1 |
| Benchmark result | 1 | Measurement semantics unchanged |
| Report metadata | 3 | Experiment status is not rewritten by rendering |
| Case assets | 1 | Shared host/worker/case hash contract |

Monitor rank events are matched to explicit bindings. Missing/duplicate ranks
or incomplete sampling cannot produce a complete monitor result. Generic reports
use evidence vendor/device identity and metric names/units, filter statistics to
the matching measurement window, and generate deterministic Markdown/SVG without
provider lookup or hardware access. Missing historical identity is not guessed.
Status semantics remain passed/skipped (exit 0), partial (exit 2), and failed
(exit 1); CLI configuration/authorization errors retain exit 2. Monitor off is
`not-run`, and report failure does not replace the recorded experiment state.

## Verification and next integration step

From the repository root, use a development Python environment with PyYAML and
CPU-only PyTorch (the PR1 environment uses torch 2.5.1+cpu):

```bash
python -m unittest discover -s base/tests -p 'test_*.py' -v
python3 -S -m unittest discover \
  -s base/vendors/kunlunxin/xpytorch_2.9_p800_candidate -p 'test_*.py' -v
python3 base/run.py benchmark run --case computation-FP32 \
  --device-ids 14 --nproc-per-node 1 --dry-run
```

The last command is Ascend static planning; ID 14 is an example, not a device
available on the P800 development host. PR1 does not run Ascend hardware or
occupy P800 cards. See the [Day 2 review](../vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day2/review.md)
for exact tested commit, commands, outcomes and retained failures.

PR2 must implement the real P800 provider/profile, xpu-smi parser and host checks,
reuse Day 1 UUID/minor resolution, validate container visibility and framework
identity, supply device bindings and monitor targets, and interlock with the PR0
`/tmp/flagperf-p800-<UUID>.lock` protocol before registering production support.
P800 driver dispatch and timed performance cases remain later work. The existing
candidate manifest stays unvalidated with no expanded validation scope.
