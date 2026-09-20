# Day 2 / PR1 review — 2026-09-20

**Result: passed for the CPU-only control-plane scope.** This is not P800
performance data or a real Ascend hardware regression. Production registration
remains Ascend-only. Kunlunxin candidate validation scope is unchanged.

## Tested code and environment

- Baseline: `e4715d77cec7908314215f49c11980673fb46f9e`.
- Provider/lease foundation: `80e2a76a`.
- Integrated executor/worker/monitor/report: `3f68dfd2`.
- Toolkit CLI/test compatibility: `fac7cb57`.
- Final tested code (cross-domain provider import fix): `893c60f94c466bf24fcfb72dcea273f079010591`.
- Python 3.12.3, CPU-only PyTorch 2.5.1+cpu, PyYAML 6.0.1. The isolated
  venv is under the ignored result directory; host packages were not replaced.
  NumPy is absent and Torch emits a warning; the Base/PR0/Toolkit suites pass without skips. The additional public-distribution
  suite retains one baseline failure, detailed below.
- Exact commands, stdout/stderr and exits are in `logs/final-tests.json` and
  the matching log triplets. `code-identity.json` hashes changed source assets.

| Gate | Baseline | Final |
| --- | --- | --- |
| Base full offline suite | 91 passed | 120 passed |
| PR0 candidate offline suite | 32 passed | 32 passed |
| Ascend Toolkit separate suite | 79 passed | 80 passed |
| Ascend case dry-run matrix | FP32 representative | 15 planned: 10 applicable, 5 skipped |
| Selectors / monitor switch | — | 3 selectors × on/off passed |
| Capacity authorization gate | — | Missing flag rejected with expected exit 2 |

Skipped cases are FP64, FP8, TF32 and both interserver cases. MPI/P2P intraserver
use the existing communication profile; P2P uses its approved smoke override.
Capacity is only planned with the explicit flag; it is never allocated or run.
See `logs/final-dry-runs.json` for complete arguments. Static sample IDs are not
claims that Ascend cards exist on this P800 host.

## Verified contracts

Tests cover static provider registration and unknown-vendor rejection, runtime
and asset path confinement, candidate/image identity gates, new physical
selection with request-order preservation, separate rank/logical/physical IDs,
legacy/new lease contention and exceptional release, and a non-Ascend fixture
provider through mocked execution. The Ascend expansion fixture verifies physical
`[6,2]` expands to logical `[12,13,4,5]` with request indexes `[0,0,1,1]`.

Four-layer assets, actual child-process configuration and environment consumption,
entrypoint routing and hashes are covered. Missing requirements reject; explicit
unsupported contracts skip without side effects. Monitor tests cover reordered
binding, missing/duplicate ranks and samples; report tests cover deterministic
old/new evidence, arbitrary metric units and window filtering, all statuses and
no provider/hardware lookup. Existing measurement formulas are unchanged.

Base executor static checks exclude Ascend device nodes, visibility variables,
fixed worker vendor and reverse dependency on Toolkit. FP32 dry-run comparison
preserves prior image/lock content, selection, permissions, worker and config
hashes; added vendor/provenance metadata is intentional. The complete preexisting
Kunlunxin profile has no diff against Day 1 before this new evidence directory.

## Retained failures and corrections

1. Initial system-Python Base run found 87 tests with 11 import errors because
   torch was absent (`baseline-base.*`). The isolated CPU environment established
   a clean original-source 91-test baseline (`baseline-cpu.*`).
2. Intermediate implementation ran 98 tests with 6 failures (`current.*`): the
   generic skipped report needed its existing human-readable marker; the future
   monitor schema test's hardcoded version 2 had become a supported version.
   It now tests max supported + 1 and retains the fail-closed assertions.
3. The first Toolkit migration run had 79 tests with 4 errors
   (`attempt01-final-toolkit.*`). Its mocks targeted re-exported legacy symbols
   instead of implementation globals, so attempted `npu-smi` execution failed
   because the command is absent on klx. No device command completed and no card
   workload ran. Tests now inject mocks at the canonical implementation, keeping
   every assertion; the added legacy CLI test checks help and early error format.
4. The first baseline comparison rejected an added `vendor: ascend` metadata
   field (`attempt01-baseline-comparison.json`). The corrected comparison verifies
   every old lock field and the new vendor field independently; hashes did not
   change. This was a comparison harness mismatch, not runtime identity drift.

5. Public-distribution smoke exposed a real cross-domain import collision:
   Operation already owns the top-level `vendors` package. Base now uses
   `base.vendors` without replacing Operation modules, with a fresh-process
   regression test. All six Operation vendor plans now pass. The public suite
   still has one preexisting failure (6/7 pass): a private Toolbox path in the
   existing Ascend host config. Baseline and final failures match; that host
   config was not modified. This is not a claim of publication readiness.
6. The initial public baseline export omitted `.github`, causing an artificial
   broken-workflow-link failure. The missing baseline tree was added and the
   suite rerun; `attempt01-baseline-public-distribution.*` retains that attempt.

All failures remain in the raw result directory and selected logs are copied
byte-for-byte here. No Docker workload, real accelerator workload, root action,
host driver change, or intervention in another user's task was performed.

## Compatibility and continuation

See [control-plane migration](../../../../../docs/vendor-control-plane.md) for
provider integration, override layering and summary 3 / monitor 2 migration.
Legacy summary 1/2 and monitor 1 remain readable. The standalone PR0 launcher,
Day 1 evidence and candidate manifest are untouched.

Day 3 implements P800 host/profile/collector/container identity gates and
registration. It must interlock with PR0 UUID locks. Driver dispatch, real
performance cases and promotion of validation scope require their own later
hardware evidence. Git commits are local backups; this session does not push.
