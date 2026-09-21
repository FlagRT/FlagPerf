# P800 native FP32 candidate qualification

The unified Base entrypoint uses the locked M1 XPYTORCH runtime, one physical
device and one rank. The host reuses the bounded preflight lifecycle, including
PR0-compatible UUID locks, lock-held postflight, container inspection and cleanup.
This is candidate development; other precisions, FlagGems and device collectives
are not qualified by this case.

The supervised child starts with the locked Python `-S`, passes node/image/package
gates, enables site hooks, joins runtime UUIDs, then runs the common FP32 case in
that same child. It provides explicit rank-zero environment and initializes Gloo
for the CPU control plane. No multi-process `torchrun` or FlagCX device process
group is needed for this strictly single-rank path. The case rechecks the current
UUID set and context-bound binding; framework ordinal is never inferred from rank.

## Run

Use an actually idle card and a fresh authorized window. Historical card 6/7
qualification is not a reservation. Card 1 has an unresolved synchronization
failure; card 2 has historical ECC/remap warnings. See `base/docs/p800-preflight.md`.

```bash
python3 -B base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case computation-FP32:P800 --physical-device-ids PHYSICAL_ID \
  --nproc-per-node 1 --timeout 300 --dry-run
```

For smoke, authenticate host privilege interactively, then use:

```bash
sudo -v
python3 -B base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case computation-FP32:P800 \
  --case-config base/benchmarks/computation-FP32/kunlunxin/P800/case_config.smoke.yaml \
  --physical-device-ids PHYSICAL_ID --nproc-per-node 1 \
  --allow-candidate-runtime --privilege-command 'sudo -n' \
  --reservation-end 'END_TIME_WITH_TIMEZONE' \
  --reservation-reference 'ACTUAL_AUTHORIZATION_REFERENCE' \
  --result-dir base/result/UNIQUE_ATTEMPT --monitor off --timeout 300
```

Qualification removes the smoke override and uses `--monitor on`. The frozen
`qualification-plan.json` specifies shape/config identity, 20,000 iterations and
five independent runs on the same UUID. Each result directory must be new.
Actual measurement must be at least 15 seconds with at least ten valid overlapping
target samples. Do not pad it with sleep or alter iterations during measurement.

The `case_config.timeout.yaml` diagnostic performs correctness then waits on CPU
until the host watchdog expires. Use a 120-second watchdog and monitor off.
Expected experiment result is failed/exit 1; successful cleanup is a separate
acceptance criterion. Do not reinterpret it as a successful performance run.

## Measurement and evidence

CPU-seeded FP32 inputs use `randn(M,N)/sqrt(N)` and `randn(N,K)`. Three small
matrices have full CPU FP64 references. Before and after timing, the actual large
shape checks fixed rows/columns using the complete reduction dimension and tests
full-output finiteness. Tolerances are 1e-4 absolute/relative; artifacts define the
relative denominator. This is sampled large-output validation, not a full check.

After warmup and full target synchronization, `perf_counter_ns` measures the
`torch.mm` loop through its last full synchronization. Tensor transfers, CPU
references and output readback are outside timing. Returned-output allocation
may be included. TFLOPS = `2*M*N*K*ITERS / elapsed_seconds / 1e12`.
`drivers/events.py` records host monotonic/UTC boundaries; CUDA Event elapsed
time remains unusable. Raw precision is preserved in the rank JSON.

CUDA `allow_tf32=false` is recorded, but its acceptance by a compatibility API
does not certify internal IEEE arithmetic. UUID, tensor placement, runtime
libraries and selected-card telemetry support the native route; no universal
CPU-fallback exclusion is claimed. Empty fallback markers are not proof.

Review `artifacts/correctness-rank-0.json`, `artifacts/metric-rank-0.json`,
`artifacts/runtime-bindings.json`, `artifacts/benchmark-events/`,
`container-inspect.json`, `container-started.json`, `benchmark-result.json`,
`benchmark-monitor/`, `cleanup.json`, and host pre/postflight. Summary schema 3 and
monitor schema 2 stay compatible with the existing offline report renderer.

```bash
python3 -B base/qualification.py RUN1 RUN2 RUN3 RUN4 RUN5 --output NEW_STATISTICS_JSON
python3 -B base/run.py report --result-root PARENT_DIRECTORY --run-id ATTEMPT_DIRECTORY
```

Statistics verify raw correctness/metrics, code/config/image/UUID consistency,
five distinct runs and complete cleanup/monitoring. All five values contribute;
sample standard deviation uses ddof=1 and CV must be at most 5%. Keep failures and
unstable groups. Report regeneration never upgrades the experiment status.
