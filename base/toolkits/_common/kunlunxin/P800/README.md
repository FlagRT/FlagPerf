# Kunlunxin P800 Toolkit

Twelve independent native Toolkit cases are dispatched through `base/run.py toolkit run`.
The implementations call the XRE/XBLAS libraries bundled with the locked candidate
image. They do not import PyTorch, run Base workloads or read old Base measurements.

## Run from the host

```bash
cd /home/kzhang519/Zhiyu/runtime-team/FlagPerf
python3 base/run.py toolkit run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids 4 --case main_memory-capacity --dry-run

# Hardware: root, or an authorized noninteractive sudo prefix for Docker/fuser.
python3 base/run.py toolkit run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids 4 --allow-candidate-runtime --smoke \
  --result-root base/result/p800-toolkit

# Four selected cards give six unordered pairs. Occupied-card use is explicit.
python3 base/run.py toolkit run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids 0,4-6 --allow-candidate-runtime --allow-busy-devices \
  --case interconnect-P2P_intraserver \
  --case interconnect-P2P_intraserver-latency \
  --minimum-seconds 15 --repeat 5 --timeout 3600 \
  --result-root base/result/p800-toolkit

# Deterministic, offline presentation rebuild.
python3 base/run.py report --result-root base/result/p800-toolkit --run-id RUN_ID
```

Card **1 is reserved for the authorized eight-card Base run only**. It remains excluded from normal Toolkit runs and all Toolkit ranges such as `0-7`; the eight-card authorization does not qualify a shared or unhealthy card. There is no default-all device selector. Check the host inventory first and prefer
idle eligible cards. Busy-card permission does not override health, identity or
lease checks, kill other processes, reset cards, or qualify a shared run as idle.
The twelve `main.sh` entries forward their arguments to the same host facade;
they require the same physical selection and candidate permission.

`--smoke` fixes a small 128 GEMM and 1 MiB copy payload with 25 samples and two
warmups. It never claims reference-size or peak performance. Normal runs default
to GEMM 2048, H2D/D2H/D2D 512 MiB, at least 50 samples and five warmups. Matrix
size, sample count, repeat count, per-command timeout and an explicit minimum
timed duration can be changed and are retained in the evidence. Large shapes are
separate measured scopes; a timeout remains a failed point, not an unsupported
dtype or a silently smaller replacement. `--privilege-command 'sudo -n'` is
available when the host user needs an existing authorized sudo configuration.

Pinned host memory defaults to `posix_memalign` + `xpu_host_register`;
`--pinned-api alloc` selects `xpu_host_alloc` explicitly. In this runtime, a
single 512 MiB pinned async submission returned error -999 with both APIs.
The working default uses `--async-chunk-bytes 1048576`: 512 native async
submissions for a 512 MiB logical payload, followed by one stream wait. This
measures the entire chunked transfer, including submission overhead. Metrics
retain `async_chunk_bytes`, `api_calls_per_sample` and `submission_scope`.
`--async-chunk-bytes 0` reproduces the whole-payload path; it is not silently
replaced by blocking copies. Smaller payloads use a single async submission.
Historical failed runs remain failed after the fix.

## Tools, origins and measured semantics

All paths below are relative to the locked image's
`$CONDA_PREFIX/lib/python3.10/site-packages/torch_xmlir`, unless stated otherwise.
`provenance.json` records presence, resolved path and SHA-256 for the actual tools.

| Case or component | Source | Purpose and limitations |
|---|---|---|
| FP32/FP16/BF16 computation | `xhpc/xblas/include/cublas_v2.h`, `libxpu_blas.so`; API pattern in vendor `samples/cublas/cublas_ex/sample_GemmEx.cpp` | Independently written `native_bench.cpp` calls `cublasGemmEx`, FP32 accumulation/output, no epilogue. Throughput is `2*M*N*K / time`; CPU reference checks run outside the timer. |
| INT8 computation | `xhpc/xblas/include/xblas_legacy_deprecated_api.h`, `libxpu_blas.so`; vendor `samples/cublas_legacy/deprecated/sample_fc_fusion.cpp` | Native `fc_fusion<int8_t,int8_t,float,int8_t,float,float>`: INT8 inputs/TGEMM, FP32 output, maxima 127, alpha 1, beta 0, no bias, LINEAR activation. It is explicitly not INT8→INT32 GemmEx; the latter and Lt route failed in this candidate. |
| H2D/D2H bandwidth/latency | `xre/include/xpu/runtime.h`, `xre/so/libxpurt.so.2` | `xpu_memcpy` or `xpu_memcpy_async` followed by stream wait. Pageable/pinned and blocking/nonblocking are independent variants. Host allocation and full-buffer verification are outside the measured interval. |
| D2D bandwidth | Same XRE API | Runtime memcpy metric counts payload once (~10 GB/s). Independent XBLAS Scopy kernel mode reports read+write (about 2.09 TB/s) and is labeled separately; the two values must not be combined. |
| P2P bandwidth | Same XRE `xpu_memcpy_peer` API | Six unordered pairs for four cards; ascending single direction and a separately measured concurrent two-direction protocol. The bidirectional denominator waits for both workers and includes their synchronization overhead. |
| P2P latency | Same XRE API | 64 KiB single-direction copy + completion waits. No inferred reverse result. |
| Capacity | `xre/bin/xpu-smi` | Per-card Memory Usage Total/Used/Free in MiB, PCI/UUID checked. This is not held-allocation or OOM stress. |
| `test_dma` | `xre/tools/test_dma` | Vendor comparator. This build's `-h` documents `--kind H2D/D2H`, in addition to round trip and host allocation options. It does not provide this runner's full sweep/evidence contract. Help is archived; no vendor performance result is fabricated from it. |
| `test_memcpy_peer` | `xre/tools/test_memcpy_peer` | Vendor unit/performance comparator, `--unit`/`--perf`. The Toolkit adapter uses the same native peer API to own timing, correctness and pair selection explicitly. |
| `fc_effciency/run_eff.py` | `xhpc/xblas/script/fc_effciency/` | Official profiling/efficiency pipeline, requiring a matching XBLAS `unittest` plus `dma_config` and evaluator. No matching binary was supplied; this pipeline is recorded as not-run, not confused with the independently implemented native API backend. |
| `perf_regression` | `xhpc/xblas/script/perf_regression/script/` | Vendor developer regression alternative; also depends on XBLAS unittest. Not a prerequisite for the native API adapter. |
| `xprofiler` | `xre/profiler/xprofiler` | Optional API/kernel tracing tool. File/help availability is archived. Current normal runs do not claim a kernel trace or enable privileged global profiling. |
| Host monitor | Host `/usr/local/bin/xpu-smi`, existing `monitoring/kunlunxin_usage.py` | Concurrent selected-UUID memory/utilization/temperature/power sampling, raw JSONL and charts. Shared telemetry includes other workloads and cannot attribute all utilization to this run. |

The bundled compatibility runtime returns an unchanged array for
`xpu_device_list`. Enumeration therefore uses local ordinals from device count,
selects each context and reads its actual PCI address. The entire PCI set must
match the selected host UUID/PCI/minor records before allocation. A physical
card number is never assumed to equal a container native ordinal. Single-card
physical 4 was observed as native ordinal 0.

## Evidence and status

```text
RUN_ID/
  summary.json, resolved-plan.json, code-identity.json, image-identity.json
  code-source/  # Exact operational source snapshots used by this run
  host-preflight/HOST/, locked-preflight/, host-postflight/
  container-plan.json, container-inspect.json, cleanup.json, lease.json
  report.md, report_monitor.md, report-assets/*.svg, sha256-index.json
  toolkit-evidence/
    manifest.json, environment.json, provenance.json, runtime-bindings.json
    toolkit-contract.json, build/, environment/, health/, topology/, diagnostics/
    monitor/samples{,.raw}.jsonl, monitor/summary.json
    cases/CASE/metrics.json
    cases/CASE/TARGET/repeat-N/
      microbenchmark.stdout, microbenchmark.stderr, microbenchmark.command.json
      metrics.json, samples.json
```

- `measurement_status=passed` requires zero exit code, correct identity, completed
  operation samples with positive finite times and correctness evidence. A tool
  file or successful help command is never sufficient.
- `monitoring_status=passed` requires at least ten complete valid samples per
  target inside the **native timed interval**, excluding setup, warmup and readback.
  Capacity uses its own static query observation. Short measurements stay partial;
  no synthetic zero samples or time-window extensions are inserted.
- `diagnosis_status=not-supported` means the equivalent Ascend DMI vendor threshold
  diagnosis is unavailable. xpu-smi ECC, temperature and power are observations,
  not replacement vendor threshold tests. Therefore overall status can remain
  partial even when measurement and monitor both pass.
- Repetition means independent processes, not iterations inside one process.
  Five-repeat CV and within-process CV are separate. Shared, short, unstable and
  shape-limited results never promote the candidate image.
- Timeout kills only the runner's owned subprocess group. Container cleanup
  verifies ownership before removing the container, performs postflight, then
  releases stable UUID leases. Existing failures and unrelated processes survive.

The P800 output matches the reference's functional categories and audit fields;
native host-synchronized timing, FP32 GEMM outputs, copy payload definitions and
the missing vendor diagnosis layer remain explicit comparison limits. It does
not imply equal hardware peak metrics or final Base eight-card qualification.

## Offline checks

```bash
python3 -m unittest discover -s base/tests -p 'test_p800_toolkit*.py' -v
python3 -m unittest discover -s base/toolkits/_common/ascend/A3/tests -v
```

Reports can be regenerated without hardware. `sha256-index.json` excludes itself
and covers the other regular files; repeated generation preserves experiment
status and produces the same report, SVG and index bytes for unchanged evidence.
