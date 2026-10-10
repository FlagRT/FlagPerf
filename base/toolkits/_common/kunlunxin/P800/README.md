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

# Hardware: authenticate sudo interactively, then run the host runner as root.
sudo -v
sudo -n env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory \
  GIT_CONFIG_VALUE_0=/home/kzhang519/Zhiyu/runtime-team/FlagPerf \
  python3 base/run.py toolkit run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids 4 --allow-candidate-runtime --smoke \
  --result-root base/result/p800-toolkit

# Four selected cards give six unordered pairs. Occupied-card use is explicit.
# Run this example from an authorized root host shell.
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

No physical card is excluded by operator policy; the former card-1 reservation ended on 2026-10-10. Every selected card still passes the same health, identity and lease gates. There is no default-all device selector. Check the host inventory first and prefer
idle eligible cards. Busy-card permission does not override health, identity or
lease checks, kill other processes, reset cards, or qualify a shared run as idle.
The twelve `main.sh` entries forward their arguments to the same host facade;
they require the same physical selection and candidate permission.

`--smoke` fixes a small 128 GEMM and 1 MiB copy payload with 25 samples and two
warmups. It never claims reference-size or peak performance. Normal runs default
to GEMM 8192, H2D/D2H/D2D 512 MiB, at least 50 samples and five warmups. Matrix
size, sample count, repeat count, per-command timeout and an explicit minimum
timed duration can be changed and are retained in the evidence. Large shapes are
separate measured scopes; a timeout remains a failed point, not an unsupported
dtype or a silently smaller replacement. `--privilege-command 'sudo -n'` is
available when the host user needs an existing authorized sudo configuration. It prefixes
Docker/fuser commands only; it does not grant the host runner access to shared
root-owned lease files or make a user-owned mode-755 evidence directory writable
by a capability-dropped container root. The verified path on this deployment host (klx) is to run the
host runner as root; this note and the cloud-observer paragraph below are site records, and other environments replay the same gates with their own inventory and paths. Preserve shared lock ownership and permissions.

No `--case` selects all twelve cases. For two cards and five repeats the complete
suite has 485 targets: 40 compute, 20 D2D, 10 capacity, 40 H2D bandwidth,
40 D2H bandwidth, 160 H2D latency, 160 D2H latency, 10 P2P bandwidth and 5 P2P
latency. Enable both compute and data-movement monitoring and select
`--minimum-seconds 15 --repeat 5` for the monitored scope. The 475 timed targets
alone need about 119 minutes; setup, readback and archiving add time. Use an
adequate `--timeout` (for example 14400 seconds), a new result root and several
GiB of available storage because every raw timing sample is retained. Capacity
remains a static query. `--p2p-payload-bytes 268435456` explicitly selects 256 MiB
bandwidth payloads; P2P latency remains 64 KiB.

If host preflight identifies a verified read-only cloud observer (recorded on the klx host),
`P800_KNOWN_READONLY_MONITORS=barad_agent` enables the existing exact observer
allowlist. Check its executable and command line first. This does not authorize
busy devices or bypass memory, utilization, health, identity or lease checks.
H2D/D2H latency is one-direction API-to-completion host time, not round-trip time.

Keep frozen runs unchanged. For an offline presentation rebuild, copy the run
first and record the renderer version used on that copy.

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
- `diagnosis_status` is the vendor toolchain compliance gate (policy 2026-10-08):
  it passes when the measured operation went through XBLAS/XRE correctness readback
  with per-point checked values plus contract coverage validation, backed by xpu-smi
  pre/post health observation. No ascend-dmi equivalent threshold
  (aiflops/bandwidth/hbm/signalQuality) tool exists in this toolchain; that fact is
  kept explicit in coverage.json and report methodology instead of failing the layer.
  Final status composes measurement x monitoring x diagnosis: a case passes only when
  all three pass; short monitoring windows still keep the case partial.
- Repetition means independent processes, not iterations inside one process.
  Five-repeat CV and within-process CV are separate. Shared, short, unstable and
  shape-limited results never promote the candidate image.
- Timeout kills only the runner's owned subprocess group. Container cleanup
  verifies ownership before removing the container, performs postflight, then
  releases stable UUID leases. Existing failures and unrelated processes survive.

The P800 output matches the reference's functional categories and audit fields;
native host-synchronized timing, FP32 GEMM outputs, copy payload definitions and
the absent DMI-equivalent threshold diagnosis (diagnosis layer is compliance
validation) remain explicit comparison limits. It does
not imply equal hardware peak metrics or final Base eight-card qualification.

## Offline checks

```bash
python3 -m unittest discover -s base/tests -p 'test_p800_toolkit*.py' -v
python3 -m unittest discover -s base/toolkits/_common/ascend/A3/tests -v
```

Reports can be regenerated without hardware. `sha256-index.json` excludes itself
and covers the other regular files; repeated generation preserves experiment
status and produces the same report, SVG and index bytes for unchanged evidence.
