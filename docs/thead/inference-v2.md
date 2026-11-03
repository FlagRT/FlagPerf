# PPU Inference（flagprof v2）适配指南

平头哥 PPU 的 Inference v2（flagprof）厂商适配。Inference v2 在目标进程内接入官方
PyTorch profiler 采集设备事件，经封存导入进入公共 Store，再由分析器生成性能/显存/通信
报告；架构见 [Inference 框架文档](../../inference/docs/README.md)，通用用法见
[使用说明](../../inference/docs/usage.md)。本页只记录 PPU 特有的身份、镜像与边界，
不重复通用流程。

## 当前状态与边界

- 适配状态：candidate（候选适配）。单卡端到端（record → import → inspect → report）
  与 CPU 回归已在 thead1 实测通过；模型预设、TP2 双 rank 与 delivery 通道在各自
  实测完成前不构成已验收结论，完成后在本页更新。
- `performance_disposition` 对 thead 采集固定为 `not_qualified`：PPU 时钟一致性未完成
  验收，报告中的性能数字只作观测，不作资格判定。
- 集合通信识别只采信 exporter 显式元数据（kernel args 的 `Collective name` 字段）；
  PPU 通信 kernel 名字族尚未经隔离 dist 实测钉住，未带显式元数据的 kernel 一律归入
  compute，不做家族猜测。
- 采集完整性不做声明：`final_evidence` 只记录可证的显式导出 flag（如 `record_shapes=1`），
  其余字段保持 unknown；这由 torch 2.10.0 exporter 的 Kineto chrome trace 结构决定
  （`schemaVersion=1`、顶层 `record_shapes`/`profile_memory`/`baseTimeNanoseconds`）。
- PPU 栈没有环境变量级的 profiler 开关，身份不走运行时探测，而走显式 sourced JSON
  （见下），导入与运行时做 UUID 交叉核验。

## 环境准备

锁定镜像（thead1 实测 ID；重建后以本地 `docker images` 输出为准）：

| 镜像 | ID（sha256 前 12） | 用途 |
|---|---|---|
| `flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64` | `1ba7ead15378` | 默认采集与运行（torch 2.10.0，含 transformers/accelerate/triton，无 flag_gems） |
| `flagperf/thead-ppu-runtime-gems:1.0.0` | `087a234b7abe` | `coverage` 预设（FlagGems 路由覆盖对比）专用 |

PPU 在 torch 内使用 CUDA 兼容命名，`--device` 传 `cuda:N`。物理卡与逻辑设备的对应以
ppu-smi 为准：`ppu-smi -i N -q` 输出 Product Name / PPU UUID / Bus Id / Minor Number /
Driver Version；torch `get_device_properties(N).uuid` 去掉 `GPU-`/`gpu-` 前缀后须与
ppu-smi 的 UUID 一致。

### sourced 身份

先生成宿主身份 JSON（每张物理卡一条 device 记录，数组顺序与 torch 逻辑序一致）：

```bash
ppu-smi -i 0 -q   # 读出型号 / UUID / Bus Id / Minor Number / 驱动版本
```

```json
{"source":"ppu-smi","driver":"1.5.1-1d747a","devices":[
  {"model":"PPU-ZW810E","uuid":"GPU-…","pci_bdf":"00000001:C9:00.0",
   "driver":"1.5.1-1d747a","minor":"1","physical":"0","host_device_node":"/dev/alixpu_ppu0"}]}
```

运行与导入时经 `FLAGPROF_THEAD_IDENTITY` 注入路径。校验规则：`source` 必须是
`ppu-smi`，`model` 必须是 `PPU-ZW810E`，所选逻辑卡的 UUID 与 torch 观测完全一致，
且 `pci_bdf`、`driver`、`host_device_node` 不得为空；任一不满足即拒绝采集。

格式契约钉在 `inference/flagprof/vendors/thead/contracts/v1.json`：
`hardware.model=PPU-ZW810E`，driver ∈ {`1.5.1-1d747a`, `1.5.3-5239a4`}，
collector = torch.profiler 2.10.0，`exporter.schema_version=1`。契约外照常导入，
但适用性标注 `outside`，不宣称已验证。

## 最小命令

宿主侧准备身份文件后，在容器内运行（`$PWD` 为仓库工作树或任意工作目录）：

```bash
IMG=flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64
DEV="--device=/dev/alixpu_ctl --device=/dev/alixpu --device=/dev/alixpu_ppu0"

# 采集（默认 final 预设：完整会话，目标正常退出后独立导入）
docker run --rm --shm-size=4g $DEV \
  -v "$PWD":/work -w /work \
  -e FLAGPROF_THEAD_IDENTITY=identity-ppu-card0.json \
  $IMG python3 -B -m flagprof record --vendor thead --preset perf \
  --store /work/stores/smoke \
  -- python3 /work/target.py

# 读取采集与导入状态（把 record 输出中的 run 路径写入 FP_RUN）
docker run --rm -v "$PWD":/work -w /work $IMG \
  python3 -B -m flagprof inspect "$FP_RUN" --verify --json

# 生成自包含报告
docker run --rm -v "$PWD":/work -w /work $IMG \
  python3 -B -m flagprof report "$FP_RUN" --out /work/reports/smoke
```

窗口模式加 `--capture-mode windowed`（按目标的 `flagprof` step 句柄推进窗口，
`--active`/`--windows` 控制窗数，`--timeout` 前传给受控目标进程）。多窗口时钟按窗口
独立投影（`source_clock=per_window`），不跨窗双计。外部自行导出的 trace 用
`flagprof seal <trace> --out <dir> --vendor thead --normal-close` 封存后经
`flagprof import` 进入 Store。

## 验证与测试

- CPU 回归：容器内 `PYTHONPATH=inference python3 -m pytest inference/tests -q`
  （2026-11-03 实测 16 passed）。覆盖 PPU 身份 UUID 交叉核验与字段缺失拒绝、契约与
  NVIDIA 互异且拒绝错误硬件栈、时钟变换仅在钉住 exporter 版本且 base 唯一时生效、
  集合通信显式元数据分类与未知名字归 compute、真实 exporter 片段导入回归与窗口投影
  防篡改、以及 pidfd/memfd/owned 超时等进程原语。
- 真机片段 fixture：`inference/tests/fixtures/thead/gemm-fragment.json` 逐字抽取自
  2026-11-03 thead1 PPU 冒烟会话（gemm kernel 及其 launch 关联、用户注释与进程元事件），
  `provenance.json` 记录源 trace 与片段 sha256、镜像、采集器与硬件版本；`.gitattributes`
  对该 fixture 关闭行尾转换。
- 真机最小验收：上述最小命令在 thead1 单卡实测，报告状态 `perf.op` / `perf.device` /
  `perf.gap` = ok，`format_applicability=within`，`compile=partial`，其余分析按原因
  unavailable；`performance_disposition=not_qualified`。
- 模型预设、TP2 与 delivery 的实测状态见顶部"当前状态与边界"；不以其他模块的历史
  结果替代本页边界。

## 与 Base / Toolkit / Operation 的关系

本适配只覆盖 Inference v2 观测链路。Base Benchmark 与 Toolkit 见
[PPU 适配指南](README.md)，算子验收见
[PPU Operation](../../operation/vendors/thead/README.md)；各模块验收状态独立维护，
互不替代。
