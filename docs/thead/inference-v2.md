# PPU Inference（flagprof v2）适配指南

平头哥 PPU 的 Inference v2（flagprof）厂商适配。Inference v2 在目标进程内接入官方
PyTorch profiler 采集设备事件，经封存导入进入公共 Store，再由分析器生成性能/显存/通信
报告；架构见 [Inference 框架文档](../../inference/docs/README.md)，通用用法见
[使用说明](../../inference/docs/usage.md)。本页只记录 PPU 特有的身份、镜像与边界，
不重复通用流程。

## 当前状态与边界

- 适配状态：candidate（候选适配）。2026-11-03 已在 thead1 真机完成：单卡端到端
  （record → import → inspect → report）、Qwen3-Embedding-0.6B 上 perf / perf-lite /
  accuracy A/A oracle / modules / perf+memory / coverage（gems 镜像）全部预设、
  TP2 双 rank（nccl 后端）与 delivery publish→verify→restore。多机、TP>2 与长期
  运行不在此范围。
- `performance_disposition` 对 thead 采集固定为 `not_qualified`：PPU 时钟一致性未完成
  验收，报告中的性能数字只作观测，不作资格判定。
- 集合通信识别只采信 exporter 显式元数据（kernel args 的 `Collective name` 字段）。
  TP2 隔离实测证实该决定：PPU 集合通信 kernel（pccl 名族）全部携带显式元数据，且
  `pcclKernel_AllReduce_RING_LL_Sum_float` 名字含 AllReduce 而元数据是 `barrier`——
  kernel 名会误导，元数据才是权威；因此不钉 kernel 名字族。
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
| `flagperf/thead-ppu-runtime-gems:1.0.0` | `087a234b7abe` | `coverage` 预设（FlagGems 路由覆盖）专用；torch 同为 2.10.0，`layer_norm` 在该构建未注册 |

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
  （2026-11-03 实测 17 passed）。覆盖 PPU 身份 UUID 交叉核验与字段缺失拒绝、契约与
  NVIDIA 互异且拒绝错误硬件栈、解释器显示名（python/python3）均在契约内、时钟变换仅在
  钉住 exporter 版本且 base 唯一时生效、集合通信显式元数据分类与未知名字归 compute、
  真实 exporter 片段导入回归与窗口投影防篡改、以及 pidfd/memfd/owned 超时等进程原语。
- 真机片段 fixture：`inference/tests/fixtures/thead/gemm-fragment.json` 逐字抽取自
  2026-11-03 thead1 PPU 冒烟会话（gemm kernel 及其 launch 关联、用户注释与进程元事件），
  `provenance.json` 记录源 trace 与片段 sha256、镜像、采集器与硬件版本；`.gitattributes`
  对该 fixture 关闭行尾转换。
- 真机预设矩阵（thead1，Qwen3-Embedding-0.6B，8 条中文输入，batch 1，bfloat16）：

| 预设 | 模式 | 实测结果（2026-11-03） |
|---|---|---|
| perf | final | `within`；perf.op/layer/device/gap = ok；61910 源记录 |
| perf-lite | windowed | 主机探针 112 事件、8 forwards；perf.layer = ok（设备采集不属该预设，`capture_status=not_requested` 是公共架构口径） |
| accuracy ×2 + oracle | windowed | `--tensor-blob all`；oracle identity_diff/layer/first_divergence = ok，`first_inequal=null`（同栈逐位一致）；op_sensitivity 需另配 `--tensor-ops-in` |
| modules | final | 5293952 源记录；op/layer/device/gap = ok；导入约 15 分钟 |
| perf + `--profile-memory` | final | memory = ok（13052 记录/11137 归一化/1915 排除） |
| coverage（gems 镜像） | windowed | route.coverage = ok；FlagGems 注册 6 函数（见下） |

  `compile.health` 在所有预设为 partial——编译监听器只覆盖成功编译与持久缓存命中，
  失败、失效与进程内复用按面板口径不观测。
- TP2：`launch --vendor thead --comm-backend nccl --nproc-per-node 2` 在 card13+14
  实测，group complete、双 rank returncode 0、双 run 导入 complete；导入分类 comm=5
  （allreduce/barrier，basis=explicit_collective_metadata）。`nccl` 是 torch 2.10.0 PPU
  构建中唯一注册的集合通信后端（flagcx/hccl/cmcc 未注册，gloo 仅 CPU）。
- delivery：publish → verify → restore 跨容器实测通过（19 files/168472 bytes，
  restore 映射保持原路径结构）。
- 模型权重可追溯：ModelScope `Qwen/Qwen3-Embedding-0.6B`（2026-11-03 master 快照），
  逐文件 sha256 记录在 `download-provenance.json`；两台 thead 使用同一镜像 ID。

### 运维注意（实测踩坑）

- `modules` 预设导入 907 s > record 默认 `--timeout 900`：显式给大 timeout；被杀的
  import 会留下 Plan-B checkpoint，复用同一 work 目录重跑会 exit 2——每轮重试用全新
  work/scratch。
- `--flaggems-include` 接受 JSON 文件路径（非内联数组）；include 要求全部注册成功，
  本构建 `layer_norm` 未注册，实测可用集为
  `["mm","add","mul","softmax","rms_norm","silu"]`。
- record 输出的 run 路径在宿主机上，传给容器内 inspect/report 时要换成容器内挂载路径。

## 与 Base / Toolkit / Operation 的关系

本适配只覆盖 Inference v2 观测链路。Base Benchmark 与 Toolkit 见
[PPU 适配指南](README.md)，算子验收见
[PPU Operation](../../operation/vendors/thead/README.md)；各模块验收状态独立维护，
互不替代。
