<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# FlagPerf Base 使用指南

本目录提供 FlagPerf 的基础规格评测能力。当前推荐入口是统一宿主命令
`python3 base/run.py`，但 Benchmark 与 Toolkit 仍使用独立执行器、权限和结果语义：

- **Benchmark**：使用各 profile 锁定的框架和原始 Base Case；Ascend 为 PyTorch 2.10/Torch-FL，P800 为 PyTorch 2.9/XPYTORCH；
- **Toolkit**：通过 MindCluster ToolBox、`ascend-dmi`、`npu-smi` 或 HCCL Test
  执行厂商测量和诊断；
- **report**：从已有证据确定性重建 Markdown/SVG 报告，不重跑硬件。

当前统一入口支持 **Ascend 单宿主**，并接入 **P800 candidate 单卡原生 FP32/FP16/BF16/INT8 计算与 H2D/D2H 传输**。原多机入口保存在
[`legacy/cluster_run.py`](legacy/cluster_run.py)，仅用于迁移兼容。
Benchmark 控制面通过静态 Vendor Provider 接入厂商策略；生产 registry 已注册
Ascend 性能执行与 Kunlunxin 单卡预检查/受控 FP32 性能执行。
P800 的启动命令、正确性和计时合同见 [P800 FP32](benchmarks/computation-FP32/kunlunxin/P800/README.md)，
实际资格结果见 [Day 4 审查](vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day4/review.md)；
FP16/BF16/INT8（8192³ 设备内核）与传输合同的资格范围、能力矩阵和已知限制见
[Day 5 审查](vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day5/review.md)。
预检查与性能验收分别记录；FP64/FP8/TF32 已有锁定栈能力结论，FlagGems、容量和通信仍待独立验收。
接口、配置兼容变化与后续接入步骤见 [控制面迁移说明](docs/vendor-control-plane.md)。


## 阅读导航

第一次使用，先完成下面的快速上手，再按需查阅：

- [环境与宿主配置](#1-环境基线)：准备镜像、ToolBox 和设备路径。
- [设备选择与静态计划](#2-先生成静态计划)：区分物理 NPU、逻辑 Device 和 rank。
- [Benchmark 使用](#3-运行-benchmark)：查找 Case、调整工作量、运行多个测试。
- [Toolkit 使用](#4-运行-ascend-toolkit)：选择厂商测量、时延和 HCCL 测试。
- [FlagCX P2P](#5-flagcx-单机-p2p)：使用专用通信环境。
- [结果与报告](#6-结果报告和退出码)：判断成功与否、找到原始数据。
- [常见问题](#7-常见问题)：定位配置、占用、运行和证据问题。
- [进一步阅读](#8-入口与职责边界)：了解实现和旧入口。

## 快速上手：完成一次 FP16 测试

以下命令在 **FlagPerf 仓库根目录**执行（当前目录应能看到 `base/`）。
示例假定环境已按第 1 节准备好；`7` 必须替换为本机已预约、空闲的物理 NPU ID。

```bash
# 1. 查看设备及物理 NPU → 逻辑 Device 映射
npu-smi info
npu-smi info -m

# 2. 检查命令与静态配置，不启动容器、不占用 NPU
python3 base/run.py benchmark run \
  --case computation-FP16 --npu-ids 7 --dry-run

# 3. 正式运行（默认开启同窗监控）
python3 base/run.py benchmark run \
  --case computation-FP16 --npu-ids 7 --allow-privileged-root
```

运行后，到命令输出对应的 `base/result/<RUN_ID>/` 查看 `report.md` 和
`summary.json`；报告中的每个 rank 对应一个选中的逻辑 Device。
`--dry-run` 成功只表示静态计划能生成，正式运行还会检查镜像、设备映射和占用。

如果你想测的是厂商工具给出的 FP16 算力，使用：

```bash
python3 base/run.py toolkit run \
  --case computation-FP16 --npu-ids 7 \
  --allow-privileged-root --allow-disruptive-dmi
```

| 你想回答的问题 | 选择 |
| --- | --- |
| 当前 PyTorch/Torch-FL 栈运行矩阵乘、内存复制有多快？ | Benchmark |
| 厂商工具测得的算力、带宽、时延或诊断结果是什么？ | Toolkit |
| 已跑过测试，只想重新生成阅读报告？ | report |

同名 Case 在两个入口下可能采用不同 workload。Base 是基础性能测试工具，
这里的利用率监控也不是模型逐算子 profiler。

## 1. 环境基线

标准 operator runtime 锁定在：

- CANN 9.0.0；
- MindCluster ToolBox 26.1.0（从宿主挂载）；
- Python 3.11.15；
- PyTorch 2.10、Torch-FL、Triton Ascend 和 FlagGems；
- 镜像 `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64`。

Torch-FL 拥有 PyTorch PrivateUse1/`torch.flagos`，不要在该镜像中安装或混用
`torch_npu`。锁文件、镜像身份和验证边界位于
[`vendors/ascend/torch_fl_2.10/`](vendors/ascend/torch_fl_2.10/)。

镜像获取方式、本地 image ID 与公开分发限制见 [Ascend 环境指南](../docs/ascend/README.md)。

运行前确认：

1. Docker daemon 可用；
2. `npu-smi info` 能看到目标 NPU，且目标 Device 空闲；
3. [`configs/ascend910_cann9_local.yaml`](configs/ascend910_cann9_local.yaml) 中的 ToolBox、驱动和设备路径与宿主一致；
4. 使用 `--npu-ids` 或 `--device-ids` 显式选择本轮资源；
5. 仅在确认高权限容器可接受后传入 `--allow-privileged-root`。

所有命令均建议从仓库根目录执行：

```bash
cd FlagPerf
python3 base/run.py --help
```

### 在另一台宿主上配置

宿主需要 Python 3、Docker、Ascend 驱动、`npu-smi` 和用于检查设备占用的 `fuser`。
PyTorch 等测试依赖在锁定容器中使用；镜像获取和构建步骤见上面的环境指南。

复制默认配置到版本库外，例如 `/tmp/flagperf-host.json`，再按本机情况修改：

```bash
cp base/configs/ascend910_cann9_local.yaml /tmp/flagperf-host.json
```

| 配置字段 | 应填写的内容 |
| --- | --- |
| `image` | 与运行时锁及镜像清单匹配的镜像；仅改 tag 不能完成环境升级 |
| `toolbox_host_path` | 本机 MindCluster ToolBox 安装目录，示例路径不可照搬 |
| `expected_device_ids` | 本机完整逻辑 Device 清单，用于检查 inventory；不是本次选卡列表 |
| `required_devices`、`host_mounts` | 本机驱动设备节点及驱动、固件等挂载路径 |
| `result_root` | 结果根目录；也可在命令中用 `--result-root` 指定 |

该宿主配置虽然以 `.yaml` 命名，当前使用 **JSON 语法**读取；修改时保留 JSON 格式。
后续运行加上 `--config /tmp/flagperf-host.json`。Benchmark 的工作量另由
`--case-config` 控制，不写在宿主配置中。

## 2. 先生成静态计划

`--dry-run` 只校验仓库配置并打印静态计划，不检查 Docker、NPU 占用或物理到逻辑
Device 映射，也不会生成虚假的 image ID：

```bash
python3 base/run.py benchmark run \
  --case computation-FP16 \
  --npu-ids 7 \
  --monitor on \
  --dry-run

python3 base/run.py toolkit run \
  --case computation-FP16 \
  --npu-ids 7 \
  --dry-run
```

示例中的 NPU 7 不是固定要求。正式运行前必须按当前宿主拓扑替换为已授权的空闲物理
NPU ID；也可改用 `--device-ids 14,15` 一类逻辑 Device 集合。Benchmark 的
`--physical-device-ids`、`--npu-ids`、`--device-ids` 三者互斥；Toolkit 保留后两者。

### 物理 NPU、逻辑 Device 与 rank

- `--npu-ids 7`：选择物理 NPU，正式 preflight 按实时映射展开其逻辑 Device。
- `--physical-device-ids 7`：Benchmark 的通用物理选择器，保留请求顺序；在 Ascend
  上与 `--npu-ids 7` 选择同一物理资源，最终 rank 数仍按展开后的逻辑 Device 数计算。
- `--device-ids 14,15`：直接选择逻辑 Device；支持逗号和范围，如 `2,3,6-9`。
- Benchmark 默认每个所选逻辑 Device 启动一个进程（rank）。若显式填写
  `--nproc-per-node`，必须等于解析后的逻辑 Device 数。

例如，**仅当本次映射确认为 NPU 7 → Device 14、15** 时，`--npu-ids 7`
才相当于选择这两个 Device，并启动两个 rank。若只测试一个逻辑 Device，使用
`--device-ids 14`；不要选完整物理 NPU 后强行写 `--nproc-per-node 1`。
Toolkit 计算测试有完整物理 NPU 的粒度要求，建议优先使用 `--npu-ids`。

正式运行会检查所选设备占用并获取协作锁（lease）；该锁只协调遵守同一协议的
FlagPerf 进程，仍需事先预约资源。监控采样按实际选中的设备记录。

## 3. 运行 Benchmark

### 3.1 有哪些 Case 可以选

`--case` 后填写下表中的完整名称，大小写需一致。**一次调用只运行一个 Case**，
必须显式指定；当前没有 Benchmark `--all`、`--suite` 或自动全量模式。

下表覆盖当前 `benchmarks/` 中的 15 个 Case 目录。“已有 Ascend 入口/配置”表示代码接入范围，
不表示所有参数都已实机通过，也不保证每次在两分钟内结束。

| Case | 测什么 / 主要输出 | 当前 Ascend 使用方式 |
| --- | --- | --- |
| `computation-FP16` | FP16 矩阵乘，TFLOPS | 常规 operator runtime，适合作为首次测试 |
| `computation-BF16` | BF16 矩阵乘，TFLOPS | 已有 Ascend 配置 |
| `computation-FP32` | FP32 输入矩阵乘，TFLOPS | 已有配置；使用后端默认精度模式，不强制严格 IEEE FP32 |
| `computation-INT8` | 缩放量化矩阵乘，TOPS | Ascend 专用实现：INT8 输入、INT32 累加、BF16 输出 |
| `main_memory-bandwidth` | 设备内 `clone()` 读写吞吐，GB/s、GiB/s | 已有 Ascend 配置；`main_memory` 在这里指设备内存 |
| `main_memory-capacity` | 持续分配至 OOM，估计可分配设备内存容量 | 高风险；必须加 `--allow-high-risk-case` |
| `interconnect-h2d` | 主机到设备传输带宽，GB/s、GiB/s | 已有 Ascend 配置 |
| `interconnect-d2h` | 设备到主机传输带宽，GB/s、GiB/s | 已有 Ascend 配置 |
| `interconnect-P2P_intraserver` | 同机设备间点对点通信带宽 | 使用第 5 节专用 FlagCX 配置，恰好两个 rank |
| `computation-TF32` | TF32 计算测试 | 未原生适配 Ascend，启动前跳过；暂不做 HF32 适配 |
| `computation-FP64` | FP64 计算测试 | 910C/A3 无本 Case 所需的原生 FP64 GEMM，启动前跳过 |
| `computation-FP8` | FP8 计算测试 | 910C/A3 无本 Case 所需的原生 FP8 GEMM，启动前跳过 |
| `interconnect-MPI_intraserver` | 同机 AllReduce SUM 带宽 | 使用既定 FlagCX 通信配置，至少两个 rank；见下方命令 |
| `interconnect-P2P_interserver` | 跨机点对点通信 | 不在当前统一入口的单宿主范围内，启动前跳过 |
| `interconnect-MPI_interserver` | 跨机集合通信 | 不在当前统一入口的单宿主范围内，启动前跳过 |

其他厂商的目录库存不能直接当作当前 Ascend 入口的支持清单。可在
[`benchmarks/`](benchmarks/) 查看源码和每个 Case 的配置。

### 3.2 运行一个 Case

下面的命令使用 NPU 7 映射出的逻辑 Device 启动原始 FP16 Case，并对实际选中设备采集
同窗 `npu-smi info -t usages` 证据：

```bash
python3 base/run.py benchmark run \
  --case computation-FP16 \
  --npu-ids 7 \
  --monitor on \
  --allow-privileged-root
```

Benchmark 保留原 Case 的 YAML 合并、warmup、计时、公式和 `[FlagPerf Result]` 输出语义。
Ascend 适配器负责 Torch-FL 初始化、`flagos:<local_rank>` 设备选择、设备同步和容量探测异常兼容；Gloo
只用于 GEMM 等非通信 workload 的控制 barrier。

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--case` | 必填的 Base Benchmark Case |
| `--case-config` | 保存到本轮结果中的 Case YAML 快照，作为配置合并的最后一层 |
| `--nproc-per-node` | 本机 torchrun rank 数；默认等于解析出的逻辑 Device 数 |
| `--monitor on\|off` | 是否采集独立状态的同窗监控，默认 `on` |
| `--timeout` | 容器硬超时，默认 3600 秒 |
| `--allow-high-risk-case` | 显式允许容量/OOM/长时间 Case |

Ascend 默认采用较短的 warmup/ITERS，容量 Case 的结束等待默认关闭；工作负载和配置快照写入结果。
两分钟是日常测试目标，尚不能保证逐 Case 耗时，容量探测仍可能超时。

设备内存带宽默认使用 4 GiB 张量、`WARMUP=2`、`ITERS=10`，以约束当前 Torch-FL
同步 D2D `clone()` 路径的运行时间。输出是该路径的实测吞吐，不代表硬件峰值 HBM 带宽。
容量 Case 会保留分配失败的原始错误并减半重试；若最小申请也从未成功，则仍报错。
Ascend 默认开启 `BOUND_REQUEST_BY_FREE_MEMORY`，查询当前设备空闲 HBM 来缩小下一次
申请量（最多取当前空闲量的一半），避免接近全部空闲量的大块申请引发底层慢重试。
查询值只指导搜索；容量结果只统计实际分配并
持有的张量，最终仍尝试 1 MiB 分配以确定结束边界。关闭该选项可复现原始逐次减半流程。

同机 AllReduce 使用项目已经确定的 FlagCX 镜像配置（文件名沿用 `p2p`）：

- 配置文件：[`configs/ascend910_cann9_p2p.yaml`](configs/ascend910_cann9_p2p.yaml)。
- 锁定镜像：`flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64`。
- 此通信镜像供 Benchmark 单机 AllReduce 和 P2P 共用；普通计算、内存及 H2D/D2H
  Case 使用常规 operator 配置。Toolkit 同名通信测试不使用此 FlagCX 配置。

```bash
python3 base/run.py benchmark run \
  --config base/configs/ascend910_cann9_p2p.yaml \
  --case interconnect-MPI_intraserver \
  --device-ids 4,6 --monitor off --timeout 105 \
  --allow-privileged-root
```

在当前 A3 宿主上，Device 4、6 分别位于物理 NPU 2、3；其他宿主先核对映射。
默认每 rank 消息 4 MiB，预热 10 次、测量 100 次。计时前用非零输入验证 SUM，
计时循环用零输入避免反复原地求和溢出，结束后再检查结果。保留原 Case 的带宽公式，
包含额外乘 2 的历史口径，不能直接与 Toolkit 同名指标混用。默认 operator 镜像不能运行此 Case。

`main_memory-capacity` 必须额外传入 `--allow-high-risk-case`。不要在共享机器上用完整参数
直接试跑该 Case。

### 3.3 调整矩阵大小和迭代次数

先区分两个配置：`--config` 选宿主/镜像，`--case-config` 选测试工作量。
Case 按 `generic < vendor < chip < override` 顺序覆盖同名字段；芯片层通过
`--case CASE:CHIP` 显式选择。`--case-config` 现在是最后一层，省略的字段保留前面
各层的值，不再替换厂商配置文件。宿主保存配置快照与 `case-assets.json`，worker
及实际 Case 消费同一解析合同，并校验来源文件 SHA256。能力 requirements 独立于
工作量 YAML，override 不能绕过 runtime、rank 或 unsupported 门禁。

例如，制作一次用于检查运行链路的小规模 FP16 测试：

```bash
cp base/benchmarks/computation-FP16/ascend/case_config.yaml /tmp/fp16-smoke.yaml
```

将 `/tmp/fp16-smoke.yaml` 的内容改为：

```yaml
M: 1024
N: 1024
K: 1024
DIST_BACKEND: "gloo"
WARMUP: 5
ITERS: 20
```

```bash
python3 base/run.py benchmark run \
  --case computation-FP16 --device-ids 14 \
  --case-config /tmp/fp16-smoke.yaml --dry-run

python3 base/run.py benchmark run \
  --case computation-FP16 --device-ids 14 \
  --case-config /tmp/fp16-smoke.yaml --allow-privileged-root
```

这里 `M、N、K` 对应 `A[M,N] × B[N,K]`，`WARMUP` 是不计入主测量的预热次数，
`ITERS` 是计时循环次数。当前默认 FP16 Ascend 配置为 `8192³`、预热 `100` 次、
计时 `5000` 次。小规模示例用于检查链路，不能当作峰值性能结果。

其他 Case 的字段需查看各自 YAML：内存带宽和传输常用 `Melements` 表示
`Melements × 1024 × 1024` 个元素；容量 Case 采用分配至 OOM 的流程，
并非修改 `ITERS` 就能限制耗时。不要把 FP16 覆盖文件用于其他类别 Case。
P2P 配置另受哈希 allowlist 限制，不能按此方法任意改参数。

### 3.4 依次运行多个常规 Case

Benchmark 每次产生独立结果目录。需要比较几项时，可在已预约设备上顺序执行：

```bash
for case in computation-FP16 computation-BF16 computation-FP32; do
  python3 base/run.py benchmark run \
    --case "$case" --npu-ids 7 --allow-privileged-root || break
done
```

遇到非零退出码（包括 `partial`）即停止，先检查对应报告。容量和 P2P 有独立资源、
权限或环境要求，应分别按各自说明运行。

## 4. 运行 Ascend Toolkit

Toolkit 会主动执行 DMI/HCCL 等性能命令，因此需要独立授权：

```bash
python3 base/run.py toolkit run \
  --case computation-FP16 \
  --npu-ids 7 \
  --allow-privileged-root \
  --allow-disruptive-dmi
```

`--case` 可重复使用；不指定时执行默认 12 个 Case 的集合，但统一入口仍要求显式选择设备。
`--compute-monitor` 和 `--data-movement-monitor` 分别控制计算与数据搬运作证，默认均为
`on`。Toolkit 数值来自厂商 microbenchmark，不应在 workload、设备范围、计时边界和公式
未对齐时直接与 Benchmark 相除或归因。

### 4.1 Toolkit Case 清单

| Case | 测量方式 / 结果 | 默认运行 |
| --- | --- | --- |
| `computation-FP16`、`computation-BF16`、`computation-FP32` | DMI 算力，TFLOPS | 是（3 项） |
| `computation-INT8` | DMI 整数算力，TOPS | 是 |
| `main_memory-bandwidth` | DMI D2D 带宽 | 是 |
| `main_memory-capacity` | `npu-smi` 查询 HBM 容量；不执行 Benchmark 的 OOM 探测 | 是 |
| `interconnect-h2d`、`interconnect-d2h` | DMI 主机↔设备带宽 | 是（2 项） |
| `interconnect-h2d-latency`、`interconnect-d2h-latency` | DMI 多消息大小时延，ns | 是（2 项） |
| `interconnect-P2P_intraserver` | DMI 同机 P2P 带宽 | 是 |
| `interconnect-P2P_intraserver-latency` | DMI 同机 P2P 时延，ns | 是 |
| `interconnect-MPI_intraserver` | HCCL Test AllReduce 带宽与正确性检查 | 否，必须显式指定 |

默认集合包含主动负载和诊断，省略 `--case` 不等于只读查询。P2P 需要至少两个
所选逻辑 Device；跨宿主的两个 `interserver` Case 不在当前 Toolkit 执行范围内。
Toolkit 的 DMI P2P 使用常规环境，不使用 Benchmark 的 FlagCX 通信配置。

### 4.2 只运行指定项目

通过重复 `--case` 选择多个项目，例如只测 H2D/D2H：

```bash
python3 base/run.py toolkit run \
  --case interconnect-h2d --case interconnect-d2h \
  --npu-ids 7 --allow-privileged-root --allow-disruptive-dmi
```

时延扫描与 HCCL 示例：

```bash
python3 base/run.py toolkit run \
  --case interconnect-h2d-latency --latency-sizes 512,4K,64K,1M \
  --npu-ids 7 --allow-privileged-root --allow-disruptive-dmi

python3 base/run.py toolkit run \
  --case interconnect-MPI_intraserver --device-ids 14,15 \
  --hccl-min-bytes 8K --hccl-max-bytes 1G \
  --allow-privileged-root --allow-disruptive-dmi
```

HCCL 需要至少两个所选逻辑 Device，且镜像包含 MPI 和 `all_reduce_test`。
P2P 延迟默认只测 64 KiB，每个无序设备对仅执行小 ID → 大 ID，不推断反向结果；未指定选卡时也使用此规则遍历可见设备。H2D/D2H 延迟仍默认 `512,4096,65536,1048576` 字节。显式 `--latency-sizes` 可覆盖上述尺寸，`K/M/G` 按二进制换算。
HCCL 默认扫描 `8K` 到 `1G`；调整上下限必须同时显式选择其 Case。
这些命令同样可先加 `--dry-run` 查看静态计划。

两个监控开关均接受 `on|off`，默认 `on`；`--timeout` 默认 3600 秒，是整个 Toolkit
容器的硬超时，不是每个 Case 的预算。`--legacy-probe` 用于旧命令输出兼容性核对，
日常测量一般无需启用。全部参数可通过以下命令查看：

```bash
python3 base/run.py benchmark run --help
python3 base/run.py toolkit run --help
```

完整参数、逐 Case 机制、证据和限制见
[`toolkits/ASCEND_A3_910C_TEST_MECHANISM.md`](toolkits/ASCEND_A3_910C_TEST_MECHANISM.md)。

## 5. FlagCX 单机 P2P

P2P 使用独立 communication runtime、单节点和恰好两个 rank，Case 配置必须命中仓库哈希
allowlist。正式协议为 `p2p-single-node-v1`，其限定范围、校准、故障检查及恢复方法集中在
[P2P 协议](../docs/ascend/p2p.md)。单次功能检查计划：

```bash
python3 base/run.py benchmark run \
  --config base/configs/ascend910_cann9_p2p.yaml \
  --case interconnect-P2P_intraserver \
  --device-ids 14,15 --nproc-per-node 2 \
  --case-config base/benchmarks/interconnect-P2P_intraserver/ascend/case_config.smoke.yaml \
  --monitor off --dry-run
```

实际执行时移除 `--dry-run` 并传入 `--allow-privileged-root`。当前锁定通信镜像在既定单机范围内
已资格化；更换为未验证镜像仍需要独立验证和候选运行授权。
功能检查结果不能当作正式性能基线，资格矩阵不是日常两分钟测试。
仓库提供的是历史资格记录；原始日志未随仓库分发，不能视为当前代码重新实测的结果。

## 6. 结果、报告和退出码

默认结果位于 `base/result/<RUN_ID>/`，核心证据包括：

- `summary.json`：外层状态、运行时、设备、权限和生命周期；
- `resolved-plan.json`：正式 preflight 后解析出的完整执行计划；
- `case-assets.json`：Benchmark 的配置、入口、环境脚本和 requirements 来源及 SHA256；
- `benchmark-result.json` 或 `toolkit-evidence/manifest.json`：领域测量事实；
- `report.md`、`report_monitor.md` 和 `report-assets/`：确定性阅读视图；
- Case 配置快照、原始日志、pre/postflight 和 SHA-256 索引。

新 Benchmark summary 使用 schema 3，同窗监控使用 schema 2；测量结果 schema 1
和报告元数据 schema 3 保持。离线报告继续接受旧 Ascend summary 1/2 和 monitor 1，
不改写历史证据。启动前 skip 的 `resolved-plan.json` 是静态计划，尚无实机绑定；
完整 Case 资产合同在进入执行阶段时保存。详细字段与兼容边界见控制面迁移说明。

离线重建报告：

```bash
python3 base/run.py report --run-id benchmark-YYYYMMDDTHHMMSSZ
```

统一状态边界：

- `0`：`passed`，或不适用 Case 的 `skipped`（跳过不代表通过）；
- `1`：执行或测量失败，或报告重建失败；
- `2`：配置/授权错误，或执行测量通过但所请求证据不完整而得到 `partial`。

Benchmark 的 `execution_status`、`measurement_status`、`monitoring_status`、
`postflight_status` 和 `report_status` 相互独立。监控不完整不能伪装成测量通过或失败；
`--monitor off` 记为 `not-run`，不使成功测量失败。

FP64、FP8、TF32 和两个跨机 Case 在启动容器和设备预检查前输出 `SKIPPED` 与原因；同时保存
`summary.json`、`benchmark-result.json` 和报告，状态为 `skipped`，不产生性能指标。
`--dry-run` 可在 `applicability` 字段查看跳过原因。批量统计时必须区分 `passed` 和 `skipped`。

### 推荐的读报告顺序

1. 先读 `summary.json` 的总体状态和失败阶段，确认本轮是否实际执行。
2. 再读 `report.md` 的设备范围、配置和指标表，确认 rank、方向、消息大小和单位。
3. 开启监控时读 `report_monitor.md`，核对采样是否覆盖测量窗口。
4. 需要复核时查 `benchmark-result.json`、`case-config/` 或 Toolkit manifest
   中的原始证据路径；不要只保留截图或单个数值。

Benchmark 的原始 Case 日志位于
`<RUN_ID>/<CASE>/<host>_noderank0/benchmark.log.txt`。失败较早时可能只有部分文件；
应以实际结果目录和 `summary.json` 为准。重建报告不会补齐缺失的测量或监控证据。

如果运行时指定了自定义结果根目录，重建时也要指定同一位置：

```bash
python3 base/run.py report \
  --result-root /tmp/flagperf-results --run-id benchmark-YYYYMMDDTHHMMSSZ
```

将占位 RUN_ID 替换为实际目录名。默认 Toolkit 的 RUN_ID 使用 UTC 时间戳，
不要自行给它添加 `benchmark-` 前缀。

### 怎样解释数值

- FP16 等 GEMM 的 TFLOPS 按 `2 × M × N × K × ITERS / 测量秒数 / 10¹²`
  计算。它包含 Case 定义的主机调用和同步边界，不能当作纯设备 kernel 时间。
- `main_memory-bandwidth` 按一次读加一次写计入字节量；H2D/D2H 按传输方向
  计量。`GB/s` 使用 `10⁹`，`GiB/s` 使用 `2³⁰`，比较前先统一单位。
- Benchmark 容量是当前环境下的可分配量；Toolkit 容量是工具查询值，二者含义不同。
- 多 rank 输出先按实际物理映射理解，不要把单 Device 数值直接称为整卡算力。
  比较 Benchmark 与 Toolkit 前，必须对齐设备范围、workload、dtype、计时边界与公式。
- `passed` 表示本轮所要求的执行/证据条件满足；不能自动推出数值精度已充分验证、
  达到硬件峰值或已形成稳定基线。性能对比还需要同配置重复测量和完整环境记录。

## 7. 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| 提示 `the following arguments are required: --case` | Benchmark 必须选一个 Case，按第 3.1 节填写；不能套用 Toolkit 默认集合行为 |
| `--dry-run` 通过，正式运行失败 | 静态计划不检查设备和 Docker；查看本轮 `failure_stage` 与 preflight 日志 |
| 镜像不存在或身份不匹配 | 按环境指南准备对应镜像，核对宿主配置和运行时清单；tag 相同不足以证明身份一致 |
| ToolBox 路径或版本不匹配 | 核对自定义宿主配置、实际安装目录和锁定版本 |
| inventory / Device 映射不一致 | 用 `npu-smi info -m` 核对完整逻辑设备清单；本轮选卡用 CLI，不通过删减 `expected_device_ids` 实现 |
| Device 被占用或 `already leased` | 等待持有资源的任务结束或选择其他已预约空闲设备；不要删除锁文件绕过正在运行的任务 |
| rank 数不匹配 | 删除手写的 `--nproc-per-node`，采用默认值，或使它等于所选逻辑 Device 数 |
| 缺少权限开关 | Benchmark 按需使用 `--allow-privileged-root`；Toolkit 还需要 `--allow-disruptive-dmi`；容量 Benchmark 另需 `--allow-high-risk-case` |
| OOM 或 timeout | 查看 Case 原始日志；普通 Case 可减小数据规模/迭代数；容量 Case 的 OOM 探测有专门语义，不应与普通测试混淆 |
| 有性能数值但总体 `partial` | 检查独立的监控、测量及报告状态；数值存在不代表所请求证据已完整 |
| P2P 提示 runtime/config 不满足 | 使用第 5 节专用宿主配置、两个 rank 和 allowlist 中的 Case 配置 |
| 报告缺失或生成失败 | 保留结果目录，检查原始证据是否齐全，再运行 `report`；报告重建不能修复原始测试失败 |

## 8. 入口与职责边界

```text
Host
  base/run.py
    ├─ BenchmarkExecutor -> Docker -> benchmark_worker.py -> torchrun -> 原 Case
    ├─ ToolkitExecutor   -> Docker -> evidence_runner.py -> DMI/npu-smi/HCCL
    └─ report            -> 已保存证据 -> Markdown/SVG
```

两个 Executor 只共享运行时锁、设备选择、preflight、lease 和外层证据约定，不共享 Case
执行、权限、计时公式、诊断或结果 schema。`run_toolkit.py`、`run_local.py` 和
`container_main.py` 是迁移兼容入口，不是新的并列主入口。

架构、适配规则和旧集群流程见
[`../docs/base/base-case-doc.md`](../docs/base/base-case-doc.md)。

## P800 standalone preflight

`python3 base/run.py benchmark preflight` checks P800 identity, a four-element
native tensor readback, observation telemetry and bounded cleanup without a
performance case. Its preflight schema 1 is separate from Benchmark results.
Real execution requires explicit single-card physical selection, candidate
opt-in, a current reservation window and a new result directory.

See the [runbook and Day 4 integration contract](docs/p800-preflight.md) and
[Day 3 evidence review](vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day3/review.md).
P800 FP32 case/driver integration is pending; missing contracts reject before
device access. This command does not produce TFLOPS or promote runtime validation.

2026-09-20 追加复验：优先在重新通过预检查的卡 6/7 上推进；卡 1 同步超时
复现并有 PCI 对应内核异常，卡 2 有既存 ECC/重映射告警。详见
[逐卡结果与恢复记录](vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day3-recheck/review.md)。
