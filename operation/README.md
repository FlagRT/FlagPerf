# Operation：算子精度、性能与诊断

Operation 选择一个算子及输入规模，在指定设备上检查结果是否正确、实际走了什么实现，并测量耗时和显存。
当前提供 52 个 Case（算子案例）；可以运行原生 PyTorch 路径、FlagGems 路径，或在同一次运行中比较两条路径。
两条路径分别对 CPU 参考检查正确性，不把其中一条直接当作另一条的真值。

| 目的 | 命令 |
|---|---|
| 找到算子与可用输入类型 | `list` |
| 查看执行计划或运行算子 | `run` |
| 从已有记录重建/汇总报告 | `report` |
| 检查失败证据，必要时显式复放 | `diagnose` |

## 1. 准备环境

规划入口需要 Linux、Python 3.10+；`list` 和 `run --dry-run` 只使用 Python 标准库，不访问 Docker 或设备。
真实执行还需要厂商驱动、设备权限和兼容的 PyTorch/厂商库；使用 FlagGems 路径时还需匹配的 FlagGems。
默认 `--execution docker`，需要 Docker 权限及预先取得的兼容镜像；`--execution local` 则使用当前 Python 环境，不启动 Docker。

Ascend 默认使用 CANN 9 / PyTorch 2.10 / Torch-FL 的锁定组合，按 [Ascend 环境说明](../docs/ascend/README.md) 准备。
该镜像不会由程序自动下载。NVIDIA 等兼容入口在 Docker 模式下没有默认镜像，需要显式传入 `--image`；
local 模式不需要 `--image`，但须自行准备当前 Python 运行环境。各入口的执行方式见第 2 节；
注册入口可用不代表每个厂商和 dtype 都已经实机通过。

[昆仑芯 P800](vendors/kunlunxin/README.md) 通过 `--runtime-profile p800-xpytorch29` 显式选择独立运行栈。
已接通 Docker/local，完成 52 case 默认类型 smoke、52 case 多类型 daily 的 Docker 评估，以及 local 七个代表 case 双路径验证；仍含数值/执行失败和路由/性能 partial。
Operation 运行栈仍为 `validated:false`；该参数与控制工作量的 `--workload` 不同。
已接入独立 allocator 显存诊断与 `--profiling timeline` 设备事件采集，并完成 Docker/local 代表验证；
`--profiling full` 的六组硬件计数器仍未支持。诊断采集成功不改变原任务的路由或共享性能资格。
P800 的路由证据边界、已知执行/数值失败和显式共享模式见上述厂商说明，不能外推为全算子验收通过。
P800 probe 分别保存首次及缓存调用的 ATen 派发、主机 launch 与关联设备 kernel；
只有复制、分配、设备输出或不完整关联不能判为路由通过。该观察不插入正式计时，
也不构成覆盖 C++ 后端全部内部指令的通用 CPU fallback 证明。

所有命令从 **FlagPerf 仓库根目录** 执行。先选当前获准使用的空闲设备，下面的 `0` 仅为示例；Docker/local 的编号含义见第 2 节。
Ascend 镜像涉及的特权权限由 `--allow-privileged-root` 显式开启；只在获准使用的主机上运行。

## 2. 按芯片跑通最小案例

以下命令均从 **FlagPerf 仓库根目录** 执行，使用 `abs / FP32 / smoke / nativetorch`。
`nativetorch` 是默认路径；选择它不要求执行 FlagGems 算子。示例编号 `0` 必须替换为实际获准使用的设备。
每组先执行 `--dry-run` 检查计划，再按对应的实际运行命令执行。
**dry-run 成功只说明参数和计划有效，不证明镜像、依赖、设备或算子可用。**

### 2.1 公共查询与参数选择

```bash
# 仅列出案例和输入类型，不需要设备运行环境；各厂商共用。
python3 operation/run.py list --names-only
python3 operation/run.py list --case abs
```

`list` 列出的是公共输入类型，不是所选硬件的通过清单。CLI 不使用 `--chip` 参数；
先按下表选择入口，再由真实执行核对或记录设备和运行环境。表中的兼容入口不代表该型号已完成新 CLI 的实机验收。

| 芯片或仓库现有硬件目录 | `--vendor` | `--runtime-profile` | 最小运行方式与前提 |
|---|---|---|---|
| Ascend 910C / A3 | `ascend` | 不需要 | Docker；准备 CANN 9 / PyTorch 2.10 / Torch-FL 锁定镜像 |
| 昆仑芯 P800 | `kunlunxin` | **必须带 `p800-xpytorch29`** | Docker；准备 P800 独立锁中的镜像和宿主权限 |
| NVIDIA A100_40_SXM | `nvidia` | 不需要 | Docker；显式指定兼容镜像，宿主支持 Docker `--gpus` |
| 寒武纪 MLU（旧目录未细分型号） | `cambricon` | 不需要 | local 兼容入口；环境提供 `torch_mlu` 和可用 MLU |
| 昆仑芯 R300p（旧栈） | `kunlunxin` | 不带 | local 兼容入口；环境提供旧栈 `torch_xmlir` 及其 CUDA 兼容 API |
| 天数智芯 BI150 | `iluvatar` | 不需要 | local 兼容入口；厂商 PyTorch 提供 CUDA 兼容 API |
| 沐曦 C550_64 | `metax` | 不需要 | local 兼容入口；厂商 PyTorch 提供 CUDA 兼容 API |

`--runtime-profile p800-xpytorch29` 选择 **P800 + XPyTorch 2.9 专用适配器和运行栈配置**，
包括锁定镜像、包版本及设备初始化规则；它不选择测试规模，也不表示验证通过。
当前只有 P800 注册了这个显式 profile。Ascend 直接由 `--vendor ascend` 选择专用适配器，不能加这个 P800 profile。
**P800 若省略 profile，会进入旧昆仑芯兼容入口，不能只把 Ascend 示例中的 vendor 改成 kunlunxin。**

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--execution docker\|local` | 启动容器，或使用当前 Python 运行环境 | `docker` |
| `--workload smoke\|daily` | 输入规模及采样预设 | `daily`；最小案例显式选 `smoke` |
| `--oplib nativetorch\|flaggems\|both` | 原生、FlagGems 或同一次运行比较两者 | `nativetorch` |
| `--profiling off\|timeline\|full` | 正式计时之外的独立诊断采集 | `off` |

旧 `--profile` 只是 `--workload` 的兼容别名，与 `--runtime-profile` 无关，且不能和 `--workload` 同时传入。
Docker 模式的 `--device-ids` 是宿主编号；local 模式是当前运行环境可见的逻辑编号，不能直接照搬宿主编号。

### 2.2 Ascend 910C / A3

先按 [Ascend 环境说明](../docs/ascend/README.md) 取得并核对锁定镜像、驱动与设备权限。
该入口使用 Torch-FL，不要换成 Inference 的 torch_npu 镜像。

```bash
# 预览：不启动 Docker 或设备。
python3 operation/run.py run --vendor ascend --device-ids 0 \
  --case abs --dtype FP32 --workload smoke --dry-run

# 实际执行：先确认环境、设备空闲和特权容器授权。
python3 operation/run.py run --vendor ascend --device-ids 0 \
  --case abs --dtype FP32 --workload smoke --allow-privileged-root
```

这里的 `--allow-privileged-root` 会给 Ascend 设备 worker 添加 Docker `--privileged`；
它不是自动 sudo，也不表示必须用宿主 root 账号。仅在获准使用特权容器时按此示例执行。
完整边界见 [Ascend Operation](vendors/ascend/README.md)。

### 2.3 昆仑芯 P800

先按 [P800 环境与镜像取得说明](vendors/kunlunxin/README.md#环境与最小运行) 准备独立锁中的镜像、
驱动、`xpu-smi`、`fuser`、设备句柄读取权限以及结果/共享锁目录权限。程序不自动下载镜像或提权。

```bash
# 预览：必须显式选择 P800 profile。
python3 operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --device-ids 0 \
  --case abs --dtype FP32 --workload smoke --dry-run

# 实际执行：默认要求设备空闲，使用 P800 自身的设备映射规则。
python3 operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --device-ids 0 \
  --case abs --dtype FP32 --workload smoke
```

**不要照抄 Ascend 的 `--allow-privileged-root`**；P800 会拒绝该参数，也不接受额外 `--container-device` 或 `--mount`。
宿主 root 启动仅是一种部署方式，不是这里要求 privileged 容器。
默认独占；只有获得共享授权后才增加 `--allow-shared-device`，共享性能资格保持 partial。
Operation 独立运行栈保持 `validated:false`，Base 的发布状态不会自动应用于算子。
相同运行栈下的 local 命令与编号规则见 [P800 local 说明](vendors/kunlunxin/README.md#local-运行)。

### 2.4 NVIDIA A100_40_SXM（兼容入口）

先安装兼容的宿主 NVIDIA 驱动和 NVIDIA Container Toolkit，取得包含所需 Python、PyTorch 和算子依赖的镜像。
将下面的示例值替换为实际已准备的镜像；此处没有默认锁定镜像，不会自动下载。

```bash
export OP_NVIDIA_IMAGE='your-nvidia-operation-image:tag'

python3 operation/run.py run --vendor nvidia --image "$OP_NVIDIA_IMAGE" \
  --device-ids 0 --case abs --dtype FP32 --workload smoke --dry-run

python3 operation/run.py run --vendor nvidia --image "$OP_NVIDIA_IMAGE" \
  --device-ids 0 --case abs --dtype FP32 --workload smoke
```

执行前确认所选宿主设备空闲；适配器通过 `nvidia-smi` 检查占用，并使用 Docker `--gpus device=...` 选卡。
兼容入口的路由证明范围有限，结果可能为 partial；这里给出启动方式，不声明当前型号/镜像已全量验收。

### 2.5 寒武纪 MLU、旧昆仑芯 R300p、天数智芯 BI150、沐曦 C550_64（local 兼容入口）

这四个入口尚未接入 Docker 宿主自动占用检查，**不能把下面命令去掉 `--execution local` 当成可用的 Docker 示例**。
使用者须先在宿主或自己的厂商容器中准备完整运行环境，进入该环境并确保 `python3` 是对应厂商解释器，
由外部预约设备、设置 visibility 并确认依赖可用。local 使用当前解释器，不替你创建环境、安装依赖或预约宿主设备；
其逻辑锁只保护当前命名空间。旧环境文件可供准备依赖时参考：
[MLU](vendors/cambricon/camtorch0830/Dockerfile)、[R300p](vendors/kunlunxin/xpytorch029/Dockerfile)、
[BI150](vendors/iluvatar/bi150-410/Dockerfile)、[C550_64](vendors/metax/ngctorch2403/Dockerfile)，
这些旧配方不构成新 CLI 的运行保证。

寒武纪 MLU：

```bash
python3 operation/run.py run --vendor cambricon --execution local --device-ids 0 \
  --case abs --dtype FP32 --workload smoke --dry-run
python3 operation/run.py run --vendor cambricon --execution local --device-ids 0 \
  --case abs --dtype FP32 --workload smoke
```

昆仑芯 R300p 旧栈（**不适用于 P800 专用栈**）：

```bash
python3 operation/run.py run --vendor kunlunxin --execution local --device-ids 0 \
  --case abs --dtype FP32 --workload smoke --dry-run
python3 operation/run.py run --vendor kunlunxin --execution local --device-ids 0 \
  --case abs --dtype FP32 --workload smoke
```

天数智芯 BI150：

```bash
python3 operation/run.py run --vendor iluvatar --execution local --device-ids 0 \
  --case abs --dtype FP32 --workload smoke --dry-run
python3 operation/run.py run --vendor iluvatar --execution local --device-ids 0 \
  --case abs --dtype FP32 --workload smoke
```

沐曦 C550_64：

```bash
python3 operation/run.py run --vendor metax --execution local --device-ids 0 \
  --case abs --dtype FP32 --workload smoke --dry-run
python3 operation/run.py run --vendor metax --execution local --device-ids 0 \
  --case abs --dtype FP32 --workload smoke
```

上述兼容入口保留真实失败和不充分的路由证据；给出命令不等于算子、dtype、FlagGems 或性能已经通过硬件验证。

### 2.6 找到本次结果

实际执行后终端会打印 `Report:`、`Summary:` 和 `Results:`。打开 `Report:` 指向的 `report.md`，
先看执行和正确性状态，再看路由与性能。输出默认保存在 `operation/result/operation-<id>/`，
每个 Case/dtype/路径/设备有自己的子目录；dry-run 只向终端输出 JSON 计划，不生成这些结果文件。
`smoke` 和 `daily` 的输入规模与采样次数不同，不能直接当作同一工作量比较。
执行失败、依赖阻塞或证据不全可能得到 failed、blocked 或 partial；成功生成报告不等于通过。

## 3. 比较原生与 FlagGems

FlagGems 是算子实现库；`--oplib both` 在相同输入和规模下分别运行原生与 FlagGems 路径：

```bash
python3 operation/run.py run --vendor ascend --device-ids 0 \
  --case abs --dtype FP32 --oplib both --workload smoke --allow-privileged-root
```

P800 对应命令如下，仍须保留专用 profile，且不带 Ascend 的特权参数：

```bash
python3 operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --device-ids 0 \
  --case abs --dtype FP32 --oplib both --workload smoke
```

其他厂商在第 2 节对应命令上增加 `--oplib both`，保留各自的环境与执行方式。
执行前可先加 `--dry-run` 检查计划。

每一侧都独立检查 CPU 参考和路由。结果不足以比较时保留原因，不产生看似有效的性能比值。
只想运行一侧，可选择 `--oplib nativetorch`（默认）或 `--oplib flaggems`。

## 常用变化

```bash
# 指定两个算子，或为同一算子指定多个规模。
python3 operation/run.py run --vendor ascend --device-ids 0 --case abs --case mm \
  --dtype FP16 --workload smoke --allow-privileged-root
python3 operation/run.py run --vendor ascend --device-ids 0 --case mm \
  --size M=128,N=256,K=128 --size M=512,N=512,K=512 --allow-privileged-root

# 正式计时之外，额外采集设备时间线；会增加执行成本。
python3 operation/run.py run --vendor ascend --device-ids 0 --case mm \
  --dtype FP16 --workload smoke --profiling timeline --allow-privileged-root

# 已有报告可离线重建。
python3 operation/run.py report --run-dir operation/result/operation-YOUR_RUN_ID
```

`--warmup`、`--iters`、`--rounds` 调整采样；`--profiling off|timeline|full` 控制独立采集，默认 off。
旧参数 `--profile` 保留为 `--workload` 的兼容别名，两者不能同时指定。
没有指定 `--case` 时会规划全部案例；第一次使用建议明确选择一个小案例。
`--spectflops` 是所选设备范围、dtype 对应的峰值分母；不填写时利用率为 N/A，不猜测硬件峰值。

## 失败后从哪里开始

先使用不占用设备的诊断，`--source-task` 指向结果中的具体 Case 子目录，而不是整个汇总目录：

```bash
python3 operation/run.py diagnose --source-task /absolute/path/to/case-result --dry-run
python3 operation/run.py diagnose --source-task /absolute/path/to/case-result
```

默认只读取已封存证据。显式 `--replay` 才会使用保存的输入和原镜像，在所选设备上重新执行一次 probe。
它不是自动修复，也不重复原性能计时循环。具体流程见 [诊断与复放](../docs/operations/diagnosis.md)。

- [如何读指标与报告](../docs/operations/results.md)
- [Ascend 适配边界](vendors/ascend/README.md)
- [厂商适配与旧集群入口](../docs/operations/operations-case-doc.md)
