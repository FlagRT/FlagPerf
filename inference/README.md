# Inference：推理观测与诊断

本目录采用 `flagprof` v2 架构。它把探针接入被测 Python 进程，将主机事件与厂商官方 profiler 的设备事件写入 Store，再通过公共读取接口生成性能、通信、显存和数值差分报告。

```mermaid
flowchart LR
  T[被测 Python 程序 / runner] --> P[主机探针与官方 profiler]
  P --> C[封存 capture]
  C --> I[厂商适配与离线导入]
  I --> S[Store 与关联证据]
  S --> V[StoreView]
  V --> A[分析器]
  A --> R[HTML / JSONL 报告]
```

先阅读 [快速使用](docs/usage.md)，参与开发时从 [框架与协作导航](docs/README.md) 开始。旧版配置、命令和结果格式的迁移方式见 [版本迁移](docs/migration.md)。

昆仑芯 P800 的镜像、设备身份和复现步骤见 [P800 Inference v2](../docs/kunlunxin/inference-v2.md)，[验证记录](../docs/kunlunxin/inference-v2-validation.md) 包含开放缺陷：OneRec 单卡/TP2 第 14 个生成 token 不同（P800-INF-007），尚不能声明 TP2 精度通过；性能保持 `not_qualified`。

## 环境与最小入口

离线导入、读取和报告使用 Python 3.9+ 标准库。采集在已经准备好的设备环境内运行：NVIDIA 使用 PyTorch profiler；Ascend 使用匹配的 PyTorch、torch_npu 与 CANN；P800 使用指南锁定的 XPyTorch/FlagRT 环境及前置 tracing 变量。内置 Hugging Face runner 另需 Transformers 和本地模型、输入。环境由使用者准备。

在仓库根目录执行：

```bash
python3 -B -S inference/run.py --help
python3 -B -S inference/run.py record --help
```

也可进入模块目录使用统一模块入口：

```bash
cd inference
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -B -S -m flagprof --help
python3 -B -S -m flagprof.runner --help
```

`run.py` 与 `python -m flagprof` 使用同一个 CLI。正式采集命令不加 `-S`，使设备环境中的 PyTorch 和厂商 SDK 可导入。需要从任意目录使用时可安装此模块：`python3 -m pip install --no-deps -e ./inference`（在仓库根目录执行）。

## 当前能力

| 能力 | 生产入口与范围 |
|---|---|
| 进程采集 | `record`、`flagprof.attach()`；默认完整会话 `final`，显式 `windowed` 可选 |
| 设备分析 | NVIDIA / Ascend / P800 适配器；设备区间、官方显存、通信和请求窗口分析；具体关联、时钟和完整性能力按厂商证据判定 |
| 主机分析 | dispatch、module、user、route、tensor 和 communication 探针；FlagGems / FlagTree 组件选择、真实编译器与 JIT 证据 |
| 数值诊断 | accuracy 模型／逐层／指定层内算子采集、单 run 或 TP2 组的 `report oracle`、带编译器身份核对的算子 replay |
| 单机双进程 | `launch --nproc-per-node 2`；NCCL、HCCL、FlagCX，逐 rank Store 与 group 报告 |
| 内置运行器 | `python -m flagprof.runner`：本地 Hugging Face embedding / generation；其他模型可通过自身 Python 程序接入 |
| 离线消费 | `seal`、`analyse`、`import`、`inspect`、`page`、`report`；Store、work、scratch 分离 |

FlagTree 单卡和 FlagCX TP2 的精度、性能与路由覆盖命令见 [组件对比](docs/components.md)。`launch` 支持与 `record` 相同的精度边界参数，以及 `coverage` 预设；当前公开版本不包含 monitor/watch。

P0/P1/P2/P4 的历史退出以 [决策记录](docs/DECISIONS.md) 中登记的范围为准。多机、8 卡扩展、在线故障恢复与热加载仍按 [路线图](docs/10-roadmap.md) 推进。关联、时钟或输入身份不满足时，报告保留 `partial`、`unknown`、`unlinked` 或 `not_comparable` 及具体原因。

## 代码地图

| 目录 / 文件 | 职责 |
|---|---|
| `flagprof/cli.py`、`record.py`、`runtime.py`、`launch.py` | 命令、进程接入与生命周期 |
| `flagprof/probes/`、`components/`、`runner/` | 探针、可选组件、模型运行器 |
| `flagprof/vendors/` | SDK 会话、格式契约、厂商分类和显式关联规则 |
| `flagprof/offline/`、`correlate/` | 封存导入、可恢复检查点、关联与派生证据 |
| `flagprof/store/` | schema、Writer、Reader、只读 StoreView 与完整性校验 |
| `flagprof/analyzers/`、`report/` | 有界分析、数值差分及离线报告 |
| `docs/` | 需求、总体框架、接口契约、阶段计划与决策记录 |
| `tests/` | CPU 冒烟与适配负例；可选 torch/Gloo 检查，PR 时由 GitHub Actions 运行 |

本目录的运行代码可独立使用，不依赖其他仓库。设备上的验收方式见 [设备最小验收](docs/usage.md#设备最小验收)。
