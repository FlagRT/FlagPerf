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

## 从一份配置得到报告

在有 CPU PyTorch 和 PyYAML 的环境运行下面两条命令，即可得到固定输入的 CTR 基准。打开 `cpu-first/report/report.html` 查看每组实际前向次数和延迟。

```bash
flagprof init --template cpu-demo
flagprof run cpu_demo.yaml --out cpu-first
```

单卡模型采集用 `flagprof init` 生成 `flagprof.yaml`，自定义 Python 脚本用 `custom-script`。填写本地路径后可用 `--check` 体检、`--dry-run` 查看命令，执行后用 `bash commands.sh <新目录>` 复跑。[单卡工作流用法](docs/usage.md#单卡工作流)

双卡 Qwen3 采集用 `flagprof init --template tp2`。要复用参考 Store 做精度对照，或比较两侧组件开关，用 `flagprof init --template compare`。采集报告会列出每个 rank 的证据，性能对照会注明指标能否比较。[双卡与对照用法](docs/usage.md#双卡与对照)

比较 FP32 与 BF16 的推理数值时，在 accuracy 工作流中设置 `compare.declare: [dtype]`，原生报告会列出最终输出误差和首个出现数值差异的边界。[跨精度对照 YAML](docs/usage.md#比较-fp32-与-bf16-的逐层数值差异)

排查某一层里的算子时，可用自定义脚本模板指定模块路径。打开两次采集的 Oracle 报告，选择可重放的算子行，复制命令即可在本机运行两侧重放并生成对照报告。[自定义脚本层内采集](docs/usage.md#用-yaml-采集自定义脚本中的一层) · [从算子行重放](docs/usage.md#从-oracle-算子行重放)

排查运行时停滞时，可以让一个节点监控会话关联多次 YAML 任务，再用退出回执、任务报告和离线重放核对同一次告警。[节点监控与 YAML 采集示例](docs/monitoring.md#节点监控yaml-采集与离线重放)

导出固定形状的 Qwen3 embedding 计算图用 `flagprof init --template export`。按名册逐模型做两次 A/A 推理用 `flagprof init --template batch`。分别从图包的 `index.html` 和批量运行的 `battery-summary.md` 阅读结果。[计算图与批量运行](docs/usage.md#计算图与批量运行)

昆仑芯 P800 的当前命令见[使用指南](docs/usage.md#p800-候选运行栈)，既有实测镜像和身份清单字段见 [P800 适配资料](../docs/kunlunxin/inference-v2.md)。[验证记录](../docs/kunlunxin/inference-v2-validation.md) 包含开放缺陷：OneRec 单卡/TP2 第 14 个生成 token 不同（P800-INF-007），尚不能声明 TP2 精度通过；该次验证未授予性能资格。

训练程序复用本采集与 Store 架构，使用 [Training v2](../docs/training/v2.md) 的独立训练协议和报告，不套用推理 TP2 规则。

## 环境与最小入口

离线导入、读取和报告使用 Python 3.9+ 标准库。采集在已经准备好的设备环境内运行：NVIDIA 使用 PyTorch profiler；Ascend 使用匹配的 PyTorch、torch_npu 与 CANN；P800 使用支持 profiler 的 XPyTorch/FlagRT，FlagPerf 在创建目标进程时设置 tracing 变量。内置 Hugging Face runner 另需 Transformers 和本地模型、输入。环境由使用者准备。

从仓库根目录安装，后续命令可以在任意目录执行。

```bash
python -m pip install ./inference
flagprof --help
flagprof record --vendor ascend --store ./results -- python my_inference.py
flagprof import "$CAPTURE" --store ./results
flagprof inspect "$RUN" --verify
flagprof page "$RUN" --limit 10
flagprof report "$RUN" --out ./report
```

昇腾 `final` 采集从本次封存的 `profiler_info.json` 读取 CANN 版本与精简设置，从同次官方导出的 SQLite 数据库读取格式版本。数据库缺失或超出采集预算时，格式版本显示为未知。

工作负载可以在实际选择执行路径的代码分支调用 `flagprof.declare_execution_mode("eager")`。编译执行和图回放分别传 `"compiled"`、`"graph"`。声明随本次采集保存，包含调用位置和代码摘要。同一次采集声明不同模式时记为 `mixed`，没有声明时保留未知。

`CAPTURE` 指已封存的采集，`RUN` 取命令输出中的 run 路径。只读查看可直接使用挂载的结果，报告写入所选输出盘。`python -m flagprof` 使用相同参数。无需设备的完整示例见[快速使用](docs/usage.md#第一次使用先读一个样例)。

## 当前能力

| 能力 | 生产入口与范围 |
|---|---|
| 进程采集 | `record`、`flagprof.attach()`；默认完整会话 `final`，显式 `windowed` 可选 |
| 设备分析 | NVIDIA / Ascend / 平头哥 PPU / 昆仑芯 P800 适配器；设备区间、官方显存、通信和请求窗口分析；具体关联、时钟和完整性能力按厂商证据判定 |
| 主机分析 | dispatch、module、user、route、tensor 和 communication 探针；FlagGems / FlagTree 组件选择、真实编译器与 JIT 证据 |
| 数值诊断 | accuracy 模型／逐层／指定层内算子采集、单 run 或 TP2 组的 `report oracle`、带编译器身份核对的算子 replay |
| 单机双进程 | `launch --nproc-per-node 2`；NCCL、HCCL、FlagCX，逐 rank Store 与 group 报告 |
| 内置运行器 | `python -m flagprof.runner`：本地 Hugging Face embedding / generation / rerank 与 CTR 模板；其他模型可通过自身 Python 程序接入 |
| 离线消费 | `seal`、`analyse`、`import`、`inspect`、`page`、`report`；自动管理检查点和临时索引 |
| 资源监控 | Ascend / NVIDIA / PPU / P800 遥测（后两者经 NVML-ABI 兼容库）、框架显存边界、阶段与资源联合时间线；`watch` 独立会话、滚动保留、归档恢复；换机先跑 `flagprof telemetry check` |
| 在线诊断 | 显式 `--diagnose on`：OOM、心跳、执行停滞、单机 TP2 同步通信超时，保留证据和覆盖 |
| 计算图导出 | `export` 独立 CPU 入口：Qwen3-Embedding 固定形状 FX / ONNX；通用 `ExportSpec` API |
| 结果交付 | 离线 HTML / Markdown、两路或三路 Oracle、runner 性能基准、条件对照、历史报告重绘 |

FlagTree 单卡和 FlagCX TP2 的精度、性能与路由覆盖命令见 [组件对比](docs/components.md)。新入口见 [资源监控与诊断](docs/monitoring.md)、[计算图导出](docs/graph-export.md) 和 [结果报告](docs/user-result-reports.md)。监控与在线诊断默认关闭；新写入的 Store 使用 schema 1.3，公共 Reader 继续读取 1.x 历史数据。

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
| `flagprof/monitor/`、`telemetry/`、`graph/` | 独立监控、采样与在线规则、CPU 图导出 |
| `docs/` | 需求、总体框架、接口契约、阶段计划与决策记录 |
| `tests/` | CPU 冒烟与适配负例；可选 torch/Gloo 检查，PR 时由 GitHub Actions 运行 |

本目录的运行代码可独立使用，不依赖其他仓库。设备上的验收方式见 [设备最小验收](docs/usage.md#设备最小验收)。
