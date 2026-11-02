> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书链接转入 [历史引用索引](history-references.md)，其出处保留。

# 01 v1 现状诊断

> 对象：`FlagPerf_advance/inference`（截至 2026-10-09，提交 `79e8840e` "cendAs Inference V4.1"）
> 上游：[00 需求](00-requirements.md) ｜ 下游：[02 架构](02-architecture.md)、[10 阶段计划 P6](10-roadmap.md)

v1 在"4 个模型、2 家设备、FlagGems/FlagTree/FlagCX 对照"的范围内是严谨可用的：身份校验、partial 语义、不按时间邻近关联等原则都做得很好。问题不在质量，而在**扩展方向**：它的扩展单位是"模型"，而目标需求的扩展单位是"厂商 × 模型 × 算子"。

## 1. 规模数据

| 项 | 数值 |
|---|---|
| `runtime/` | 70 个模块，12,414 行 |
| `benchmarks/` + `vendors/` + `components/` + `inference_engine/` | 约 4,100 行 |
| `ModelAdapter` 基类（`runtime/workload.py`） | 35 个方法 + 约 10 个类属性 |
| 单个模型适配器覆写数 | Qwen3-Embedding 35、OneRec 26、DeepFM 16 |
| 报告/分析模块（`*_reporting.py`、`*_analysis.py`） | 14 个，其中 `reporting.py` 630 行、`tp_reporting.py` 609 行 |
| 顶层 CLI 命令（`runtime/config.py`） | 10 个：accuracy / performance / preview / export / list / report / oracle / performance-baseline / analyze / repack-reference |
| 设计文档 | `implementation_docs/` 19 篇，约 60 万字节 |

## 2. 四个耦合点

### 2.1 `ModelAdapter` 同时承担七种职责

| 职责 | 代表方法 | 是否真的与模型有关 |
|---|---|---|
| 资产校验与指纹 | `validate_assets`、`asset_fingerprints` | 大部分是通用文件检查 |
| 输入编码与封存 | `prepare_inputs`、`validate_batches`、`input_layout` | 按输入**类型**（文本/稀疏/多模态）通用，不按模型 |
| 模型加载 | `load`、`load_cpu`、`load_via_cpu` | HuggingFace 类模型可通用 |
| 执行语义 | `forward`、`open_execution`、`execution_steps` | 按任务类型（单步/生成）通用 |
| **观测语义** | `boundary_specs`、`selected_modules`、`instrumentation` | **真正与模型有关** |
| 参考包契约 | `reference_contract`、`reference_boundaries`、`prepared_specs` | 本应由框架统一 |
| 报告文案 | `report_descriptions`、`metadata` | 本应由框架统一 |

后果：接入一个模型要实现 16–35 个方法，再在 `registry/models/*.json` 中列出全部源文件闭包。按 Qwen3 适配器约 250 行估算，500 个模型约需 10 万行适配代码。

### 2.2 采集与被测对象是同一段代码

`inference_engine/pytorch.py:forward()` 一个函数内同时完成：算子清点（`Inventory`，TorchDispatchMode）、module forward hook 张量捕获、FlagGems 路由包装、scope 栈、有限性检查、证据落盘、batch 循环。

后果：**没有"被测目标"这个概念**。只能观测由本工具自己 `adapter.forward()` 发起的前向；vLLM 服务、用户脚本、没有写适配器的模型都无法观测。

### 2.3 分析绑定运行目录形状，而不是数据模型

14 个报告/分析模块各自按文件名读取 run 目录中的特定产物；同一个"关联"逻辑在 `vendors/ascend/trace.py`、`vendors/nvidia/trace.py` + `trace_index.py`、`vendors/ascend/communication.py` 中各写一遍。

后果：新增一种分析 = 同时修改运行期产物和报告代码；不同厂商的同类分析难以保证口径一致。

### 2.4 CLI 按"分析种类"扩展

每增加一种能力（preview、oracle、performance-baseline、repack-reference……）就增加一个顶层命令和一段 config schema。命令数随能力数线性增长。

## 3. v1 中值得继承的原则

这些不是代码，而是已被实测验证的**规则**，v2 必须保持：

| 原则 | v1 出处 | v2 落点 |
|---|---|---|
| 时间邻近绝不作为关联依据；关联缺失/多义即 partial | `vendors/ascend/trace.py` 文件头、`implementation_7.md` | [04 §1](04-correlation.md) |
| CANN：flow 起止端点或唯一 `connection_id` → 同主机线程 scope 归属 | `vendors/ascend/trace.py:attribute` | [04 §3.3](04-correlation.md)、[11](11-spike-cann-correlation.md) 的基线 |
| CANN HCCL：marker → Enqueue/Dequeue `correlation_id` → `Node@launch` → `connection_id` → 设备事件 | `implementation_3.md §4` | [04 §3.5](04-correlation.md) |
| CUDA：CUPTI `correlation` 唯一关联 launch，必要时 Kineto `External id` | `implementation_7.md` | [04 §3.3](04-correlation.md) |
| 设备 kernel 时间不必落在主机 marker 窗口内 | `implementation_14.md` | [04 §3](04-correlation.md) |
| worker 级身份校验（包版本、环境变量、源码哈希、输入哈希） | `runtime/worker.py:identity` | seal 模式 manifest，见 [03 §6](03-event-schema.md) |
| 监控是独立进程，profiler 冲突区间屏蔽而非删除，缺失不填零 | `implementation_15.md` | counter 探针，见 [05 §2](05-probes-and-vendors.md) |
| 精度只报指标、不设阈值 | `plan.md` | [07 §3.3](07-analyzers-and-reports.md) |
| 适配器 descriptor + 源码闭包校验 | `runtime/adapters.py` | v2 registry，见 [05 §5](05-probes-and-vendors.md) |
| 同条件开关对照确认开销 | `implementation_15.md` | 非功能需求 N1 |

## 4. v1 资产处置

按决策 D3（历史产物不兼容，见 [DECISIONS](DECISIONS.md)），以下只处置**代码与方法**，不迁移 `oracles/`、`performance_baselines/`、`result/` 中的历史数据。

| 处置 | v1 位置 | v2 去向 |
|---|---|---|
| **迁移改造** | `inference_engine/pytorch.py:Inventory` | dispatch 探针（TLS corr 分配、sig 字典化） |
| | `components/flaggems.py:GemsRoute` | route 探针（读 TLS corr，回退原因枚举化） |
| | `runtime/layer_capture.py:Scopes`、`RouteScopes` | module 探针 → span 事件 |
| | `vendors/ascend/trace.py`、`vendors/nvidia/trace.py`、`trace_index.py` | vendor-trace **导入器**：只负责把原始 trace 转为事件，关联逻辑上移到 `correlate/` |
| | `vendors/ascend/communication.py` | comm 关联规则 + 导入器 |
| | `vendors/ascend/telemetry.py`、`vendors/telemetry.py`、`runtime/monitor.py` | counter 探针 + health 心跳 |
| | `runtime/metrics.py` | oracle 分析器的数值指标库（公式保持不变） |
| | `runtime/trace_utils.py:ScopeIndex` | `correlate/` 中的区间索引 |
| | `runtime/worker.py:identity` | seal 模式身份采集 |
| | `runtime/adapters.py` 的 descriptor 与源码闭包校验 | v2 registry |
| **降级** | `benchmarks/*` 与 `ModelAdapter` | Tier-2 插件；只保留执行语义与观测语义两类方法，见 [06 §4](06-model-onboarding.md) |
| | 14 个报告/分析模块 | 拆为分析器 + 报告面板，按需重写 |
| | `runtime/preview*.py`（5 个模块） | route 分析器 + 可选的隔离验证 run |
| **不迁移** | `retention.py`、`report_replay.py`、`repack-reference` 等面向历史产物格式的代码 | 由 store 的保留策略与 manifest 取代 |
| | `legacy/` | 保持原样，不进入 v2 |

v1 在 v2 能够替代之前保持可用（决策 D5），退役条件见 [10 P6](10-roadmap.md)。
