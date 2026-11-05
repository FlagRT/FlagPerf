> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书出处见删除前固定 Git 提交中的 [历史引用索引](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md)，其出处保留。

# 05 探针与厂商适配

> **当前采集默认（2026-10-26，D33）**：CLI/Python record/attach 使用 final 官方 start→正常 stop/export；handle.step() 仅兼容、不驱动 SDK。显式 windowed 保留原窗口采集。所有者退出后由 worker 分析/导入，裸 attach 的 atexit 保持 unknown/partial。部署、恢复及已验收边界见[默认 B 使用说明](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-e854eaa4d4fc)。

> 上游：[02 架构](02-architecture.md) L1、[03 schema](03-event-schema.md)、[D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合)–[D24](DECISIONS.md#d24--导入契约身份分层来源账本完成标记证据索引) ｜ 下游：[06 模型接入](06-model-onboarding.md)、[07 分析器](07-analyzers-and-reports.md)

## P800 当前实现补充（2026-11-03）

本节更新当前入口，以下历史表格保留其原登记日期。P800 使用独立
`kunlunxin` registry、格式契约和硬件身份；共享已验证的 Kineto 字段解释代码，
不继承 NVIDIA 资格。运行条件、能力表、模型实测与限制统一见
[P800 使用与验收](../../docs/kunlunxin/inference-v2.md)。

采集前由使用者显式设置 `XPU_ENABLE_PROFILER_TRACING=1` 和所选逻辑设备的
`XPU_CUPTI_ENABLE_DEVICE`；launcher 为各 worker 设置对应 rank。它们是诊断采集前提，
不调整被测算子、队列策略或模型。windowed 会话仅批准这两项精确条件。
无采集基线和主机探针不据此声明设备轨完整。SDK 未提供的丢失计数仍是 unknown。

通信探针在原 imperative 接口外，观察 PyTorch
`_functional_collectives.all_reduce/all_gather_tensor`；通过精确注册的 group name
区分模型组与控制组，保持原调用结果、序列和逻辑字节数，卸载恢复原入口。
功能性通信的主机返回不证明设备完成；后台线程 launch 缺少与调用线程标记的因果键时，
设备关联保留 unavailable。独立实测确认的有限 `kl3_all_reduce<float|bfloat16>`
kernel 名称族只用于设备类别，不生成调用归属。

`flagprof.scope()` 是当前已实现的可选用户区间入口；未 attach 或探针未启用时不采集。
当前五预设为 perf/modules/coverage/perf-lite/accuracy；counter/monitor、在线恢复等
路线图功能不因本适配而变成已实现。

## 1. 探针的定义与硬约束

探针 = 一个独立的采集单元，负责安装/卸载某种 hook 并产出若干类事件。

硬约束（违反即视为设计错误）：

1. **不得 import 任何模型或 benchmark 代码**（[02](02-architecture.md) P2）。
2. **不得 import 其他探针**。共享状态只走 `flagprof/context.py` 的 TLS 接口。
   > **当前实现（T9，2026-10-18）**：`Context` 是**显式对象**（`Context(writer)`，随 `attach()` 创建，由句柄持有），不是模块级全局；栈基于 `threading.local`，并按 asyncio Task 各自独立；**fork 后失效**（子进程里继承的 Context 拒绝分配 corr，须在子进程重新 attach）。`push_scope(name)` 返回 token，`pop_scope(token)` 必须按 **LIFO** 弹出自己的 token，否则抛 `ValueError`。模块级便捷函数（`alloc_corr`/`push_scope`/`pop_scope`/`current_scope`/`current_corr`）经当前 attachment 转发到它。
3. **必须可单独启停**，且卸载后进程状态可恢复（hook 摘干净）。
4. **必须声明 descriptor**（§5），包含产出事件类型、开销等级、依赖、数据来源 API。
5. **写事件必须非阻塞**（[03 §6](03-event-schema.md)）。
6. 安装失败只降级不中断：写一条 `health(sev=warn)` 并把自己标为未启用，run 继续。
   > **当前实现约定（2026-10-27，[D46](DECISIONS.md#d46) 第 12 条）**：`health.kind` 的枚举里没有表示“探针降级”的值，P1 不扩充枚举；降级记入 attachment `state.json` 的 `capabilities.unavailable`（带 `reason`）与 `errors`，经 `inspect` 显示。枚举的扩充留到 P3。

## 2. 探针目录

| 探针 | 机制 | 产出 | 开销等级 | 阶段 |
|---|---|---|---|---|
| `dispatch` | `TorchDispatchMode`（`summary` 等级下用 C++ `RecordFunction` 观察者，见 §2.1） | `op`/`opsum`、`sig` | **高**（`each`）/ 中（`summary`） | P1 |
| `module` | `nn.Module` 的 `register_forward_pre_hook` / `register_forward_hook` | `span(cat=module)` | 低 | P1 |
| `route` | 包装 FlagGems 的注册表/分发入口（继承 v1 `components/flaggems.py:GemsRoute`） | `route` | 低 | P1 |
| `compile` | Triton/FlagTree 的编译回调或缓存层包装 | `compile` | 低 | P1 |
| `comm` | 包装 `torch.distributed` 的集合通信入口 + 厂商侧 enqueue 记录 | `comm` | 低 | P2 |
| `tensor` | 在 module hook / op 边界按白名单抓取张量，算 stats（可选落 blob） | `tensor` | **高** | P4 |
| `counter` | **独立旁路进程**，周期轮询厂商遥测、框架显存统计、主机资源 | `counter` | 低（不在目标进程内） | P2 |
| `health` | 目标进程内的心跳线程 + 异常钩子（`sys.excepthook`、OOM 捕获、信号处理） | `health` | 极低 | P3 |
| `vendor.<v>.session` | **官方 profiler 会话**（复用，不改源码）：单一会话所有者、按窗口采集与分段导出、产物有效性判定（[D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合)） | 原始 trace（不写事件） | 由官方 profiler 决定，按配置实测登记 | P0（T8） |
| `vendor.<v>.trace` | **离线导入器**：先枚举来源账本，再把官方 profiler 的原始产物转成事件，同时写出原生关系等派生文件（[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)、[D24](DECISIONS.md#d24--导入契约身份分层来源账本完成标记证据索引)） | `kernel`、`memop`、`comm(phase=device)`；派生文件 | — | P0 起（T7、T8），通信等补全在 P2 |
| `user` | 公开 API `flagprof.scope("name")`，供用户/插件手工标注（**设计目标（未实现）**：当前没有这个顶层函数，只有 `Context.scope(name)` 上下文管理器，见 §1 第 2 条） | `span(cat=user)` | 极低 | P1 |

### 2.1 dispatch 探针的两个等级

| 等级 | 实现 | 产出 | 适用 |
|---|---|---|---|
| `each` | `TorchDispatchMode` | 每次调用一条 `op` | 单步前向、逐层分析、精度差分 |
| `summary` | 聚合计数器（见下） | 周期性 `opsum` | 自回归生成、长稳运行、vLLM 服务 |

`summary` 等级下的实现选择（按优先级）：
1. C++ `at::addGlobalCallback` / `RecordFunction` 观察者 —— 开销远低于 Python 层 mode；
2. 若不可用，仍用 `TorchDispatchMode` 但只累加不写事件。

> **开销必须实测**（N1）：P1 的验收包含"同条件下探针 off/on 的端到端耗时对照"，结论写入报告。v1 `implementation_15.md` 已有这个做法，继承。

### 2.2 counter 探针为什么是独立进程

继承 v1 `runtime/monitor.py` 的结论：

- 在目标进程内轮询 npu-smi/DCGM 会被 GIL 和设备锁干扰，且进程崩溃后采样一并丢失；
- 厂商 profiler 开启期间，部分遥测 API 与 profiler 互斥 → 这段区间的采样标 `blocked=true`，**屏蔽而非删除，缺失不填零**；
- 独立进程是同一 store 中**自己的 writer**（`kind=counter`，有自己的 `writer_id`，[03 §3](03-event-schema.md#3-corr关联-id)）。`rank` 字段沿用被监控进程的 rank，只作拓扑属性；`p=counter`。

### 2.3 tensor 探针的扰动

抓张量会同步设备、分配主机内存，必然破坏计时。因此：

- 预设层面与性能探针**默认互斥**（§3）；
- 若用户强行同时开启，报告的 `perturbation.host` 与 `perturbation.device` 都标 `present`（[07 §2.1](07-analyzers-and-reports.md)），性能结论降级为"仅供参考"；
- 抓取白名单由 Tier-1 profile 或 Tier-2 插件给出（[06](06-model-onboarding.md)），没有白名单时 tensor 探针不抓任何东西（不做"全抓"默认值）。

> **2026-10-28 按 [D50](DECISIONS.md#d50) 第 5 条**：P4 首批尚无 profile YAML，`accuracy` 预设的默认边界按结构规则给出：输入嵌入模块、最大同类型 `ModuleList` 的每个元素、runner 的 `pooled` 与 `output`。这仍是有限的边界集合，不是全抓；`--tensor-boundaries <json>` 可替换默认集合。

> **2026-11-02 用户批准的边界选择入口**：`--tensor-boundaries model` 关闭模块边界张量、保留 runner 提交的模型终点；`layer` 与省略参数一致，沿用默认边界。JSON 路径列表保持兼容。`--tensor-ops-in` 独立增加指定模块内部算子采集；`layer` 加下钻时保留其他层边界，`model` 加下钻时其他层不采张量。完整值仍由 `--tensor-blob` 控制。CLI 与 Python API 共用配置归一化，使用示例见 [P1 accuracy](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-a7ee45133ae2)。

## 3. 预设（preset）

用户不直接勾选探针，而是选预设。预设是 registry 中的 JSON，可被用户覆盖。

> **2026-10-27 按 [D46](DECISIONS.md#d46) 第 1 条修订**（P1 首批；当前实现待 [T10.1](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-29067dfd8983) 交付，之前以下方 T9 的“当前实现”为准）。修订前的表原文保留在本节末尾。

| preset | 官方 profiler | dispatch | module | route | compile | 目的 |
|---|---|---|---|---|---|---|
| `perf`（默认） | final | 关；`--dispatch each-window` 开有界 each 窗口并写桥接 marker | 开 | FlagGems on 且 dispatch 开启时观察；否则见下文 | 开（hook 可用时） | 单次诊断采集（R53/R31）。算子明细取自官方 profiler 的 cpu_op（D22 第 5 条）。FlagGems off 下 `--probes none` 对应 P0 档 |
| `perf-lite` | 无 | 关（2026-10-28 [D49](DECISIONS.md#d49) 第 2 条 c；原为 summary） | 开 | unavailable（dispatch 关）；回退清单由 `coverage` 产出 | 开 | 低开销、长时间运行；层耗时头条的来源 |
| `coverage` | 无 | 全程 summary + 有界 each 窗口 | 开 | FlagGems on 时开 | 开 | 算子清单与回退清单扫描（R1/R47） |
| `modules` | final（`with_stack`/`with_modules`） | 关 | 开 | 同 perf | 同 perf | 兼容 T9 档 |
| `accuracy` | 无 | 关；`--tensor-ops-in` 时 tensor 探针在指定 scope 内拦截深度 0 调用 | 开，`layers=all` | 不观察调用；FlagGems on 时写 `route_table` | 开 | Oracle–Device 差分（R52），另开 tensor 探针（2026-10-28 [D50](DECISIONS.md#d50) 第 4 条，实现见 T11.1） |
| `monitor` | — | — | — | — | — | **设计目标（未实现，P3）** |

comm、counter 属 P2，health 探针属 P3，P1 首批的预设不含这三者。FlagGems 是组件开关（`--flaggems off|on`，`on` 时必须给 include 列表），默认 `off`。

**2026-10-27 按 [D48](DECISIONS.md#d48) 补充依赖**：FlagGems 组件与 route 观察者分别管理。默认 perf/modules 的 dispatch 关闭时，组件仍按配置启用，route 为 `unavailable/dispatch_disabled`；打开 each-window 后才具备 route 调用判定所需的帧。`--probes none` 关闭项目探针，组件开关独立生效。根模型登记、惰性发现与首轮窗口由 [T10.1 §3.2–§3.4](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-e14e5516d9b9) 规定。

> **当前实现（T9，2026-10-18）**：`record --preset` 只有 **`perf`、`modules`** 两个预设（`flagprof/runtime.py` 的 `configuration()` 解析，其他名字报 `unsupported preset`）；它们不是上表意义的"探针组合"，只决定向官方 profiler 请求哪些选项。
>
> - **`perf`**：当前 = 官方 profiler 会话 + writer（含显式 `Context`）。module、dispatch、route、compile 探针**不可用**——句柄的 `capabilities.unavailable` 列出 `module_probe`、`dispatch_probe`、`route_probe`、`compile_probe`、`eager_module_tree`（`runtime.py`）；上表 `perf` 行中的 module、route、compile、comm、counter、health 全部是设计目标，没有对应的在线生产者。
> - **`modules`**：只向官方 profiler **请求选项**（`with_stack=True`、`with_modules=True`），模块层信息是否出现取决于官方 profiler 自己的输出；**不提供自动探针，也不提供 eager 模块树**。
> - **`perf-lite`、`accuracy`、`coverage`、`monitor`**：**设计目标（未实现）**，所含探针的阶段见 §2。

> **2026-10-14（[D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合)）**：2026-10 的跨厂商实验表明，不加逐调用项目探针时，官方 profiler 的 cpu_op 与原生 id 已能回答"哪个调用产生了哪个 kernel、发生在哪个模块"；而全量 `dispatch(each)` 的配对增量为 +34% 到 +263%。因此 `perf` 不再默认开启 `dispatch(each)`。另外：
> - 官方 profiler 本身也有开销（同步前向 +36% 到 +116%），**性能基准 run 与诊断采集要分开**；
> - `perf-lite`、`coverage` 等预设的调整随 P1 一并修订。

**精度与性能分开采集**是 v1 的既有结论（[00 §4](00-requirements.md)），这里体现为 `perf` 与 `accuracy` 两个预设，而不是在代码里写死互斥逻辑。

分开的理由要说准：精度探针（tensor 取数、逐 op 同步）**同时扰动主机与设备两侧** —— 取张量要同步设备、分配主机内存，会改变 kernel 的排队与并发；性能探针的标记与计时则主要拉长主机侧时间。两者叠加时，任何一侧的计时都不再代表被测系统。

至于「标记是否扰动设备时长」，[Spike-C](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/11-spike-cann-correlation.md) 只给出一组有限的证据：17 组 marker/plain 对照中，单 kernel 设备时长之比为 0.926–1.034。它说明**在这些配置下**标记对设备时长的影响落在这个范围内，不说明「设备时长天然不受扰动」，也不覆盖 tensor 探针、通信等待、并发调度或以后的 workload。因此：

- **预设能报告哪些指标，由它实际启用的来源决定。** `accuracy` 不开 vendor profiler，所以**没有** kernel/memop/comm 设备事件，设备性能指标一律 `unavailable`；它的产出是张量、路由与主机侧的调用结构。
- 任何 run 的设备时长都按「**该采集配置下观察到的值**」报告，并附上采集配置。扰动是否受控由 [07 §2.1](07-analyzers-and-reports.md) 的 `perturbation` 单独标注：没有对应的对照 run 时标 `not_evaluated`，**不自动视为无扰动**。
- 需要主机侧性能结论时，要求 `perf` 与一次同配置的无标记基线配对（口径见 07 §2.1）。
- 用户显式让 `accuracy` 同时开启 vendor trace 时，设备事件可以产出，但 `perturbation.device` 标 `present`（tensor 探针在场），性能结论降级为"仅供参考"。
- 预设不负责"是否配对"的调度，这是 [09](09-cli-and-campaign.md) 的 campaign 职责。

**修订前的预设表（原文，至 2026-10-27）**：

| preset | 启用 | 目的 |
|---|---|---|
| `perf` | 官方 profiler 会话 + vendor trace（主机侧算子记录取自 profiler 自带的 cpu_op）、module、route、compile、comm、counter、health；`dispatch(each)` **按需**（[D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合) 第 5 条；本表的完整修订随 P1） | 单步前向的性能画像（R53/R31）。**启用项为设计目标，当前实现见下** |
| `perf-lite` | dispatch(`summary`)、module、route、comm、counter、health | 生成式/长稳/服务观测，低开销。**设计目标（未实现）** |
| `accuracy` | module、tensor、route、health（**不开** vendor profiler，dispatch 为 `each` 但不计时） | Oracle–Device 差分（R52）。**设计目标（未实现，tensor 探针属 P4）** |
| `coverage` | dispatch(`each`)、route、compile、health | 只要算子清单与回退清单，不要设备 trace —— 最快的 500 模型扫描（R1/R47）。**设计目标（未实现）** |
| `monitor` | counter、health | 只监控，不碰目标进程（R77–79）。**设计目标（未实现）** |

## 4. 厂商契约

新增一家厂商 = 实现 `VendorAdapter` 接口 + 填一份能力矩阵 + 提供真实 trace 测试样本。**不允许**改动 schema、关联框架或分析器。

```python
class VendorAdapter(Protocol):
    name: str                      # "ascend" | "nvidia" | ...

    # --- 环境与身份（三层身份分别给出，附录 A #27）---
    def available(self) -> bool: ...
    def identity(self) -> dict: ...                 # 硬件身份：设备型号、驱动、固件、SDK 版本（须有来源，否则 unknown）→ meta.env
    def collector_identity(self, cfg) -> dict: ...  # 采集器身份：官方 profiler 名称、版本、实际配置
    def required_env(self, preset: str) -> dict: ...# 必须为空：不得改变被测系统配置（D13）
    def collector_prerequisites(self) -> list[dict]: ...  # 采集前置条件：变量、理由、作用范围（D22 第 4 条；主线两家预期为空）
    def clock_sync(self) -> dict: ...               # 主机↔设备时钟对齐点

    # --- 采集会话（D22：官方 profiler 原样复用，单一会话所有者）---
    def profiler_session(self, cfg) -> ContextManager: ...   # 启停官方 profiler；按窗口导出；每窗口判定产物有效性
    def telemetry_poll(self) -> dict: ...                     # counter 探针调用，返回 values
    def telemetry_blocked_during_profile(self) -> bool: ...

    # --- 离线导入与关联（D23、D24、D25）---
    def device_taxonomy(self, version=None) -> SourceFormat: ...  # 厂商格式解释，见下（D25、D27）
    def import_trace(self, raw_dir, writer) -> None: ...      # 来源账本 → 规范化 → kernel/memop/comm 事件 + 派生文件（解析身份入登记）
    def link_rules(self) -> list[LinkRule]: ...               # H3/H4 的项目归属规则，带 via id；全部求值取并集（04 §3.6）
    def native_rules(self) -> list[NativeRule]: ...           # 原生归属的主机侧规则（04 §3.7）；设备侧复用 link_rules 中 side=device 的部分
    def comm_identity(self, raw) -> dict: ...                 # 还原 (group, group_seq)

    # --- 能力自述 ---
    def capabilities(self) -> dict: ...


class SourceFormat(Protocol):      # 厂商格式解释，由 vendors/<v>/format.py 提供，注入公共层（T8.1 交付）
    version: str
    def identity(self) -> dict: ...          # {version, sha256}，写进解析身份；指纹覆盖下面全部钩子（D27 第 1 条）
                                             # T8.2 实现为格式模块的源码闭包哈希：注释改动也会改变指纹；
                                             # 指纹变化不必然是语义变化，语义变化必须在指纹中可见
    def context(self, events) -> Any: ...    # 一次性构造的格式上下文（如 pid → 进程名）
    def is_device(self, event, ctx) -> bool: ...     # 是否设备执行记录：决定 R（附录 A #12）
    def category(self, event, ctx) -> str: ...       # compute / comm / memcpy / unknown
    def normalizable(self, event, ctx) -> bool: ...  # 能否转成规范事件：决定 E
    def task_type(self, event) -> Any: ...
    def host_pid(self, event, ctx) -> Any: ...       # 主机进程身份还原，须有格式依据（D27 第 2 条）
    def is_launch(self, event) -> bool: ...
```

示意接口与生产接口的对应：上面的 `VendorAdapter` 是职责划分示意；生产代码中，适配器由 `vendors.format.load_adapter(vendor)` 发现，当前入口面是 `vendors/<v>/importer.py` 上的 `device_taxonomy / default_registry / field_context / device_fields / build_rules`。设备判定、类别、能否规范化是三个独立判断：设备轨上类型未知的记录仍以 `unknown` 进入 R，已知类别但缺字段的记录为 `unrepresentable`，都不会从分母中消失。

### 4.1 能力矩阵

每家厂商自报能力，分析器据此决定能出什么结论、报告据此说明限制。

| 能力键 | 取值 | 腾昇（v1 已知） | NVIDIA（v1 已知） |
|---|---|---|---|
| `kernel_trace` | yes/no | yes | yes |
| `host_device_link` | `per_call` / `per_kernel_name` / `interval_only` | `per_call`（queue=1 经 Enqueue/Dequeue 桥，Spike-C 已验证） | `per_call` |
| `native_owner` | `per_call` / `scope_only` / `none` —— 不加项目探针时，官方 profiler 能否把设备记录归到原生调用（04 §3.7） | 历史格式（2026-10 实验，eager）：`per_call`。**当前格式**（torch_npu 2.10.0 / CANN 9.0.0，**Level0 + 数据精简**，格式 **v3**，T8 新采集三窗）：**`per_call`**，每窗 10,120 条全部 owned，30,360 条经独立原始链路证明（[T8.2](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-3d910d3fb495)，2026-10-15 Gate 通过）。同一采集在格式 v2 下为 `none`（每窗 10,120 unowned / `no_bridge`，主机进程编号未还原），v2 可显式重放 | 历史格式（2026-10 实验，eager）：`per_call`。**当前格式**（torch.profiler 2.8.0+cu126，分类表 v1，T8 新采集两窗）：`per_call`，每窗 9,265 条全部 owned |
| `session_conflict_detect` | `detectable` / `not_detectable` —— 能否检测用户已自行开启的 profiler（D22 第 2 条） | `not_detectable`（T8 实测：当前 torch_npu 没有可靠的活动会话查询接口；不以启动重叠 profiler 的方式探测） | `detectable`（T8 实测；经私有符号 `torch.autograd.profiler._is_profiler_enabled`，已在 capture 清单登记） |
| `memop_trace` | yes/no | yes | yes |
| `comm_device_trace` | yes/no | yes（HCCL） | yes（NCCL） |
| `kernel_metrics` | 列表 | 待定 | 占用率等 |
| `telemetry` | 指标列表 | npu-smi：显存/利用率/功耗/温度 | DCGM/nvml |
| `telemetry_during_profile` | yes/blocked | **blocked**（v1 已知冲突） | yes |
| `graph_mode_visibility` | none/partial/full | none | none |
| `required_env` | dict | **空**（见下） | **空** |
| `collector_prerequisites` | list | **空**（T8 实测） | **空**（T8 实测） |
| `device_taxonomy` | 版本号 + 覆盖状态 —— 当前采集格式能否被分类表识别（D25） | 代码默认 **v4**（T8.3，2026-10-16；分类与 v3 相同，另含格式契约与 unknown 入选；返工轮设计者 Gate 通过，2026-10-16）；v3 是 T8.2 通过 Gate 的版本（分类表与 v2 相同，另含主机身份还原）；标签与内容修订的关系见下方"版本标签与内容修订"；v2 起当前格式已覆盖（[T8.1](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-5fcd4691137d)，2026-10-15 Gate 通过）：T8 新采集每窗 compute R = E = 9,980、memcpy R = E = 140、unknown 0；`KERNEL_AIVEC/AICORE/MIX_AIV/AICPU` 的映射依据见 [taxonomy-ascend-v2](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-44bc5a296b18)。v1 可显式重放，历史输出逐字节不变 | 代码默认 **v2**（T8.3；分类与 v1 相同，另含格式契约与 unknown 入选；返工轮 Gate 通过，见下方"格式契约"）；**v1** 当前格式已覆盖：T8 新采集每窗 compute R = E = 9,260、memcpy R = E = 5 |

**格式契约**（2026-10-15，[D28](DECISIONS.md#d28--格式适用范围显式化采集配置入身份导入判定适用性静默失败转显式) 第 2 条）：指纹只回答"用了哪套解释"。每个格式版本另带一份契约，放在 `vendors/<v>/` 旁，写明适用条件（含采集配置）、依赖的厂商标签及失效类型、每项解释的依据等级；导入时据此给出"契约内 / 契约外 / 无法判定"。契约外照常导入，不被当作已验证。

> **现状（2026-10-16，T8.3 返工轮 Gate 通过，附遗留 L1–L6 转 T9 §0）**：T8.3 已实现六份契约（腾昇 v1–v4、NVIDIA v1–v2）、三值判定与只读入口 `StoreView.format_assessment()`。首轮 Gate 未通过时的四个阻塞项（[首轮审核](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-ed6a5f5f7031) B1–B4：NVIDIA 设备执行轨被判为 host；unknown 入选不看轨道角色；设备轨整条缺失仍判契约内；契约约一半字段未被执行）**已在返工轮修复**，75 个来源路径（67 份唯一原件）上的修复经独立复核确认（[复审](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-81d74b957ee6) §0 R1–R8，[返工 Gate 输入](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-7e2e7910cbbf)）。返工遵循 [D29](DECISIONS.md#d29--判定一致性结构证据优先于名称契约条目须被执行或被测试)：结构证据优先于名称、冲突显式；每份契约至少一条必需结构条件（存在有完整事件的设备执行轨）；契约条目须被执行或被测试绑定。T8 当前五窗仍为"无法判定"（腾昇每窗 10 项、A100 每窗 7 项必需条件缺证据），主要是证据没有喂进判定链路，不是格式判不了（复审 §2；[D30](DECISIONS.md#d30--契约条件的证据先查产物自述每个必需条件登记生产者)，已确认（用户 2026-10-16））。
>
> **T9 关闭结果（2026-10-18；上文"每窗缺 10/7 项"保留为 10-16 的历史口径，不再代表现状）**：L1–L6 由 T9.0 关闭（证据：[T9 退出矩阵](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-e0872549b0c8)、[T9.3 执行记录](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-61e29505d51c)）。**旧五窗**（T8 采集）仍判"无法判定"，缺口已收窄为缺少有来源的 `execution.mode`，NVIDIA 另缺独立的 `exporter.version`——二者保留 required，无导出来源就不得为判 `within` 而下调要求（D30 补充，设计者裁决 2026-10-18）。T9.3 新采集：腾昇窗 `within`（`execution.mode` 取自 workload 源码声明的 eager 来源；通用 `record` 没有这类来源，不得套用）；A100 窗 `indeterminate`，缺 `exporter.version` 精确解释了这一点。
>
> **连接键的语义域**（T8.3 C7）：腾昇 queue 键只接受官方 `correlation_id` 字段，键为（`torch_npu.queue` 域，trace，主机进程，值）。字段名含 `correlation`、或数值相同，都不足以连接。修复适用于全部格式版本，旧版本选择不恢复不安全的连接路径。

**返工后的读法**（2026-10-16，[T8.3 返工复审](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-81d74b957ee6) §3、§4）：

1. **轨道角色与活动记录分层。** 轨道角色（设备执行轨 / 主机 / 元数据 / 未识别）与"轨道上的每条记录是否设备活动"分开判断。主机事件类别、设备活动判据、设备轨上允许的非设备类别分开声明：例如 Kineto 把 GPU 侧的 `ProfilerStep#N` 等用户注释写在设备轨上（`cat=gpu_user_annotation`），契约登记为"设备轨上允许的非设备类别"，不进 R/E、不算主机冲突、单独计数。设备轨上出现主机类完整事件、或主机轨上出现 kernel，按冲突告警并判契约外。角色索引由原件与契约构建，不由已导入的 E 倒推。
2. **腾昇的结构见证影响角色，不影响 R/E。** 腾昇的设备判据仍是已登记的进程名依据；`Task Type` 与 `Physic Stream Id` 的结构见证用于冲突审计：`witnessed_conflict` 把该轨角色写回为 device，所以它进角色索引，不进 R/E。设备进程被改名为 Python 时，该轨 role=device，必需结构条件 pass，status 由 `name_structure_conflict` 判 outside。行为正确，但读者须知道角色与 R/E 的依据不同源。
3. **`process_labels`、`process_name` 不作设备存在证据。** 本机实测 Kineto 给 GPU 0..15 全部发出 `process_labels`，实际只有 GPU 0 有活动。进程元数据只能说明"声明了"，设备存在只由设备执行事件与轨道结构证明。
4. **两家支持范围不对称。** `device_count` 数的是不同编号的个数，不是编号值。NVIDIA 契约有 `scope.device_ordinals`，单个编号 1 判 outside；腾昇契约（v4）没有这一条，单个 NPU 编号 1 判 within（5 份唯一腾昇来源的编号为 1）。同一设备有 2 条设备执行轨时仍是 count=1、within；Kineto `pid ≠ args.device` 时生产静默判 within，第二信号目前只在测试侧。补法已裁决（复审 §3.4）：第二信号做成支持范围条件、不做角色判据；腾昇补 `device_ordinals` 或登记不对称理由；统计设备执行轨条数与每个编号的轨数。转 T9.0 §0.4。
5. **版本标签与内容修订。** 版本标签（腾昇 v1–v4、NVIDIA v1–v2）是厂商格式代际，不是内容修订号：返工中标签不变而内容大改（例如腾昇 v4 的格式身份 sha `88ede396…` → `f6798be2…`；磁盘上旧 store 与新 run 的 sha 确实不同），目前只有解析身份的 sha 能区分修订前后，没有 contract revision 字段。解析身份也只覆盖 `SourceFormat` 协议的闭包，账本处置、规则与共享判定代码不在任何代码哈希身份内（[D27](DECISIONS.md#d27--格式解释的身份覆盖全部钩子兼容验收到连接键) 更正注记，复审 §3.5）。是否加内容修订号已在 T9.0 任务单裁决：每份契约加整数 `revision`，并以 `revisions.lock.json` 钉住（[T9.0 §4.5](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-1fb43a35144d)）。

`host_device_link` 的取值直接映射到 [04 §6](04-correlation.md) 的降级级别。

**能力取值绑定什么**（2026-10-15，[D25](DECISIONS.md#d25--冻结语义与输出不冻结厂商分类表)）：表中每个取值只对一组具体条件成立，即（硬件，采集器及版本，分类表版本，采集配置，样本范围）。它不是厂商的永久属性。T8.1 给出了实例：同一台腾昇机器上，历史格式的 `native_owner` 是 `per_call`，当前格式在分类补齐之后仍是 `none`。T8.2 进一步表明，这次差异的成因是**采集配置**（会话未声明 profiler level，默认 Level0 裁掉了 `CANN` 记录），而不是 SDK 版本（[D28](DECISIONS.md#d28--格式适用范围显式化采集配置入身份导入判定适用性静默失败转显式) 第 1 条）。所以本表按条件分行填写，能力不跨采集条件继承；根因分析先比较配置，再归因于版本。registry 登记能力时，须同时登记这组条件。

> **`required_env` 必须为空 —— 工具不得为提高关联率而改变被测系统的运行配置。** v1 强制 `TASK_QUEUE_ENABLE=0`，代价是测的不是用户生产环境的真实形态。[Spike-C](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/11-spike-cann-correlation.md) 已证明 `TASK_QUEUE_ENABLE=1` 下经 `Enqueue → Dequeue` 显式桥仍可逐调用关联（8,016 个 kernel 经此路径关联，无歧义），该强制前提**取消**。`TASK_QUEUE_ENABLE` 与 `TRITON_ENABLE_TASKQUEUE` 是两个独立开关，其**实际取值**必须原样记入 `meta.env`，不得由工具设定。若某厂商某配置确实无法关联，正确做法是按 §4.1 降级并如实声明，不是改配置。

### 4.2 接入一家新厂商的验收清单

1. `identity()` 能唯一标识设备与软件栈；
2. `import_trace` 在提供的样本上产出的事件通过 schema 校验；
3. `link_rules` 在样本上的 `linked_time_ratio` 有明确数值（哪怕很低），且无任何基于时间邻近的规则；
4. `capabilities()` 与实测一致（CI 中有断言）；
5. 至少一个模型的 `perf` 预设端到端跑通并出报告；
6. 开销对照数据（N1）；
7. **原生关系**（04 §3.7）在样本上有明确的已归属、歧义、未归属数值，04 §3.7 的反例行为全部成立，`kernel.link` 不因原生关系而改变；
8. **来源账本守恒**：每条来源记录恰有一个处置，设备执行记录不处置为 `source_only`，单条无法表达的记录不导致整份 trace 失败（[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #26）；
9. **身份**：在 store 中只读登记表就能得到真实硬件、采集器与解析身份，不依赖外层报告补救（#27）；
10. **当前采集器 → 当前导入器贯通**（[D25](DECISIONS.md#d25--冻结语义与输出不冻结厂商分类表) 第 5 条）：用该厂商**当前版本**官方 profiler 的新采集产物走一遍当前导入器，compute 的 R 与 E 大于 0，原生关系有明确数值。只验证"采集产物有效"或只用历史样本验证导入，都不算通过。每次采集器或 SDK 版本变化都要重做，并把抽取的小片段加入 CI。**逐跳解释**（[D27](DECISIONS.md#d27--格式解释的身份覆盖全部钩子兼容验收到连接键) 第 3 条）：预登记每条关联规则用到的连接键；验收报告设备 → launch → 主机线程组 → 桥接 → 原生调用各跳的命中数，任一跳为 0 须指出断在哪个键。逐跳计数须同时登记格式身份、调用/注释选择配置（ScopeConfig）与统计单位（设备条数 / 唯一 launch 数）；前后对照必须同配置，替代路径不得按相邻列相除成覆盖率（T8.2 F1）；
11. **格式解释版本化**：`device_taxonomy()` 返回的格式对象带版本号，指纹覆盖全部注入钩子（D27 第 1 条，以变异测试保证）；新版本附新旧解释在全部已有样本上的差异预期，历史样本的项目 link、原生关系与处置逐字节不变（输出锁）；
12. **适用范围与静默失败**（[D28](DECISIONS.md#d28--格式适用范围显式化采集配置入身份导入判定适用性静默失败转显式)）：会话显式声明影响导出结构的采集配置并写进采集身份；格式契约随版本维护；导入报告给出适用性判定、轨道角色审计（未识别轨道上的完整事件计数并告警）与 unknown 归并。至少用两个变异片段验收：设备轨改名、删除设备轨元数据——R 可以变化，但必须出现告警并判为契约外，不得以 `complete` 且无任何异常的形式通过。另须（[D29](DECISIONS.md#d29--判定一致性结构证据优先于名称契约条目须被执行或被测试)）：在全部真实样本上，每条规范设备事件所在轨道的角色为设备执行轨；设备执行轨整条缺失时判契约外；对角色判定与适用性判定做变异，至少一项测试失败。验收写法（复审 [§3.2](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-a9b975fef97b)、[§3.5](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-5067078b07e5)）：性质门禁加同类输入上的阳性对照；**代码变异与契约数据变异**（删条件行、改值，条件 id 集合与 `requirement` 与预登记逐项比较并进 CI）并列，复核方另设计一批、不预先告知；导入时实际加载的判定模块须全部落在某个身份闭包内（身份层面的谓词清点）；每个必需条件登记证据生产者（[D30](DECISIONS.md#d30--契约条件的证据先查产物自述每个必需条件登记生产者)，已确认（用户 2026-10-16））。

### 4.3 生命周期义务（当前实现，T9，2026-10-18）

真实官方 SDK 暴露了假后端测不到的生命周期问题（`stop` 改写原件、分析子进程继承注入、跨窗日志等）。它们已转成厂商无关的公共机制（`flagprof/vendors/session.py` 与 `record`），厂商适配器须遵守下列义务：

| SDK / 进程事实 | 公共机制（当前实现） | 证据状态 |
|---|---|---|
| `stop` 之后 SDK 可能改写 working 目录里的原件 | 窗口导出后复制成 sealed-window 快照（逐文件 SHA + fsync，复制后与校验后核对身份清单）；窗口记录的 `path` 指快照，`working_path` 保留 SDK 目录；导入只读快照（[02 §5](02-architecture.md#5-一次-run-的数据流)） | 真机；主机缺"SDK.stop 改写 → import → verify"端到端回归 |
| SDK 起分析子进程，继承注入的环境 | 注入只按**完整 proc_key** 匹配父身份，不按进程名；后代不自动采集（[D32](DECISIONS.md#d32--record-自动注入范围只注入-wrapper-直接启动的-python-目标)，见 [02 §3](02-architecture.md#3-目标接入attach--wrap--launch)） | 真机；主机缺 spawn / forkserver 多分析子进程测试 |
| 窗口产物无效（空设备轨、截断、日志报错） | 窗口记 `invalid` 并附 reasons；`invalid_policy=stop` 抛错并停止后续窗口，`continue` 保留 invalid 窗口并继续下一窗 | 真机 stop / continue 均跑通；"`continue` 吞掉 SDK 未导出窗口"的变异存活（C01），待测试钉住 |
| 跨窗日志（同一日志文件被多个窗口共享） | 逐窗日志字节区间（`log_slice`：begin/end/sha256，前缀须保持，只数新出现的匹配起点）；日志被改写判 `collector_log_rewritten` | **record 路径尚未接线日志来源**（默认空日志，T9.4 M3）；默认 Traceback 与 fatal 分支有变异存活（E03/E04/E05/E07） |
| 进程被 kill（含窗口中途 SIGKILL） | 未封存窗口停在 `collecting`/`exporting`，读取方在 owner 已死时推断为 `lost`，不修复 manifest；已封存窗口完整；run 级 `truncated` | 真机 SIGKILL 跑通；主机缺"窗口中途 SIGKILL"回归；`inspect` 目前不显示 run 级 `truncated` 与窗口 `lost`（T9.4 M1） |
| 复制过程中断（复制中被杀、IO 失败、期间原件变化） | 只在暂存目录里工作，留下未登记的 `.sealing-*`，窗口保持 `exporting`，不得声明有效；不对 SDK 文件做 hardlink | 真机；主机缺"复制中途被杀"回归 |

只有真机证据、尚无主机回归的四项（SDK.stop 改写端到端、窗口中途 SIGKILL、复制中途被杀、SDK 多分析子进程）**待 T9.4（M5）补主机回归**；复核方已写出并验证三个最小 fake 测试（`c3_stop_rewrite_e2e.py`、`c3_kill_midwindow.py`、`c3_sigkill_midcopy.py`，归档在 `docs/reviews/claude_reviews/2026-10-18-t9-review-evidence/`），尚未进入 `tests/`。

**窗口停顿的构成与解析位置（当前实现，T9.3 复核实测）**：窗口边界 step 的中位耗时，A100 为 2.21 s = SDK 导出 0.26 s（12%）+ 我们的复制/哈希/fsync/校验 1.20 s（54%）+ 回调外 profiler stop 0.73 s（34%）；腾昇为 7.40 s = SDK 导出 5.65 s（76%）+ 复制/校验 1.27 s（17%）+ 余 0.50 s。窗口记录里的 `validation_ns`（复核汇总中称 `validation_snapshot_ns`）把复制、身份检查、fsync、校验（以及日志扫描与发布准备）合在一起，无法再拆。腾昇会话当前以 `tensorboard_trace_handler(analyse_flag=True, async_mode=False)` 导出（`flagprof/vendors/ascend/session.py:48`），官方解析因此在窗内**同步**执行，与"76% 在导出"相符。

> **【调研，待 spike】离线解析方向**：窗内只封存原始采集目录，把官方解析移到导入端（torch_npu 提供 `analyse_flag=False` + 事后离线 analyse，以及 `async_mode`），与"SDK.stop 改写原件"的教训一致——先封存原始字节。代价：导入端需要 torch_npu/CANN 环境，离线解析器版本须进解析身份。A100 侧封存成本（复制时同步哈希、单次读）同属待验证方向。目标是缩短窗口停顿，不改封存语义；以上均来自外部工具资料与本项目的停顿分解，**未在本项目实测**，须真机 spike 验证后再定。

## 5. registry

继承 v1 `runtime/adapters.py` 的 descriptor + 源码闭包校验思路，但**范围扩大**：探针、分析器、厂商、组件（FlagGems/FlagTree/FlagCX）都有 descriptor。

```json
// registry/probes/dispatch.json
{
  "id": "dispatch",
  "version": "1.0",
  "module": "flagprof.probes.dispatch",
  "emits": ["op", "opsum", "sig"],
  "cost": "high",
  "levels": ["each", "summary"],
  "requires": {"framework": "torch", "min_version": "2.1"},
  "sources": [
    {"metric": "op.dur", "api": "torch.utils._python_dispatch.TorchDispatchMode", "note": "主机侧 wall time，不含异步设备执行"}
  ],
  "code_closure": ["flagprof/probes/dispatch.py", "flagprof/context.py"]
}
```

> **当前实现（T9，2026-10-18）**：`flagprof/registry.py` 的 `Registry` 目前只做**懒加载的组件描述符**，字段为 `id`、`kind`、`version`、`module`、`factory`、`requires`、`format_vendor`。`kind` 取 `vendor`/`probe`/`analyzer`/`component`/`responder`；`requires` 是模块名字符串列表（不是上例的 `{framework, min_version}` 对象）；`format_vendor` 只对 `kind=vendor` 必需；当前取 `ascend`、`nvidia` 或候选 `kunlunxin`。来源有两个：内置 `BUILTINS`（三个 vendor）与 `record --plugins <JSON>` 传入的描述符列表（传入后**取代**而不是追加 `BUILTINS`）。`create()` 按 `module` + `factory` 延迟导入，并把实现文件 sha256 与厂商源码闭包身份写入实现身份（外部依赖不在认证范围）。上例的 `emits`、`cost`、`levels`、`sources`、`code_closure` 字段、`registry/` 目录下按类别分文件的 JSON，以及下面"作用"中的指标来源矩阵自动生成、`list` 子命令，均为**设计目标（未实现）**。

作用：

1. **指标来源矩阵自动生成**（[07 §5](07-analyzers-and-reports.md)）—— v1 要求每个指标写明来源 API，v2 把它变成 registry 的机器可读字段，报告直接生成，杜绝文档与实现脱节。
2. **身份校验** —— `code_closure` 的哈希进 manifest.identity，保证 seal 模式下"采集代码本身"也是可比较的一部分。
3. **依赖检查** —— 分析器按 [07 §1](07-analyzers-and-reports.md#1-分析器契约) 声明 `requires`：可有多条替代路径（`any_of`），每条路径列事件类型、派生视图与能力位，并声明自己的可用条件。都不满足时输出 `unavailable`，部分满足时输出 `partial`，不报错（继承 v1 partial 语义）。
4. **能力发现** —— CLI 的 `list` 子命令直接读 registry。


## 6. Spike-C 能力声明回填（2026-10-10；**已裁决，已并入 §4**）

> **读法**：本节保留为证据索引。"queue=1 用显式入队/出队桥"的规范表述已并入 [§4.1](#41-能力矩阵) 与 §4 的 `VendorAdapter` 协议，并据此**取消** `required_env` 强制项（[D13](DECISIONS.md#d13--工具不改变被测系统的运行配置)）。`per_call` 声明的适用边界由 [04 §6](04-correlation.md) 的二维 segments 表达（[D12](DECISIONS.md#d12--关联降级是二维的主机归因--设备链接)），不再是单一能力值。正文与本节冲突时以正文为准。

[实测报告](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-e1bd695b52c5)建议：已测 CANN 9.0 / torch_npu 2.10 / FlagTree cendAs eager 配置声明 `host_device_link=per_call`，附版本、队列配置和逐 run 覆盖率；queue=1 使用显式入队/出队桥。
这不替代导入时的丢失/冲突检查，不扩大到厂商 Triton 3.2 失败环境或未执行的 8 卡/图回放场景；TP2 已完成两后端三重复。此处为实测建议，规范性确认见 DECISIONS Q7。
