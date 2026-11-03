> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书出处见删除前固定 Git 提交中的 [历史引用索引](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md)，其出处保留。

# 07 分析器与报告

> 2026-10-20 授权修订：[D34](DECISIONS.md#d34--t94t95-读取批次与测量边界补充)与 T9.4 M8/M9、T9.5 §4.7 补充公共读取校验、首批屏障、质量/身份和测量/关闭边界。P0 未退出；历史验收与旧状态保留。当前任务具体规则以两份修订任务单为准。


> 上游：[03 schema](03-event-schema.md)、[04 关联](04-correlation.md) ｜ 下游：[08 在线/故障](08-online-and-fault.md)、[09 CLI](09-cli-and-campaign.md)
> 制作报告文档/生成器前须先读 `report_rules.md` 与 `内容讲解方法论.md`（项目规则）

## 1. 分析器契约

```python
class Analyzer(Protocol):
    id: str
    requires: Requires           # 见下：事件、派生视图与能力位，可声明多条替代路径
    requires_min_schema: str     # 如 "1.0"
    def run(self, store: Store, params: dict) -> Result: ...

@dataclass
class Need:
    events: list[str] = ()       # 事件类型，如 ["kernel"]、["op", "sig", "route"]
    views: list[str] = ()        # 公共只读接口的派生视图，如 ["native_relations", "ledger"]
    capabilities: list[str] = () # 能力位与数据条件，如 ["clock_valid"]、["module_scope"]、["sig_complete"]

@dataclass
class Requires:
    any_of: list[Need]           # 满足其中任一条路径即可运行；按顺序取第一条满足的路径
```

`requires` 在 2026-10-15 由"事件类型列表"扩展为上面的结构（[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立) 第 4 条、[D25](DECISIONS.md#d25--冻结语义与输出不冻结厂商分类表)）。原因是同一个分析结论可能有多条数据来源。例如 `perf.op` 的"设备时间按调用聚合"：
- 有项目探针时，走 `Need(events=["kernel", "op", "sig"])`，按项目 corr 与规范签名聚合；
- 只有官方 profiler 原件时，走 `Need(events=["kernel"], views=["native_relations"])`，按原生调用聚合，没有规范签名。

只列事件类型时，第二条路径会被"缺 op/sig"一律挡掉，原生关系带来的价值就到不了分析器。实际走的路径与未满足的条件写进 `Result.coverage` 和 `notes`。

`Result` 必须包含：

```python
@dataclass
class Result:
    id: str
    status: Literal["ok", "partial", "unavailable"]
    data: Any                  # 机器可读结论（表/字典）
    coverage: dict             # 见 §2
    notes: list[str]           # 人读的限制说明
```

硬约束：

1. 分析器是**纯函数**：只经 store 的**公共只读接口**读取（事件 + 版本化只读元数据与派生视图：来源身份、项目绑定、原生关系、作用域、来源账本、两套覆盖率、批量证据回查；[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)）。不直接读运行目录里的文件，不读 `ext`，不解析 `vendor_raw/`，不访问网络，不 import torch、厂商 SDK 或 `vendors/`（N3）。
2. `requires` 的任何一条路径都不满足 → `status="unavailable"`，**不报错、不中断**其他分析器（继承 v1 partial 语义）；`notes` 列出每条路径缺什么。
3. 部分满足（如 kernel 事件存在但关联率低）→ `status="partial"`，结论照出但 `notes` 说明限制。
4. 任何结论都能追溯到原始事件（N6）：`data` 中的行携带定位信息（corr、sid、call_key、ts 区间），以及事件身份 `{w, seq}` / `{trace_uid, loc}`（[03 §6.5](03-event-schema.md)）。
5. **依赖的逐事件数据已被精简时降级**：分析器读 `manifest.retention`（[03 §7](03-event-schema.md#7-保留与压缩策略)），所需事件类型已被精简时输出 `unavailable`/`partial` 并说明原因，不在聚合数据上给出逐调用结论。
6. **时间轴类结论要求有来源的时钟**：导入的设备事件在没有可信时钟换算时 `ts=null`（附录 A #5，T8 导入结果即如此）。此时设备**时长**可用，但不得给出跨来源时间轴、设备空闲归因、wall-clock 占比等结论，这类分析器在 `requires` 中声明 `capabilities=["clock_valid"]`。`ts=null` 说的是我方导入尚未做换算，不等于来源没有时钟信息：Kineto 的主机与 GPU 已在同一 `ts` 轴上（生产者统一，文档），腾昇的时钟锚点齐全；T9 登记时钟来源证据，P2 决定"生产者统一时基"的有效档位（[复审 §6.3](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-fda8bbc0b9b5)）。在此之前仍按 `ts=null` 处理。
7. **设备时长之和不是层的耗时**：按原生调用或模块作用域把设备时长求和，得到的是"该范围发起的设备工作量"，不是该层的独占墙钟时间（多流并发、主机侧开销、同步等待都不在其中）。报告中两者必须分开命名。
8. **导入状态与持久化范围要一起读**：公共接口的 coverage 当前以 `scope=prepared_events` 计算。导入为 `partial` 时，派生视图可能完整而设备事件只落盘了一部分，分析器须同时检查导入状态，不得用准备阶段的覆盖率描述已持久化的事件全集。
9. **路径"有数据"不等于"可用"**（2026-10-15，T8.1 审核）：`any_of` 选路时，一条路径的视图存在但其中没有有效关系（例如 `native_relations` 存在、owned 为 0），不能算满足，也不能因此挡住后面真正可用的路径。每条路径须声明自己的**可用条件**（如"至少一条 owned"），按条件选路，并在 `notes` 中写明为什么选这条、其余路径缺什么。T8.1 的腾昇当前采集正是这种情况：设备清单可用，按原生调用聚合不可用。字段级的降级规则在 P1 任务单中细化。
10. **导入完整、格式适用、分析可用是三件事**（2026-10-16，T8.3 审核；[D28](DECISIONS.md#d28--格式适用范围显式化采集配置入身份导入判定适用性静默失败转显式)、[D29](DECISIONS.md#d29--判定一致性结构证据优先于名称契约条目须被执行或被测试)）：`complete` 只说明落盘完整；`StoreView.format_assessment()` 的内层判定（契约内 / 契约外 / 无法判定）说明解释前提是否成立，旧 store 返回 `unavailable`，不等于"无异常"。契约外时不得把 R = 0 读成"没有设备负载"，不得给出无条件的全量占比；无法判定时，只在所需字段与关系不受缺项影响的范围内给出 `partial` 结论。unknown 组按原生类型名分开报告：profiler 自身的控制任务（如腾昇 `PROFILING_ENABLE/DISABLE`）单列，不混入"未知的模型计算"。逐结论的消费矩阵以 [codex T8.3 复盘](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-d94c86bbac29) §4.2 与[返工复盘](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-711fa68dfd18) §5.4 为输入，在 P1 任务单中定稿；§5.4 把"统一时间轴"一行写成"不可提供"过于保守，见第 6 条（[复审 §7](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-3f7249ef335c)）。

## 2. coverage：每个结论自带的可信度

```json
{
  "level_distribution": {                // 逐事件级别的分布，按类别分列，见 04 §6
    "compute": {"A":  {"count": 10432, "time_ns": 8104223000},
                "B+": {"count": 211,   "time_ns": 297310000},
                "C":  {"count": 27,    "time_ns": 1904000}},
    "comm": {"...": "..."}, "memcpy": {"...": "..."}, "unknown": {"...": "..."}
  },
  "launch_attr_ratio": 0.991,            // 归属唯一的 launch / 全部 launch（B 级），见 04 §6
  "kname_aggregate": true,               // H4 能力位，见 04 §6
  "op_granularity": "each|summary",      // op 还是 opsum
  "kernel_link": {                       // 集合与指标的定义见 04 §2
    "compute": {"raw": 10760, "imported": 10758, "exempt": 88,
                "linked": 10432, "ambiguous": 31, "unlinked": 207,
                "reasons": {"conflict": 31, "no_bridge": 180, "no_device_key": 27},
                "linked_count_ratio": 0.978, "linked_time_ratio": 0.964,
                "import_completeness": 0.9998, "dur_missing": 0},
    "comm": {"...": "..."}, "memcpy": {"...": "..."}, "unknown": {"...": "..."},
    "excluded_by_time": 14,              // 被时间规则排除的候选数，见 04 §1
    "ts_null": 0                         // 时钟换算不成立的设备事件数，见 03 §5.2
  },
  "profiled_window": 0.92,               // vendor profiler 开启窗口 / measure 阶段主机墙钟
  "scope_coverage": {"scoped_ops": 0.998, "unscoped_threads": ["tid:4412"]},
  "tensor_align": {"aligned": 1824, "unscoped": 0},   // call_key 不可对齐计数，见 03 §4.9
  "tensor_grade": {"full": 0, "sampled": 1824, "aggregate": 0},  // 证据等级，见 §3.3
  "layer_folding": {"folded": true, "kept": [0, 1, 31]},
  "dropped_events": {"op": 0, "counter": 12, "exact": true},  // writer 未正常关闭时 exact=false，为下界，见 03 §6.3
  "sampling_gaps": [["t0","t1","profiler_blocked"]],
  "clock_drift_warning": false,
  "retention": {"pruned": false},        // 精简后依赖逐事件的结论降级，见 03 §7
  "perturbation": {                      // 见 §2.1
    "probes_injected": ["dispatch:each", "module"],
    "device": "not_evaluated",           // present / controlled / not_evaluated
    "host": "present",
    "e2e_overhead_vs_baseline": 1.13,    // 无配对基线时为 null
    "baseline_run": "r_20261010_m3_nomark"
  }
}
```

**coverage 不是脚注**：报告首屏展示各类别的级别分布（条数与时长）、`linked_count_ratio`/`linked_time_ratio`、`import_completeness`、`profiled_window`，以及 `perturbation` 的两项。用户先知道这份数据能说多少话，再看数字。

> 关联率与 `import_completeness`、`profiled_window` **必须同屏出现**。前者是「导入的部分里归因了多少」，后两者是「原始记录导入了多少、采集窗口占多长」。只报前者会把「范围内 100%」读成「整体 100%」。

> **原生与项目覆盖分列（2026-10-14，[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)）。** `kernel_link` 只衡量项目归属。原生归属的覆盖（[04 §2](04-correlation.md#2-关联状态) 的 `native_owned_*`、`scope_*`）与来源账本的处置汇总，要在 coverage 中各自成块，并与 `kernel_link` 并列展示，不合并成一个比率。
> - 没有项目探针的 run，`linked_count_ratio` 为 0 是正确的；同屏应能看到原生归属与作用域覆盖，否则会被误读为"这家厂商无法关联"。
> - 具体的键名随 P1 分析器契约一起定稿。
>
> 上一版在这里用 Spike-C 的「100% 与 51.7% 并存」作例子，并定义了 `marked_scope_ratio`。51.7% 是 campaign 中 marked 与 plain 两类对照 run 之间的 kernel 条数比，不是单个 run 的覆盖率，该指标已删除（[04 §2](04-correlation.md#2-关联状态) 的更正）。campaign 层面需要类似的量时，用带前缀的名字（如 `campaign_marked_kernel_count_share`），不得与单 run 指标同名。

### 2.1 扰动：按指标类单独标注，不自动判定为无扰动

v1 的 `perturbed: bool` 把"这份数据被探针污染了"当成整份 run 的属性，开了任何探针就整份不可信。按指标类拆开是对的：主机侧时长（`op.dur`/`span.dur`/端到端）与设备侧时长（`kernel`/`memop`/`comm` 设备段）受到的扰动来源不同，强度也不同。

但拆开之后，**每一类仍然需要证据**，不能从「这是设备时间」直接推出「没有扰动」。[Spike-C](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/11-spike-cann-correlation.md) 在同配置、同 kernel 数的 17 组"有标记 / 无标记"配对中，测得单 kernel 设备时长之比 0.926–1.034，端到端耗时之比 0.97–3.86（3.0–3.9 倍只出现在 M1 这类 kernel 极短的微基准上，真实模型 M3 为 +13%）。它支持的结论是：**在这些配置下，标记对设备时长的影响落在这个范围内**。它不覆盖 tensor 探针（会同步设备、改变并发）、通信等待、并发调度，也不能外推到别的 workload。

> **规范**：
> - 任何时长都按「**该采集配置下观察到的值**」报告，单次 run 即可报告，并附采集配置。
> - `perturbation.host` / `perturbation.device` 各取一值：`present`（已知有扰动源作用于这类指标，如标记之于主机时长、tensor 探针之于两侧）/ `controlled`（有同配置的对照 run，且对照覆盖了这类指标；同时给出对照 run、指标与观测到的比值范围）/ `not_evaluated`（没有对照证据，**默认值**）。**没有「天然无扰动」这个取值。**
> - 端到端吞吐与时延要作为性能结论，需要同配置的无标记基线；没有时照常报告观察值，并在 `notes` 注明"含采集开销"。
> - 采集开销是 **per-workload** 的量，不是工具的固定常数，禁止用某次测得的倍率去校正另一个 workload。
> - 预设能产出哪些指标，取决于它实际启用的来源（[05 §3](05-probes-and-vendors.md)）。`accuracy` 预设不开 vendor profiler，设备性能指标为 `unavailable`。

**`status` 与扰动分开。** 分析器的 `status` 只由完整性与归属决定（依赖是否满足、关联率、丢弃与精简）；扰动是并列的独立标注，报告首屏同时展示二者。上一版让 `device_metrics="unperturbed"` 自动支撑 `status="ok"`，把「数据完整、归属明确」与「代表不加探针时的性能」两个不同的判定混在了一起（[D14 修订](DECISIONS.md#d14--扰动按指标类分流不是整份-run-的布尔标签)）。

## 3. 分析器目录

### 3.1 性能（R53、R31）

| 分析器 | 输入 | 输出 |
|---|---|---|
| `perf.op` | op/opsum + sig + route + kernel.link | 按 (算子, 签名, 路由决策) 的调用次数、主机时间、设备时间、占比；Top-N 瓶颈。**聚合层级必须声明**（2026-10-15，T8.2 审核）：原生 owner 是最内层原生调用（腾昇常为 `aclnn*`，外层才是 `aten::*`），两家芯片不在同一层，不得按 owner 名称直接横比；只有原生 owned、没有项目 sig/route 时，可输出"原生调用发起的设备工作量"（含重叠，不是独占墙钟），回退率与跨厂商签名对比为 unavailable |
| `perf.layer` | span(cat=module) + op + kernel | 逐层耗时树（主机/设备分列）、层内算子构成、重复层的方差 |
| `perf.phase` | span(cat=phase/step/batch) | 阶段划分：加载/预热/测量；生成式的 prefill vs decode、每 token 耗时分布 |
| `perf.timeline` | 全部带时间的事件 | 可供 UI 渲染的时间轴切片（主机线程 × 设备流 × counter 对齐） |
| `perf.gap` | kernel + comm + span | 设备空闲区间定位：是等主机、等通信、还是等内存 |
| `memory` | memop + counter(src=device/framework) | 显存峰值与时间点、分配/释放归因到层、搬运量 |
| `comm` | comm 两个 phase + 跨 rank group_seq | 每次集合通信的各 rank 到达差、总线带宽、等待时间占比 |

"关联图节点、Backend 和运行时状态"三要素分别由 `perf.layer`（图节点/层）、`perf.op` 的 route/compile 列（Backend）、`perf.timeline` 的 counter 轨（运行时状态）承载。

### 3.2 路由与覆盖（R47、R18/R22、R45）—— 取代 v1 的 preview

v1 用独立的 preview 阶段（5 个模块、专门的搜索流程）找出 FlagGems 不可用算子。v2 中这**只是一次普通 run 的分析结果**：

| 分析器 | 输出 | 替代 v1 的 |
|---|---|---|
| `route.coverage` | 按算子×签名的 FlagGems 命中率；回退清单（算子、签名、`reason` 枚举、发生次数、涉及的层） | preview 的主要产物 |
| `route.impact` | 回退带来的时间代价估计：回退算子的设备时间占比；若有 off/on 对照 run，给出实测差值 | preview_budget |
| `compile.health` | 编译耗时分布、缓存命中率、失效原因、编译失败清单 | — |

**隔离验证的位置变了**：v1 需要逐个算子开关去搜索哪个会失败；v2 一次 run 就从 `route.reason` 得到**这次实际执行到的**全部回退与失败项。没有执行到的路径（如前面的算子抛错，后续代码没跑）保持未知，`route.coverage` 不把它们列为可用，也不列为不可用。只有当需要确认"某算子单独启用是否可用"时，才由 `route.coverage` 的结论**驱动**一次二次 run（CLI：`flagprof verify-route --from <run> --ops ...`），这是分析器的下游动作，不是前置阶段。

### 3.3 精度（R52、R86/R88）

> **P4.1 实现口径（2026-10-29）**：full 浮点配对按原 dtype 单次读取数值与 ULP；相等极值作为成对哨兵排除，有限极值失配另计。层级直方图与分母累加；max_abs／ULP／样本／位置同源。张量 stride/contiguous 单列，ULP 描述量级，具体成因结合实现或定点实验。[T11.3 验收](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-4ea9cd3ce263)。


> **2026-10-28 按 [D50](DECISIONS.md#d50) 第 8 条（P4 首批，T11.2 实现）**：
> - 实现 `oracle.identity_diff`、`oracle.layer`、`oracle.first_divergence`、`oracle.op_sensitivity` 与对齐清点；`oracle.replay` 留到第二批。
> - “首个发散层”不设阈值，分两项给出：`first_inequal` 是首个 digest 不等的边界；`largest_jumps` 是相对误差跳变的前 5 名，只排序。
> - `op_sensitivity` 增加 `introduced`：全部来源逐字节相等、输出在 `full` 等级下数值不等，即判为差异由该算子产生，这一判断不需要 replay。来源已有差异时仍只报放大倍数。

| 分析器 | 输入 | 输出 |
|---|---|---|
| `oracle.layer` | 两个 run 的 tensor 事件，按 `call_key` join | 每个边界的误差指标，**按证据等级输出**（见下）：`full` 给逐元素的绝对/相对误差 max/mean/p99、余弦相似度、超阈值元素占比；`sampled` 给样本上的同类指标；`aggregate` 只给 stats/chunks 的差异。NaN/Inf 计数各等级都有 |
| `oracle.first_divergence` | 同上 + span 树 | **首个检出的发散层**：在现有证据等级下，沿执行顺序找第一个误差显著跃升的边界，并给出其上游层的误差水平作对照。等级低于 `full` 时，结论是「此等级下首个检出」，不是「真正的首个出错点」 |
| `oracle.op_sensitivity` | tensor(role=in/out) + sig + route | 按算子类型/签名/路由聚合误差放大量，给出**敏感算子线索**（R52 原文）。只说「误差在此处放大」，不判定「误差由本算子引入」 |
| `oracle.replay` | 某个 call_key 的 in/out blob | 用同样输入在两边重放单个算子，隔离浮点顺序与舍入差异。需要 blob，属按需触发 |
| `oracle.identity_diff` | 两个 manifest | 环境差异清单；差异未被 `experiment` 解释时阻止出结论 |

**不设阈值**（v1 原则，继承）：分析器只给指标与排序，不输出 pass/fail。报告可以展示"误差最大的 10 个边界"，但不说"不合格"。容差配置、异常算子判定与门禁是分析器**之上**的独立判定层，属扩展（[00 §6](00-requirements.md#6-需求池参考文档补充) X8），不进入分析器；精度修正不做（[D21](DECISIONS.md#d21--范围边界暂不做与不做清单) B2）。

**证据等级。** 每个 tensor 比较结果带 `grade`，取两侧共同具备的最高等级：

| 等级 | 来源（[03 §4.9](03-event-schema.md)） | 能说 | 不能说 |
|---|---|---|---|
| `full` | 两侧都有 `blob` | 逐元素误差、首个出错元素 | — |
| `sampled` | 两侧都有 `probe`，且 `probe_alg` 相同 | 样本上的逐元素误差 | 样本之外的元素一致 |
| `aggregate` | 只有 `stats`/`chunks`，或两侧 `probe_alg` 不同 | 聚合量的差异，作为筛查线索 | 张量一致；差异的位置与大小 |

聚合量或样本没有差异，**不能**证明整个张量一致。

**放大不等于引入。** 即使两侧都精确地计算 `y = 1000·x`，输入差 1e-6 也会变成输出差 1e-3 —— 输出误差显著大于输入误差，可能只说明这个算子的条件数大。因此 `oracle.op_sensitivity` 只报告放大倍数与排序，作为线索；要确认误差**来源于**某个算子的实现，需要 `oracle.replay`：用同一份输入在两侧重放该算子并比较输出。没有 blob 时 replay 不可用，结论停留在线索层面。无输入捕获时放大倍数也无法计算，标注为"不可区分"。

### 3.4 故障与长稳（R32、R33）

见 [08](08-online-and-fault.md)。离线侧的分析器：`fault.timeline`（故障前最后 N 个事件的现场）、`stability.drift`（长时间窗口内的耗时/显存漂移）、`lifecycle`（热加载代次与切换耗时）。

## 4. 面向组件开发组的导出清单（R31 的"反馈至算子库、编译层和模型适配工具链"）

这是本工具对 FlagGems / FlagTree / FlagCX 开发组的**直接产出**，格式必须机器可读且稳定。每项都带回指原始 run 与事件的定位信息（R106–R108 的复现要求）。

| 导出 | 消费方 | 内容 |
|---|---|---|
| `missing_ops.jsonl` | FlagGems | 算子名、签名（完整 shape/dtype）、`reason`、出现频次、涉及模型与层、设备时间占比（即"修这个值多少"）、run_id + corr 样本 |
| `op_perf.jsonl` | FlagGems | 同一算子在 aten / FlagGems / 厂商原生三条路径下的设备时间对比（需要对照 run） |
| `op_accuracy.jsonl` | FlagGems | 按签名的数值误差与证据等级（来自 oracle 分析），与 `op_perf.jsonl` 用**同一个 sig 哈希**对齐 → 直接支撑 R37 的能力矩阵选算 |
| `compile_issues.jsonl` | FlagTree | 编译失败与超时清单、缓存失效模式、各阶段耗时、后端版本 |
| `comm_profile.jsonl` | FlagCX | 按 (op, bytes, group 大小) 的通信耗时与带宽、各 rank 不同步程度、超时事件 |
| `vendor_gaps.jsonl` | 厂商/适配工具链 | 该芯片上不可用或显著劣化的算子与通信模式 |

> `op_perf` 与 `op_accuracy` 用同一 sig 哈希对齐是 R37 的硬要求（[00 §2](00-requirements.md)）—— 这也是 §3.1 与 §3.3 必须共用 `sig` 字典的原因。

面向编译优化层与模型适配工具链的反馈（实测 Shape 分布、显存峰值、配置建议）属扩展，登记在 [00 §6](00-requirements.md#6-需求池参考文档补充) X13；上表是主线与 P5 的范围。

## 5. 指标来源矩阵（自动生成）

v1 要求"每个指标写明来源/API"，但靠人工维护文档。v2 中每个探针/分析器在 registry 的 `sources` 字段声明来源（[05 §5](05-probes-and-vendors.md)），报告的"指标来源"一节由 registry **生成**：

| 指标 | 来源 API | 口径说明 | 探针 |
|---|---|---|---|
| `op.dur` | `TorchDispatchMode` | 主机 wall time，不含异步设备执行 | dispatch |
| `kernel.dur` | CANN profiler kernel 记录 | 设备时间轴上的时长；`kernel.ts` 换算到主机单调时钟，不成立时为 null | vendor.ascend.trace |
| `counter.mem_used` | npu-smi | 设备上报，profiler 期间不可用（标 blocked） | counter |

文档与实现因此不会脱节：改了采集来源就必须改 descriptor，否则 CI 的一致性检查失败。

## 6. 报告

- **shell + panels**：报告框架只负责导航、coverage 展示与导出；每个 panel 对应一个分析器的 `Result`。分析器 `unavailable` 时 panel 显示原因而非消失（用户需要知道"为什么没有这个数据"）。
- **两种输出**：HTML（人读，自包含单文件）与 `result.json` + `*.jsonl`（机器读，即 §4 的导出）。两者由同一批 `Result` 生成，不存在"报告里有但 JSON 里没有"的数据。
- **批量汇总**（R89）：campaign 级报告由单 run 的 `result.json` 聚合而成，不重新读事件。
- 制作报告模板前必须先读 `report_rules.md`，并参考 `内容讲解方法论.md` 的讲解结构。
