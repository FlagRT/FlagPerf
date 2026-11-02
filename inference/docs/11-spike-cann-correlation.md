> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书链接转入 [历史引用索引](history-references.md)，其出处保留。

# 11 Spike-C：CANN 算子级关联验证

> 状态：**partial：单卡与 A100 完成，TP2 HCCL/FlagCX 18 次完成、8 卡暂缓、schema 待确认**；[报告与证据](history-references.md#ref-e1bd695b52c5)。是 [10 阶段计划](10-roadmap.md) 的 S1，**早于一切主包代码**
> 回答的问题：在腾昇 + CANN 9.0 上，一次 ATen 算子调用能否唯一地关联到它触发的设备 kernel？**尤其是 FlagGems 经 Triton 生成的 kernel。**
> 结论的去向：[04 §3.3](04-correlation.md) 的规则表、[04 §6](04-correlation.md) 的降级级别、[05 §4.1](05-probes-and-vendors.md) 的 `host_device_link` 能力值、[03](03-event-schema.md) 的 `ext` 键集合与冻结

## 1. 为什么必须先做这件事

R53 要求"关联图节点、**Backend** 和运行时状态定位瓶颈"。如果一次 op 调用无法对应到设备 kernel，那么：

- 无法回答"FlagGems 的这个算子实现在腾昇上比 aten 快还是慢"；
- 无法把设备耗时归因到层；
- 给 FlagGems 开发组的 `op_perf.jsonl`（[07 §4](07-analyzers-and-reports.md)）就只能是空的或聚合的。

v1 只验证到**模块级**（`vendors/ascend/trace.py:attribute` 用 flow 端点与 `Node@launch` 的 `connection_id` 把设备事件归到主机线程的 scope 区间），且 HCCL 通信链路已验证（`implementation_3.md §4`）。**算子级、尤其是 Triton kernel 的路径从未验证过。** 如果在冻结 schema 之后才发现做不到，`kernel.link`、`ext` 键、分析器的归因口径都要返工。

## 2. 假设与待判定项

| 编号 | 假设 | 判定标准 |
|---|---|---|
| **H-A** | CANN profiler 的 flow 事件（`s`/`f` 端点）覆盖 **Triton 生成的 kernel**，不只覆盖 aclnn 算子 | 在 FlagGems on 的 run 中，Triton kernel 的 flow 端点存在率 |
| **H-B** | `Node@launch` 的 `connection_id`（或等价字段）在整个 run 内唯一，可作为 launch↔kernel 的一对一键 | 同一 `connection_id` 对应的设备事件数分布；是否出现跨算子重用 |
| **H-C** | 主机侧的 launch 记录可以**确定性地**归属到我们注入的 op marker（同线程嵌套），而不是靠时间最近 | 每个 launch 落在恰好一个 op marker 区间内的比例 |
| **H-D** | `TASK_QUEUE_ENABLE=1`（任务队列开启，默认配置）下上述关联仍然成立 | 对照实验；v1 强制为 0 正是因为任务队列线程打断归属 |
| **H-E** | 多流、多设备、多 rank 下 `connection_id` 的唯一性范围不被破坏 | 多卡实验中的冲突率 |
| **H-F** | `compile.kname`（Triton kernel 符号名）与 trace 中的 `kernel.name` 可精确匹配（或规范化后精确） | 名称匹配率；是否带不可预测后缀 |

**判据（2026-10-10 修订）**：关联链由两段组成，必须**同时闭合**才构成逐调用归因：

- **前段 `op → launch`**：H-C（同线程严格调用包含）或 queue 桥（`Enqueue.correlation_id → Dequeue`）任一成立；
- **后段 `launch → kernel`**：H-A（flow 端点）或 H-B（`connection_id`）任一成立。

两段同时闭合且三次重复稳定 → **级别 A**；仅后段闭合 → **级别 B+**；两段均不闭合但 H-F（kname）成立 → **级别 B**；均不成立 → **级别 C**。

> **原判据"H-A/H-B/H-C 任一成立即 A 级"已作废。** H-A/H-B 只闭合后段，H-C 只闭合前段，任一单独成立都推不出逐调用归因。级别按两段分别记录，见 [04 §6](04-correlation.md)。

## 3. 环境

| 项 | 要求 |
|---|---|
| 镜像 | `flagrt/ascend-operator-runtime:0.1.0-cann9.0-py311-torch2.10-arm64`，或其 0.2.0 / flagCX 分支（与 `VERSIONS.md` 对齐） |
| 软件栈 | CANN 9.0.0、Python 3.11.15、PyTorch 2.10 + torch_npu、triton_ascend、FlagGems |
| 硬件 | 单卡用于 §5.1–5.4；多卡（≥2，最好 8）用于 §5.5 |
| 对照 | 同一组实验在 NVIDIA 上跑一遍作为**基线**（CUPTI 的 `correlationId` 已知可用），用于区分"CANN 做不到"与"我们的方法有问题" |

记录环境身份（包版本、环境变量、驱动固件）作为报告的一部分 —— 结论只在该版本上成立。

## 4. 被测对象（三档，必须全做）

| 档 | 内容 | 目的 |
|---|---|---|
| **M1 微基准** | 纯 PyTorch 的 `mm`、`add`、`layer_norm`、`softmax` 各执行 100 次，shape 固定且各算子不同 | 最干净的环境，确定关联机制的基本形态。**各算子 shape 刻意不同**，便于用 shape 反查 kernel 是否归对 |
| **M2 FlagGems 算子** | 同样的算子，FlagGems 开启，确保走 Triton kernel；包含至少一个**必然回退**的算子作对照 | 判定 H-A/H-F —— 这是 Spike 的核心 |
| **M3 真实模型** | 一个小规模 HF 模型（如 Qwen3-Embedding-0.6B）单次前向，FlagGems on | 验证在真实算子密度与嵌套下结论是否保持 |

M1/M2 必须包含**重复执行**（同一算子同一 shape 跑 100 次），因为关联歧义恰恰出现在重复调用上 —— 单次调用谁都能关联对。

## 5. 实验设计

### 5.1 E1：marker 注入与 launch 归属（H-C）

**做法**：用 `TorchDispatchMode` 包住每次算子调用，在进入/退出时通过 torch_npu 的 marker/`record_function` 注入带**唯一 id** 的标记（id 即未来的 corr）。开启 CANN profiler，导出 trace。

**测量**：
- 每个 op marker 区间内落入的主机侧 launch 记录数（期望：每个算子 1 个或少数几个，且与算子语义相符）；
- 跨区间或区间外的 launch 占比；
- **同一线程**内 marker 是否严格嵌套（若注入的 marker 自身乱序，一切免谈）。

**注意**：marker 注入本身有开销且可能改变调度。E1 需要同时跑一次"不注入 marker"的 profiler，对比 kernel 总时间与数量，确认注入没有显著改变执行（若改变显著，要在报告里说明）。

### 5.2 E2：flow 与 connection_id 的覆盖与唯一性（H-A/H-B）

**做法**：解析 profiler 导出的全部产物（`ASCEND_PROFILER_OUTPUT/` 下的 `*.json`/`*.csv`、以及原始 `PROF_*` 目录），枚举所有可能的关联字段。

**必须逐项统计**（这是本 spike 最重要的交付数据）：

| 统计项 | 说明 |
|---|---|
| 设备 kernel 总数、总时间 | 分母 |
| 具有 flow `f` 端点的 kernel 数与时间占比 | H-A，**按 kernel 来源分列：aclnn vs Triton** |
| flow `s` 端点能落到主机线程 + 时间区间的比例 | 端到端可行性 |
| `connection_id` 的取值分布：出现次数直方图 | H-B。若大量 `connection_id` 对应 >1 个 kernel，要看是"一次 launch 多个 kernel"（合法）还是"不同 launch 重用了 id"（致命） |
| `Node@launch` 记录数 vs 主机 launch 记录数 | 是否一一对应 |
| 其他候选字段 | `task_id`、`stream_id`、`batch_id`、`Enqueue/Dequeue correlation_id`、ACL API 记录中的任何 id。**全部枚举，不要只验证已知的那几个** |

**关键判别方法（必须用，不要只看字段存在性）**：用 **shape 反查**。M1 中各算子 shape 不同，因此 kernel 的 grid/block 或名称中的形状特征可以独立验证关联是否正确 —— 即"关联说这个 kernel 属于 `mm(1024,1024)`，那它的规模就应当与 `mm(1024,1024)` 相符"。**不要用时间接近来自我验证**，那会得出循环论证的结论。

### 5.3 E3：Triton kernel 专项（H-A/H-F）

FlagGems 的 Triton kernel 走的是与 aclnn 不同的下发路径，必须单独验证：

- Triton kernel 在 trace 中以什么名字出现？是否带哈希/特化后缀？后缀是否在多次 run 间稳定？
- 它是否产生 flow 端点与 `Node@launch` 记录？
- `triton_ascend` 的编译产物（kernel 二进制/符号名）与 trace 中的 `kernel.name` 能否精确匹配（H-F）？
- JIT 首次编译发生在第 1 次调用，第 2–100 次是缓存命中 —— 两者的 trace 形态是否一致？

### 5.4 E4：TASK_QUEUE_ENABLE 对照（H-D）

v1 强制 `TASK_QUEUE_ENABLE=0`，这是一个**重大限制**：它改变了被测系统的行为，意味着我们测的不是用户生产环境的真实形态。

**做法**：E1–E3 在 `TASK_QUEUE_ENABLE=0` 与 `=1` 下各跑一遍，同时记录 `TRITON_ENABLE_TASKQUEUE` 的取值组合。

**测量**：
- 两种配置下 §5.2 的全部统计量对比；
- `=1` 时主机侧 launch 发生在哪个线程（任务队列线程 vs 主线程）—— 这决定 TLS corr 是否在线程边界断开；
- 若 `=1` 下关联失效，**定位失效的确切环节**（是 marker 归属断了，还是 flow 端点消失了）。这个定位结果决定我们能否用"在入队点传递 corr"的方式修复。
- 两种配置下的端到端性能差异（量化"为了可观测性付出的代价"）。

### 5.5 E5：多流 / 多卡（H-E）

- 单卡多流：构造并发的算子流（多个 Python 线程或显式多 stream），检查 `connection_id` 是否跨流冲突；
- 多卡：2 卡与 8 卡的 TP 场景，加入 HCCL 通信，检查 `connection_id` 的唯一性范围是进程内、设备内还是全局；
- 同时验证 [04 §3.5](04-correlation.md) 的 `(group, group_seq)` 跨 rank 对齐在真实 HCCL trace 上可还原（v1 已验证 marker→Enqueue/Dequeue→`Node@launch`→`connection_id` 链路，这里是复核 + 扩展到 FlagCX）。

### 5.6 E6：NVIDIA 基线

同样的 M1/M2/M3 在 NVIDIA 上跑，用 CUPTI `correlationId` 建立关联，统计同样的指标。作用有二：

1. 给出"关联做对了应该长什么样"的参照（关联率、歧义率的合理范围）；
2. 验证我们的实验方法本身没问题 —— 如果在 NVIDIA 上也测不出高关联率，那是方法错了，不是 CANN 的问题。

## 6. 注意事项（踩过或可预见的坑）

1. **绝不用时间邻近做验证**。本 spike 的结论会固化成全工程的关联规则，如果验证方法本身依赖时间匹配，就会把一个错误的方法论写进地基。用 shape 反查、用计数一致性、用注入的唯一 id。
2. **设备 kernel 时间不必落在主机 marker 窗口内**（`implementation_14.md` 的实测结论）。不要因为"kernel 在 marker 之后才执行"就判定关联失败 —— 异步执行下这是正常的。
3. **profiler 本身会改变行为**。Level1 + `record_shapes` 的开销不可忽略。每组实验都要有"profiler off"的耗时对照。
4. **不要只看 `ASCEND_PROFILER_OUTPUT` 的加工产物**。torch_npu 导出的 json 是二次加工的，可能丢字段。原始 `PROF_*` 目录下的 `.data`/`.csv` 里可能有更多 id。两者都要翻。
5. **首次调用 ≠ 稳态调用**。第 1 次有 JIT 编译、内存分配、流创建，形态与第 2–100 次不同。统计时分开，不要把预热期混进去。
6. **多次 run 的稳定性**。同一实验至少跑 3 次，确认关联率不是偶然。若波动大，说明机制不可靠，即使单次成功也不能定为级别 A。
7. **回退路径也要覆盖**。M2 中必须有一个 FlagGems 回退到 aten 的算子 —— 实际 run 中两条路径混合存在，关联规则必须同时处理。
8. **记录否定结果**。"字段 X 存在但全为 0"、"flow 只覆盖 aclnn 不覆盖 Triton" 这类否定结论与肯定结论同等重要，必须写进报告。不要只报喜。
9. **不要在 spike 里写主包代码**。spike 的产物是知识与报告，代码放 `spikes/cann_correlation/`，允许粗糙，但**解析逻辑要在 P2 中被正式重写**，不要直接搬运。
10. **环境变量全记录**。`TASK_QUEUE_ENABLE`、`TRITON_ENABLE_TASKQUEUE`、`ASCEND_*`、`ACL_*` 的实际取值全部写进报告 —— 结论只在这组取值下成立。
11. **多卡实验注意端口与残留进程**。失败的多卡 run 会留下僵尸进程占用设备，影响下一轮测量。每轮前确认设备干净（`npu-smi info`）。

## 7. 交付物

codex 完成 S1 时交付以下内容，放在 `FlagPerf_advance/inference_v2/spikes/cann_correlation/`：

| 文件 | 内容 |
|---|---|
| `REPORT.md` | **主交付物**，见 §8 的必答清单 |
| `run_e1.py` … `run_e6.py` | 各实验的可重跑脚本，参数化（`TASK_QUEUE_ENABLE`、FlagGems 开关、模型、卡数） |
| `parse_trace.py` | trace 解析与统计脚本，输出 §5.2 的统计表 |
| `data/<exp>-<config>/` | 每组实验的原始 trace（或其最小可复现子集）+ 统计 CSV |
| `fixtures/` | 挑选出的**真实 trace 片段**，供 P2 的 linker 单元测试直接使用（[04 §7](04-correlation.md) 要求用真实片段而非人造数据） |
| `env.json` | 全部环境身份信息 |

## 8. REPORT.md 必须回答的问题

逐条回答，**每条都要有数据支撑，不允许只写结论**：

1. H-A 到 H-F 各自成立与否，支撑数据是什么？
2. CANN 9.0 的 trace 中，**所有**可用于关联的字段是哪些？各自的语义、唯一性范围、覆盖率？
3. Triton kernel 与 aclnn 算子在关联能力上有无差异？若有，差在哪一环？
4. 最终 `host_device_link` 应取 `per_call` / `per_kernel_name` / `interval_only` 中的哪个？
5. 推荐的关联规则有哪几条？优先级？每条的 `via` id、输入字段、判定逻辑、已知失效条件？
6. `TASK_QUEUE_ENABLE=1` 下能否工作？若不能，失效的确切环节在哪？是否有 workaround（如在入队点传递 corr）？若无，我们是否必须把"要求 `=0`"写成工具的硬性前提并在报告中向用户声明这一偏差？
7. 多流/多卡下 `connection_id` 的唯一性范围？
8. [03 §5](03-event-schema.md) 的 `ext` 应包含哪些 `cann.*` 键？
9. `kernel.link` 的 `status` 判定在实测中的分布（linked/ambiguous/unlinked 各占多少，按时间加权）？
10. 与 NVIDIA 基线相比，差距在哪？哪些差距是机制性的、哪些是我们方法可改进的？
11. 对 [03](03-event-schema.md) 和 [04](04-correlation.md) 的**具体修改建议**（哪一节、改成什么）。
12. 有没有发现任何**当前设计未考虑**的情况？（这一条最容易被忽略，但往往最有价值）

## 9. 完成判据

- §8 的 12 个问题全部有数据支撑的回答；
- `fixtures/` 中的真实片段可被独立使用；
- [04](04-correlation.md) 与 [03](03-event-schema.md) 完成回填，[DECISIONS](DECISIONS.md) 新增一条记录 schema 冻结；
- 结论在 [10 §9](10-roadmap.md) 的状态表中登记。
