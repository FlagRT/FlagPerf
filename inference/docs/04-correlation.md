> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书链接转入 [历史引用索引](history/history-references.md)，其出处保留。

# 04 关联脊梁（correlation spine）

> 状态：**草案；§3.3/§3.4 的 CANN 部分依赖 [Spike-C](history/11-spike-cann-correlation.md) 的实测结论**
> 上游：[02 架构](02-architecture.md) P5、[03 schema](03-event-schema.md) §3 ｜ 下游：[07 分析器](07-analyzers-and-reports.md)、[11 Spike-C](history/11-spike-cann-correlation.md)

R53 的原话是"**关联图节点、Backend 和运行时状态**定位瓶颈"。本文件定义这条关联链怎么建立、建立不了时怎么说实话。这是整个工具最难、也最有价值的部分。

## 1. 第一原则

> **时间邻近永远不是关联依据。**

这条从 v1 继承（`vendors/ascend/trace.py` 文件头、`implementation_7.md`），在 v2 升级为全局硬约束：

1. 关联必须由**显式 id**建立：框架分配的 corr、厂商 profiler 的 correlation/connection_id、flow 事件的起止端点配对、通信域内的序号。
2. 时间区间只能用于**排除**候选（候选 kernel 的设备时间若早于 launch 的主机时间，可排除），不能用于**选择**候选。排除本身也有代价，因此受三条附加约束：
   - 必须带**显式的保守容差**（规则中写死并记录在 `link.evidence`）。时钟换算有残差，零容差的排除会把正确候选剔掉，结果是 `unlinked` 而不是 `ambiguous` —— **误排除比歧义更危险，因为它既丢了数据又不自知**。
   - 每条规则的排除次数必须计数并进 coverage；某规则排除率异常升高即是换算或容差出了问题的信号。
   - 事件的 `ts` 为 null 时（时钟换算不成立，[03 §5.2](03-event-schema.md)），**连排除也不允许**。
3. 设备 kernel 的时间**不必**落在主机 marker 窗口内（`implementation_14.md` 的实测结论）—— 异步执行下这很正常，因此"窗口包含"也不是关联依据。
4. 候选多于一个且无法用 id 判别 → `ambiguous`，**不猜**。
5. 关联率是报告的一等公民，不是脚注。

违反这条原则的代价不是"数字略有偏差"，而是"给 FlagGems 开发组一个错误的算子耗时归因"，比没有数字更糟。

## 2. 关联状态

每个需要跨层关联的事件（主要是 `kernel`、`memop`、`comm.device`）带 `link` 对象：

| `status` | 含义 | 分析器的处理 |
|---|---|---|
| `linked` | 唯一确定地关联到一个 corr | 正常归因 |
| `ambiguous` | 候选 ≥ 2 且无法判别（含「一个候选 + 一条归属未知的 launch」，§3.6） | 计入"未归因"，在 coverage 中单列；可在报告中展示候选集 |
| `unlinked` | 无候选 | 计入"未归因"，单列 |
| `not_applicable` | 本来就不该有关联（如设备侧后台 kernel、profiler 自身开销），**只能由 registry 白名单判定** | 不计入未归因分母 |

`link.via` 记录使用的规则 id，便于排查：`cann.flow` / `cann.connection_id` / `cann.enqueue_corr` / `cuda.cupti_correlation` / `cuda.external_id` / `kname.compile` / `comm.group_seq`。

**关联指标先用集合定义。** 对同一 run、同一设备事件类别 k ∈ {`compute`, `comm`, `memcpy`, `unknown`}：

| 集合 | 定义 |
|---|---|
| R_k | **原始清单**：导入器**独立于 linker**，从厂商原始产物逐条枚举的设备记录（按 task type / kernel 记录计数，不经过任何关联筛选）。落盘在 `traces/<trace_uid>.json` 的 `inventory`（[03 §6.2](03-event-schema.md#62-manifest-与-writer-片段谁写哪个文件)） |
| E_k | 已导入为 `kernel` / `memop` / `comm.device` 事件的记录 |
| X_k | E_k 中命中 registry 白名单的记录（`not_applicable`） |
| A_k = E_k − X_k | 应当归因的记录 |
| L_k | A_k 中 `status=linked` 的记录 |

记 d(e) 为事件 e 的有效设备时长（[03 §4.8](03-event-schema.md) 的 `dur`）。

| 指标 | 定义 | 回答 |
|---|---|---|
| `linked_count_ratio` | \|L_k\| / \|A_k\| | 应当归因的设备事件，有多少条归到了唯一调用 |
| `linked_time_ratio` | Σd(L_k) / Σd(A_k) | 同一问题按设备时长加权 —— 少数未关联的长 kernel 比大量未关联的短 kernel 严重得多 |
| `import_completeness` | \|E_k\| / \|R_k\| | 原始设备记录有多少被导入。导入器无法独立枚举时为 `unknown` |
| `dur_missing` | A_k 中 `dur` 无效的条数 | 这些事件不进时间比的分子与分母，单独报出 |

规则：

- 分母为空时指标为 null —— 不写 0，也不写 1。
- 四类分别统计，不合并成一个总数。
- **时长求和不是墙钟占用。** Σd 是各事件时长之和，并发流上会重叠。它回答「多少设备工作量可归因」，不回答「设备有多少时间在忙」，后者由 C 级的忙闲区间给出（§6）。
- 采集范围另报 `profiled_window`（[07 §2](07-analyzers-and-reports.md)）：vendor profiler 开启窗口占 measure 阶段主机墙钟时间的比例。范围是**主机侧的时间窗事实**，不由设备事件反推 —— 设备时间落不落在主机标记窗口内，说明不了它属于哪个范围（§6 的 511 个晚启动 kernel）。

**原生关联指标（2026-10-14，[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)）。** 上面四个指标只衡量**项目归属**（`link`）。原生归属（§3.7）另有一组指标，与之并列报告，不复用上面的指标名，也不合并成一个比率（[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #25）。

对同一类别 k，沿用上面的 A_k：

| 集合 | 定义 |
|---|---|
| N_k | A_k 中原生归属唯一（§3.7 判定为"已归属"）的记录 |
| S_k | A_k 中有唯一模块作用域的记录 |

| 指标（名称为 L-字段 `1.0`，2026-10-27 起） | 定义 |
|---|---|
| `native_owned_count_ratio` / `native_owned_time_ratio` | \|N_k\| / \|A_k\|；Σd(N_k) / Σd(A_k) |
| `scope_count_ratio` / `scope_time_ratio` | \|S_k\| / \|A_k\|；Σd(S_k) / Σd(A_k) |

这组指标的规则：
- 空分母同样为 null。
- 同一条设备事件可以是"项目未归属、原生已归属"（B+ 级的常见情形）。报告必须把两组数并列展示，不得只展示其中一组。

**来源账本（[D24](DECISIONS.md#d24--导入契约身份分层来源账本完成标记证据索引)）。** 每条来源记录的处置（imported / source_only / unrepresentable / unknown）另行汇总，回答"原件里有什么、哪些进了 store、哪些没进以及为什么"。它与 R_k 的关系由 [附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #26 约束：账本不缩小 R_k；设备执行记录不得处置为 `source_only`。

> **更正（2026-10-11 第二轮，[D15](DECISIONS.md#d15--t1t3-之后的契约回填r1r8)）。** 上一版在此定义了 `marked_scope_ratio`（分母 = 本 run 全部设备事件时间，分子 = 落在标记范围内的设备事件时间），并用 Spike-C 的 46,145/89,312 = 51.7% 说明「整段 run 有近一半的设备时间没进过分母」。这有两处错误：
>
> 1. 这个比例是 **campaign 中 63 条 marked 记录与 57 条 plain 记录之间的计算 kernel 条数比**。plain 是不开标记的对照 run。它既不是单个 run 的量，也不是时间比（[T3 派生统计](history/history-references.md#ref-a58ea01614ea)）。
> 2. 「落在标记范围内的设备时间」本身就不可算：用设备时间是否落入主机窗口来判定范围，正是 §6 已经否定的做法。
>
> 「范围内 100% 不等于整体 100%」这条教训仍然成立；表达它的正确量是 `import_completeness` 与 `profiled_window`。

## 3. 五跳关联链

```text
       ┌────────────────────────────────────────── scope (span)
       │   H0: TLS scope 栈
       ▼
  [ op ] ──H1(TLS corr)──► [ route ] ──H2(TLS corr)──► [ compile ]
     │                         │                           │
     │ H3: 厂商 launch id       │                           │ H4: kname
     ▼                         ▼                           ▼
  [ kernel ] ◄───────────────────────────────────────────────┘
     ▲
     │ H5: (group, group_seq)
  [ comm ] 跨 rank
```

### 3.1 H0：op → scope（TLS scope 栈）

module 探针在 `nn.Module` 的 pre/post hook 中压栈/弹栈，栈存于 TLS。dispatch 探针写 `op` 时读栈顶 span 的 corr 填入 `op.scope`，并自增该 scope 的 `seq_in_scope`。

风险与处理：

- **线程边界**：TorchDispatchMode 与 TLS 栈都是线程局部的。若模型在多线程中前向（如数据并行的 Python 线程、DataLoader worker 中的预处理算子），子线程的 scope 栈为空 → `op.scope` 缺失。处理：探针在首次遇到"无 scope 的 dispatch"时写一条 `health(kind=info)` 记录该线程（**当前实现约定，[D46](DECISIONS.md#d46) 第 12 条**：`health.kind` 没有对应枚举值，P1 不写这条 health，改由分析器从缺 `scope` 的 `op` 加 `tid` 统计），分析器把这些算子归入 `scope=<unscoped:tid>`，并在 coverage 中报告无 scope 算子占比。**不跨线程继承栈**（继承会给出错误的归属）。
- **异步/协程**：同理，contextvars 可随协程传播，TLS 不行。v1 暂不处理协程边界，在 coverage 中暴露。（**当前实现（T9，2026-10-18）**：`flagprof/context.py` 的栈已按 asyncio Task 各自隔离，不跨 Task 继承；contextvars 传播仍未采用。）

### 3.2 H1/H2：op → route → compile（TLS corr）

dispatch 探针在进入算子时分配 corr 并压入 TLS（`flagprof/context.py`）。FlagGems 的路由包装器（route 探针）与编译后端钩子（compile 探针）直接读取当前 corr。

- 嵌套：算子库内部再次触发 dispatch 时，TLS corr 保持为**最外层**算子的 corr，内层 `op` 事件另分配自己的 corr，并在 **`op.parent_corr`**（[03 §4.3](03-event-schema.md)，顶层为 `0`）中记录外层 corr。v1 `GemsRoute` 的 `depth` 计数迁移为 `route.depth`。
- compile 可能发生在首次调用（lazy JIT），也可能在预热期独立触发 —— 后者 `corr=0`，靠 H4 的 kname 关联。

这两跳是**纯 Python 内**的，无跨语言问题，可靠性高。P1 即可交付。

### 3.3 H3：op → kernel（host → device）**最难的一跳**

这是 Spike-C 的核心。两家厂商的机制不同：

#### NVIDIA（参考实现，已在 v1 验证）

CUPTI 为每次 kernel launch 分配 `correlationId`，runtime API 记录与 concurrent kernel 记录共享它。Kineto 还提供 `External id`，可由 `torch.profiler.record_function` 注入。

规则：
1. 主要：`cuda.correlation_id` —— runtime launch 记录 ⟷ kernel 记录，一对一。
2. 把主机侧 launch 记录按线程与时间区间归到我们的 `op`（launch 发生在 op 的 dispatch 区间内，**同一线程**）。这里用时间区间是合法的，因为是**同线程内的嵌套关系**而非跨异步边界的猜测。
3. 兜底：`External id`（我们在 op 探针中注入 `record_function`）。

#### CANN（Spike-C 已验证，2026-10-10）

v1 只验证到**模块级**（`vendors/ascend/trace.py:attribute`）：flow 事件的起止端点配对 + `Node@launch` 的 `connection_id`。[Spike-C](history/11-spike-cann-correlation.md) 把三个未知项逐一测掉：

| 原未知项 | 实测结论 |
|---|---|
| FlagGems 的 **Triton kernel**（非 aclnn）是否也走 flow/connection_id | **是**，与 aclnn 同一套机制，M1/M2/M3 均闭合 |
| `TASK_QUEUE_ENABLE=0` 之外的配置是否仍成立 | **成立**，queue=1 时经 `Enqueue(correlation_id) → Dequeue` 桥跨线程，8,016 个 kernel 全部关联；v1 的强制 `=0` 因此取消（[D13](DECISIONS.md#d13--工具不改变被测系统的运行配置)） |
| 多流、多设备下 `connection_id` 的唯一性范围 | **仅导出会话内局部**：它由 CANN decoder 自增合成，跨 rank 裸值大量重复（TP2 实测两 rank 交集 300/300），跨 trace 必须先隔离命名空间（[03 §5.1](03-event-schema.md)） |

仍未验证（不得外推）：8 卡、长稳下的 ID 回绕、事件丢失、图回放、多 Python 线程。

候选规则（**全部规则取并集**，见 §3.6；可用性已由 Spike-C 逐条判定）：

| 规则 id | 依据 | v1 验证状态 |
|---|---|---|
| `cann.flow` | profiler 导出的 flow 事件 `s`/`f` 端点配对 | 模块级已验证 |
| `cann.connection_id` | `Node@launch` 与设备事件的 `connection_id` 相等。**只在同一导出会话内成立**，由 decoder 自增合成（[03 §5.1](03-event-schema.md#51-厂商原生-id-的命名空间)） | 模块级已验证 |
| `cann.enqueue_corr` | Enqueue/Dequeue 记录的 `correlation_id`（HCCL 链路上已用） | 通信路径已验证 |
| `cann.acl_api_ts` | ACL API 记录与设备任务的显式 id（若 CANN 9.0 暴露） | **未知** |

**若全部不可用**：降级方案见 §6。

### 3.4 H4：compile → kernel（kname）

Triton/FlagTree 产出的 kernel 有确定的符号名。`compile.kname` 与 `kernel.name` 做精确匹配可以把设备 kernel 归到编译产物，进而归到 FlagGems 的某个实现（R37、R45）。

注意：
- 这是**多对多**关系（一次编译产出的 kernel 被调用 N 次），只能回答"这个 kernel 来自哪个编译单元"，**不能**替代 H3 回答"这次 kernel 属于哪次 op 调用"。
- 名字可能带后缀（哈希、特化参数）。匹配规则按后端在 `vendors/` 中实现，必须是**精确或规范化后精确**，不做前缀模糊匹配。
- 如果 H3 失败而 H4 成功，结果是"算子级聚合可归因，单次调用不可归因" —— 这对应 §6 的能力位 `kname_aggregate`，不改变事件的级别。

### 3.5 H5：comm 跨 rank 对齐

```text
rank0: comm(phase=enqueue, group=G, group_seq=17, corr=A) ──► comm(phase=device, corr=A) ──► kernel(ktype=comm)
rank1: comm(phase=enqueue, group=G, group_seq=17, corr=B) ──► comm(phase=device, corr=B)
          └──────────── 同一次集合通信：匹配键 (G, 17, op) ────────────┘
```

- `group` 按 [03 §4.6](03-event-schema.md) 取 `hash(成员 rank 列表 ‖ 通信域创建序号)`，各 rank 独立计算结果一致。只用成员列表哈希会把成员相同的不同通信域折叠成一个。
- `group_seq` 是各 rank 在该通信域内的调用计数器 —— 集合通信语义保证各 rank 调用顺序一致，因此序号对齐是**语义保证**而非时间猜测。
- 若某 rank 的 `group_seq` 与其他 rank 不一致（出现缺口），说明发生了不对称调用或丢事件 → 标 `ambiguous` 并生成一条 `health(kind=comm_timeout)` 候选证据（这恰好是检测挂死的有力信号，见 [08 §2](08-online-and-fault.md)）。
- CANN 侧 enqueue→device 的链路继承 v1 `implementation_3.md §4`：marker → Enqueue/Dequeue `correlation_id` → `Node@launch` → `connection_id` → 设备事件。
- 分析器可据此计算各 rank 的到达时间差（等待时间归因），这是**跨 rank 的时钟对齐问题**，见 §4。

### 3.6 linker 算法：候选并集与判定

H3 的各条规则不是「按优先级找到第一条成立的」，而是**每条都算、结果取并集**。[T1](history/history-references.md#ref-867dfc27c4dd) 的 X7 证明了前一种做法的问题：人为让两个不同调用的 launch 共享同一个 `connection_id` 后，spike 的 linker 因为「候选不唯一」整体跳过了 connection 规则，flow 规则先成功，冲突就此消失，结果报成 `linked`。

对每个设备事件 e：

1. **白名单**：e 命中厂商 registry 的白名单 → `status=not_applicable`、`reason=excluded`，结束。linker 不得自行推断（[03 §4.8](03-event-schema.md)）。
2. **找 launch**：对每条适用的设备侧规则（flow 端点、`connection_id`、CUPTI `correlationId`……），在**同一命名空间**（[03 §5.1](03-event-schema.md#51-厂商原生-id-的命名空间)）内找出与 e 显式键相等的**全部** launch 记录。一个键命中多条 launch 时全部保留，不因「不唯一」而跳过。结果记为 launch 集合 S(e)。
3. **找归属**：对 S(e) 中的每条 launch，按全部适用的主机侧规则（同线程严格嵌套、Enqueue/Dequeue 的 `correlation_id` 桥、External id……）求出它的 op 候选，同样全部保留。没有任何候选的 launch 记为「归属未知」。
4. **合并**：C(e) = S(e) 中所有 launch 的 op 候选的并集，按 corr 去重；指向同一 corr 的多条路径合并进 `via`，互为佐证。U(e) = S(e) 中是否存在归属未知的 launch。
5. **判定**：

| 条件 | `status` | `reason` | `candidates` |
|---|---|---|---|
| \|C\| ≥ 2 | `ambiguous` | `conflict` | C 升序；U 为真时末尾追加 `null` |
| \|C\| = 1 且 U | `ambiguous` | `conflict` | `[c, null]` |
| \|C\| = 1 且非 U | `linked`（`corr = c`） | — | — |
| \|C\| = 0，S 非空 | `unlinked` | `no_bridge`；能确定归属所需的主机事件未导入或已丢弃时为 `source_missing` | — |
| \|C\| = 0，S 为空 | `unlinked` | `no_device_key`；能确定 e 带键、但对应的 launch 记录未导入或已丢弃时为 `source_missing` | — |

`via`、`candidates` 一律升序写出；`evidence` 列出 S(e) 中各 launch 及其命中的主机记录的身份（[03 §6.5](03-event-schema.md)）。

6. **时间只做排除，且在合并之前。** 规则可以用时间排除候选（如 launch 晚于设备事件开始），但须带保守容差、计数（§1）；e 或 launch 的 `ts` 为 null 时**不做排除**。

**\|C\| = 1 且 U 为什么不是 `linked`**：存在一条与 e 共享显式键、却不知道属于谁的 launch，它可能属于另一个调用。此时把 e 归给 c，就是在证据不足时选了一个答案。

**两段状态由此派生**（§6）：`device_link = ok` 当且仅当 S(e) 非空；`host_attr = ok` 当且仅当 `device_link = ok`、\|C\| = 1 且非 U —— 也就是恰好 `status = linked`。

**结果与规则顺序无关。** 打乱规则的检查顺序、打乱输入事件的顺序，输出必须逐字节相同。这是对「优先级不决定结果」的机械检验。

### 3.7 原生归属（原生调用关系）

> 2026-10-14 新增（[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)，[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #25）。

**为什么需要**

§3.6 的 linker 只把设备事件归到项目 corr。没有项目生产者时（只开官方 profiler、没有 marker），设备事件的 `link` 停在 B+（`device_link=ok`、`host_attr=broken`），但官方 profiler 自己已经记录了 launch 属于哪个原生调用。

2026-10 的实验中，腾昇、A100、P800、PPU 四家计算类设备记录的原生调用关系，都可以由显式 id 或同线程嵌套证明；而项目 corr 为 0。原生归属把这段已经存在的关系以证据形式保存下来，**不改变 `link` 的任何字段**。

上面是**历史格式**上的结论，不跨采集条件继承（[05 §4.1](05-probes-and-vendors.md#41-能力矩阵)）。T8.1 中，腾昇当前格式（torch_npu 2.10.0）的设备 → flow、Dequeue → Enqueue、Enqueue → cpu_op 三跳证据都在，原生归属仍为 0，原因见下文"前提：主机进程身份"。

**输入**

来自同一次导入的规范化结果：
- 设备记录；
- launch 记录。S(e) 与 §3.6 第 2 步完全相同，由各厂商规则中 `side=device` 的部分求出；
- 原生主机记录：原生调用（如 cpu_op）、作用域注释、Enqueue/Dequeue 等。

**算法**（对每个不在白名单中的设备事件 e）

1. **找 launch**：S(e)，同 §3.6 第 2 步。
2. **找原生 owner**：对 S(e) 中每条 launch，按全部适用的原生主机侧规则求原生调用候选，全部保留。
   - **显式 id**：如 launch 的 `External id` 指向的原生调用，命名空间按 [03 §5.1](03-event-schema.md#51-厂商原生-id-的命名空间)（同一 trace、同一主机进程）。CANN 的 Enqueue/Dequeue 桥同理。
   - **同来源、同线程的严格嵌套**：取**最内层**的原生调用。只有已被证明的嵌套链才能去掉祖先；候选之间部分重叠或起止相同、无法判定谁在内层时，全部保留。
   - **作用域注释**（如 `record_function` 区间、模块锚点）**永远不作 owner**。
3. **合并与判定**：
   - C_n(e) 为候选的并集，按 `(trace_uid, loc)` 去重；U_n(e) 表示 S(e) 中是否存在没有任何原生候选的 launch。
   - 判定与 §3.6 第 5 步的表同构：|C_n| ≥ 2，或 |C_n| = 1 且 U_n → 歧义；|C_n| = 1 且非 U_n → 已归属；|C_n| = 0 → 未归属（区分 S 是否为空）。
   - 状态名不得与 `link.status` 重名，具体名称属 L-字段（T8 Gate 1）。
4. **直接边**：来源直接给出"原生调用 → 设备记录"的关系（不经 launch，如 PPU 的 External id）时，记为直接边并单独标注，**不补造 launch 节点**。
5. **作用域链**：
   - 已归属原生调用的祖先（同线程严格嵌套的外层原生调用与作用域注释）作为作用域链输出。
   - 模块作用域由可配置的注释前缀或类别识别，**不得**硬编码模型结构名（如只认 `layers.N`）。
6. **时间**：与 §1、§3.6 第 6 步相同。时间只用于同一来源线程内的嵌套判断，不做跨来源比较，不按最近时间选择。

**前提：主机进程身份**（2026-10-15，[D27](DECISIONS.md#d27--格式解释的身份覆盖全部钩子兼容验收到连接键)）

第 2 步的同线程嵌套按 `(trace, 主机进程, 线程)` 分组，所以"同一线程"成立的前提是两条记录的主机进程身份已经按同一命名空间解释（[03 §5.1](03-event-schema.md#51-厂商原生-id-的命名空间)）。厂商格式可能对进程号编码，例如腾昇的 HostToDevice flow 起点：

```text
Dequeue (pid 1, tid 102)  ⊃?  flow 起点 (pid 1279 = (1 << 10) | 255, tid 102)  ─flow id─►  设备记录
```

- 进程号的还原属于厂商格式解释（`SourceFormat.host_pid`），必须有格式依据，并计入格式指纹；
- 依据不足或可能冲突时不还原，结果保持未归属。**不得**因为线程号相同、时间区间包含，就跨进程组拼接；
- T8.1 中，腾昇当前采集没有编码进程的进程名（原因是采集会话未声明 profiler level，torch_npu 默认 Level0 把 `CANN` 记录裁掉，见 [D28](DECISIONS.md#d28--格式适用范围显式化采集配置入身份导入判定适用性静默失败转显式) 背景），还原条件不成立，三窗共 30,360 条设备记录全部为 `no_bridge`。
- T8.2（2026-10-15 Gate 通过）以腾昇格式 v3 解决：依据为镜像内 CANN 9.0.0 的编码写法 `(pid << 10) | (layer << 5) | device`（CANN 主机 layer = 7、device = 31，低 10 位 = 255）与 37 份样本的结构一致；只在低位正确、元数据无冲突、原 pid 上没有真实主机记录、目标为已登记的 Python 进程时还原。三窗各 10,120 条变为 owned。
- **"保留原值"的含义**：原始 pid 保留在来源与规范记录中作为**证据**，**不作为连接键**。无法确认的记录使用一个不可能等于任何真实 pid 的内部隔离键；否则一条同号的完整竞争链会把设备工作抢走（T8.2 竞争链测试关掉保护后确实得到错误的 owned）。
- **owner 的层级**：owner 是严格嵌套链中**最内层**的原生调用。腾昇上通常是 `aclnn*` 一层（外层才是 `aten::*`），A100 由 NVIDIA 原生规则决定；两家 owner 不在同一抽象层。消费方必须声明按哪一层聚合（07 §1 perf.op），不得按 owner 名称直接横比两家芯片。

**输出**

每个设备事件对应一条原生关系记录，内容为：状态、owner（若唯一）、候选、launch 集合、直接边、作用域链、`via`、`evidence`，全部以 `(trace_uid, loc)` 引用。这条记录写进导入器的派生文件，不进入事件（D23 第 2 条）。

**与 §3.6、§6 的关系**

- `link.status`、`segments` 与级别（§6）不受原生归属影响。B+ 级设备事件可以同时有一个已归属的原生 owner：报告应同时展示"项目归属：无"与"原生归属：某原生调用"，而不是只展示 `unlinked`。
- 原生 owner 能回答"由哪个原生调用发起、在哪个作用域内"，不能回答"实际走了哪个 FlagGems 实现、为何回退"。后者需要 route 事件（§3.2）。
- 原生 owner 不能升级为项目 corr。要得到项目归属，必须有真实的项目生产者（§3.3）。

**必须保持的反例行为**

| 输入 | 预期 |
|---|---|
| 删除必要的 launch | 断链 |
| 同一设备记录得到两个不同的原生候选 | 歧义 |
| 已知候选 + 归属未知的 launch | 歧义 |
| 同值 id 跨 trace / 跨 capture | 不关联 |
| 打乱规则或输入顺序 | 输出逐字节不变 |
| 设备时间整体平移 | 不变 |
| 剥离全部原生调用记录 | 无 owner |
| 作用域注释与原生调用起止相同 | 注释不作 owner |
| launch 的主机进程号未还原，与原生调用不在同一进程组（T8.1 腾昇当前格式） | 未归属（`no_bridge`）；不按线程号或时间跨进程组拼接 |
| 编码进程号还原后与另一个真实主机进程冲突 | 不还原，保持未归属或歧义 |
| 编码进程号与另一个真实主机进程同号，且该进程有一条完整的竞争链（T8.2） | 隔离，不归属给同号进程；保持未归属 |

### 3.8 P1 设计输入：两跳归属与连接键域（借鉴 nsys，2026-10-16）

> **状态（2026-10-27，[D46](DECISIONS.md#d46)）**：P1 首批用项目 module 探针加 each 窗口内的项目桥接，做设备时间到层的归属（kernel.link → op.scope → span 树）。第 1 点的原生两跳路径、第 3 点“消费官方 profiler 的模块作用域”、第 2、5、6 点都转 P2。第 4 点：T10.1 不改 linker，桥接靠带前缀的 marker 事件名与同线程包含关系（腾昇另经 Enqueue/Dequeue 桥），不用流事件。
>
> **原状态**：设计输入，不是规范；**规范细节在 P1 任务单定稿**。来源：[T8.3 返工复审 §6.3](history/history-references.md#ref-fda8bbc0b9b5)。nsys 的做法来自官方文档（本机没有 nsys）；"本机"数据来自设计者会话对 A100 与腾昇原件的实测。

§3.7 的作用域链由同线程严格嵌套得到。P1 要把设备工作归到作用域（cpu_op、user_annotation、module）时，借鉴 nsys 的 GPU 区间投影（`nvtx_gpu_proj_*`）：

1. **两跳规则**：kernel →（correlation，精确 ID）→ 发起它的主机调用 →（同一线程的调用栈包含）→ cpu_op、user_annotation、module。**设备跳只用精确 ID，主机跳只用同线程调用栈包含。** nsys 的规则是：GPU 操作属于某区间，当且仅当发起它的 CUDA API 调用位于同线程的该区间内；不按 kernel 执行时间与区间重叠；嵌套区间都包含。本机 A100 窗口上，两跳结果与 `External id` 逐条一致（9,260/9,260）。这不违反 §1"关联绝不单用时间邻近"：设备跳只用 ID，主机跳是同线程的栈嵌套，与 §3.7 第 6 步一致。
2. **graph replay 无构建期证据时记不可归属（unattributable）。** nsys 的 graph 在构建期归属；我们没有构建期证据时，不按执行时间重叠去猜。具体状态名与 §5 现有的 `reason=graph_replay` 的关系，在 P1 任务单定稿。
3. **模块覆盖为 0 是采集配置缺口，不是匹配算法缺口。** 本机 A100 窗口没有 `python_function`、`module` 事件，因为 `with_stack`、`with_modules` 未开。T9 的 `record` 提供带模块作用域的采集档（声明进身份，开销进 Q9），P1 再消费。
4. **ac2g 流事件不作连接键。** 本机 A100 窗口里 ac2g 流事件的 `f` 比 `s` 多 684，所以不用流事件，用 `args.correlation`。P1 须核对 linker 是否依赖流事件。
5. **连接键带域。** nsys 的 `globalTid` = HW:8 / VM:8 / PID:24 / TID:24，用硬件与 VM 位避免碰撞。Kineto 的设备 pid 是设备序号、主机 pid 是 OS pid，落在同一个整数空间。真实数据目前无碰撞：Kineto 26 份的设备 pid 全为 0，主机 pid 最小 36，A100 当前窗为 9594。但腾昇 41 份中有 30 份主机 pid = 1（容器内 PID 1 采集是本项目常态，当前窗 55、56 即是），将来在容器里用 PID 1 采集多卡 Kineto 时，设备序号 1 会与主机 pid 1 重合；该条件现在没有任何测试（复审变异 M-n）。T9.0 §0.4 补碰撞条件测试；多设备阶段把连接键改为带域的（pid 命名空间，device，correlation）。与 [05 §4.1](05-probes-and-vendors.md#41-能力矩阵) 的 C7 语义域一致：同一个数不等于同一个键。
6. **torch_npu 自带的 db 作为 X17 的评估对象。** torch_npu 自带 `ascend_pytorch_profiler.db`（`SCHEMA_VERSION` 1.2.0），与 nsys SQLite 近乎同构（`TASK`、`CANN_API`、`CONNECTION_IDS`、`PYTORCH_API`，以 connectionId 相连）。本机：`TASK` 10,685 行的 connectionId 100% 命中 `CANN_API`；`PYTORCH_API` 中有 connectionId 的 27,670 行 100% 命中 `CONNECTION_IDS`。P1 评估把 db 作为结构化导入源之一，即既有的 [X17](00-requirements.md#6-需求池参考文档补充)"JSON / 官方 DB 对照"；T9 不做。

以上六点的规范细节在 P1 任务单定稿；在此之前不改 §3.3、§3.7 的现行规则。

## 4. 时钟

| 域 | 时钟 | 对齐方式 |
|---|---|---|
| 同进程内 | `CLOCK_MONOTONIC_RAW` | 天然一致 |
| 同主机跨进程 | 同上 | 天然一致 |
| 主机 ↔ 设备 | 厂商 profiler 自带换算 | 导入器负责换算到主机单调时钟，换算参数写入 `kernel.ext` |
| 跨主机（多节点） | 无共同时钟 | **不做跨节点时间对齐**。跨节点只用 `(group, group_seq)` 的逻辑对齐；若要报跨节点等待时间，必须先用 barrier 标定并标注不确定度 |

`meta.clock` 在 run 首尾各采一次并在运行期间按周期（默认 60 s）追加 `(mono_ns, wall_ns)`（[03 §4.1](03-event-schema.md)）。离线换算按**最近的两个标定点分段插值**，残差写入 `clock_transform`；若相邻标定点推出的频率偏差超过阈值，分析器在 coverage 中标注时钟漂移警告并指出漂移发生的**区间**，而不只是报告整段平均。

## 5. 已知的关联盲区

| 盲区 | 成因 | 首版处理 |
|---|---|---|
| **图回放**（CUDA Graph / ACL Graph / torch.compile 的 inductor 图） | 回放时不经过 Python dispatch，`op` 事件不存在 | 捕获期的 op 正常记录；回放期由 vendor 适配层发 `span(cat=engine, name=graph_replay)`；期间的 kernel 记 `link.status=unlinked`、`reason=graph_replay`，**仍计入分母**，分析器可按该 span 聚合为"图内设备时间"并在 coverage 中单列其占比。**首版不承诺图内算子级归因**。上一版在这里写的是 `not_applicable`，那会把图内的真实计算整块移出分母，与 [03 §4.8](03-event-schema.md) 的白名单规则冲突 |
| 后台线程算子 | TLS scope/corr 不跨线程 | §3.1 的 `<unscoped:tid>` |
| C++ 层直接发起的 kernel（不经 ATen） | 无 op 事件 | `unlinked`，计入未归因 |
| profiler 自身的 kernel/内存开销 | — | `not_applicable`（由厂商 registry 白名单识别） |
| 多流并发下的 `ambiguous` | 同一 corr 对应多 kernel 是正常的（一个 op 多个 kernel）；反向多对一才是问题 | 一对多是合法的 `linked`；多 corr 争抢同一 kernel 才标 `ambiguous` |

## 6. 降级与可信度：两段链路，四个级别

（原标题"若 H3 在 CANN 上不可行：三级降级"已不适用：[Spike-C](history/11-spike-cann-correlation.md) 已证明 H3 在 CANN 上可行；本节现在定义的是**各种失效组合下仍能说什么**，而非备用方案。）

**四个级别都必须能产出对 FlagGems 开发组有用的结论**，区别只在精细度与可归因的维度。

关联链是**两段**：op→launch（`host_attr`）与 launch→设备事件（`device_link`）。**两段状态与级别都是逐条判定的**，按 §3.6 派生，记录在每个设备事件的 `link.segments` 中：

| `host_attr` | `device_link` | 级别 | 所属对象 | 对应的 `status` | 能回答 | 不能回答 |
|---|---|---|---|---|---|---|
| ok | ok | **A 完整** | 设备事件 | `linked` | 单次 op 调用的设备耗时；按 op×sig×route 的精确设备时间；瓶颈定位到具体调用 | — |
| broken | ok | **B+ 设备侧** | 设备事件 | `ambiguous`，或 `unlinked` 且找到了 launch | kernel 级设备耗时、stream/task 归属、设备时间轴完整 | 该 kernel 属于哪一次 op 调用 |
| ok | broken | **B 主机侧** | **launch 记录** | — | 主机侧 op/层耗时完整、路由与编译链完整 | 设备耗时的**任何**调用级或层级归属 |
| broken | broken | **C 区间** | 设备事件 | `unlinked` 且没找到 launch | 设备时间轴的总量与忙闲 | 任何算子级或层级的设备耗时归属 |

- **设备事件只有 A / B+ / C 三种。** `host_attr` 说的是「这个设备事件的 launch 归属唯一」，没找到 launch 就谈不上归属，所以设备事件上不会出现 ok/broken。
- **B 是主机侧 launch 记录的级别**：launch 的归属唯一，但没有匹配到任何设备事件 —— 设备 trace 未导入、被截断、或设备侧键缺失时出现。run 级报告 `launch_attr_ratio`（归属唯一的 launch / 全部 launch）与 launch 记录按两段的分布。
- `not_applicable` 的事件没有级别，不进任何分布。

**run 不贴单一级别标签。** 同一个 run 里不同事件的级别可以不同：[T1](history/history-references.md#ref-867dfc27c4dd) 的 X1 在 queue=1 的 M3 片段上去掉 queue 桥之后，仍有 1,316 个 kernel 经直接路径归属，另 572 个经队列的 kernel 降为未关联。给整个 run 贴 A 会掩盖局部断链，贴 B+ 会丢掉本来完整的部分。run 级输出的是**级别分布**：每个事件类别下各级别的条数与设备时长（`level_distribution`，[07 §2](07-analyzers-and-reports.md)）。分析器按自己实际用到的那些事件的级别，决定能说什么。

> **修订（2026-10-11 第二轮，[D12 修订](DECISIONS.md#d12--关联降级是二维的主机归因--设备链接)）**：上一版把级别写成 run 级的单一标签，[03](03-event-schema.md) 又允许每段取 `not_applicable`。这留下两个漏洞：两个候选各自都有完整链时，「两段都存在」会把 ambiguous 包装成 A；混合路径的 run 该贴哪个标签无从定义。现在 ok 要求归属唯一，级别逐事件判定，run 级只报分布。

H4（kname → 编译单元）**不参与级别判定**，它是一个独立的能力位 `kname_aggregate: true/false`：

| | 作用 |
|---|---|
| `kname_aggregate=true` | 在 B+ / C 级的事件上额外提供「kernel → 编译单元 → FlagGems 实现」的**设备时间聚合**，支持算子级性能对比与回退收益评估 |
| 仍不能提供 | 单次调用归因；同名 kernel 在不同 scope 的区分；同名多特化的区分 |

> **为什么要把 H4 拆出来**：更早的版本把「device_link broken 但 kname 可用」直接写成级别 B，而 `host_attr` 列写成「任意」，结果 `broken/broken + kname` 同时满足 B 行和 C 行 —— 级别不再是两段状态的函数，同一份数据可以被判成两个级别。拆成「级别 + 能力位」后判定唯一。

> **为什么必须分两段**：`TASK_QUEUE_ENABLE=1` 且未接 queue 桥时，经队列的设备事件正好落在 B+ —— 设备侧数据完全可用，只是上不到 op。一维级别会把它与「两段全断」混为一谈，白白丢掉本来可用的设备侧结论。

降级不是静默发生的：级别分布、`launch_attr_ratio` 与 `kname_aggregate` 写入每个性能分析器的 `coverage`（[07 §2](07-analyzers-and-reports.md)），报告首屏即展示。

> **C 级不得按主机 span 窗口切分设备时间。** Spike-C 的 20 份 fixture 中有 **511 个已关联 kernel 在其主机 marker 结束之后才开始执行**。异步下这是正常现象（见 §5），不说明关联错；但它说明按主机窗口分配设备时间会把异步积压的工作错记到后面的层。因此 C 级只输出"该区间内设备的总忙闲"，**不输出层级设备耗时**；缺少归属证据时，层成本必须标记为不可获得，而不是给一个有数字的错误答案。

## 7. 实现要点（给 codex）

1. `flagprof/context.py` 提供：`alloc_corr()`、`current_corr()`、`push_scope(name)`、`pop_scope(token)`、`current_scope()`。全部基于 `threading.local`。**不要**用全局字典 + 锁（竞争会污染性能数据）。
   > **当前实现（T9，2026-10-18）**：`push_scope(name)` 返回 token，`pop_scope(token)` 必须按 **LIFO** 弹出自己的 token，否则抛 `ValueError`（此前文档写作 `push_scope(span)/pop_scope()`，已按实现更正；不允许无参弹出）。`Context` 是显式对象（`Context(writer)`，由 `attach()` 创建），线程局部栈按 asyncio Task 各自独立，fork 后失效；细节见 [05 §1](05-probes-and-vendors.md#1-探针的定义与硬约束)。
2. `correlate/linker.py` 是**离线**的：输入事件流与厂商规则表，输出带 `link` 字段的 kernel/memop/comm 事件。它由导入器在导入时调用，`link` 随设备事件一次写出（[D18](DECISIONS.md#d18--关联在导入时完成厂商主机侧记录不转成-store-事件)）。规则以可插拔函数组织，每条声明 `via` id 与所需的命名空间；linker 按 §3.6 对全部规则取并集。**规则优先级只决定检查顺序，不决定结果。**
3. 规则必须是**确定性**的：同样的输入产生同样的 link 结果，与规则顺序、输入顺序无关。禁止任何依赖遍历顺序或浮点时间比较容差的逻辑（时间只用于区间排除，容差写死在规则里并记录）。
4. linker 必须输出完整的统计（§2 的集合与指标、§6 的级别分布），且能 dump 未关联与歧义事件的样本供人工排查。原始清单 R_k 由导入器独立产出，不由 linker 的输入反推。
5. 所有厂商规则的单元测试用**录制的真实 trace 片段**（放 `tests/fixtures/`），不用人造数据 —— 人造数据会掩盖真实 trace 的字段怪癖。对真实片段做受控变异（如 T1 的 X1–X8）是允许的，变异本身必须可复现。
6. **输出契约就是门禁。** 任一设备事件缺 `status`、`segments`（`not_applicable` 除外）、`reason`（非 `linked` 时）或 `evidence`，回归测试即失败，而不是记一条告警。
7. **原生归属（§3.7）在新模块中实现**（`correlate/native.py`，厂商主机侧原生规则放在 `vendors/<v>/native_rules.py`）：
   - 只复用现有规则中 `side=device` 的部分求 S(e)；
   - 不调用 `link()`（它以 corr 为 owner 键）；
   - 不修改 T4 门禁锁定的文件（`linker.py`、`inventory.py`、`chrome.py`、`rules_common.py`、各厂商 `rules.py`）；
   - `kernel.link` 的输出逐字节不变（[T8](history/history-references.md#ref-24a77204cd82)）。

## 8. Spike-C 实测回填（2026-10-10；**已裁决，规范部分已并入正文**）

> **读法**：本节保留为**证据索引**，不再是独立的规范来源。下列第 2、3、6 条的规范内容已并入 [§6](#6-降级与可信度两段链路四个级别)（二维 segments，[D12](DECISIONS.md#d12--关联降级是二维的主机归因--设备链接)）与 [05 §4.1](05-probes-and-vendors.md)（不改被测配置，[D13](DECISIONS.md#d13--工具不改变被测系统的运行配置)）；第 4、5 条并入 [03 §4.8/§5.1/§5.2](03-event-schema.md)。正文与本节冲突时以正文为准。
>
> **口径**：下文"闭合 op→launch→kernel"指 campaign 中 63 条 marked 记录的 46,145 个计算 kernel；另 57 条 plain 记录（43,167 个）是不开标记的对照 run，不在关联统计内。46,145/89,312 = 51.7% 只是这两类记录之间的条数比，不是任何单个 run 的覆盖率（§2 更正）；M3 的形状回查另有 90 处不一致（4.8%），见 [DECISIONS Q7](DECISIONS.md#q7--spike-c-实测回填与冻结边界) 的口径修正。

[报告](history/history-references.md#ref-e1bd695b52c5)与[真实 fixtures](history/history-references.md#ref-975e3ad0fde3)支持以下范围内的修订建议，不自动构成全版本能力保证或 schema 冻结：

1. §3.3：CANN 9.0 / torch_npu 2.10 / FlagTree cendAs 下，M1/M2/M3 的 Triton 与 aclnn 均可闭合 op→launch→kernel；M2 覆盖 TASK_QUEUE_ENABLE=0/1 × TRITON_ENABLE_TASKQUEUE=false/true。
2. queue=1 时 Node 不直接嵌套在 op 所在线程，需 `Enqueue(correlation_id) → Dequeue → Node@launch → connection_id/HostToDevice → kernel`。不能因直接包含为 0 就判定整条链不可用，亦不必强制 queue=0。
3. §1/§3.3：允许主机同线程内的严格调用嵌套；跨线程/设备始终要求显式键。flow 与 connection 的候选冲突应保留 ambiguous，而非让优先级吞掉冲突。缺 ID 的真实片段测试必须仍为 unlinked。
4. §3.4：kname 不能唯一确定 compile.key，也不能替代重复调用的 corr。保存逻辑名、编译 key、二进制 SHA 和实际 entry 符号映射；不进行任意前缀模糊匹配。
5. §4：导出局部 ID、合成 pid 与时钟转换必须有来源/版本；保存 raw 值及 trace 命名空间，跨 rank 不裸拼 ID。
6. §6：A 级要求完整的 op→launch 与 launch→kernel 两段链路；11 §2 原有“H-A/H-B/H-C 任一成立”应修订。已测配置建议 `per_call`，同时输出实际覆盖率；未测配置仍按证据降级。
7. §3.5/§5：单卡真实多流已验证；TP2 两后端各三重复完成，跨 rank 的裸 connection ID 重用已实测；group/seq 600 对对齐通过，见报告 §7。8 卡、长稳 ID 回绕、事件丢失、图回放和多 Python 线程不能从本轮短 eager run 外推。

完整 via、输入、优先级、失效条件见报告 §5；待确认项见 [DECISIONS Q7](DECISIONS.md#q7--spike-c-实测回填与冻结边界)。
