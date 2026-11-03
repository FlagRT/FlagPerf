> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书出处见删除前固定 Git 提交中的 [历史引用索引](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md)，其出处保留。

# 03 事件 schema v1 与 store

> 2026-10-20 授权修订：[D34](DECISIONS.md#d34--t94t95-读取批次与测量边界补充)与 T9.4 M8/M9、T9.5 §4.7 补充公共读取校验、首批屏障、质量/身份和测量/关闭边界。P0 未退出；历史验收与旧状态保留。当前任务具体规则以两份修订任务单为准。


> 状态：**L-核心待 P0 开工冻结；L-字段草案**（分层规则见 §8 与 [D11](DECISIONS.md#d11--schema-分三层冻结不是一次性整体冻结)）
> 上游：[02 架构](02-architecture.md) P1 ｜ 下游：[04 关联](04-correlation.md)、[05 探针](05-probes-and-vendors.md)、[07 分析器](07-analyzers-and-reports.md)
> 决策依据：[DECISIONS](DECISIONS.md) D4（schema v1 一次性包含全部事件类型，不分期冻结）

本文件是全工程唯一的冻结契约。探针、分析器、报告都可以推倒重写，schema 不行。

## 1. 为什么是"一次冻结全部类型"

分期冻结的代价不对称：新增**字段**便宜（读取方忽略未知字段即可），新增**事件类型**也还行，但新增一类**关联语义**会反向污染已有事件的关联设计。`tensor`（数值）、`route`（路由）、`health`（故障）三者如果后补，就必须回头给 `op`/`kernel` 加关联字段并重新生成全部历史数据。因此 v1 一次性定义 13 类事件，允许某些类型在早期阶段"定义了但还没有探针产出"。

| 组 | 类型 | 首个产出阶段 |
|---|---|---|
| 基础设施 | `meta`、`sig` | P0 |
| 核心五类 | `span`、`op`（+`opsum`）、`kernel`、`comm`、`counter` | P1–P2 |
| 路由与编译 | `route`、`compile` | P1 |
| 数据搬运 | `memop` | P2 |
| 数值 | `tensor` | P4 |
| 故障 | `health` | P3 |

## 2. 事件信封

每条事件是一行 JSON 对象，**信封字段对所有类型相同**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `v` | str | schema 版本，`"1.0"`。只在每个分片的第一条 `meta` 事件中出现一次，其余事件省略 |
| `t` | str | 事件类型（`span`/`op`/…）。**必填** |
| `ts` | int \| null | 统一时间轴上的时间戳，纳秒。**在线探针**直接取单调时钟（Linux `CLOCK_MONOTONIC_RAW`），与墙钟的映射记在 `meta.clock`，**必填**。**离线导入的事件**（`kernel`/`memop`/`comm.device`）只有在**有来源的时钟换算成立**时才写本字段，原始值与换算依据保留在 `ext`（§5.2）；换算不成立时 `ts` 为 null，事件**照常写出**，并留在一切计数与分母中 |
| `clk` | str | `ts` 所属的时钟域，如 `host_mono:<boot_id>`。默认值由分片首条记录 `meta.shard_header` 声明（同下文 `pid` 等）。**只有同一 `clk` 的 `ts` 可以互相比较**：不同主机的 `CLOCK_MONOTONIC_RAW` 互不相干（[04 §4](04-correlation.md#4-时钟)）。离线导入的事件取**被测进程所在主机**的时钟域，而不是导入器所在主机的 |
| `seq` | int | 本 writer 内严格递增序号，用于检测丢弃。**由产生侧分配，且「取号」与「入队或丢弃计数」在同一个临界区内完成**：入队失败的事件也消耗序号。由此单个 writer 的**文件顺序就是 `seq` 顺序**，读取方见到跳变即可**确定**中间有丢弃，缺口不会被之后的记录补上（§6.3）。只取号、不同时入队是不够的：线程 A 取到 10 后被挂起、线程 B 取到 11 先落盘，读者看到的「缺 10」其实只是乱序。产生侧事件**必填**；控制记录不带，见下 |
| `pid` | int | 进程 id |
| `tid` | int | 线程 id（Linux `gettid`） |
| `rank` | int | 分布式 rank；非分布式为 `0` |
| `p` | str | 产出该事件的探针 id（如 `dispatch`、`vendor.ascend.trace`） |
| `corr` | int | 关联 id，见 §3。无关联语义的事件可省略 |
| `ext` | obj | 厂商原生 id 与非通用字段的唯一容身处，见 §5 |

`pid`/`tid`/`rank`/`p`/`clk`（以及 `op.dev`）在同一分片内恒定或高度重复 —— 分片首条记录 `meta.shard_header` 声明这些字段的默认值，单条事件中与默认值相同时可省略（读取方负责回填）。这把典型事件行压到 120–200 字节。

**两类记录**：

| | 由谁产生 | `seq` | 身份（§6.5） | 重复出现 |
|---|---|---|---|---|
| **产生侧事件** | 探针经队列交给 writer | 有 | `(run_id, writer_id, seq)`，分配后**永不改变** —— compact、排序、精简都不得重新编号 | 读取方按身份去重 |
| **控制记录** | writer 的写盘线程自己写：`sig`、`meta` 的 `shard_header`/`defaults`/`clock`/`drop_report` | 无，不参与缺口检测 | 按内容键（`sig` 以 `sid`，`drop_report` 以 writer + 分片序号） | 幂等：`sig` 内容相同，`drop_report` 是累计值 |

信封字段在控制记录上的适用范围：`sig` 只有 `t` 与 §4.2 的字段，**不带** `ts`/`clk`/`seq`/`corr`/`pid`/`tid`/`rank`/`p` —— 它是内容寻址的定义，不是发生在某一时刻的事件。`meta` 控制记录带 `ts`（写盘线程写出时取的单调时钟）与 `clk`，不带 `seq`。

**为什么用 JSONL 而不是二进制**：P0 阶段要的是可尾读、可人肉 grep、崩溃后仍可读。单位事件体积的代价由 §7 的 compact 阶段（转列式/压缩）偿还。一旦证明吞吐不够，可以在不改字段定义的前提下换编码（`meta.encoding` 字段已预留）。

## 3. corr：关联 id

```text
corr: u64 = [ writer_id : 16 bits ][ counter : 48 bits ]
```

- **writer = 一个写入进程实例**（进程内多线程共享一个 writer）。被测进程、它的 counter 进程（[05 §2.2](05-probes-and-vendors.md)）、离线导入器各自都是独立的 writer。
- **`writer_id` 在 run 内唯一、永不复用、与 `rank` 无关。** 同一 rank 上的被测进程和 counter 进程是两个 writer；进程重启后再次 attach 是一个**新的** writer。`writer_id` 在进程首次产生事件时分配：单机用 store 下 `writers/` 目录的文件锁取号，取号后立即原子创建自己的 writer 片段（§6.2），片段存在即占号；多机时高位取启动器给出的**节点序号**，低位用节点内文件锁取号。位宽划分属 L-字段，读取方把 `writer_id` 当作不透明值。
  > 上一版写的是「多机取 `rank`」。但 rank 不是进程实例的身份：同一 rank 本来就可能有被测进程和 counter 两个 writer，重启后也会复用同一 rank。用 rank 作 `writer_id` 会让 `corr` 高位、`seq` 的归属和分片路径同时冲突。`rank` 只作为事件的拓扑属性留在信封里（§2），不参与任何身份。
- `counter` 为该 writer 内的原子递增计数，由**产生事件的线程**自行原子加。
- `corr == 0` 表示「无关联」。
- `corr` 在 JSON 中是整数。导出到不能精确表示 64 位整数的环境（如浏览器 JS 的 Number，精确上限 2⁵³）时，**必须转成十进制字符串**：多机时 `writer_id` 高位是节点序号，`corr` 很容易超过 2⁵³，按数字解析会静默丢精度、把不同调用合成一个（[D17](DECISIONS.md#d17--sid-写为十六进制字符串)）。
- **attach 到已有 run**：`run_id` 由先到者创建，后到者读 `manifest.json` 加入，再按上一条取得新的 `writer_id`。多个进程 attach 到同一 run 合法；**同一进程被 attach 两次不合法**（会产生两套 corr 分配器），writer 须以进程锁拒绝第二次。
- **容量**：48 位 counter 在 10⁶ corr/s 下约 8.9 年不回绕，长稳运行（R33）安全。16 位 `writer_id` 上限 65,536，超出即报错，**不得回绕复用** —— 复用会让同一个 corr 值在一个 run 内指向两次不同的调用。

**分配规则**：corr 由**最外层知情者**分配并通过线程局部存储（TLS）向内传递。详见 [04 §3](04-correlation.md)。一条关联链上的 `op`、`route`、`compile` 事件共享同一 `corr`；`kernel` 事件则通过 **`link` 字段**（§4.8，单数）**引用** `corr` 而非共享，因为一次 op 可能对应多个 kernel。

**跨线程/跨进程**：TLS 不跨线程。若框架把算子派发到后台线程（如 ACL 的 task queue 线程），Python 层记录的 corr 会在线程边界处断开 —— 这正是 [Spike-C](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/11-spike-cann-correlation.md) 要测的核心问题。

## 4. 事件类型

下列字段表中，**粗体**为必填。

### 4.1 `meta` — 身份、环境、配置

每个分片的第一条事件，以及 run 期间任何环境变化（如模型热加载）时追加。

| 字段 | 说明 |
|---|---|
| **`kind`** | `run_start` / `shard_header` / `env` / `defaults` / `model` / `clock` / `run_end` / `drop_report` |
| `data` | 由 `kind` 决定的对象 |

关键 `kind`：

- `clock`：`{mono_ns, wall_ns, tsc_hz?, vendor_device_clock?}` —— 单调时钟与墙钟、与设备时钟的对齐点。**run 首尾各采一次，且运行期间按固定周期（默认 60 s）追加**。
  > 两点标定只能给出整段的平均漂移，既无法定位漂移发生在何时，也无法对长 run 做分段换算。R33 的长稳场景可能连续跑数日，设备时钟与主机时钟的相对频率并不恒定。周期标定让离线换算可以**按最近的两个标定点做分段插值**，并把插值残差写进 `clock_transform`（§5.2）。代价是每分钟两次取时，可忽略。
- `env`：包版本（torch、torch_npu/CUDA、triton、FlagGems、FlagTree、FlagCX）、**全量**环境变量、设备型号与固件、OS/内核。继承 v1 `runtime/worker.py:identity` 的采集口径。
  > **全量采集 + 显式脱敏，而不是白名单。** 白名单必然漏掉新厂商的关键开关，而 `identity` 正是 seal 模式判定「两次 run 可比」的依据（§6）—— 漏一个变量就是一次假阴性。脱敏按名字模式（`*TOKEN*` / `*SECRET*` / `*KEY*` / `*PASSWORD*` / `*_AK` / `*_SK` 等，模式表在 registry 中可扩展），被脱敏的变量**保留键名，值记为 `"<redacted:" + sha256 前 8 位 + ">"`**，使 diff 仍能发现「值变了」而不泄露值本身。
- `run_start`：除 `run_id` 与启动参数外，必须含**进程唯一键** `{boot_id, pid_ns, pid, proc_start_ticks}` 与 `heartbeat_period_ns`。
  > 容器下不同容器的 `pid` 会重复，§5.1 还指出厂商导出的 pid 可能是合成值 —— 裸 `pid` 不足以标识进程。`boot_id`（`/proc/sys/kernel/random/boot_id`）+ pid namespace + 进程启动 ticks（`/proc/<pid>/stat` 第 22 字段）三者合起来才唯一。`heartbeat_period_ns` 必须进 store：卡死判定在进程外做（[08 §2.1](08-online-and-fault.md)），判定方若不知道被测方**实际**用的周期，`k × period` 阈值就是错的。
- `shard_header`：每个分片的**第一条记录**，`data = {writer_id, shard_index, defaults}`；`v` 字段只出现在这里。
- `defaults`：分片中途默认值（`pid`/`tid`/`rank`/`p`/`clk`/`op.dev`）发生变化时追加（见 §2）。
- `drop_report`：writer 丢弃的事件数，按类型分组，**累计值**，并带 `next_seq`（写出时下一个待分配的序号）。**丢弃必须被记录**（N6）。写出时机：**每个分片封口时、run 结束时，以及运行中计数有变化时按周期（默认 10 s）**。
  > 只在结束时写，等于在最需要诊断的场景（崩溃、被 kill、OOM）下恰好写不出来 —— 分析器会把「丢了很多事件」误读成「本来就没有这些事件」。周期写出把崩溃时不可见的丢弃限制在最后一个周期内；剩下的部分只能报下界，规则见 §6.3。
- `model`：模型标识、权重哈希、版本号、热加载代次（R33）。

### 4.2 `sig` — 算子调用签名字典

`op` 事件如果每条都带完整 shape/dtype，500 模型 × 百万次调用会把 store 撑爆。签名进字典表，`op` 只引用 `sid`。

| 字段 | 说明 |
|---|---|
| **`sid`** | 签名 id：规范内容的 64 位哈希，写为 16 位小写十六进制字符串，算法见下文「`sid` 的计算」（**冻结**） |
| **`role`** | `in`（调用的输入签名，由 `op.sid` 引用）/ `out`（输出签名，由 `op.osid` 引用） |
| **`name`** | 算子名，如 `aten::mm` / `aten::scaled_dot_product_attention` |
| **`overload`** | overload 名，无则空串 |
| **`tree`** | 规范参数树，节点数组，见下文 |
| `trunc` | bool，默认 `false`。签名字典超出预算后写出的截断签名为 `true`，此时 `tree` 为空数组 |

**规范参数树。** `tree` 是按先序遍历排列的节点数组，每个节点为 `{p, k, ...}`：`p` 是参数路径，`k` 是节点类型。

| 路径 `p` | 含义 |
|---|---|
| `"0"`、`"1"` | 第 0、1 个位置参数 |
| `"1[0]"`、`"1[0][2]"` | list/tuple 的元素，可嵌套 |
| `"2{key}"` | dict 中键为 `key` 的值，键按字典序遍历 |
| `"kw:dim"` | 名为 `dim` 的关键字参数，关键字参数按名字排序 |
| `"r"`、`"r[0]"` | 返回值，只出现在 `role=out` 的签名中 |

| 类型 `k` | 其余字段 |
|---|---|
| `tensor` | `shape`、`dtype`、`layout`、`contiguous`；仅当 `contiguous=false` 时有 `stride` |
| `list` / `tuple` / `dict` | `n`（元素个数）；子元素作为后续节点出现 |
| `none` | 无 |
| `bool` / `int` / `str` | `v` |
| `float` | `v`，取 Python `repr()` 字符串，避免不同 JSON 库的浮点序列化差异 |
| `dtype` / `layout` / `memory_format` | `v`，规范名（如 `torch.float16`） |
| `device_obj` | 无。参数本身是 device 对象时只记类型，不记值（理由见下） |
| `other` | `type`，类型的限定名 |

规则：

- **参数形式以 dispatcher 层看到的为准**（`__torch_dispatch__` 收到的 `args`/`kwargs`，PyTorch 已按算子 schema 规范化），探针不自行重排或补默认值。两颗芯片上同一次调用经过同一套 schema，得到同一棵树。
  > 设计者在已验证镜像的 torch 2.10.0（CPU）上核对过这一层的规范化：`cat([a,b], dim=1)` 与 `cat([a,b], 1)` 到达时都是 `args=([a,b], 1)`；取默认值的参数被去掉（`cat([a,b], dim=0)` 到达时只有 `([a,b],)`，`sum(a, 1, keepdim=False)` 只有 `(a, [1])`）；tuple 实参变成 list；只有 keyword-only 参数（如 `add` 的 `alpha`）留在 `kwargs`。所以「去掉默认值」是 dispatcher 已经做好的，探针若再补回默认值，反而会让 schema 新增带默认值的参数时 `sid` 跨 torch 版本漂移。跨 torch 版本是否完全一致尚未验证，由 [T5](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-dd13c52d9c06) 记录。
- **容器一律递归**，不限深度。`None` 也是一个节点，不能省略 —— 省略会让 `f(x, None, y)` 与 `f(x, y)` 得到同一棵树。
  > [T2](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-d3abef3c723a) 的 151 条 unavailable 中有 143 条来自 `aten::cat`：历史采集器只遍历顶层 Tensor 参数，`TensorList` 整个被跳过，cat 的输入形状全部丢失。没有路径的扁平参数列表还无法区分「第 0 个参数是含两个张量的列表」与「两个张量参数」。
- **签名中没有任何设备信息。** `tensor` 节点不含 `device`；参数本身是 device 对象时只记 `k=device_obj`。设备记在调用上（§4.3 的 `op.dev`），不记在签名上。
  > R52/R86/R88 要求同一个算子调用在英伟达与国产芯片之间能对齐。设备若进签名，同一次调用在两边得到不同 `sid`，`tensor.boundary` 的 `op:<sid>` 形式就无法跨芯片 join，精度差分直接失效。上一版把设备从哈希中剔除、却仍留在签名记录里，结果是**同一 `sid` 在一个 run 中可能对应两份内容不同的定义**（同一逻辑签名在两张卡上被调用），读取方按 `sid` 建表时后写的会覆盖先写的。把设备挪到调用上，两个问题一起消除。
- **`stride` 仅在 `contiguous=false` 时记录**。连续张量的 stride 由 shape 唯一决定，写入只会引入厂商差异。
- **输入与输出是两个签名。** 输出签名 `role=out`，由 `op.osid` 引用。数据相关的输出形状（`nonzero`、`unique` 等）只改变 `osid`，不会让同一次调用在两颗芯片上得到不同的 `sid`。`arange` 这类没有张量输入的算子，输入树只有标量节点，输出形状由 `osid` 给出（T2 中的 5 条）。
- **shape 不做 bucketing、不截断**。R1（动态 shape 分析）和 R37（按形状选实现）都依赖真实 shape。需要降低基数时，由**分析器**在读取侧泛化（如把 batch 维标注为动态），不在采集侧丢信息。

**`sid` 的计算（冻结）。** 取 `{name, overload, role, tree}` 四个字段组成的对象，序列化为规范 JSON（键按字典序、无空白、UTF-8、非 ASCII 字符不转义），取 SHA-256 摘要的前 8 字节，写为 **16 位小写十六进制字符串**（即十六进制摘要的前 16 个字符）。`trunc` 不参与哈希：截断签名的 `sid` 仍由完整的树算出，与不截断时相同。探针、导入器、离线工具按同一算法都能独立算出同一 `sid` —— 这是跨芯片、跨 run 对齐的基础。

**可解码不变式（L-核心）：每一条被保留下来、引用了 `sid`/`osid` 的事件，在同一分片中出现之前，对应的 `sig` 必须已经写出。** 由此任何一个分片都能独立解码；在线尾读读到引用时，定义一定已经可读；崩溃留下的半截分片中，存活的记录也都能解码。

实现机制不冻结，参考做法如下：

> 1. **`sig` 不走队列。** 探针在把引用事件入队之前，先把签名放进进程内的 `sid → sig` 字典。字典写入在产生侧完成，不经过可能丢弃的队列。
> 2. **写盘线程是分片文件的唯一写者**，并为当前分片维护「已写出的 `sid` 集合」。写一条事件前，若它的 `sid`/`osid` 不在集合中，先从字典取出定义写一条 `sig`。分片轮转时集合清空，新分片按需重写 —— 重复是幂等的，`compact`（§7）会去重。
> 3. **字典有显式预算**（默认 64 MB）。超出后新签名以 `trunc=true` 写出（只保留 `name`/`overload`，`sid` 仍按完整树计算），计入 `meta.drop_report` 的 `sig_truncated`；截断签名再达到第二级上限（默认 16 MB）后，引用新签名的事件按普通溢出丢弃并计数。不论哪种情况，都不允许出现「事件已写出、签名不可读」。
>
> 「典型 10³–10⁵ 种签名」只是经验值，不是上界：动态 shape 与标量参数（如 `arange(n)`）会持续增加签名基数，所以必须有预算和耗尽策略。
>
> 上一版的机制是「每个分片的头部写出本分片引用到的全部 `sig`」。这做不到：写头部时无法知道之后才出现的新签名；封口步骤实际是在**尾部**补写；进程在封口前崩溃时，补写根本不会发生。在线尾读与崩溃这两个最需要它的场景，恰好都不成立。

### 4.3 `op` — ATen 算子调用

| 字段 | 说明 |
|---|---|
| **`sid`** | 指向输入签名（`sig.role=in`） |
| `osid` | 指向输出签名（`sig.role=out`）。算子抛错、或探针等级不采输出时缺省 |
| **`corr`** | 本次调用的关联 id |
| `dev` | 调用所在设备，torch 设备串（如 `npu:0`）。参数分布在多个设备上时为 `{参数路径: 设备串}` 对象。与分片默认值（§2）相同时省略。**取值规则（2026-10-27，[D47](DECISIONS.md#d47) 第 1 条）**：依次收集张量参数、device 类型参数、输出张量的设备；全部相同写设备串，不止一个写对象（输出路径为 `out`、`out.<i>`）；没有任何设备信息时不写。P1 的 dispatch 探针不为 `dev` 设分片默认值，总是显式写出 |
| **`dur`** | 主机侧耗时（ns），即 dispatch 返回减进入 |
| `scope` | 所属 scope 的 span `corr`（见 §4.7）；由探针从 TLS scope 栈读取 |
| `seq_in_scope` | 该算子在当前 scope 内的调用序号，用于构造 `call_key`（见 [06 §5](06-model-onboarding.md)） |
| `fw` | framework，`torch`（预留 `tf`/`ort`，见 [00 §3](00-requirements.md)） |
| `parent_corr` | 外层算子的 `corr`；顶层调用为 `0`。算子库内部再次触发 dispatch 时（[04 §3.2](04-correlation.md)），内层 `op` 分配自己的 `corr` 并在此记录外层的。取代 04 §3.2 中「具体字段在冻结前确定」的占位 |
| `async` | bool，调用是否异步返回（设备执行尚未完成） |
| `err` | 异常类型与消息（算子抛错时） |

### 4.3b `opsum` — 算子聚合（低开销模式）

长时间运行或 token 级生成场景下，逐次记录 `op` 的开销不可接受。探针的 `summary` 等级改为按 (scope, sid, dev, nested, route decision) 周期性聚合：

| 字段 | 说明 |
|---|---|
| **`sid`**、`scope`、`dev`、`nested`、`route_decision` | 聚合键。`dev` 与 `op.dev` 同规则；`nested` 只在为真时写出（值为 `true`），含义同 `op.parent_corr ≠ 0`。`dev`、`nested` 自 L-字段 `1.1` 起（[D47](DECISIONS.md#d47)） |
| **`n`** | 调用次数 |
| **`dur_sum`** | 主机耗时总和（ns） |
| `dur_min`/`dur_max`/`dur_p50`/`dur_p99` | 分布（用定宽直方图估计，直方图边界记在 `meta`） |
| **`win`** | 聚合窗口 `[start_ts, end_ts]` |

`opsum` **没有 corr**，不参与关联链。分析器必须能同时处理 `op` 与 `opsum` 两种来源，并在结论中标注粒度（见 [07](07-analyzers-and-reports.md) 的 `coverage.op_granularity`）。

> **当前实现约定（P1，[D46](DECISIONS.md#d46) 第 2 条、[D47](DECISIONS.md#d47)）**：P1 的 dispatch 探针按 (`sid`, `dev`, `nested`, `route_decision`) 聚合，不写 `scope`、`dur_p50`、`dur_p99`。`route_decision` 只为深度 0 且为加速器的调用填写；具体 `reason` 与 `impl` 不进 opsum，由分析器按 attachment 状态中的 `route_table` 补全。

### 4.4 `route` — FlagGems 等算子库的路由决策

| 字段 | 说明 |
|---|---|
| **`corr`** | 与触发它的 `op` 相同 |
| **`lib`** | 算子库标识，`flaggems` / `vendor_native` / `aten` |
| **`decision`** | 枚举：`hit`（走了该库） / `fallback`（回退） / `excluded`（被策略排除） / `error`（尝试后异常回退） |
| **`reason`** | 枚举，仅当 `decision != hit`。见下表 |
| `impl` | 命中的具体实现标识（kernel 名/模板名），供 R37 使用 |
| `depth` | 嵌套深度（算子库内部再次触发 dispatch 时 > 0），继承 v1 `GemsRoute` 的 depth 计数 |
| `detail` | 自由文本，**仅供人读**。分析器不得解析它 |

`reason` 枚举（对应 R47、R18/R22 的三级回退）：

| 值 | 含义 |
|---|---|
| `not_registered` | 该算子未在库中注册 —— 算子级缺失 |
| `unsupported_signature` | 已注册但本次 shape/dtype/layout 不支持 —— 实现级 |
| `device_unsupported` | 该芯片上不可用 —— 芯片级 |
| `policy_excluded` | 用户/配置显式排除 |
| `compile_failed` | 编译失败（配套一条 `compile` 事件） |
| `runtime_error` | 执行时抛错后回退 |
| `numerical_guard` | 库内数值保护主动回退 |

> **约束来源**：R47 要求回退原因可聚合，R18/R22 要求区分算子级/实现级/芯片级 —— 所以 `reason` 必须是枚举而非自由文本（[00 §2](00-requirements.md)）。新增枚举值走 §8 的 minor 演进。

### 4.5 `compile` — JIT 编译

| 字段 | 说明 |
|---|---|
| **`corr`** | 与触发编译的 `op` 相同（可能为 0，如预热期独立编译） |
| **`backend`** | `triton` / `flagtree` / `aclnn` / `tvm` … |
| **`key`** | 编译缓存 key（内容哈希） |
| **`cache`** | `hit` / `miss` / `invalid` |
| **`dur`** | 编译耗时（ns），`cache=hit` 时为查表耗时 |
| `kname` | 产出的 kernel 名/符号名 —— **这是 compile→kernel 关联的候选键**，见 [04 §3.4](04-correlation.md) |
| `stage_dur` | 各阶段耗时 `{frontend, opt, codegen, assemble}`（后端支持时） |
| `ver` | 编译后端版本（R45 要求） |
| `err` | 编译失败信息 |

### 4.6 `comm` — 集合通信

| 字段 | 说明 |
|---|---|
| **`op`** | `allreduce` / `allgather` / `broadcast` / `reducescatter` / `alltoall` / `send` / `recv` / `barrier` |
| **`group`** | 通信域标识 = `hash(成员 rank 列表 ‖ 该成员集合上的通信域创建序号)`。**仅用成员列表哈希是不够的** |
| **`group_seq`** | 该通信域内本次调用的序号 —— **跨 rank 对齐的唯一依据**，见 [04 §3.5](04-correlation.md) |
| **`bytes`** | 传输字节数 |
| `dtype`、`count` | 元素类型与个数 |
| `lib` | `hccl` / `nccl` / `flagcx` |
| `algo` | 算法/拓扑（库暴露时） |
| `phase` | `enqueue`（主机侧入队）/ `device`（设备侧执行） —— 两条事件，用 `corr` 串起来 |
| **`dur`** | 本 phase 的耗时 |
| `corr` | 主机入队事件分配，设备事件引用 |

**跨 rank 对齐只用 `(group, group_seq, op)`，绝不用时间戳匹配**（[04 §1](04-correlation.md)）。

> **为什么 `group` 要带创建序号**：同一组 rank 上可以存在**多个不同的通信域**（混合并行下 TP 组与 DP 组的成员集合可能恰好相同，或同一组成员先后建了两个 communicator）。若 `group` 只是成员列表哈希，这些域会被折叠成一个，各自独立的 `group_seq` 计数器混在一起，跨 rank 对齐将稳定地配错 —— 而且错得很安静。创建序号在各 rank 上一致，依据与 `group_seq` 同源：通信域的创建顺序是集合通信语义的一部分。8 卡场景（当前暂缓）正是会暴露此问题的配置。

### 4.7 `span` — 区间

scope、phase、batch、generation step、模型生命周期全部统一为 span，不再有第二套区间概念。

| 字段 | 说明 |
|---|---|
| **`ph`** | `b`（begin）/ `e`（end）/ `x`（完整区间，带 `dur`） |
| **`cat`** | 类别枚举：`phase`（load/warmup/measure）/ `module`（nn.Module 层）/ `user`（用户标注）/ `engine`（引擎内部阶段，如 vLLM 的 prefill/decode）/ `batch` / `step`（生成步）/ `lifecycle`（模型加载、热切换、卸载 —— R33） |
| **`name`** | 名称。`cat=module` 时为 module 的限定路径，如 `model.layers.3.self_attn` |
| **`corr`** | 本 span 自身的 id（`b`/`e` 成对共享） |
| `parent` | 父 span 的 `corr`，由 TLS scope 栈给出 |
| `attrs` | 对象，`cat` 相关的属性（如 `module` 的类名、`step` 的 token 下标、`lifecycle` 的版本号） |

`e` 事件只需 `ph`、`corr`、`ts`。span 树的重建规则见 [04 §4](04-correlation.md)。

### 4.8 `kernel` / `memop` — 设备执行

由厂商 trace 导入器产出（离线），不是在线探针。导入时调用 linker，`link` 随事件一次写出，之后不再改写；厂商主机侧记录（launch、Enqueue/Dequeue、marker 等）不转成 store 事件，留在 `vendor_raw/` 由 `evidence` 引用（[D18](DECISIONS.md#d18--关联在导入时完成厂商主机侧记录不转成-store-事件)）。

`kernel`：

| 字段 | 说明 |
|---|---|
| **`name`** | kernel 名 |
| **`dev`** | 设备序号 |
| **`stream`** | 流/队列标识 |
| **`ts`**、**`dur`** | 设备时间轴上的起止。`ts` 按 §5.2 换算到统一时间轴，换算不成立时为 null；`dur` 是设备时间轴上的差值，只要原始单位可知就有效，与 `ts` 是否成立无关 |
| **`link`** | 关联结果对象，结构见下。`status` 枚举与 [04 §2](04-correlation.md) 完全一致：`linked` / `ambiguous` / `unlinked` / `not_applicable` |
| `ktype` | `compute` / `comm` / `memcpy` / `unknown` |
| `grid`、`block`、`regs`、`smem` | 启动配置（可得时） |
| `metrics` | 设备计数器（占用率、带宽等），厂商支持时 |

`link` 的完整结构 —— **只有 `{corr, status, via}` 不足以复核**，queue 桥与证据冲突都需要能回到原始事件：

| 字段 | 说明 |
|---|---|
| **`status`** | `linked` / `ambiguous` / `unlinked` / `not_applicable`。四值与 [04 §2](04-correlation.md) 同一套枚举 |
| `corr` | `status=linked` 时唯一确定的关联 id |
| `candidates` | `status=ambiguous` 时的候选 corr 数组，升序。元素可以是 `null`：表示存在一条按显式键匹配到、但归属无法解析的 launch（如 queue 桥缺失）。**不得任取其一升级为 linked** |
| **`segments`** | `{host_attr, device_link}`，**逐事件**判定，各取 `ok`/`broken`；`status=not_applicable` 时整个字段为 null。判定规则与级别见 [04 §6](04-correlation.md) |
| **`via`** | 已成立的规则 id **数组**，升序（如 `["cann.enqueue_corr","cann.flow"]`）。多条规则指向同一 corr 时合并记录，互为佐证；指向不同 corr 时全部保留，候选进入 `candidates` |
| `reason` | `status != linked` 时的原因枚举：`no_bridge`（缺跨线程桥）/ `no_device_key`（缺 flow/connection）/ `conflict`（候选集合含两个以上不同元素，`null` 也算一个）/ `source_missing`（主机侧事件未导入或已丢弃）/ `excluded`（registry 白名单命中的非调用任务，与 `not_applicable` 同时出现）/ `graph_replay`（图回放期间的 kernel，[04 §5](04-correlation.md)） |
| `evidence` | 支撑本次判定的事件身份数组（§6.5）：本 store 的事件写 `{w, seq}`，厂商原始记录写 `{trace_uid, loc}`（`loc` 是导入器定义的原始记录定位，如 `trace_view.json#traceEvents[17688]`）。**不得使用分片内偏移** —— 轮转、压缩、compact 之后偏移就变了 |

`not_applicable` **仅用于已识别的、本来就不该有调用归因的事件**（如 profiler 自身的控制任务）。未知设备任务一律进 `unlinked`，**不得为提高 coverage 而归入 `not_applicable`**。计算、通信、搬运三类的分母分别统计。

判定来源必须是**厂商 registry 中可枚举的白名单**（kernel 名模式 / task type / stream 属性），该白名单随 manifest 一并落盘，分析器与人工都能复核「这条凭什么不算分母」。**linker 不得自行推断 `not_applicable`** —— 它是唯一能合法缩小分母的状态，一旦可由代码临场判断，关联率就失去意义。

一次 op 对应**多个** kernel 是合法的 `linked`（原生 LayerNorm 实测一次调用产出 5 个 kernel）；只有反向关系 —— 一个 kernel 存在多个不同的有效 op 候选 —— 才构成 `ambiguous`。

**候选取并集，不按优先级短路。** linker 对每条适用的规则求出候选集合，再取并集；规则优先级只决定检查顺序，**不允许**某条规则先成功就跳过其余规则。一个键在原始记录中命中多条 launch 时全部计入，不得因「不唯一」而整体跳过。[T1](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-867dfc27c4dd) 的 X7 正是在这两处把人为制造的冲突报成了 `linked`。完整算法见 [04 §3.6](04-correlation.md#36-linker-算法候选并集与判定)。

`memop`：`{kind: h2d/d2h/d2d/alloc/free, bytes, dev, stream, ts, dur, link}`。显存峰值由 `counter` 承担，`memop` 承担搬运归因。

### 4.9 `tensor` — 张量引用与统计（Oracle–Device 差分的载体）

| 字段 | 说明 |
|---|---|
| **`call_key`** | 跨 run 对齐键，见 [06 §5](06-model-onboarding.md)。格式 `<scope_path>#<seq_in_scope>#<boundary>#<sample_idx>`。**scope 缺失时不得产出**，见下 |
| **`role`** | `in` / `out` / `grad` |
| **`boundary`** | 边界名（Tier-1 YAML 声明的层边界名，或 `op:<sid>`） |
| `corr` | 若由 op 边界捕获，引用该 op 的 corr |
| **`shape`**、**`dtype`** | |
| **`stats`** | 全局统计 `{count, nan, inf, min, max, mean, absmean, l2, hist?}` |
| **`chunks`** | 分块统计：按规范规则把张量切成 K 块（默认 K=64），每块记 `{min, max, absmean, l2, nan, inf}` |
| **`probe`** | 确定性抽样：M 个元素（默认 M=256）的**原值**，下标由 `hash(call_key) → 固定步长` 决定，与数据内容无关 |
| `digest` | 原始字节的 `xxh3_64`。**仅用于判等**：相同则两侧完全一致可直接跳过；不同**不构成**任何结论（`-0.0`/`+0.0`、NaN payload 会让数值相等的张量字节不等） |
| `blob` | `{path, offset, nbytes, hash, fmt}` —— 显式开启时才有，指向 run 目录下的 blob 文件 |
| `mask` | 有效区域掩码引用（变长输入场景，避免 padding 污染误差，继承 v1） |
| **`probe_alg`** | `chunks` 切分与 `probe` 抽样的算法版本（如 `v1`）。版本不同的两侧不得逐块、逐元素比较 |

> **2026-10-28 按 [D50](DECISIONS.md#d50) 第 6、7 条（L-字段 `1.2`，T11.1 实现时生效）**：
> - `digest` 用 `sha256:<hex>`，对象是有效区域原 dtype 的小端连续字节。原文的 `xxh3_64` 作废，因为镜像没有 xxhash，规则又不允许装新包；语义仍是只用于判等。
> - 新增可选字段 `src`：下钻时每个张量输入的来源，取值为上游算子输出的 call_key、`param:<路径>`、`in/<i>` 或 `unknown`。
> - `call_key` 的 boundary 取 `out`、`in/<i>`、`op:<name.overload>[/<i>]`；sample 取固定输入的样本 id。
> - `probe_alg="p4v1"`：K=64、M=min(256, N)、下标由 sha256(call_key) 定相位，统计在主机上以 FP64 计算。

**`call_key` 的产出前提**：`scope_path` 来自 span 树。若该张量所在线程没有 scope 栈（[04 §3.1](04-correlation.md) 的 `<unscoped:tid>` 情形），**不得产出 `call_key`** —— `tid` 在两次 run 中不同，拼出来的键看着合法却永远 join 不上，比缺失更糟。此时 `tensor` 事件照常写出并标 `call_key: null` 与 `align: "unscoped"`，由分析器计入"不可对齐"并在 coverage 中单列。

**默认不落 blob**。原因：500 模型全量中间结果的体积不可接受。差分流程是「先用 stats 定位首个发散层 → 再对该层开 blob 重跑」。blob 保留策略见 §7 与 [Q5](DECISIONS.md#q5--tensor-blob-的保留策略)。

> **为什么全局 stats 不够 —— 张量越大越不够。**
> 设 N 个元素、`absmean ≈ 1`，某一个元素错了 1.0：它对 `mean` 的影响是 1/N，对 `l2` 的影响约 1/(2N)。N = 10⁶ 时这是 10⁻⁶ 量级，**低于两家芯片之间的正常数值噪声**，全局统计一定看不见。而精度 bug 恰恰常常是局部的（某个 tile 边界、某段 padding、某个特化分支）。所以"张量很大时这种情况极少出现"的直觉是反的：**张量越大，全局聚合量对局部错误越不敏感**，漏报概率越高。
>
> 三个新字段各自补一个窟窿，总开销仍远小于 blob（K=64、M=256 时约 1–2 KB/张量，blob 是 MB 级）：
> - `chunks` 把稀释倍数从 N 降到 N/K —— 局部错误落在某一块内，该块的 `max`/`l2` 会明显跳变；
> - `probe` 给出可**逐元素**比较的真值，是小号 blob，能直接回答"差多少"而不只是"有没有差"；
> - `digest` 让完全一致的层以近乎零成本被排除，把注意力留给真正有差异的层。
>
> 三者都**不依赖另一侧的数据**，各 run 独立产出，差分仍在离线按 `call_key` join 完成。
>
> 但它们的**证据等级**不同，分析器必须按等级措辞（[07 §3.3](07-analyzers-and-reports.md)）：`blob` 是完整证据，能给逐元素指标；`probe` 是抽样证据，只能给样本上的指标；`stats`/`chunks` 是聚合证据，只能给筛查线索。聚合量或样本没有差异，**不能**证明整个张量一致。
> `chunks` 的切分规则必须是**规范且与设备无关**的（按 flatten 后的元素序等分，余数并入最后一块），否则两侧分块边界不同，比较无意义。

### 4.10 `counter` — 周期采样

| 字段 | 说明 |
|---|---|
| **`src`** | `device`（厂商遥测，如 npu-smi/DCGM）/ `framework`（torch 显存统计）/ `host`（CPU、内存、IO） |
| **`dev`** | 设备序号（`src=device` 时） |
| **`values`** | 字典，如 `{"mem_used": 12345, "util": 0.91, "power": 210.5, "temp": 61}` |
| `blocked` | bool —— 采样与 profiler 冲突而被屏蔽的区间标记。**屏蔽而非删除、缺失不填零**（继承 v1 `implementation_15.md`） |

值的单位与语义由 registry 中的探针 descriptor 声明（[05 §5](05-probes-and-vendors.md)），schema 本身不枚举指标名 —— 厂商可增，但必须在 registry 中登记来源 API。

### 4.11 `health` — 故障信号

| 字段 | 说明 |
|---|---|
| **`kind`** | `heartbeat` / `hang` / `oom` / `comm_timeout` / `device_error` / `process_exit` / `ecc` / `link_down` |
| **`sev`** | `info` / `warn` / `error` / `fatal` |
| `detail` | 对象，kind 相关 |
| `evidence` | 支撑该判断的事件引用（如最后一个 `op` 的 corr、超时的 `comm` 的 `group_seq`） |

`heartbeat` 由目标进程以固定周期写出，**卡死判定由离线/在线的规则引擎做**，不由探针判断（[08 §2](08-online-and-fault.md)）。

## 5. `ext`：厂商原生 id 的唯一容身处

厂商特有字段一律放 `ext`，键名加厂商前缀：

```json
{"t":"kernel","name":"triton_mm_f16","ext":{
  "cann.connection_id": 10241, "cann.task_id": 77, "cann.stream_id": 3,
  "cann.flow_start_ts": 1699..., "cann.batch_id": 5}}
```

```json
{"t":"kernel","name":"ampere_sgemm","ext":{
  "cuda.correlation_id": 88213, "cuda.external_id": 4412}}
```

规则：

- 分析器**不得**读 `ext`。只有 `vendors/<v>/` 下的导入器与关联规则可以读写。原生调用关系由导入器写进派生视图，经 store 的公共只读接口暴露（[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)），不经 `ext`。
- `ext` 中的 id 是关联链的**证据**：`link.via` 指明用了哪条规则，`link.evidence` 指向原始事件，排查时回到 `ext` 核对。
- 新增厂商只增加 `ext` 键，不改 schema 版本。
- **原始缺失保留为 null 或缺省，不得填 0。** 厂商的哨兵值按各自来源解释，不得套用框架 `corr=0` 的含义。

### 5.1 厂商原生 id 的命名空间

**厂商原生 id 一律是局部的，直到被证明不是。** 任何 `ext` 中的 id 在 join 之前必须先被限定到它真实成立的范围：

| 范围 | 组成 | 例 |
|---|---|---|
| 导出会话内 | `(run_id, vendor, trace_uid)` | `cann.connection_id` |
| 进程内 | 上者 + `pid` | `cann.enqueue_correlation_id` |
| 设备上下文内 | 上者 + `device_id`、`stream_id` | `cann.task_id` |

`trace_uid` 由**导入器**在每次导入厂商 trace 时分配，并登记到 `traces/<trace_uid>.json`（§6.2）：`{trace_uid, vendor, source_path, sha256, imported_at, device_ids, rank}`。没有这张登记表，上表的命名空间无处落地 —— 事件 `ext` 里的 `trace_uid` 将是一个无法回溯到具体文件的裸数字。2026-10-14 起，登记表另含三层来源身份，见 §6.2。

**原生关系中的引用**（[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)）有两条规则：
- 原生关系视图中的每个节点（原生调用、launch、设备记录、作用域注释）一律以 `(trace_uid, loc)` 引用。
- 原生 id（`correlation`、`External id`、`connection_id`……）只在上表的命名空间内参与 join，**不得**作为项目 `corr` 写入任何事件（[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #25）。

`capture_id`（§6.5）与 `trace_uid` 分立：前者标识一次采集窗口，后者标识一份原始文件（#27）。

`cann.connection_id` 的实测事实（[Spike-C](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/11-spike-cann-correlation.md)）：它由 CANN decoder 的 `ApiEventParser.connection_id` **从 0 自增合成**，不是硬件自带的调用号。TP2 实测中两个 rank 的裸值集合交集达 300/300；而**同一次** `all_reduce` 在两个 rank 上的 `connection_id` 反而不同（600 对中仅 189 对恰好相同）。因此：

> **裸值相等既不足以证明是同一次调用，也不是同一次调用的必要条件。** 跨 rank/跨 trace 一律先隔离命名空间，再按逻辑键（[04 §3.5](04-correlation.md) 的 `(run, group, group_seq, op)`）对齐。

厂商导出的 pid/tid 可能是合成值（实测 CANN pid=1279 对应 Python pid=1），**不得要求各层原始 pid 相等**，映射关系由厂商导入器声明并记入 `ext`。

### 5.2 时钟来源

`ts`（§2）是换算**之后**的统一时间轴值。换算的输入与依据必须同时保留：

| 键 | 说明 |
|---|---|
| `<v>.raw_ts` | 厂商时间轴上的原始值，不做任何转换 |
| `<v>.raw_ts_unit` | 原始单位（`us`/`ns`/`cycle`） |
| `<v>.clock_domain` | 原始值所属时钟域（`device`/`host_epoch`/`host_mono`/`unknown`） |
| `<v>.clock_transform` | 换算依据：标定点、公式、来源 API 与版本 |
| `<v>.clock_valid` | 换算是否成立（bool）。不成立时信封的 `ts` 为 null（§2）—— 通用消费者只看 `ts` 是否为 null，不读 `ext` |

> **单位换算不是时钟域转换。** epoch-like 微秒 ×1000 只是换单位，得到的不是 `CLOCK_MONOTONIC_RAW`。`ts` 为 null 时，该事件仍可参与**基于 id 的关联**并输出自身 `dur`，**仍计入覆盖率的所有分母**，但**禁止**参与任何跨来源的时间比较、排序与 gap 计算；[04 §3](04-correlation.md) 的"用时间排除候选"规则在该事件上同样失效。

## 6. store 布局、manifest 与 seal 身份

### 6.1 目录布局

```text
<store>/<run_id>/
  manifest.json              run 级登记，原子写入（写 .tmp 后 rename）
  writers/
    w-<writer_id>.json       writer 片段：由该 writer 自己写（§6.2）
    .lock                    writer_id 取号锁
  captures/
    <capture_id>/            官方 profiler 会话的产物与会话清单：由采集会话写（D22；名称待 T8 Gate 1）
  traces/
    <trace_uid>.json         厂商 trace 登记：由导入器写（§5.1）
  <派生目录>/                 导入器写出的派生文件：来源账本、原生关系、证据索引、完成标记
                             （[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)/[D24](DECISIONS.md#d24--导入契约身份分层来源账本完成标记证据索引)；目录名与格式待 T8 Gate 1，不得匹配 traces/*.json）
  events/
    w-<writer_id>/
      000001.jsonl           活跃分片：明文，可尾读
      000000.jsonl.gz        已封口分片：压缩编码以分片条目的 codec 为准，默认 gzip（D16）
    w-<writer_id>/...
  blobs/                     tensor blob（显式开启时）
  vendor_raw/                厂商 profiler 原始产物（保留以便复核）
  logs/
```

`run_id`：`<utc_compact>-<host>-<short_uuid>`，可排序、可检索（N5）。

分片按 writer 分目录，不按 rank：同一 rank 可以有多个 writer（§3）。

### 6.2 manifest 与 writer 片段：谁写哪个文件

多个进程同时改一个 `manifest.json`，`.tmp + rename` 只能保证读到完整版本，挡不住「各自读旧版、改完写回、互相覆盖」。所以**每个文件只有一个写者**：

| 文件 | 写者 | 内容 |
|---|---|---|
| `manifest.json` | run 的创建者与 finalizer（run 结束时汇总的那一方） | run 级字段，见下表 |
| `writers/w-<id>.json` | 该 writer 自己 | `{writer_id, kind, rank, proc_key, clk, shards[], next_seq, dropped, state}` |
| `traces/<trace_uid>.json` | 该次导入的导入器 | `{trace_uid, vendor, source_path, sha256, imported_at, device_ids, rank, inventory, linker}`，以 `O_EXCL` 创建，同一 run 中同一 `trace_uid` 只导入一次。`linker` 是本次关联所用的 `{rules_version, registry_sha256}`（[D18](DECISIONS.md#d18--关联在导入时完成厂商主机侧记录不转成-store-事件)）。`inventory` 是原始清单 R（[04 §2](04-correlation.md#2-关联状态)）的落盘位置：按类别 `compute`/`comm`/`memcpy`/`unknown` 的条数，以及按厂商原始类型（如 CANN `Task Type`）的条数；由导入器独立于 linker 枚举，无法枚举时为 null，`import_completeness` 随之为 `unknown`。 2026-10-14 起另含（字段名为 L-字段 `1.0-rc`，由 T8 Gate 1 定稿；内容按 [D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立) 第 5 条、[D24](DECISIONS.md#d24--导入契约身份分层来源账本完成标记证据索引)）：三层来源身份（硬件 / 采集器 / 解析，[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #27）、`capture_id`、来源账本的处置汇总、原生规则版本与作用域识别配置、各派生文件的 SHA-256。现有 `vendor` 字段保留为规则族含义，不再表示硬件厂商 |
| `captures/<capture_id>/` 中的会话清单 | 该采集会话自己 | 窗口、实际配置、硬件与采集器身份、各窗口的有效性记录（窗口 k 的记录落盘后才开始 k+1）。会话异常终止后，未封存窗口的 `lost` 由 finalizer 记在 `manifest.json` 的汇总中，**不写回会话文件**（与 writer 的 `lost` 同理） |
| 派生文件（来源账本、原生关系、证据索引、完成标记） | 该次导入的导入器 | 只新增、不改写 `events/` 与 `vendor_raw/`；记录输入 SHA 与解析器、规则版本；完成标记在事件全部持久化之后写出（[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #26、#28）。2026-10-16 起另含**格式适用性评估**：作为 `format_assessment` 键写在派生摘要 `summary.json` 中（L-字段 `1.0-rc`），是导入时的快照，经 `StoreView.format_assessment(trace_uid)` 读取，读取时不重新运行厂商解释；不新增事件字段，不改 kernel 字段 |

writer 片段的字段：

| 字段 | 说明 |
|---|---|
| `kind` | `probe`（被测进程内）/ `counter`（独立采样进程）/ `importer`（离线导入器） |
| `rank`、`proc_key` | 拓扑属性与进程唯一键（§4.1 `run_start` 的 `{boot_id, pid_ns, pid, proc_start_ticks}`）。**都不参与 writer 身份**，只用于把 writer 对回物理进程 |
| `shards[]` | 每个分片 `{index, path, codec, records, first_seq, last_seq, sealed, sha256?}`。`records` 是已提交的记录条数（含控制记录），尾读据此判断是否读完（§6.4）。`codec` 取 `none`（明文）/ `gzip` / `zstd`，读取方按它解码而不按后缀猜；默认 `gzip`，读取环境不支持的 `codec` 报告为不可读，不得静默跳过（[D16](DECISIONS.md#d16--封口分片的压缩编码记在分片条目上默认-gzip)）。本表及 §6.4 的 `sealed` 指**分片封口**，与官方 profiler 窗口的 sealed-window 快照（[02 §5](02-architecture.md#5-一次-run-的数据流)）无关 |
| `next_seq` | 下一个待分配的 `seq` |
| `dropped` | 按类型的累计丢弃数，与最近一条 `meta.drop_report` 一致 |
| `state` | 片段里只有 `active` / `closed`（正常关闭），都由 writer 自己写。`lost`（finalizer 发现进程已不在而片段未关闭）**不写回片段** —— 否则片段就有了两个写者 —— 而记在 `manifest.json` 的 `writers` 汇总中。读取方的有效状态：片段为 `closed` 即 `closed`；否则 manifest 记为 `lost` 即 `lost`；否则 `active` |

读取方以 `writers/` 与 `traces/` 目录的内容为准，`manifest.json` 中的汇总只是 finalizer 写出的快照。

`manifest.json` 的字段：

| 字段 | 说明 |
|---|---|
| `run_id`、`created_at`、`schema_version` | |
| `status` | `running` / `complete` / `truncated` / `failed` |
| `probes[]` | 启用的探针及其配置与版本 |
| `identity` | 见下 |
| `mode` | `observe` / `seal` |
| `experiment` | seal 模式的实验变量声明（如 `{flaggems: "on", device: "ascend"}`） |
| `collection` | 采集期自报的事实：丢弃数、采样空洞、标记范围、脱敏变量名单。**不含关联率** |
| `retention` | 保留策略的执行记录，见 §7 |
| `writers` | finalizer 写出的 writer 汇总 `{<writer_id>: {state, records, dropped, tail_unknown}}`。`lost` 只记在这里（见上表 `state`） |

> `collection` 此前叫 `coverage`，与 [07 §2](07-analyzers-and-reports.md) 分析器结论里的 `coverage` 同名异义 —— 一个是采集侧事实，一个是结论可信度。冻结前改名，冻结后改就是 major。

`identity`（继承 v1 `runtime/worker.py:identity` 的口径）：

```json
{"packages": {"torch":"2.10.0","torch_npu":"...","flag_gems":"...","triton":"..."},
 "env": {"TASK_QUEUE_ENABLE":"0","TRITON_ENABLE_TASKQUEUE":"true", "...": "..."},
 "device": {"vendor":"ascend","model":"...","driver":"...","firmware":"..."},
 "code": {"flagprof_commit":"...", "plugin_hashes": {...}},
 "inputs": {"recipe": "...", "sha256": "..."}}
```

- **observe 模式**：`identity` 照记不校验。
- **seal 模式**：`inputs` 必填；对照的两次 run 做 identity diff，差异必须被 `experiment` 解释，否则分析器拒绝出对比结论并说明原因（N4）。

### 6.3 写入、丢弃与崩溃

**writer 实现约束**：探针调用 writer 是非阻塞的（有界队列 + 后台写盘线程）。队列满时**丢弃并计数**，绝不因为写盘慢而阻塞被测进程 —— 那会让性能数字失真（N1）。`sig` 不经过队列（§4.2）。

**丢弃可见性**：`seq` 的取号与入队（或丢弃计数）在同一临界区内完成（§2），所以单个 writer 的文件顺序就是 `seq` 顺序。读取方在**分片内部**见到 `seq` 跳变，即可确定这些序号已被丢弃，不会再补上。

**崩溃语义（N2）**：活跃分片是明文 JSONL，逐行 flush（可配 `fsync` 周期）。读取方遇到最后一行不完整时丢弃该行并标 `truncated`，而不是整个 run 作废。封口第 3 步之后、第 4 步完成之前崩溃的分片保持明文（`codec=none`），读取方照常读取；其他进程**不得**代为压缩并改写该 writer 的片段（§6.2 单写者）。

崩溃时最后一段的丢弃量只能部分知道：如果崩溃前持续丢弃、之后再没有成功写出的事件，读者看不到跳变，最近一条 `drop_report` 也可能早于这些丢弃。因此报告的丢弃量分两种：

| writer 状态 | 报告 |
|---|---|
| `closed`（正常关闭，最后一条 `drop_report` 在关闭时写出） | 精确值 |
| `active` / `lost`（未正常关闭） | **下界**：最近一条 `drop_report` 的计数 + 其后可见的 `seq` 跳变；并标 `tail_unknown=true`。**不得报告「零丢失」** |

周期写出 `drop_report`（§4.1）把不可见部分限制在最后一个周期之内。

### 6.4 封口与尾读交接

[08 §3](08-online-and-fault.md) 的尾读方按 writer 片段列出的分片读取。封口必须按以下顺序，否则尾读方会在读一个已被改名的文件：

1. 创建新分片，写出 `meta.shard_header`，在 writer 片段中登记为活跃（片段原子写）；
2. 停止向旧分片写入，写出旧分片的 `meta.drop_report`；
3. 在 writer 片段中把旧分片标 `sealed=true` 并写定 `records`、`last_seq`，**此时它仍是原文件名的明文**；
4. 后台压缩（默认 gzip），**完成后**才在片段中同时更新 `path` 与 `codec`，再删除明文。

> **术语区分**：本节的"封口 / `sealed`"只指 writer 分片的封口；它与 SDK 产物的 **sealed-window 快照**（官方 profiler 窗口目录的逐文件 SHA 复制品，[02 §5](02-architecture.md#5-一次-run-的数据流)）和 P4 的实验 seal（[02 §4](02-architecture.md#4-两种运行模式observe-与-seal)）是三件不同的事。分片 `sealed=true` 不表示窗口产物已封存，反之亦然。

尾读方的规则：

- **按记录数判断读完，不按时间戳。** 只有 `sealed=true` 且自己已读的记录数等于 `records` 的分片，才可释放。时间戳可能重复、可能为 null（§5.2），不能表示「读完」。
- 读到的文件消失（已压缩改名），按片段中的新 `path` 重新打开，**按记录序号跳过已读的前 N 条**后继续。明文的字节偏移不能用于压缩后的文件。
- 重复消费按事件身份去重：产生侧事件以 `(writer_id, seq)`，控制记录按 §2 的内容键。

### 6.5 事件身份

| 对象 | 身份 | 性质 |
|---|---|---|
| 产生侧事件 | `(run_id, writer_id, seq)` | 分配后**永不改变**。轮转、压缩、compact、排序都不得重新编号 |
| `sig` | `sid` | 内容寻址 |
| `meta.drop_report` | `(writer_id, 分片 index)` 取最后一条 | 累计值 |
| 厂商原始记录 | `(trace_uid, loc)` | `loc` 由导入器定义，指向 `vendor_raw/` 中不可变的原始文件 |
| 采集会话 | `capture_id` | 由采集会话分配，标识一次采集窗口；一个 capture 可对应多个 `trace_uid`（[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #27） |
| 派生文件中的记录 | 不产生新身份 | 只引用上表身份；证据索引只是加速结构（#28） |

`link.evidence`（§4.8）、报告中的回指、跨 run 引用一律使用上表的身份，不使用文件路径加偏移。

## 7. 保留与压缩策略

| 阶段 | 动作 |
|---|---|
| 运行中 | 活跃分片明文；达到大小阈值（默认 64 MB）或时间阈值后封口，后台压缩（默认 gzip，[D16](DECISIONS.md#d16--封口分片的压缩编码记在分片条目上默认-gzip)） |
| 导入时 | 派生文件（来源账本、原生关系、证据索引）随导入写出。它们与 compact 同类：不改 `events/` 与 `vendor_raw/`，记录输入 SHA 与规则版本，同一输入可逐字节重建（[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #28） |
| run 结束 | `compact`：排序、去重 `sig`、生成索引（按 type/scope/sid 的位置表），可选转列式（Parquet）。**compact 不删除任何产生侧事件，也不改变任何身份（§6.5）** |
| 精简（prune） | **显式的单独步骤**，不是 compact 的一部分；单个 run 默认不自动精简。执行后在 `manifest.retention` 记录 `{pruned_at, policy, kept_types, dropped_types, window}` |
| 精简后保留 | `meta`、`sig`、`opsum`、`span`、`route`、`compile`、`comm`、`counter`、`health`、`kernel` 聚合 |
| 异常保留 | run 失败或 health 出现 `error`/`fatal` 时**不精简**，保留全量逐事件与 `vendor_raw`（继承 v1"失败保留现场"） |
| 长稳运行（R33） | 按时间窗精简：老窗口只保留 `opsum`/`counter` 聚合与全部 `health`，窗口记入 `manifest.retention` |
| blob | 默认关闭；开启时按 `call_key` 白名单，且设总量上限，超限即停止并记 `drop_report` |

**精简后分析器降级。** 分析器读 `manifest.retention`，依赖的逐事件类型已被精简时输出 `unavailable` 或 `partial`，并说明原因（[07 §1](07-analyzers-and-reports.md)）；不得在聚合数据上继续给出逐调用结论。被精简事件的身份不复用，指向它们的 `evidence` 解析为「已按策略删除」，而不是报错或悬空。

## 8. 版本与演进规则

版本号 `major.minor`：

| 变更 | 版本 | 允许性 |
|---|---|---|
| 新增可选字段 | minor+1 | 随时 |
| 新增事件类型 | minor+1 | 随时 |
| 新增枚举值（如 `route.reason`） | minor+1 | 随时；读取方必须把未知枚举值归入 `other` 而非报错 |
| 新增 `ext` 键 | 不变 | 随时 |
| 字段改名、改语义、改单位 | major+1 | **需在 [DECISIONS](DECISIONS.md) 登记，且需要迁移工具或明确宣告历史数据不可读** |
| 删除字段 | major+1 | 同上 |

读取方规则：
1. 忽略未知字段与未知事件类型（但计数并在 coverage 中报告）。
2. 遇到 `major` 大于自身支持版本时拒绝分析并明确报错。
3. 分析器声明自己需要的最低 minor 版本。

**冻结时点（按 [D11](DECISIONS.md#d11--schema-分三层冻结不是一次性整体冻结) 分层，不是整体冻结）**：

| 层 | 冻结对象 | 时点 | 本轮状态 |
|---|---|---|---|
| **L-核心** | §2 信封与两类记录、§3 corr 与 writer 身份、§4 的事件类型集合、§4.2 `sid` 算法与可解码不变式、§4.8 `link` 语义与候选并集、§5.1 命名空间、§5.2 时钟、§6 store 的文件所有权/丢弃/崩溃/封口/身份语义、§7 compact 不改身份；2026-10-14 扩充：原生关系与项目归属分立、来源处置、身份三层、导入完成标记（附录 A #25–#28，不改事件字段） | **P0 开工即冻结**（#25–#28 自 2026-10-14 起） | 清单见 [DECISIONS 附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结)（2026-10-11 第二轮按 [D15](DECISIONS.md#d15--t1t3-之后的契约回填r1r8) 回填） |
| **L-字段** | 各类型的具体字段与枚举值（含 `compile` 的多 entry、warmup `phase`、参数树节点的字段细节）；`writer_id` 位宽划分；字典预算默认值；T8 的表示形式（派生文件布局与格式、descriptor 新字段、处置原因枚举） | P0 结束定为 `1.0` | 随 P0 实测补全；T8 表示形式于 Gate 1 定为 `1.0-rc`；2026-10-27 随 P0 退出定为 `1.0`（[D45](DECISIONS.md#d45)）；同日 opsum 新增 `dev`、`nested` 两个可选字段，升为 `1.1`（[D47](DECISIONS.md#d47)） |
| **L-厂商** | `ext` 键、规则 id、能力矩阵取值 | 不冻结 | 随厂商接入增加 |

L-核心的改动一律走上表的 major，并需在 [DECISIONS](DECISIONS.md) 新增条目；L-字段走 minor；L-厂商不影响版本号。

> L-字段的 minor 记录：`1.1` 由 [D47](DECISIONS.md#d47) 引入（opsum `dev`/`nested`）；`1.2` 由 [D50](DECISIONS.md#d50) 批准（tensor 可选字段 `src`），T11.1 实现时生效。

**导入解释的版本与旧数据读法**（2026-10-16，[T8.3 返工复审](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-81d74b957ee6) §3.5、§6.1）：
- **旧 store**：T8.3 之前导入的 run 没有 `format_assessment`，`StoreView.format_assessment()` 返回 `unavailable`，读作"未记录"，不得显示为"无异常"。
- **采用新解释**：新的角色规则或契约只对**原件导入新 run** 生效，保留旧 run 并比较完整解析身份；不就地改写旧 store 的评估或派生文件（[D30](DECISIONS.md#d30--契约条件的证据先查产物自述每个必需条件登记生产者) 第 3 条，已确认（用户 2026-10-16））。
- **版本 + 指纹**：`device_taxonomy` 的版本标签（腾昇 v1–v4、NVIDIA v1–v2）是厂商格式代际，不是内容修订号；同一标签下内容可以改变，修订前后只靠解析身份的 sha 区分。判断两次导入是否用了同一套解释，须同时看版本与 sha。当前 sha 只覆盖 `SourceFormat` 协议的闭包，账本处置、规则与共享判定代码不在任何代码哈希身份内，这一缺口转 T9.0 §0.5（[D27](DECISIONS.md#d27--格式解释的身份覆盖全部钩子兼容验收到连接键) 更正注记）。


## 9. Spike-C 实测回填（2026-10-10；**已裁决，多数已并入正文**）

> **读法**：本节保留为**证据索引**，不再是独立的规范来源。正文与本节冲突时以正文为准。裁决见 [DECISIONS Q7](DECISIONS.md#q7--spike-c-实测回填与冻结边界)。
>
> | 原建议 | 去向 |
> |---|---|
> | 原生 ID 命名空间 | 已并入 [§5.1](#51-厂商原生-id-的命名空间) |
> | 时钟来源与原始值保留 | 已并入 [§2](#2-事件信封) `ts` 行与 [§5.2](#52-时钟来源) |
> | link 枚举与 04 §2 统一、候选/证据 | 已并入 [§4.8](#48-kernel--memop--设备执行) |
> | **compile 多 ELF entry 与逻辑名** | **未并入**：`compile` 类本轮无实测，按 [D11](DECISIONS.md#d11--schema-分三层冻结不是一次性整体冻结) 属 L-字段，P0 期间按实测补 |
> | **JIT/autotune 首次迭代额外 kernel** | **未并入**：须在 P0 设计 `phase`（warmup/steady）标注后落字段 |

证据：[实验报告](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-e1bd695b52c5)、[字段清单与逐事件统计](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md#ref-59546a5ca33d)。TP2 已验证跨 rank 裸 ID 重用及 group/seq 对齐；8 卡仍暂缓。

- §5 的 `cann.connection_id` 是 CANN decoder 合成的导出局部 ID，须补 `cann.trace_uid`、主机 pid/tid、device 命名空间，不能跨 trace 裸值 join。
- queue=1 的计算路径需要 `cann.enqueue_correlation_id`；保留 flow 类别/id、task/stream/batch/context、globalTaskId 及原始事件引用。完整候选键与缺失语义见报告 §8。
- §4.5 的 `compile.kname` 与 metadata 逻辑名可精确匹配 trace，但一个名字对应多个编译 key；混合内核还有 `_mix_aic/_mix_aiv` 实际 ELF entry。建议另存 `triton.entry_symbols`、`triton.binary_sha256`，不任取唯一编译产物。
- §4.8 的 link 枚举与 04 §2 的 `not_applicable` 尚不一致；应统一，并明确缺证据/冲突的处理。原生 LN 一次 op 对五个 kernel 是合法 linked。
- 原始 exporter 时间轴并非可直接标注的 CLOCK_MONOTONIC_RAW。建议保留 `cann.raw_ts/raw_ts_unit/clock_domain/clock_transform`，先核对换算来源，再写统一 ts。
- JIT/autotune 首次迭代有大量额外 kernel，必须保留阶段信息；不能按“一调用固定一 kernel”冻结契约。

设计确认入口：[DECISIONS Q7](DECISIONS.md#q7--spike-c-实测回填与冻结边界)。P0 仍未开始。
