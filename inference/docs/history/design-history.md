> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](../README.md)，可执行命令见 [使用说明](../usage.md)。未发布的本地实验与任务书链接转入 [历史引用索引](history-references.md)，其出处保留。

> **当前（2026-11-02）：P2 已按 D54 登记范围通过并退出（D55）。** [退出记录](history-references.md#ref-0bb32f584712)。P0/P1/P2/P4 主线阶段退出条件已齐备；P5 准入准备与对外交付前的阅读检查另行登记。下方带日期的旧状态保留为历史。

> **2026-11-01 工程交付完成**：T14.1/T14.2 登记样本、双解释器回归、A100 TP2、读取计时与归档公共重开已完成。完整 P2 退出及组件团队交付前的设计者阅读裁决另行登记。 [交付](history-references.md#ref-1dc484c35db2)。

> **2026-10-31 晚：D54 批准，T14 下发。** 完整 P2 的退出范围收窄为：已登记 TP2 + 设备通信明确 ID（或显式不可用原因）+ 时钟来源判定 + `perf.gap`。8 卡改为按需扩展，counter 移至 P3。P6-0（昆仑芯 P800 契约探针）与 T14 同批；对外交付前做读报告检查。[D54](../DECISIONS.md#d54) · [T14.1](history-references.md#ref-7271dc8d7c92) · [T14.2](history-references.md#ref-1fa80036fb14) · [T13 之后复核](history-references.md#ref-3b5745cc81f4)。
>
> **2026-10-31 T13 联合工程验收完成。** P4 按 D50–D52 登记小范围退出；OneRec 单卡生成诊断、匹配 A100 基线及 TP2 HCCL／FlagCX 对照完成。四条件矩阵、五组数值复算、双解释器各 634 项回归、最终 12 个入口的 dtfs 恢复通过；完整 P2 保持未退出。[验收与证据](history-references.md#ref-409c11becc2f) · [任务书](history-references.md#ref-a272319ae7a0) · [恢复接手](history-references.md#ref-7741c52a401f)。

# inference_v2 设计文档集（PLAN）

> **2026-10-30：D52 四任务工程验收完成。** T12.4/T12.5 两厂商设备与官方显存视图成立，T11.4 六目标 replay 和双侧性能连接成立，T11.5 三条件下钻、独立复算及双解释器 CI 通过。[联合交付](history-references.md#ref-9596bcfb2a96)。P2b 与 OneRec 匹配门限留后续；P4 退出提交设计者复审。

> **此前状态（2026-10-29，T12.3 工程交付）**
> - [T12.3](history-references.md#ref-f7a5d79a28b5)：已完成校验遍历合并与跨进程读取复用；S 首次导入约快 5%，重开全遍历约快 90%，双解释器回归通过。
> - P0/P1 沿用 D45/D49；P4 首批沿用 D51，T11.3 工程交付待设计者复审，完整 P4 待第二批。
> - [T12.2](history-references.md#ref-61027047fd40) 完成两轮 prepare 优化：M 320.59→204.19 s，外层 407.92→323.75 s；三个 M 配对、S、Reader 与双解释器 CI 通过。[验收](history-references.md#ref-c7d5393110c1)。
> - 200/300 s 软目标未完全达到；更新 profile 后按收益收敛停止，未进入第三/四轮生产优化。
> - OneRec 32 token 的匹配导出及真实门限继续待实测；P2a 设备/显存及 P4 第二批工程交付已完成，P4 退出待复审，P3 排序保持。

> 历史前置：S1 与 T1–T8.3 已执行，T8.3 返工轮设计者 Gate 通过（2026-10-16，[复审](history-references.md#ref-81d74b957ee6)）；遗留 L1–L6 已由 T9.0 关闭。当前实施遵循用户授权的 T9.0→T9.1→T9.2→T9.3 顺序，各段保留自己的独立 Gate。
> 角色分工：设计（本文档集）→ 用户评审 → codex 实现 → 证据回填本文档集
> 上游约束：[AGENTS.md「本人的最终任务」](history-references.md#ref-a54ff182c7e8)、[需求追溯](../00-requirements.md)
> 需求参考：`<私有开发工作区>/docs/` 下的实施方案与运行时功能拆分表（2026-10-13 新增）。它们是参考，不绑定设计（[D19](../DECISIONS.md#d19--需求文档是参考不绑定设计)）
> 推进方式：主线先行，当前只做推理全链路诊断（性能 + 精度）；扩展按 [D20](../DECISIONS.md#d20--分阶段推进主线先行) 排序，范围边界见 [D21](../DECISIONS.md#d21--范围边界暂不做与不做清单)

## 1. 一句话定位

inference_v2 是一个**与被测对象解耦的观测工具**：探针挂在进程上而不是模型上；设备侧复用官方 profiler（[D22](../DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合)），导入时同时写出规范事件与带来源的派生证据（原生关系、来源账本、证据索引，[D23](../DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)、[D24](../DECISIONS.md#d24--导入契约身份分层来源账本完成标记证据索引)）；性能、精度、故障诊断都是经公共只读接口读取这些数据的分析器。设计参照 nsys 的"schema 即产品"思路，但在同一条关联链上额外承载**数值**（张量）与**路由**（FlagGems 命中/回退），这是 nsys 不具备的。

## 2. 为什么要做 v2

v1（`FlagPerf_advance/inference`）把"被测模型"和"采集逻辑"写在同一个接口里：每接入一个模型需要实现约 35 个方法的 `ModelAdapter`。要覆盖 5 家厂商、500 个搜广推模型，这条路线的成本线性增长且不可接受。详见 [01-diagnosis.md](01-diagnosis.md)。

## 3. 文档地图

| # | 文档 | 回答的问题 | 状态 |
|---|---|---|---|
| 00 | [需求追溯](../00-requirements.md) | 要满足哪些需求？每条需求落在哪个设计/阶段？ | 草案 |
| 01 | [v1 现状诊断](01-diagnosis.md) | v1 哪里耦合？哪些资产保留、降级、重写？ | 草案 |
| 02 | [总体架构](../02-architecture.md) | 分几层？层间契约是什么？两种运行模式怎么并存？ | 草案 |
| 03 | [事件 schema v1](../03-event-schema.md) | 事件长什么样？store 怎么落盘？怎么演进？ | L-核心已冻结（附录 A 28 条）；L-字段 2026-10-27 随 P0 退出定为 `1.0`（[D45](../DECISIONS.md#d45)） |
| 04 | [关联脊梁](../04-correlation.md) | 一次 aten 调用如何关联到 FlagGems、Triton、设备 kernel 和 layer？ | 已按 Spike-C 与 T1–T8 回填；项目关联与原生关系分立 |
| 05 | [探针与厂商适配](../05-probes-and-vendors.md) | 有哪些探针？厂商需要提供什么？ | 草案 |
| 06 | [语义层与模型接入](../06-model-onboarding.md) | "逐 layer"从哪来？接入一个模型要写多少东西？ | 草案 |
| 07 | [分析器与报告](../07-analyzers-and-reports.md) | 性能/精度/回退等结论如何从事件中算出来？ | 草案 |
| 08 | [在线通道与故障诊断](../08-online-and-fault.md) | 卡死/OOM/通信中断怎么检测？恢复与热加载的边界在哪？ | 草案 |
| 09 | [CLI 与批量运行](../09-cli-and-campaign.md) | 用户怎么用？500 模型 × 5 厂商怎么组织？ | 草案 |
| 10 | [阶段计划](../10-roadmap.md) | 先做什么后做什么？每阶段的验收标准？ | 草案 |
| 11 | [Spike-C：CANN 算子级关联验证](11-spike-cann-correlation.md) | 架构的最大风险如何先验证？交给 codex 的实验方案 | 已执行并审核（2026-10-10） |
| 12 | [codex 交接约束](12-codex-handoff.md) | 实现方必须遵守的工程规则与交付格式 | 草案 |
| — | [决策记录](../DECISIONS.md) | 已定的决策、理由、待用户拍板的问题 | 持续更新 |

## 4. 推荐阅读顺序

**评审者（用户）**：README → 00 → 01 → 02 → DECISIONS → 03 → 04 → 11 → 10，其余按需查阅。

**实现者（codex）**：12 → 02 → 03 → 04 → 当前阶段在 [10](../10-roadmap.md) 中指向的文档。执行 Spike-C 时只需 12 + 11 + 04。

## 5. 文档间依赖

```text
00 需求 ──► 02 架构 ──┬─► 03 schema ◄── 04 关联 ◄── 11 Spike-C（验证 04 的假设，结论回填 03/04）
   ▲          ▲      ├─► 05 探针/厂商
   │          │      ├─► 06 模型接入
01 诊断 ──────┘      ├─► 07 分析器/报告 ──► 08 在线/故障
                     └─► 09 CLI/批量
10 阶段计划：按依赖顺序编排 03–09、11 的交付；12：所有实现工作的约束
DECISIONS：所有文档中"已定/待定"事项的唯一登记处
```

## 6. 术语表

| 术语 | 含义 |
|---|---|
| run | 一次采集。对应 store 中一个目录和一个 `manifest.json` |
| writer | 向 store 写事件的一个进程（或进程内一个写入通道）。多进程/多 rank 各自是独立 writer |
| probe（探针） | 一个独立的采集单元，声明自己产出的事件类型和开销等级 |
| event | store 中的一条记录，带统一信封字段，见 [03](../03-event-schema.md) |
| span | 带起止时间的区间事件。scope / phase / batch / generation step 都统一表达为 span |
| corr | 关联 id。同一次逻辑调用在不同层（aten → FlagGems → Triton → 设备）的事件共享或互相引用 |
| sig | 算子调用签名（shape/dtype/stride/device 等）的字典项，op 事件只引用其 id |
| call_key | 跨 run 对齐张量的确定性键（scope 路径 + 调用序号 + 边界名 + 样本），用于 Oracle–Device 差分 |
| attach 模式 | 不改目标模型代码，将探针接入进程。D33 默认 final 已交付：关闭时导出、目标退出后独立导入，不要求目标调用 `step()`；显式 windowed 仍按 step 推进。`record` 自动接入限直接 Python 目标（D32）。公共配置与命令见 [P1 使用说明](history-references.md#ref-0a9e456a7414)、[当前架构](../02-architecture.md#3-目标接入attach--wrap--launch)。 |
| seal 模式 | `record --seal`：封存输入并校验环境身份，用于可复现对比（如 FlagGems off/on、A100 vs 腾昇） |
| Tier 0/1/2 | 模型接入成本阶梯：零配置 / 约 20 行 YAML / Python 插件，见 [06](../06-model-onboarding.md) |
| analyzer（分析器） | store → 结论的纯函数，声明依赖的事件类型；依赖不满足时输出 `partial` |
| linked / ambiguous / unlinked | 关联状态。**绝不以时间邻近作为关联依据**，关联不上即标 `unlinked` 并计数 |

## 7. 文档维护规则

- 设计变更先改文档、后改代码；schema 冻结后的任何改动必须走 [03 §8 演进规则](../03-event-schema.md#8-版本与演进规则) 并登记到 [DECISIONS](../DECISIONS.md)。
- 每个阶段完成后，codex 在 [10](../10-roadmap.md) 的状态表中回填证据链接。
- 按项目规则，v2 的功能变化同步到 `personal/FlagPerf-advance.md` 与 `personal/guides/`（见 [12 §6](12-codex-handoff.md#6-文档同步义务)）。

## 历史状态记录

<details>
<summary>2026-10-30 更新前的 attach 术语说明（历史原文）</summary>

| 术语 | 当时的说明 |
|---|---|
| attach 模式 | 不改被测代码，把探针注入目标进程。广覆盖，不保证可比。**默认设计恢复"不改被测代码"（[D33](../DECISIONS.md#d33--恢复attach-不改被测代码)，用户 2026-10-20）**：显式调用 `handle.step()` 推进窗口降为可选项（用户接受官方 profiler 固有扰动作为采集代价）；默认路线的自动窗口推进尚待细化，且须给出超限行为或窗口上限的显式设计（单个 trace 超约 65,536 设备事件导入为 partial；导入耗时与内存随事件量线性增长，6 万设备事件约 110 s / 2.5 GiB）。**当前实现仍是显式 step（T9）**（`record --help` 写 "explicit profiler steps"），默认模式落地前 record 并非"不改被测代码"的接入 |

</details>



<details>
<summary>本轮交付前状态（原文）</summary>

> **当前状态（2026-10-28，P1 退出与 P4 首批下发）**
> - 设计者完成 P1 退出复审：[D49](../DECISIONS.md#d49)（用户 2026-10-28 批准）判 P1 在登记范围内通过并退出；耗时头条与回退清单的口径由 P1.1 [T10.4](history-references.md#ref-3f47daaaa459) 修补，修补前这两类产出不对外交付。入口：[退出矩阵](history-references.md#ref-567cd9829d3a)、[复盘页](history-references.md#ref-8578565fdbb8)。
> - P4 首批已下发：[D50](../DECISIONS.md#d50)（用户 2026-10-28 批准，含 Q10 裁决“不复用 msprobe 作为数据来源”）与任务单 [T11.1](history-references.md#ref-6deef0520efd)、[T11.2](history-references.md#ref-97b0743ec8d6)；L-字段拟升 `1.2`（tensor 增加 `src`）。P1.1 与 P4 首批并行。
> - P3 不提前（[Q11](../DECISIONS.md#q11--p3故障检测与热加载是否提前)，用户 2026-10-28 裁决）：当前目标是先实现 v1 的已有功能，再让 v1 退休。P1 已知限制只在 D49 第 4 条与退出矩阵列出。
> - 此前状态原文保留于历史区。

</details>

### 2026-10-28 P1 退出复审前的状态块（原文）

> **当前状态（2026-10-27，P1小规模工程验收交付）**
> - T10.1探针/接入与T10.2有界分析/四面板报告已实现；T10.3小规模正式功能矩阵与N1初步观察已完成，**待设计者P1退出复审**。入口：[退出矩阵](history-references.md#ref-567cd9829d3a)、[正式报告](history-references.md#ref-86ccff91a5b8)、[使用指南](history-references.md#ref-0a9e456a7414)。
> - 按用户最新修订，模型、输入、镜像、include与采集配置保持；最终源码共用四条件smoke，正式R1–R3/C0G/C0，N1-a三对、N1-b一对。保留原2h NPU/4h墙钟与全部历史用量；未使用12h扩容。
> - 新run使用D47的schema1.1/dev/nested与route_table；D48生命周期、同次身份、公共读取和有界导出已接入。R3 linked=1692/54792；R3非零compile分母为空，单列未覆盖。
> - N1-a中位比0.043079，显示本负载的perf-lite开销较大；N1-b单对1.002043只描述本次。系统性性能统计与大规模容量验证另行安排。编译未覆盖、时钟、240万记录的有效超时和镜像归档失败均保留。
> - P0已有D45裁决保持；P2/P4的排期交由P1复审决定。此前状态原文保留于历史区。


### 2026-10-27 P1执行前状态（原文）

> **当前状态（2026-10-27，P1 首批下发）**
> - **P1 首批已下发，待实施**：首批三张任务单 [T10.1](history-references.md#ref-29067dfd8983)（探针框架与主机侧关联链）、[T10.2](history-references.md#ref-51d0d35bc759)（分析器、span 树与最小报告）、[T10.3](history-references.md#ref-d60f95337bf0)（真实模型验收与退出证据） 一次下发，设计决定见 [D46](../DECISIONS.md#d46)。中途不设 Gate，终点为设计者 P1 退出复审；日历时限依次为 11-03、11-05、11-07，超时即上报。
> - **2026-10-27 首批修订（[D47](../DECISIONS.md#d47)，用户批准方案 B）**：opsum 聚合键增加 `dev`、`nested`，L-字段升为 `1.1`；route 探针写静态判定表 `route_table`，判定范围为深度 0 且加速器调用；只在 summary 出现的回退行按 `summary_only` 降级。三张任务单已同步修订，尚未开工。
> - **执行契约已补齐（[D48](../DECISIONS.md#d48)）**：按用户本轮要求保留 D47，补齐模型/组件生命周期、final 配置、FlagGems-on 无探针基线与有界分析；关联错绑等问题由少量开发验收反例定位，修复后经针对性测试关闭。三张任务单及设计索引已同步。
> - P0 已于 2026-10-27 退出，L-字段 `1.0`（[D45](../DECISIONS.md#d45)，[退出矩阵](history-references.md#ref-e0872549b0c8)）；已知限制照旧适用。D32 已确认。
> - FlagGems 环境：用 v1 基准所用的镜像 `1147704d…`（`flagperf/inference-ascend:20260923-tp-flagtree-flagcx`，已装 FlagGems `f7ae8e6b`），不另装包。
> - 本次下发之前的状态块原文移至文末[历史状态记录](#历史状态记录)。


### 2026-10-27 P1 首批下发前的状态块（原文）

> **当前状态（2026-10-27）**
> - **P0 已退出，L-字段定为 `1.0`**（[D45](../DECISIONS.md#d45)，用户 2026-10-27 批准；[退出矩阵](history-references.md#ref-e0872549b0c8)）。③ 按登记低负载范围判定达标，附已知限制；T9.7 的 W7 有效失败降为跟踪基准，W9 以 94.44% 供给下的零回退观察作为 aarch64 A5 证据；原失败与无效样本均保留。D32 已确认。
> - 已交付：T9.7 Writer Stage 1（进程外代理写者接入生产，离线入口复用共享写入优化）；T9.8 真实推理闭环校准（真实轻量记录约 70 条/s，轻量记录扰动测不出）。见 [T9.7 报告](history-references.md#ref-0ffd66d1ab4a)、[T9.8 报告](history-references.md#ref-e93d1c4566d0)。
> - 下一步：P1（Python 侧关联链）首个任务单，按 D45 第 7 条的设计约束与第 9 条的轻量验证协议编写；并行准备装有 FlagGems 的环境。
> - 2026-10-27 之前的状态块原文移至文末[历史状态记录](#历史状态记录)。

### 2026-10-27 P0 收尾前的状态块（原文）

> **当前状态（2026-10-27）**
> - 已完成：T9.4 生产入口收口（独立代理 Gate，2026-10-20）；T9.6 Reader 流式化登记验收通过（2026-10-23）；T9.5d ext4 正式验收后 S9 已收口，G1 极差 0.035036 与 F11 由用户按 [D42](../DECISIONS.md#d42) 处置为非阻塞。
> - 未完成：T9.5c 硬条件原失败保留；T9.7 W7 有效回归失败、W9 缺有效 aarch64 A5，正式验收未通过；P0 未退出，L-字段保持 1.0-rc（[退出矩阵](history-references.md#ref-e0872549b0c8)）。
> **2026-10-27 D44 本轮交付**：T9.8 原始 e385f495 正式校准与独立身份整合后闭环均已完成；T9.7 在线代理／离线共享优化、O2/O3、最终双解释器／134 故障条件、W3–W6/W8、X86 A5 与真实入口已交付。新四对 A/A 合格，正式十对 W7 中位 0.946409 低于原门 0.963973，依用户决定非阻塞且不再修复；aarch64 W9 供给仅 94.44%，低于 98%，原五分钟余量不足重验，停止并保留 blocker。W1 65 分钟、W8 30 分钟已获批准，总八小时／十二小时不变。工程原门、完整测量通过、P0③与 P0 退出均未成立。 见 [T9.8 报告](history-references.md#ref-e93d1c4566d0)、[T9.7 报告](history-references.md#ref-0ffd66d1ab4a)及 [Gate 材料](history-references.md#ref-a1968734253f)。
> - 2026-10-26 前叠放在文首的带日期状态横幅，原文移至文末[历史状态记录](#历史状态记录)。

### 2026-10-26 状态块合并前叠放的横幅（原文）

以下为 2026-10-26 状态块合并前叠放在文首的带日期状态横幅，按原叠放顺序（新→旧）原文保留，仅作历史语境；当前状态以文首“当前状态”块为准。

> **2026-10-25 最新：S9 已收口；G1 极差与 F11 已由用户接受为非阻塞项，可继续后续生产推进。** a04/a05 补证：store 329、correlate 79、conformance 21 全过，417-case/50,040 次比较与最终字节一致完成。原正式结果保留；本轮未执行生产接入，P0 未退出。以下早期失败/pending 条目保留历史语境，不再以当前 G1/F11 阻碍推进。 [D42](../DECISIONS.md#d42)；[收口证据](history-references.md#ref-03dff9b5e6ad)。


> 2026-10-20 授权修订：[D34](../DECISIONS.md#d34--t94t95-读取批次与测量边界补充)与 T9.4 M8/M9、T9.5 §4.7 补充公共读取校验、首批屏障、质量/身份和测量/关闭边界。P0 未退出；历史验收与旧状态保留。当前任务具体规则以两份修订任务单为准。


> **2026-10-18 设计者 Gate 结论（T9.0–T9.3，[设计者审核](history-references.md#ref-79d0492278f2)）**：T9.0 通过；T9.1 通过，附测试缺口；T9.2 有条件通过（record 自动注入范围由实现方收窄而未进 DECISIONS，以 [D32](../DECISIONS.md#d32--record-自动注入范围只注入-wrapper-直接启动的-python-目标) 追认）；T9.3 工程/测量闭环成立，但 ⑩ 的日志项与 ② 的 truncated/lost 只在验收 harness 路径成立、生产入口未贯通，另有 inspect 对不存在路径退出 0（生产缺陷）与验收工具退出码缺口，开收口轮 T9.4。**P0 未退出，L-字段保持 1.0-rc**；③ 仍为 failed（旧口径）；设计者提出的 [D31](../DECISIONS.md#d31--p0-吞吐验收改为三项并报退役单一-accepted-速率)（改口径）**已由用户采纳（2026-10-19）**：③ 改为 ③a/③b/③c 三项并报。
> **下一步**：[T9.4](history-references.md#ref-e525188761b6)（生产入口收口，任务单已编写，**待用户确认后下发**）；**D32 待用户确认**；D31 已采纳（用户 2026-10-19），据此已写 [T9.5](history-references.md#ref-06e993298ce0)（Writer 消费端移出 GIL 与 ③ 三项验收，**待用户确认后下发**，Stage 0 可与 T9.4 并行）；提交当前工作树文档也待用户决定。顺序与细节见 [12 §8](12-codex-handoff.md#8-当前待办)。

> 2026-10-18 执行更新（codex 实现方口径，保留；其中“独立 Gate”为实现方内部复核，设计者 Gate 结论见上）：T9.0–T9.2 已实现并通过独立 Gate；T9.3 已实现窗口策略、独立快照封存及 record 子进程边界修复，真机最终版本验收执行完成。Q9 三组预算完成，500,000 accepted/s 未达标；**P0 未退出，L-字段保持 1.0-rc**。当前状态和证据以 [退出核查](history-references.md#ref-e0872549b0c8) 与 [T9.3 执行记录](history-references.md#ref-61e29505d51c) 为准。


<details>
<summary>T12.2 交付前入口状态（原文）</summary>

> **当前状态（2026-10-29，T11.3/T12.1 工程交付）**
> - P0 按 D45 在登记低负载范围退出；P1 按 D49 退出，P1.1 T10.4 已交付。
> - P4 首批在登记范围内通过 D51 复审；[T11.3](history-references.md#ref-3c84559c539d) 六报告、字段映射和双解释器 CI 完成，待设计者复审。[验收](history-references.md#ref-4ea9cd3ce263)。完整 P4 退出仍需第二批。
> - P2a 进行中：[T12.1](history-references.md#ref-6ce6c9c166a0) S/M 成本、profile 与单参数对照完成，缓存增大未观察到提速。[交付](history-references.md#ref-a103fc7f312b)。下一步由用户定门限后登记 T12.2。
> - OneRec 目标为 32 个新 token，当前仅有明确假设的敏感性区间；L 代表性条件未满足而不执行。P3 不提前，P5 前按 v1 能力清单排序。决策见 [D51](../DECISIONS.md#d51)。

</details>
