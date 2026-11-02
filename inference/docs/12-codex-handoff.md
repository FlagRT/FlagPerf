> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书链接转入 [历史引用索引](history-references.md)，其出处保留。

> **当前（2026-11-02）：P2 按 D54 登记范围通过并退出。** [D55](DECISIONS.md#d55) · [退出与接手](history-references.md#ref-0bb32f584712)。完整请求窗口已可用，未知误差界保留；对外交付阅读检查单独登记。下方 10-31 状态为历史。

> **2026-10-31 T13 联合工程验收完成。** P4 按 D50–D52 登记小范围退出；OneRec 单卡生成诊断、匹配 A100 基线及 TP2 HCCL／FlagCX 对照完成。四条件矩阵、五组数值复算、双解释器各 634 项回归、最终 12 个入口的 dtfs 恢复通过；完整 P2 保持未退出。[验收与证据](history-references.md#ref-409c11becc2f) · [任务书](history-references.md#ref-a272319ae7a0) · [恢复接手](history-references.md#ref-7741c52a401f)。

# 12 codex 交接约束

> **历史状态（2026-10-31，D54 批准与 T14 下发；当时原文）**
> - 用户批准 [D54](DECISIONS.md#d54)，下发 [T14.1](history-references.md#ref-7271dc8d7c92)（P2b 设备通信、时钟、设备忙闲；含一次已批准的 A100 2 卡 TP2 采集，环境由用户准备）与 [T14.2](history-references.md#ref-1fa80036fb14)（设备头条与窗口身份、冷热读只测、使用页、personal 重组）。两单的冻结输入、读法规则与下发前自检记录以任务单为准。
> - 设备忙闲读法（D54 b）：头条用 compute（+memcpy）忙时间并集；通信任务占用单列“跨 rank 下界 + 超出部分”；`NOTIFY_*` 不计忙；每个并集旁给出时长和。跨窗口并列由导出的窗口身份经代码检查（D54 c）。
> - P0/P1/P4 的 D45/D49/D50–D53 结论保持；完整 P2 未退出，退出范围按 D54 第 1 条。L-字段为 `1.2`。P3 不提前；P6-0 任务单待 P800 窗口确认。
> - 读报告检查（对外交付前，D54 第 3 条）、端到端估时、输出逐值相等、来源守恒、失败不改判沿用。常规工程在已授权范围连续完成，契约、语义、采集配置、资源范围的实质改变只暂停受影响分支。
> - FlagPerf_advance 只在本地管理，不 push（D52 第 6 条）；保留其他工作流的未提交材料。此前状态原文移至历史区。

> 对象：实现方（codex）｜ 本文件优先于所有其他文档中的实现建议
> 阅读顺序：本文件 → [02 架构](02-architecture.md) → [03 schema](03-event-schema.md) → [04 关联](04-correlation.md) → [10](10-roadmap.md) 中当前阶段指向的文档

## 1. 仓库与分支规则（项目硬规则）

- `main` 只用于同步 fork 上游，**不要**往里提交我们的修改；
- `dev-1.0` 只接受 PR，**不可直接 push**；
- **FlagPerf_advance** 在 `dev-zkm` 做本地开发与版本管理，**不 push**（用户仓库策略与 D52 第 6 条）；正式云端交付走 **FlagPerf** 自建分支与 PR；
- 工作目录：`FlagPerf_advance/inference_v2/`。**不要修改 `FlagPerf_advance/inference/`（v1）**，它在 P6 之前保持可用（决策 D5）。

## 2. 实现顺序

严格按 [10 阶段计划](10-roadmap.md) 推进。schema 不整体冻结，按 [D11](DECISIONS.md#d11--schema-分三层冻结不是一次性整体冻结) 分层：**L-核心**（[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结)）冻结后即可开始 P0 主包代码；**L-字段**在 P0 结束时定为 1.0；**L-厂商**不冻结。S1 与其后续 T1–T3 的结论已回填（[D15](DECISIONS.md#d15--t1t3-之后的契约回填r1r8)）。2026-10-14 附录 A 扩充为 28 条（#25–#28：原生关系与项目归属分立、来源处置、身份三层、导入完成标记；不改事件字段）。

每个阶段的退出条件在 10 中逐条列出，未达成不进入下一阶段。阶段完成后在 [10 §9](10-roadmap.md) 的状态表回填证据链接。

## 3. 不可违反的设计约束

这些来自设计文档，在此集中重申。违反其中任何一条，即使功能可用也要求重做：

1. **探针不得 import 模型/benchmark 代码**，不得 import 其他探针（[05 §1](05-probes-and-vendors.md)）。
2. **分析器与报告不得 import torch 或厂商 SDK**，必须能在无 torch 的环境中运行（N3）。CI 里要有这个环境。
3. **关联不得使用时间邻近作为判定依据**（[04 §1](04-correlation.md)）。时间只能用于排除候选，且排除须带保守容差并计数。除代码审查外，这一点由 [T1](history-references.md#ref-867dfc27c4dd) 的 `X8`（设备时间整体平移后 `linked` 必须逐条不变）机械地验证。
4. **厂商差异只出现在 `vendors/`**。`store/`、`correlate/` 框架、`analyzers/`、`report/` 中不允许出现 `if vendor == "ascend"` 这类分支（[02](02-architecture.md) P6）。
5. **写事件不得阻塞目标进程**。队列满就丢弃并计数（[03 §6.3](03-event-schema.md#63-写入丢弃与崩溃)）。**`sig` 不走队列，但也不阻塞**：产生侧把定义放进进程内字典再入队引用事件；写线程是唯一落盘者，在某分片首次写出引用某 sid 的事件之前先写出该 `sig`。由此保证**可解码不变式**：同一分片内，任何存活的引用事件之前都已有其 `sig`（[03 §4.2](03-event-schema.md#42-sig--算子调用签名字典)）。字典超预算时按 03 §4.2 截断或丢弃引用事件并计数，不阻塞。
6. **探针安装失败只降级不中断**：写 `health(sev=warn)`，标记未启用，run 继续（[05 §1](05-probes-and-vendors.md)）。
   > **当前实现约定（2026-10-27，[D46](DECISIONS.md#d46) 第 12 条）**：`health.kind` 的枚举里没有表示“探针降级”的值，P1 不扩充枚举。降级记入 attachment `state.json` 的 `capabilities.unavailable`（带 `reason`）与 `errors`，并经 `inspect` 显示；枚举的扩充留到 P3。
7. **分析器依赖不满足时输出 `unavailable`/`partial`，不抛异常**（[07 §1](07-analyzers-and-reports.md)）。
8. **不设精度阈值**，分析器只给指标不给 pass/fail（[07 §3.3](07-analyzers-and-reports.md)）。
9. **schema 分三层冻结**（[D11](DECISIONS.md#d11--schema-分三层冻结不是一次性整体冻结)）：**L-核心**（[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) 的 28 条，#25–#28 为 2026-10-14 扩充）P0 开工即冻结，改动走 major 并须先改 PLAN 正文；**L-字段** P0 结束定 1.0，加字段/加枚举走 minor；**L-厂商**（`ext` 键、规则 id、厂商分类表）不冻结，可自由演进。厂商代码的长期门禁锁**输出**而不锁源码：历史样本的 link、原生关系与处置逐字节不变；`correlate/linker.py` 保留逐字节锁（[D25](DECISIONS.md#d25--冻结语义与输出不冻结厂商分类表)）。
10. **不要重构 FlagPerf 原有框架**。inference_v2 是新增目录，与 FlagPerf 原型的既有结构保持兼容，不做跨目录的大规模改动（项目规则）。

## 4. 代码规范

| 项 | 要求 |
|---|---|
| 语言 | Python 3.11（与镜像一致） |
| 运行环境 | 已验证镜像 `flagperf/inference-ascend:20260922`（设计者 2026-10-11 核对）：`/usr/local/python3.11.15/bin/python3`、torch 2.10.0+cpu、pytest 8.3.2，**无 `zstandard`**（故封口压缩默认 gzip，[D16](DECISIONS.md#d16--封口分片的压缩编码记在分片条目上默认-gzip)）。宿主机 Python 是 3.9.9，只用于 spike 脚本的离线重放。**不得为本项目往镜像里安装新包**；确需新依赖时按 [README §1](history-references.md#ref-f967c07859b9) 提出。A100 实机验证（[T8](history-references.md#ref-24a77204cd82)）使用此前已验证的 A100 环境，同样不装包，解释器与 torch/CUDA 版本随产物登记；`flagprof` 须能在该解释器上导入 |
| 类型 | 公开接口必须有类型标注；`Protocol` 定义契约 |
| 依赖 | `store/`、`analyzers/`、`report/` 只依赖标准库 + 少量纯 Python 包；torch/厂商 SDK 只能被 `probes/`、`vendors/`、`runner/` 依赖 |
| 日志 | 工具自身的日志与事件流分开；不要把调试信息写进事件 |
| 配置 | 不要新增全局单例；配置从 `attach()` 参数流入。允许的例外只有两个进程级守卫：writer 的 attach 守卫，以及采集会话守卫（[D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合) 第 2 条）。两者都只记录本进程状态，并在 fork 后清空 |
| 错误处理 | 采集路径上的异常一律捕获并降级，绝不让探针的 bug 杀死被测进程 |
| 命名 | 事件字段名以 [03](03-event-schema.md) 为准，**一个字母都不要改** |

## 5. 测试要求

| 层 | 要求 |
|---|---|
| schema | 对全部 13 类事件的校验器测试；版本兼容测试（读取方忽略未知字段/类型）；`ts=null` 事件合法 |
| store | 写入-读取往返；**kill -9 后可读**；截断行处理；seq 缺口检测；溢出丢弃计数；以及 [10 P0 验收](10-roadmap.md#2-p0--store-与-schema) ④–⑦ 列出的反例：分片中途新签名、双线程受控乱序、尾部连续丢弃后 kill、同 rank 多 writer 与重启、封口/压缩期间尾读 |
| correlate | 用 `spikes/*/fixtures/` 中的**真实 trace 片段**，不用人造数据（[04 §7](04-correlation.md)）；断言确定性：同输入同输出，且**规则顺序与输入顺序的排列不改变结果**（[04 §3.6](04-correlation.md#36-linker-算法候选并集与判定)）；门禁同时检查不变量、已审核预期表和输出契约字段（`segments`/`level`/`reason`/`evidence`），任一缺失即失败 |
| probes | 安装/卸载幂等；卸载后进程状态恢复；失败降级路径 |
| analyzers | 在缺少某类事件的 store 上返回 `unavailable` 而非报错；coverage 字段完整 |
| 端到端 | 每个阶段至少一个真实模型的端到端 run + 报告 |
| 性能 | 探针 off/on 的开销对照（N1），结果进报告 |
| 新增判定（2026-10-16，[D29](DECISIONS.md#d29--判定一致性结构证据优先于名称契约条目须被执行或被测试) 第 5 条） | 写出必须成立的**性质**，在全部真实样本上以门禁检查，不只断言计数；"预期无告警"配同类输入的阳性对照；**变异杀伤**：把新增判定改为常值或交换分支时至少一项测试失败，Gate 输入附变异结果。同一事实有两个判断时，须同源或有一致性门禁（D29 第 1 条） |
| 普遍规则的落地（2026-10-16，T8.3 返工复审 [§3.1](history-references.md#ref-fb9020d89b7e)、[§3.2](history-references.md#ref-a9b975fef97b)） | **谓词清点**：涉及"某事实由谁判断"时，Stage 0 列出全仓该事实的全部判断位置，验收要求每处同源或有一致性门禁，任务单点名的文件只是例子；**双方变异清单**：复核方在验收时另设计变异，每个新增条件 id 至少有一个"改为常值"的变异；**契约数据变异**：对契约文件删条件行、改值，与代码变异并列，各契约的条件 id 集合与预登记逐项比较并进 CI；**输入隔离**：测试改一个字段时，排除其他能接住结果的信号（如队列键须两端同值非法，设备信号须逐一隔离） |

CI 中必须有一个**无 torch**的 job 跑 store/analyzer/report 的测试。做法是在上述镜像中以 `python3 -S` 运行（禁用 site-packages 后 torch 与 pytest 都不可见），因此**测试一律用标准库 `unittest` 编写**；它们同样能被 pytest 收集运行。

## 6. 文档同步义务

按项目规则，代码或功能变化必须同步文档：

| 变化 | 同步到 |
|---|---|
| v2 的功能新增/变更 | `personal/FlagPerf-advance.md` |
| 使用方式变化 | `personal/guides/` 下对应文档 |
| 设计变更（接口、schema、关联规则） | **先改 PLAN 文档，再改代码**；schema 变更另记 [DECISIONS](DECISIONS.md) |
| 阶段完成 | [10 §9](10-roadmap.md) 状态表 |
| 对 FlagPerf 原型机制的新认识 | `personal/FlagPerf-origin.md`（**只记原始内容，不含我们的优化**） |

制作汇报文档/生成器（report.md/html 等）前，先读 `report_rules.md`，必要时参考 `内容讲解方法论.md`。

## 7. 提交与汇报

- 提交粒度按阶段内的可验收单元，commit message 说明对应的阶段与需求 ID；
- 每个阶段结束时提交一份简短的阶段报告：做了什么、验收项逐条结果、**未达成的部分与原因**、对设计文档的修改建议；
- 遇到与设计文档冲突的实现障碍，**不要自行改设计**：在 [DECISIONS](DECISIONS.md) 中提出问题并等待确认。设计文档的修改权在设计者。

**交接规则补充（2026-10-18，T9.0–T9.3 设计者审核；流程规则，无 D 号，登记于 [DECISIONS 变更记录](DECISIONS.md#变更记录)）**：
1. **Gate 的 reviewed_input 必须是已提交的 blob。** 干净检出必须得到被认证的文档与代码；未提交的工作树版本不得作为 Gate 输入（T9 总 Gate 认证的 PLAN 文档只存在于未提交工作树，见[设计者审核](history-references.md#ref-79d0492278f2) §1.1）。
2. **改设计范围、动用设计者保留裁决，须先在 DECISIONS 提议**（可先由实现方提议）。"授权顺序执行"不等于"授权改设计"。已发生的两例：record 自动注入范围（[D32](DECISIONS.md#d32--record-自动注入范围只注入-wrapper-直接启动的-python-目标) 追认）、NVIDIA `exporter.version` / `execution.mode` 的取舍（[D30 补充](DECISIONS.md#d30--契约条件的证据先查产物自述每个必需条件登记生产者)追认）。
3. **每条验收写明验证所经的生产入口。** 在验收 harness 路径成立不等于生产入口成立；验收工具不得使用生产入口没有的配置；store 记录的每种失败状态都必须经公共只读接口可见。
4. **复核脚本进 `tests/` 或 CI。** 留在 `docs/validation/` 的复核脚本不算回归保护。
5. **多 case 工具的退出码由逐 case 状态汇总**，零 case 非零退出。
6. **任务单沿用 codex 自评 §5.2 的六段大纲**（[自评](history-references.md#ref-1a5b874163ee)）：1 要支持的判断与完成边界；2 输入身份与现有证据；3 机制与不变量；4 预登记实验；5 验收和反例；6 回填和下一决策。
7. **（2026-10-19）任务单须附设计者下发前自检记录**（[codex-tasks/README §5](history-references.md#ref-6fe64a9751e0)；流程规则，无 D 号，登记于 [DECISIONS 变更记录](DECISIONS.md#变更记录)）。本条约束设计者；实现方遇到缺少自检记录的任务单，可拒绝开工并在 DECISIONS 提问。

<a id="collab-20261026"></a>

**协作纪律补充（2026-10-26，用户决定）**：
1. **一个任务一个提交**，或以任务号为前缀的连续提交。提交只按路径暂存本任务的文件（`git add <路径>`，不用 `git add -A`/`git commit -a`），不得把其他工作流的未提交文件一起提交。反例：`9a6cc661` 把 T9.5d 候选与 D33 的 `flagprof/` 改动混在一个检查点提交里，导致 T9.6、T9.5c 封存包的源码身份同时过期。
2. **D33 不另开分支**，继续在 `dev-zkm` 推进；提交同样按任务分开，D33 与 Writer/T9.x 的文件不混入同一提交。
3. **入口文档的状态块只替换、不叠加**：PLAN 入口文档与任务单文首只保留一个“当前状态（日期）”块；新状态替换旧块，旧块原文移入该文件的“历史状态记录”小节，不删除、不改写。
4. **冻结证据的身份过期**：后续提交后，冻结证据的校验脚本报“源码身份不符”（如 `source:…`、`frozen_input:…`）属于身份过期，不等于原结论失效。应另做重新绑定（新目录、新的输入 SHA 与必要的差分重跑，如 [T9.6 重新绑定](history-references.md#ref-4f341f4eb690)），不改原封存包、原校验脚本与原结论。

## 8. 当前待办

> **2026-10-27 更新（P1 首批下发）**：P1 首批三张任务单已下发——[T10.1](history-references.md#ref-29067dfd8983) → [T10.2](history-references.md#ref-51d0d35bc759) → [T10.3](history-references.md#ref-d60f95337bf0)，设计决定见 [D46](DECISIONS.md#d46)，终点为设计者 P1 退出复审。
>
> **2026-10-27 更新**：P0 已退出（[D45](DECISIONS.md#d45)），8.3 第 5 条“P0 退出复审 → P1”已完成，D32 已确认。下一步为 P1 首个任务单（设计者编写，约束见 D45 第 7、9 条）。下文 8.1–8.3 保留为历史待办。
>
> 2026-10-18 更新：T9.0–T9.3 的设计者审核已完成（[设计者审核](history-references.md#ref-79d0492278f2)，codex 自评见[此处](history-references.md#ref-1a5b874163ee)）。本节先给出当前待办（审核 §7 的顺序）；其后（8.4）是 2026-10-18 前的待办与已完成任务，作为历史保留。
>
> 2026-10-19 更新：用户采纳 [D31](DECISIONS.md#d31--p0-吞吐验收改为三项并报退役单一-accepted-速率)（P0 ③ 改为 ③a/③b/③c 三项并报，Q9 关闭）；[T9.5](history-references.md#ref-06e993298ce0) 任务单已写，用户 2026-10-19 确认下发（待用户提交工作树后开工）（Stage 0 可与 T9.4 并行）。8.3 已按此更新；**仍待用户**：提交当前工作树文档、确认 [D32](DECISIONS.md#d32--record-自动注入范围只注入-wrapper-直接启动的-python-目标)、下发 T9.4 与 T9.5。
>
> 2026-10-20 更新：用户主动提出并确认 [D33](DECISIONS.md#d33--恢复attach-不改被测代码)：默认模式恢复"attach 不改被测代码"，显式 `handle.step()` 推进窗口降为可选项（用户接受官方 profiler 固有扰动作为采集代价）；路线细化须正视两个数据完整性边界（单个 trace 超约 65,536 设备事件导入为 partial；导入耗时与内存随事件量线性增长），所选路线须给出超限行为或窗口上限的显式设计。8.3 新增第 6 条待办；"仍待用户"各项不变（D33 无需用户再动作）。
>
> 2026-10-21 更新：T9.5 Stage 0 设计者 Gate 已完成（[审核](history-references.md#ref-44739a19cd50)）：选定同进程候选 A，淘汰 B；Stage 1 待用户确认 [D37](DECISIONS.md#d37--t95-gate-裁决在线-emit-按模式分契约弃用以现行-writer-为参照的时间比较设计者提议待用户确认) 后放行，届时设计者修订 T9.5 Stage 1 部分再请用户下发。**新增待用户**：确认 D37。（同日晚修订：B 类重做 smoke 后选型重新打开，D37 第 1 条暂缓；待用户确认是否让 codex 对 B2X 做登记协议补充取证，见[审核 §8](history-references.md#ref-fe14eae2f1ea)。） 用户随后确认下发 [T9.5b](history-references.md#ref-34cf6c3c0cce)（待用户提交工作树后开工）；[D38](DECISIONS.md#d38--附录-a-19-澄清写者--writer-实例的唯一物理写盘进程允许登记代理设计者提议待用户确认) 待用户确认。
>
> 2026-10-22 更新：T9.5b 设计者 Gate 已完成（[审核](history-references.md#ref-2a47357d5a94)）。用户裁决：方向选 B2X 类、不保留 A；Stage 1 前插入 [T9.5c](history-references.md#ref-cdbf854eab27)；Reader 流式化 [T9.6](history-references.md#ref-6d7e83e32eee) 为 P0 退出前置（[D39](DECISIONS.md#d39--t95b-gate-裁决选-b2x-类不保留-astage-1-前插-t95creader-流式化t96为-p0-退出前置)）。D38 标题已按状态行同步为用户已确认。顺序：T9.6 → T9.5c → T9.5 Stage 1 → P0 退出复审（T9.6 不得在 T9.5c 预登记至测量结束之间合入（source_identity 覆盖整个 flagprof），Stage 1 在 T9.5c Gate 通过且 T9.6 合入后开工）。见 8.3 第 5 条与本节末的 P0 退出条件。

> 2026-10-23 更新（用户已确认）：后续默认执行环境为新工作区的 dtfs 共享目录，见 [D40](DECISIONS.md#d40)；T9.5c 的容量报告与验收按修订任务单执行。历史 ext4 记录保留。本次仅调整文档中的环境与性能处置口径，任务顺序、镜像登记前置及 P0 退出条件中的其他要求沿用现行规定。

### 8.1 两类复核的区分

- **codex 内部复核**：由 codex 主会话派出的子代理执行（各段记录中称"独立 Gate"），属实现方内部的弱独立，可作 Gate 输入，**不替代设计者 Gate**。T9.0 §6 原文是"设计者派出的复核代理"，执行中被改为"独立复核代理"；子代理还按主会话授权修改了 acceptance.json / p0-exit.json 的 Gate 字段。旁证它不是橡皮图章：T9.0 初轮 R1/R2 未过，T9.1/T9.2/T9.3 均查出问题。
- **设计者 Gate**：设计者审核，复核代理由设计者派出。T9.0–T9.3 的设计者 Gate 是 2026-10-18 一次覆盖四段的事后审核。

### 8.2 T9.0–T9.3 设计者 Gate 结论（审核 §0）

- T9.0 通过：L1–L6 全部有关闭证据；revision 锁在完整历史下拒绝同 revision 改写，浅克隆 fail-closed。
- T9.1 通过，附测试缺口（证据 SHA / 未登记 / 路径逃逸 / 重复路径 / 第二信号类型未钉）。
- T9.2 有条件通过：record 自动注入范围由实现方在 T9.3 收窄而未进 DECISIONS（本轮以 D32 追认）；corr 唯一性、fork 后附件可见、record 完成判定各层谓词缺独立测试。
- T9.3 工程/测量闭环成立，但两项 P0 条件只在验收 harness 路径成立，生产入口未贯通：⑩ 的"日志报错判 invalid"在 record 路径未接线；② 的 truncated/lost 不经 inspect 暴露。另 inspect 对不存在路径退出 0（生产缺陷），验收工具退出码仍有同族漏洞。开收口轮 T9.4。
- P0 未退出，L-字段保持 1.0-rc（与 codex 结论一致）。③ 仍为 failed；复核证明 ③ 的指标本身测错了对象，设计者提出 D31（待用户裁决；**2026-10-19 用户已采纳**，见 8.3）。
- 流程级问题：总 Gate 认证的 PLAN 文档只存在于未提交工作树（审核 §1.1）。

### 8.3 下一步（顺序，审核 §7）

0. **用户决定提交当前工作树文档。** 工作树混有三源：用户全局用词替换、设计者 10-16 未提交稿、codex 10-18 回填与未跟踪自评；本轮文档更新完成后再提交，使 Gate 输入可复现。设计者不主动 commit。
1. **[D31](DECISIONS.md#d31--p0-吞吐验收改为三项并报退役单一-accepted-速率) 已采纳（用户 2026-10-19）；仍待用户确认 [D32](DECISIONS.md#d32--record-自动注入范围只注入-wrapper-直接启动的-python-目标)。**（此前该项为“用户裁决 D31、确认 D32”。）
2. **[T9.4](history-references.md#ref-e525188761b6) 收口**（任务单已写，2026-10-19 下发前核查后三轮修订，用户 2026-10-19 确认下发（待用户提交工作树后开工）；不依赖 D31）：
   - M1 inspect 显示 run / 窗口状态（manifest status、valid / invalid / lost、exit、tail_unknown）；
   - M2 inspect 路径不存在 / 空目录 / 非 run 目录非零退出（修复，当前退出 0，生产缺陷）；`--verify` 失败非零退出、失败行 integrity 不默认 verified 当前已正确（`cli.py:39`、`cli.py:52`），只需用测试钉住（G05–G08）；
   - M3 record 日志来源接线（显式 `--log-file` 或登记的 SDK 日志位置；未提供时窗口日志检查记为 unchecked 并经 inspect 可见，不得静默视为无错）；
   - M4 验收工具退出码；
   - M5 用测试杀死复核方存活清单中的真实缺口，把 check_session.py / check_sealing.py 并入 tests，补 3 个 fake 端到端测试与 active=2 窗口测试；
   - M6 Gate 输入入库；
   - M7 导入器分批 flush（每批 K < 队列容量，逐批解析 receipts，completion 只在最后屏障后写），使任意规模 trace 导入 complete，ledger summary / 覆盖 / capture 行状态按实际持久化表述，验收工具不得再用与生产不同的导入器容量。
   - 验收：生产入口端到端、复核方新一轮自设变异（报告杀伤率，不设 100% 门槛）、417 核心与输出锁不变。
3. **T9.5 Writer 消费端移出 GIL 与 ③ 三项验收**（[T9.5](history-references.md#ref-06e993298ce0)，2026-10-19 设计者编写，**2026-10-19 下发前核查后三轮修订，用户 2026-10-19 确认下发（待用户提交工作树后开工）**；Stage 0 可与 T9.4 并行，Stage 1 起须在 T9.4 合入之后）：设计者不指定实现，Stage 0 在登记环境（py3.11、`--cpus 4`、ext4）重建基线，并比较至少两类候选（A 同进程成块写入、每事件零系统调用；B 进程外消费者经共享内存环或管道成块写盘），给出 ③a/③b/③c 与 kill 损失窗口的数据，设计者在 Stage 0 Gate 选型后才开始 Stage 1 生产改动；必须保住 seq / drop 线性化、签名先于引用且可解码、flush 屏障、kill 后可读、多 Writer 隔离、尾读与离线一致、store 无 torch 可导入、417 核心与输出锁不变；按 D31 ③a / ③b / ③c 预登记。
   - 历史（至 2026-10-19）：本条原为“待 D31 裁决后编写任务单：消费者移出 GIL（独立写进程 + 共享内存环 / 管道，成块写），每签名一次校验或校验后移；导入器改为 Writer 端阻塞背压（容量检查通过后才取 seq，在线 emit 仍永不阻塞），并测离线导入的耗时 / RSS 规模曲线（60k 事件 110 s / 2.5 GiB 需解释）；……”。任务单编写时，离线导入规模成本归入相邻 spike（见第 4 条），导入器阻塞背压为可采用项且须与 T9.4 M7 分批 flush 衔接。
4. **相邻 spike（可并行、需真机）**：腾昇官方解析移到导入端（`analyse_flag=False` + 离线 analyse），A100 封存成本（复制时同步哈希、单次读）。目标是缩短窗口停顿，不改封存语义。另：离线导入规模成本（复核实测合成 trace 60k 设备事件 110 s / 2.5 GiB、100k 180 s / 4.1 GiB）另立 spike。上述 spike 都不阻塞 T9.5。
5. **P0 退出复审 → P1**：若 T9.4 与 T9.5 均通过且 ①–⑬ 全部满足（2026-10-22 起，T9.5 含 [T9.5c](history-references.md#ref-cdbf854eab27)，另需 [T9.6](history-references.md#ref-6d7e83e32eee) 合入；顺序 T9.6 → T9.5c → T9.5 Stage 1 → P0 退出复审，T9.6 不得在 T9.5c 预登记至测量结束之间合入（source_identity 覆盖整个 flagprof），Stage 1 在 T9.5c Gate 通过且 T9.6 合入后开工；见 [D39](DECISIONS.md#d39--t95b-gate-裁决选-b2x-类不保留-astage-1-前插-t95creader-流式化t96为-p0-退出前置)），设计者提交 P0 退出复审，通过后 L-字段由 1.0-rc 定为 1.0。设计者在 T9.5 期间编写 P1 首个任务单（一个 embedding 模型；route / module 事件 + opsum 聚合器；真实在线事件率与 emit 开销实测；off / on 开销）。
6. **[D33](DECISIONS.md#d33--恢复attach-不改被测代码) 默认接入路线细化（用户 2026-10-20 决策；任务单待编写，编写时按下发前自检流程）**：默认模式恢复"attach 不改被测代码"——窗口推进不再依赖被测代码显式调用 `handle.step()`（显式 step 降为可选项），自动推进机制（如时间 / 事件量阈值切窗或官方 schedule 自动步进）由设计者细化。约束：用户接受官方 profiler 固有扰动，但两个数据完整性边界的处置不在此范围内，所选路线须给出**超限行为或窗口上限的显式设计**：① 单个 trace 超约 65,536 设备事件导入为 partial（丢按位置排序的连续尾部；[T9.3 设计者审核 §1.3](history-references.md#ref-79d0492278f2) 发现 5；分批 flush 与显式化由 T9.4 M7 承担）；② 导入耗时与内存随事件量线性增长（6 万设备事件约 110 s / 2.5 GiB，合成 trace；规模成本已列相邻 spike，见第 4 条）。与 T9.4 M7、相邻 spike 的结论衔接后编写任务单，再请用户确认下发。

**仍待用户（2026-10-20）**：① 提交当前工作树文档（第 0 条）；② 确认 [D32](DECISIONS.md#d32--record-自动注入范围只注入-wrapper-直接启动的-python-目标)。（[D33](DECISIONS.md#d33--恢复attach-不改被测代码) 已由用户 2026-10-20 主动确认，无需再动作；第 6 条的任务单编写后再请用户下发。原③"确认并下发 T9.4 与 T9.5"已由用户按 [D34](DECISIONS.md#d34--t94t95-读取批次与测量边界补充)（2026-10-20）授权修订任务单并实现，不再待办，以文件头 2026-10-20 注记与两份修订任务单为准。）

### 8.4 历史待办与已完成任务（2026-10-18 前，保留）

S1（Spike-C）及其后续 T1–T8.3 已执行并经设计者 Gate 审核（T8.3 首轮未通过，返工轮 Gate 通过，P0 未退出）；T9.0–T9.3 已实现并通过 codex 内部复核（下文及任务单中的"独立 Gate"），设计者 Gate 结论见 8.2：
- T4 门禁通过；
- T7 交付了 P0 ①②④–⑨ 的功能闭环，③ 吞吐未达标；
- T8 交付了 ⑩–⑬ 的实现：官方 profiler 会话、来源账本、三层身份、原生关系、完成标记、证据索引与公共只读接口 `StoreView`。L-字段定为 `1.0-rc`（[Gate 2](history-references.md#ref-fe33c25a65f2)）。设计者审核见 [2026-10-15 review](history-references.md#ref-b6ab26ea7cbc)。
- T8.1 通过设计者 Gate（[审核](history-references.md#ref-573c75ba342e)）：厂商格式解释经 `SourceFormat` 注入并带版本，腾昇当前格式 compute 每窗 9,980 条进入规范路径；源码锁改为输出锁；生产默认值不再带实验命名；当前采集片段进入 CI。腾昇当前格式的原生归属仍为 0，断点已定位到主机进程编号。
- T8.2 通过设计者 Gate（[审核](history-references.md#ref-dc425489f08c)）：格式指纹改为两厂商格式模块的源码闭包；腾昇格式 v3（默认）有官方依据地还原主机进程编号并隔离冲突，当前采集三窗各 10,120 条原生 owned，A100 与历史输出不变。根因是我们的会话未声明 profiler level（默认 Level0 裁剪 `CANN` 记录），不是 SDK 版本。遗留：F1（历史逐跳对照的 ScopeConfig 口径）待补 reviewed 对照；v3 依赖目标进程名 `Python`（仅结构依据）。
- T8.3 首轮工程交付（2026-10-16），设计者 Gate **未通过**（[审核](history-references.md#ref-ed6a5f5f7031)）：NVIDIA 设备执行轨被判为 host（B1）、unknown 入选未检查轨道角色（B2）、设备轨整条缺失仍判契约内（B3）、契约约一半字段未被执行（B4）。已成立的部分：腾昇采集会话显式声明 Level0 + 数据精简，请求/实际/来源按窗口进身份；六份格式契约进指纹闭包；三值适用性判定与 `StoreView.format_assessment()`；C7 队列键按语义域与作用域修复；F1 补正；14 条 profiler 控制任务按 D28 第 4 条进入 E。
- T8.3 返工轮 Gate 通过（2026-10-16，附遗留 L1–L6 转 T9 §0；[复审](history-references.md#ref-81d74b957ee6)）：Kineto 角色由结构判定，unknown 入选消费同一份角色索引，空设备轨、角色冲突、支持范围显式进入评估；新旧代码独立重放 307 组导入，除格式评估与解析身份外逐元素相同。遗留：L1 同一事实多处判断、L2 15 个存活变异与契约条件行删除存活、L3 少数契约字段未绑定、L4 支持范围两家不对称且第二信号只在测试侧、L5 身份闭包不覆盖账本处置与共享判定代码、L6 证据卫生，全部转 T9 §0 前置清理。五窗仍为 `indeterminate`，主要是证据没有喂进判定链路，不是格式判不了（复审 §2）；P0 未退出。

设计侧据此新增或修订了：
- [D25](DECISIONS.md#d25--冻结语义与输出不冻结厂商分类表)：冻结语义与输出，不冻结厂商分类表；源码锁转为输出锁（`linker.py` 除外）；生产默认值不带实验命名；永久验收"当前采集器 → 当前导入器"；
- [D26](DECISIONS.md#d26--采集成本的验收口径)：采集成本分三层报告；腾昇 A6 保留"不可判定"并以结构论证豁免（2026-10-18 起仅为历史适用：豁免只覆盖原 Session 普通 step，当前 attach 以 T9.3 实测为准）；
- [Q8](DECISIONS.md#q8--导入期解析与索引是否复用-perfetto-trace_processor) 关闭（REJECT 当前接入）；[Q9](DECISIONS.md#q9--p0-吞吐门槛在原生优先路径下如何推导) 三组预算已实测；accepted约20.46–21.28万/s，原500k/s门槛未通过，见[T9.3预算](history-references.md#ref-f07fc39a4dfa)（2026-10-18 设计者审核更正：accepted 速率随循环长度漂移，"持续排空约2.32–2.38万/s"是独占尾部排空而非并发速率；改口径提议见 [D31](DECISIONS.md#d31--p0-吞吐验收改为三项并报退役单一-accepted-速率)，2026-10-19 用户已采纳，Q9 关闭）；
- [D27](DECISIONS.md#d27--格式解释的身份覆盖全部钩子兼容验收到连接键)：格式指纹覆盖全部注入钩子；主机身份还原须有格式依据；格式兼容验收报告逐跳命中数，0 必须指出断点；
- [D28](DECISIONS.md#d28--格式适用范围显式化采集配置入身份导入判定适用性静默失败转显式)：采集配置显式声明并入身份；每个格式版本带契约，导入时判定适用性；轨道角色审计与 unknown 归并让静默失败变显式；第 4 条（已证实的设备活动在类别未知时进入 E）用户已确认。
- [D29](DECISIONS.md#d29--判定一致性结构证据优先于名称契约条目须被执行或被测试)（2026-10-16，T8.3 审核）：同一事实的多个判断须同源或有一致性门禁；结构证据优先于名称，冲突显式；每份契约至少一条必需结构条件；契约条目须被执行或被测试绑定；新增判定以性质门禁、阳性对照与变异杀伤验收。
- [D30](DECISIONS.md#d30--契约条件的证据先查产物自述每个必需条件登记生产者)（2026-10-16，T8.3 返工复审；**已确认（用户 2026-10-16）**）：契约的每个必需条件登记证据生产者；判定"不可观察"之前，先清点产物自述（导出文件头、封存的配置文件）；已封存采集的派生事实写进新 run，不就地改旧 store；完整性的认证只来自采集端，导入端只做精确 ID 的单向负证据。2026-10-18 设计者补充裁决：NVIDIA `exporter.version`、`execution.mode` 保留 required，接受 indeterminate。

当前任务单在 **[codex-tasks/](history-references.md#ref-e29c4a8c676e)**：

| 编号 | 任务 | 依赖 | 说明 |
|---|---|---|---|
| [T8.1](history-references.md#ref-5fcd4691137d) | 厂商分类表版本化、门禁改锁输出、清除实验默认值、当前采集器贯通 | T8、D25 | **Gate 通过**（2026-10-15） |
| [T8.2](history-references.md#ref-3d910d3fb495) | 格式指纹覆盖全部钩子；腾昇主机进程身份还原（有官方依据时） | T8.1、D27 | **Gate 通过**（2026-10-15） |
| [T8.3](history-references.md#ref-18cd774a5eb9) | 格式契约、采集配置显式化与静默失败显式化：F1 补正；采集配置显式化；格式契约与导入适用性判定；轨道角色审计；unknown 归并；已证实设备活动在类别未知时进入 E | T8.2、D28 | 首轮 Gate 未通过；返工轮（[§9](history-references.md#ref-eeed26a375af)，D29）**Gate 通过**（2026-10-16），遗留 L1–L6 转 T9 §0（[§9.9](history-references.md#ref-4c7c9713f7d1)） |
| [T9.0](history-references.md#ref-3e7f0f466f04) | P0 收尾第一段：前置清理——判定收敛、测试补强、身份闭包与格式一致性套件（T8.3 遗留 L1–L6；离线，不新采集） | T8.3 | 已实现；2026-10-18独立Gate（codex 内部复核）通过，初轮失败与修复均保留；设计者 Gate 通过（2026-10-18） |
| [T9.1](history-references.md#ref-6bda9d9f5ed8) | 证据生产者（D30）+ `inspect` 只读展示（离线，用封存原件写新 run） | T9.0 Gate、D30 | 已实现，独立Gate（codex 内部复核）通过；见[执行记录](history-references.md#ref-bb2b957b04aa)；设计者 Gate 通过，附测试缺口 |
| [T9.2](history-references.md#ref-70fd0c2bb1c8) | `context.py`、`attach()` 骨架（持有 T8 的会话）、registry、CLI `record` 骨架 + 采集档声明（含模块作用域档，不做分析） | T9.1 Gate | 已实现，独立Gate（codex 内部复核）通过；T9.3补充root-only自动注入与显式worker规则；设计者 Gate 有条件通过（注入范围以 D32 追认，待用户确认） |
| [T9.3](history-references.md#ref-cbbbb3b8acbe) | 真机闭环：显式 Level0 复核、窗口停顿、invalid 策略、③ 按 Q9 方法复测（三组预算）；之后才是 P0 总验收 | T9.2 Gate | 已实现；最终真机验收执行完成，Q9③未达标，P0未退出；设计者 Gate：工程/测量闭环成立，生产入口未贯通，转 T9.4 |
| [T9.4](history-references.md#ref-e525188761b6) | 生产入口收口：inspect 状态、record 日志来源、导入器分批与测试缺口（M1–M7） | T9.0–T9.3 设计者 Gate；不依赖 D31 | 2026-10-19 下发前核查后三轮修订，**用户 2026-10-19 确认下发（待用户提交工作树后开工）** |
| [T9.5](history-references.md#ref-06e993298ce0) | Writer 消费端移出 GIL 与 ③ 三项验收（Stage 0 比较至少两类候选，设计者 Stage 0 Gate 选型），按 D31 ③a / ③b / ③c 预登记 | D31（已采纳，用户 2026-10-19）；Stage 0 可与 T9.4 并行，Stage 1 起须在 T9.4 合入之后 | 2026-10-19 下发前核查后三轮修订，**用户 2026-10-19 确认下发（待用户提交工作树后开工）** |

**T9 编写备忘（2026-10-16 历史输入）**（已落实到分段任务单，当前结果以各段验收为准）：
- T9 **不提供人可见的诊断输出**（用户 2026-10-15 裁决）。`inspect` 只显示身份、导入状态、处置、两套覆盖率与证据入口，不做热点表等分析结论；
- 按 D26 测量每个窗口阻塞被测进程的时长，并实现可配置的无效窗口策略（标 invalid 后继续 / 立即停止）；
- 按 Q9 先预登记在线负载画像，再测持续排空与离线消费，导入、首次打开、重复查询、分页与完整核验分开测量；
- `StoreView.traces()` 拆成"只读元数据"与"完整核验"两个入口：现在列一次 trace 就会解析全部派生文件并校验原件 SHA；
- attach 生命周期把"采集停止并导出""导入进程完成""run 汇总"作为三个独立交接点；目标进程退出码不决定证据是否完整；
- CI 拆成快速集（每次提交，不含 417 门禁）与全量集（Gate 前），两者耗时都写进执行记录（2026-10-15，T8.1 审核）；
- `inspect` 展示原生归属时同时展示 reason 分布。T8.2 的逐跳诊断若证明有用，在 P0 退出定 L-字段 1.0 时决定是否进入正式输出（D27 第 3 条）；当前倾向先不进入（T8.2 审核）；
- `record` / `attach` 显式设置并记录影响导出结构的采集配置（D28 第 1 条，T8.3 先在会话层落地，但未上机）；T9 的硬件运行须复核"显式 Level0 与 T8 封存的导出结构一致"（T8.3 遗留）；`inspect` 展示适用性判定与 unknown 归并（D28 第 2、3 条）。
- **T9 拆分**（设计者建议，**已确认（用户 2026-10-16）**；[复审 §6.4](history-references.md#ref-a2621015cf7c)；T9.0 已完成独立Gate；历史编写入口：[T9.0](history-references.md#ref-3e7f0f466f04)）：T9.0 前置清理（离线）→ T9.1 证据生产者 + `inspect` 只读展示（离线，用封存原件写新 run；腾昇三窗预期 within；四栏展示见下）→ T9.2 `context.py`、`attach()`、registry、`record` 骨架 + 采集档声明（含模块作用域档，不做分析）→ T9.3 真机闭环（显式 Level0 复核、窗口停顿、invalid 策略、Q9 三组预算），之后才是 P0 总验收。每段一个 Gate，顺序执行。理由：T8.x 每轮都把多件事捆在一个 Gate 里，返工时整轮重验；T9.1 不需要硬件，就能让用户第一次看到 within 的完整判定链。
- **§0 前置清理 = T9.0**（2026-10-16，T8.3 返工复审 [§6.1](history-references.md#ref-ca0dcd080310)；T9 的第一个提交，先于任何功能实现；任务单：[T9.0](history-references.md#ref-3e7f0f466f04)）：
  - 0.1 谓词收敛：复审 §3.1 中所有不同源的位置（队列键合法性三处；NVIDIA 类别字面量在 `disposition.py`、`session.py`、`chrome.py` 的副本；`cpu_op` 字面量；`ScopeConfig.annotation_categories` 默认值）改为调用同一函数或读同一契约字段，`nvidia/disposition.py` 首当其冲；更新 `vendors/ascend/README.md` 中过期的 `correlation_keys` 描述。
  - 0.2 测试补强：§3.2 的 15 个存活变异各补一项能杀死它的测试；M-i 改为行为断言；新增契约数据变异（各契约条件 id 集合、`requirement` 与预登记表逐项比较并进 CI）；C7 测试让两端同值非法；NVIDIA 设备信号逐一隔离。
  - 0.3 契约字段：`taxonomy_entries[*].evidence/status`、腾昇 `expected_process_names`、`structural_records[].basis`、`unchecked` 的取值，要么绑定，要么移入 `notes`。
  - 0.4 支持范围：第二信号做成范围条件（`pid ≠ args.device` → 范围失败，不做角色判据）；腾昇补 `device_ordinals` 或登记不对称理由；增加设备执行轨条数与每个编号的轨数；PID 碰撞条件补测试（M-n）。
  - 0.5 身份闭包：覆盖导入路径执行的全部判定模块（账本处置、规则、原生规则、共享的 chrome、inventory、prepare），加"实际加载模块 ⊆ 身份闭包"测试；契约文件加内容修订号，或在 05 写明"版本标签 = 厂商格式代际，内容修订由 sha 区分"，由设计者在 T9 任务单中二选一（已在 T9.0 任务单裁决：每份契约加整数 `revision`，并以 `revisions.lock.json` 钉住，见 [T9.0 §4.5](history-references.md#ref-1fb43a35144d)）。
  - 0.6 证据卫生：ci 记录同时登记 HEAD blob 哈希；验收脚本支持仓库外 `--out`；定向测试日志写模块清单。
  - 0.7 把返工的性质门禁、变异 runner、全来源重放收编为 `tests/`（Git 片段，CI 跑）+ `tools/`（原件重放，验收跑）下的可复用格式一致性套件。
  - 验收：复核方变异清单（本轮）全部被杀，另附复核方新设计的一批；75 份真实来源的输出与状态不变。
- **证据生产者表**（[D30](DECISIONS.md#d30--契约条件的证据先查产物自述每个必需条件登记生产者)，已确认（用户 2026-10-16）；更正：上一版此处写"NVIDIA 没有独立的实际配置"，是设计者未核对原件就采纳的错误前提）：契约中每个必需条件登记生产者、产生时机、来源引用、缺失策略与影响的结论。先清点产物自述：
  - Kineto 导出顶层带 `schemaVersion`、`cupti_version`、`cuda_runtime_version`、`cuda_driver_version`、`deviceProperties` 与为真的采集开关。torch 2.7.0 与 2.9.0 的 `_KinetoProfile.start_trace` 逐字相同，五个开关只在为真时写入、值为 `"1"`（2.8.0 未逐字核验）；缺键须结合 collector 版本判断是"false"还是"未记录"，collector 版本来自我方会话记录。导出里没有 torch 版本字符串（由我方 `session.json` 记录），`exporter.version`、`execution.mode` 也不在导出里，仍无生产者，交设计者逐条裁决：保留 required 并接受 indeterminate，或改为 conditional 并写明理由。NVIDIA 另用 `hardware-nvidia.json`（nvidia-smi 的 uuid、pci、驱动）。
  - 腾昇封存的 `profiler_info.json`（`common_config`、`experimental_config`、`torch_npu_version`、`cann_version`）带实际配置与版本，`PROF_*` 下另有 `sample.json`、`info.json`。正对照：喂入后腾昇 index 55 为 within（16/16）。
  - T8 已封存的窗口用新 run 写派生事实，不就地改旧 store；不得复制请求值充当实际值。
- `inspect` 分四栏展示：外层导入状态；内层格式适用性（`format_assessment()` 的 `status` 与 `assessment.status` 容易读错）；`outside`/`indeterminate` 的原因分组——**前提被破坏**（无设备执行轨、角色冲突、未识别轨道、标签依赖失败）/ **超出已验证范围**（硬件、版本、配置值、设备编号）/ **缺证据**；unknown 类别占比与归并组（类别全部 unknown 的输入可以是 within，见 07 §1 第 10 条）。旧 store 的 `unavailable` 显示为"未记录"，不显示为"无异常"。codex T8.3 复盘 §4.2 与返工复盘 §5.4 的消费矩阵作为 `inspect` 与 P1 `requires` 的输入。
- **完整性证据只来自采集端**（D30 第 4 条；对应 CUPTI `cuptiActivityGetNumDroppedRecords`、nsys `DIAGNOSTIC_EVENT`）：Kineto 只在 VLOG(1) 时查 dropped 并只写日志，导出 JSON 不含丢数据计数；torch_npu 是否暴露丢弃计数或序号仍待查明。有则作为 `capture_completeness` 的证据，无则保持 unchecked。导入端只允许精确 ID 的单向检测作为"发现丢失"的负证据：关联闭包孤儿（A100 两窗反向孤儿 0）、收尾标记缺失（`traceName` + Record Window End；腾昇 `*.done`、`analyse.done`、`SESSION_TIME_INFO` 结束时间）。没有发现不等于完整，不据此宣称完整，也不加启发式。
- **借鉴 nsys 的其余点（归 T9，[复审 §6.3](history-references.md#ref-fda8bbc0b9b5)）**：
  - 时钟来源证据：Kineto 的主机与 GPU 已在同一 `ts` 轴上，由生产者统一（文档）；`baseTimeNanoseconds` 为整秒锚，精度约 1 s（数据验证，无官方语义）。腾昇锚点齐全（`start_info`、`end_info`、`host_start`、`dev_start` 日志、`SESSION_TIME_INFO`）。T9 登记时钟来源证据，P2 决定"生产者统一时基"的有效档位；codex 返工复盘 §5.4 说统一时间轴"不可提供"，过于保守（07 §1 第 6 条已补注）；
  - 导出 schema 版本作适用性条件：Kineto `schemaVersion`、torch_npu `ascend_pytorch_profiler.db` 的 `SCHEMA_VERSION` 登记为契约的适用性条件（契约修订语义见上 0.5）；
  - 模块作用域采集档：A100 窗口没有 `python_function`、`module` 事件，因为 `with_stack`、`with_modules` 未开，这是采集配置缺口，不是匹配算法缺口。T9.2 的 `record` 提供带模块作用域的采集档（声明进身份，开销进 Q9），不做分析；P1 再消费。
- 归 P1 的借鉴点（两跳归属、连接键带域、torch_npu db 作为导入源的评估等）见 [04 §3.8](04-correlation.md#38-p1-设计输入两跳归属与连接键域借鉴-nsys2026-10-16)，在 P1 任务单定稿。
- Q9 离线消费预算计入"每次导入构建 3 次角色索引"，只有测量显示必要时才合并上下文。

P0 退出（L-字段 1.0）需要以下全部完成：T7 已交付的部分、T8（含 D26 豁免）、T8.1、T8.2、T8.3（均已通过；T8.3 返工轮 2026-10-16 通过）、T9（含 T9.0 前置清理；拆分为 T9.0–T9.3，已确认（用户 2026-10-16）），以及 ③ 按 Q9 方法的实测验收。**P0 未退出。**（2026-10-18 设计者审核后另需 T9.4 收口，且 ③ 按 D31（已采纳，用户 2026-10-19）的三项并报口径，取决于 T9.5；见 8.3。。2026-10-22 起另需 [T9.5c](history-references.md#ref-cdbf854eab27)（B2X 语义修复与确认测量，T9.5 Stage 1 的前置）与 [T9.6](history-references.md#ref-6d7e83e32eee)（Reader 流式化与有界内存），顺序 T9.6 → T9.5c → T9.5 Stage 1 → P0 退出复审（[D39](DECISIONS.md#d39--t95b-gate-裁决选-b2x-类不保留-astage-1-前插-t95creader-流式化t96为-p0-退出前置)）。）

P0 的实现对象是 [附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) 的 L-核心契约。开工前请完整读一遍那 28 条，它们是 major 级约束；L-厂商的边界见附录 A.6 与 D25。

> 2026-10-23 [D41](DECISIONS.md#d41)：用户授权先冻结任务单并实现 T9.6、T9.5c。Reader 退出项限于登记画像下消除普通在线事件全量常驻且满足预算；T9.5c 阶段性 CI、计数高水位与参照资格按修订任务单。Stage 1 与 P0 仍需各自 Gate。

> **2026-10-23 T9.6 验收完成**：生产流式读取及 finalize 在登记画像下通过全部内存与响应门，独立旧实现差分通过；[验收包](history-references.md#ref-318933687dbe)。随后在此冻结生产树上完成了 T9.5c 取证；其硬条件未通过，见文首最新回填。

## 历史状态记录

<details>
<summary>2026-10-30 D52 工程交付完成时的交接状态（原文）</summary>

> **当前状态（2026-10-30，D52 工程交付完成）**
> - T12.4/T12.5、T11.4/T11.5 登记范围内工程验收完成；[联合报告](history-references.md#ref-9596bcfb2a96)、[公共使用入口](history-references.md#ref-5ef59ff0c610)、[P4 退出复审输入](history-references.md#ref-49da047de8fe)。P4 退出待设计者复审，完整 P2 仍有 P2b。
> - P0/P1 的 D45/D49、P4 首批 D51 与本批 D52 各自结论保持。L-字段为 `1.2`；当前功能包括 tensor 采集、同输入 replay、设备／显存报告与双侧性能连接。
> - 后续按 v1 能力清单推进 OneRec 等场景；P3 不提前。新的实验范围与预算由后续任务登记，D52 已结束的预算不自动续用。
> - 读报告检查、端到端估时、输出逐值相等、来源守恒、失败不改判沿用。常规工程修复在已授权范围连续完成，只有契约、语义、资源范围等实质改变才提交对应决策。
> - FlagPerf_advance 只在本地管理，不 push；保留其他工作流的未提交材料。此前状态原文移至历史区。

</details>

<details>
<summary>2026-10-28 P1／P4 首批交接状态（原文）</summary>

> **当前状态（2026-10-28，P1 退出与 P4 首批下发）**
> - 用户 2026-10-28 批准 P1 退出复审结论 [D49](DECISIONS.md#d49) 与 P4 首批设计 [D50](DECISIONS.md#d50)。任务单 [T10.4](history-references.md#ref-3f47daaaa459)（P1.1）、[T11.1](history-references.md#ref-6deef0520efd)、[T11.2](history-references.md#ref-97b0743ec8d6)（P4 首批）已下发，共用本文件的全部约束。P3 不提前（Q11）：当前目标是先实现 v1 的已有功能，再让 v1 退休。
> - D49 第 6 条的流程调整适用于此后全部任务：读报告检查；预登记写端到端耗时估算；已知限制只写在阶段决策与退出矩阵；任务单约 200 行以内，自检表每问一行。
> - L-字段当前为 `1.1`；按已批准的 D50，T11.1 实现时 tensor 增加可选字段 `src`，升为 `1.2`，按 [03 §8](03-event-schema.md#8-版本与演进规则) 走 minor。
> - D45 第 9 条的流程与“输出逐值相等、来源守恒、失败不改判”三条底线沿用。D33 方案 B 的未提交文件不得混入提交。
> - 此前状态块原文移至文末[历史状态记录](#历史状态记录)。

</details>


### 2026-10-28 P1 退出复审前的状态块（原文）

> **当前状态（2026-10-27，P1 首批下发）**
> - **P1 首批已下发**：[T10.1](history-references.md#ref-29067dfd8983)（探针框架与主机侧关联链）、[T10.2](history-references.md#ref-51d0d35bc759)（分析器、span 树与最小报告）、[T10.3](history-references.md#ref-d60f95337bf0)（真实模型验收与退出证据），设计决定见 [D46](DECISIONS.md#d46)。三张单共用本文件的全部约束；探针降级的记录位置按 D46 第 12 条（§3 第 6 条的注记）。
> - **2026-10-27 首批修订（[D47](DECISIONS.md#d47)，用户批准方案 B）**：opsum 聚合键增加 `dev`、`nested`，L-字段升为 `1.1`；route 探针写静态判定表 `route_table`，判定范围为深度 0 且加速器调用；只在 summary 出现的回退行按 `summary_only` 降级。三张任务单已同步修订，尚未开工。
> - **本轮用户授权修订（[D48](DECISIONS.md#d48)）**：T10.1–T10.3 与相关设计说明已同步接入时机、C0-G 无探针基线及有界分析约定。关联错绑、漏发等用少量开发验收反例定位，修复并通过针对性测试后关闭；T10.3 保留一次正式验收，生产运行不增加重复证明。D47 原方案沿用，任务待实施。
> - P0 已退出，L-字段 `1.0`（[D45](DECISIONS.md#d45)）。L-字段改动按 [03 §8](03-event-schema.md#8-版本与演进规则) 走 minor；L-核心改动仍须先在 DECISIONS 提议。W7 仍是跟踪基准：改动在线 Writer 时继续用同一 harness 报告比值，不作阻塞门。
> - 流程按 D45 第 9 条：门限须追溯到需求或用户场景，与需求无关的数值门首轮失败即上报；任务单设日历时限；一页预登记、smoke 与正式各一次、单一提交冻结；保留“输出逐值相等、来源守恒、失败不改判”。
> - D33 方案 B 是独立工作流，已于 2026-10-26 被用户停止；其未提交文件不得混入 P1 提交。
> - 本次下发之前的状态块原文移至文末[历史状态记录](#历史状态记录)。


### 2026-10-27 P1 首批下发前的状态块（原文）

> **当前状态（2026-10-27）**
> - **P0 已退出，L-字段定为 `1.0`**（[D45](DECISIONS.md#d45)，[退出矩阵](history-references.md#ref-e0872549b0c8)）。此后 L-字段改动按 [03 §8](03-event-schema.md#8-版本与演进规则) 走 minor；L-核心改动仍须先在 DECISIONS 提议。
> - T9.7 已把进程外代理写者接入生产；W7 有效失败降为跟踪基准：改动在线 Writer 时继续用同一 harness 报告比值，不作阻塞门。T9.8 真实闭环校准已交付。
> - P1 起按 D45 第 9 条执行：门限须追溯到需求或用户场景，与需求无关的数值门首轮失败即上报决策；任务单设日历时限；验证协议轻量化（一页预登记、smoke 与正式各一次、单一提交冻结、分钟级记账），保留“输出逐值相等、来源守恒、失败不改判”。
> - D33 方案 B（官方最终交付 → 封存 → 有界离线导入）是独立工作流，状态以 `docs/validation/d33-plan-b-*` 各报告为准，本块不作判定。
> - 2026-10-27 之前的状态块原文移至文末[历史状态记录](#历史状态记录)。

### 2026-10-27 P0 收尾前的状态块（原文）

> **当前状态（2026-10-27）**
> - 已完成：T9.4 生产入口收口（独立代理 Gate，2026-10-20）；T9.6 Reader 流式化登记验收通过（2026-10-23，已在生产 `flagprof/store`）；T9.5d ext4 正式验收后 S9 已收口，G1 极差 0.035036 与 F11 由用户按 [D42](DECISIONS.md#d42) 处置为非阻塞。
> - 未完成：T9.5c 硬条件原失败保留；T9.7 W7 有效回归失败、W9 缺有效 aarch64 A5，正式验收未通过；P0 未退出，L-字段保持 1.0-rc（[退出矩阵](history-references.md#ref-e0872549b0c8)）。
> - D33 方案 B（官方最终交付 → 封存 → 有界离线导入）是独立工作流，状态以 `docs/validation/d33-plan-b-*` 各报告为准，本块不作判定。
> **2026-10-27 D44 本轮交付**：T9.8 原始 e385f495 正式校准与独立身份整合后闭环均已完成；T9.7 在线代理／离线共享优化、O2/O3、最终双解释器／134 故障条件、W3–W6/W8、X86 A5 与真实入口已交付。新四对 A/A 合格，正式十对 W7 中位 0.946409 低于原门 0.963973，依用户决定非阻塞且不再修复；aarch64 W9 供给仅 94.44%，低于 98%，原五分钟余量不足重验，停止并保留 blocker。W1 65 分钟、W8 30 分钟已获批准，总八小时／十二小时不变。工程原门、完整测量通过、P0③与 P0 退出均未成立。 见 [T9.8 报告](history-references.md#ref-e93d1c4566d0)、[T9.7 报告](history-references.md#ref-0ffd66d1ab4a)及 [Gate 材料](history-references.md#ref-a1968734253f)。
> - 2026-10-26 前叠放在文首的带日期状态横幅，原文移至文末[历史状态记录](#历史状态记录)。

### 2026-10-26 状态块合并前叠放的横幅（原文）

以下为 2026-10-26 状态块合并前叠放在文首的带日期状态横幅，按原叠放顺序（新→旧）原文保留，仅作历史语境；当前状态以文首“当前状态”块为准。

> **S9 补证接管约定（2026-10-25）**：候选源码/数据、测试资产与执行器分别冻结；长回归前验证证明目录及真实落盘/读回。修复工具另起系列，只有全部相关输入和运行条件一致时才复用未受影响的成功套件，逐项标注执行来源；失败原件不覆盖。本轮 a04 Store/conformance + a05 correlate 的复用依据和两轮归档见[收口交付](history-references.md#ref-03dff9b5e6ad)。


> **2026-10-25 最新：S9 已收口；G1 极差与 F11 已由用户接受为非阻塞项，可继续后续生产推进。** a04/a05 补证：store 329、correlate 79、conformance 21 全过，417-case/50,040 次比较与最终字节一致完成。原正式结果保留；本轮未执行生产接入，P0 未退出。以下早期失败/pending 条目保留历史语境，不再以当前 G1/F11 阻碍推进。 [D42](DECISIONS.md#d42)；[收口证据](history-references.md#ref-03dff9b5e6ad)。


> **2026-10-25 T9.5d ext4正式验收未通过**：129项已尝试、128项有效；关键故障/稳态条件通过，G1参照δ=0.035035957＞0.030。原S9输入缺项失败保留，独立CI补证仍有两项一致性失败和一项打包漏项且不替换正式样本。见[正式结果与交接](history-references.md#ref-e3b585c45e85)。完整性复算通过；设计者Gate待裁决，Stage1/P0未放行。

> **2026-10-25 D33 方案 B 可选整合回填**：用户明确允许原吞吐 gate 失败时推进可选生产整合。本阶段已接入 final/import/有界 inspect/page 和失败恢复，81 项矩阵、17 项 CLI 检查、57 项回归及短 Ascend 链路通过。5k 真实 CLI RSS 45.32 MiB、192 MiB 容器峰值 128.77 MiB；完整入口时延预算待定。1k/5k 原吞吐 gate 仍失败，默认路径和 Writer 未改，未生产验收或大规模验收。本阶段停止；不得据功能通过自动扩量。见[预登记与报告](history-references.md#ref-1051ec6e6994)、[使用说明](history-references.md#ref-e854eaa4d4fc)。下列同日吞吐停止记录保留前一阶段语境。

> **2026-10-25 D33 方案 B 吞吐迭代停止**：按用户授权最多两轮已完成，最终 1k/5k 2.019/9.978 s 对配对基线 1.287/6.349 s，吞吐 gate 失败；81 项功能/故障及资源检查通过。保留原型和失败，停止调优，不触发条件性生产整合；Writer 和生产源未改，不开展大规模验收。见[预登记与验证报告](history-references.md#ref-c081b7866806)。

> **2026-10-25 D33 方案 B**：用户已确认“官方最终交付 → 封存 → 有界离线导入”，本轮仅做 smoke。方案见[独立设计](history-references.md#ref-474bb80934d8)；81 项功能/故障检查通过，吞吐 gate 未通过，生产整合与大规模验收均未开始。证据见[smoke 报告](history-references.md#ref-168c9b0700a4)。与 T9.5d Writer 工作分离。

> **2026-10-25 T9.5d ext4整体smoke回填**：13/13阶段、34/34测量case、95次回归/边界测试及24次变异全部通过；独立复算、334个文件及224个归档成员核验通过。当前输入冻结一致，生产源码本轮未改。按用户任务单§8.7，在登记ext4范围继续推进；dtfs历史失败与偶发启动异常保留。G2短参考吞吐比约0.854503，仅单对观察，正式参照尚未建立。本轮未启动145-case正式矩阵，Stage 1/P0未放行。见[本轮交接](history-references.md#ref-9e64b5cca951)。

> **2026-10-25 T9.5d先前dtfs系列回填（历史）**：[任务单](history-references.md#ref-de1a85316545)及局部修复、smoke取证已交付，**整体smoke未通过**。新增契约/变异通过，最新回归仍有启动超时；诊断捕捉到fsync24.709s与8976条拒收。[交接](history-references.md#ref-534e6377847d)及任务单§8给出存储侧排查与有界异步持久化的边界。全部问题通过smoke前不得进入大规模验收，本轮正式矩阵未执行。原T9.5c失败和T9.6结论保留，Stage 1/P0未放行。

> **2026-10-23 实施回填**：T9.6 登记验收通过；T9.5c 实现与取证已交付，候选硬条件未通过。145 个登记 case 全部执行、155 个观测；V2 有 9 次 I3a 失败，S2 仅 8/15；S7 70k 有效覆盖不足，S5 参照不合格。最终 CI 292/79/21 全部通过不覆盖这些反证。Stage 1 与 P0 未放行。 [T9.5c Gate 输入](history-references.md#ref-3124ca1c1b92)。

> 2026-10-20 授权修订：[D34](DECISIONS.md#d34--t94t95-读取批次与测量边界补充)与 T9.4 M8/M9、T9.5 §4.7 补充公共读取校验、首批屏障、质量/身份和测量/关闭边界。P0 未退出；历史验收与旧状态保留。当前任务具体规则以两份修订任务单为准。

<!-- D44 superseded entry, retained history -->
> - **下一步：Writer Stage 1 与真实闭环校准：[T9.7](history-references.md#ref-418019448c44)（Stage 1 生产接入）、[T9.8](history-references.md#ref-104a0f28ea18)（真实闭环校准）已下发（[D43](DECISIONS.md#d43)），待 codex 开工。** 协作纪律见 [§7 补充（2026-10-26）](#collab-20261026)。
