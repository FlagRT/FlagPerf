> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书链接转入 [历史引用索引](history-references.md)，其出处保留。

> **2026-11-01 请求时钟补充完成**：[八 rank 的完整请求窗口与对应关系](history-references.md#ref-aaa05675a16c)已验证。硬件同步误差界仍未知，完整 P2 裁决不变。

> **2026-11-01 工程交付完成**：T14.1/T14.2 登记样本、双解释器回归、A100 TP2、读取计时与归档公共重开已完成。完整 P2 退出及组件团队交付前的设计者阅读裁决另行登记。 [交付](history-references.md#ref-1dc484c35db2)。

> **2026-10-31 T13 联合工程验收完成。** P4 按 D50–D52 登记小范围退出；OneRec 单卡生成诊断、匹配 A100 基线及 TP2 HCCL／FlagCX 对照完成。四条件矩阵、五组数值复算、双解释器各 634 项回归、最终 12 个入口的 dtfs 恢复通过；完整 P2 保持未退出。[验收与证据](history-references.md#ref-409c11becc2f) · [任务书](history-references.md#ref-a272319ae7a0) · [恢复接手](history-references.md#ref-7741c52a401f)。

# 10 阶段计划

> **2026-10-30：D52 四任务工程验收完成。** T12.4/T12.5 两厂商设备与官方显存视图成立，T11.4 六目标 replay 和双侧性能连接成立，T11.5 三条件下钻、独立复算及双解释器 CI 通过。[联合交付](history-references.md#ref-9596bcfb2a96)。P2b 与 OneRec 匹配门限留后续；P4 退出提交设计者复审。

> **此前状态（2026-10-29，T12.3 工程交付）**
> - [T12.3](history-references.md#ref-f7a5d79a28b5)：已完成校验遍历合并与跨进程读取复用；S 首次导入约快 5%，重开全遍历约快 90%，双解释器回归通过。
> - P0/P1 沿用 D45/D49；P4 首批沿用 D51，T11.3 工程交付待设计者复审，完整 P4 待第二批。
> - [T12.2](history-references.md#ref-61027047fd40) 完成两轮 prepare 优化：M 320.59→204.19 s，外层 407.92→323.75 s；三个 M 配对、S、Reader 与双解释器 CI 通过。[验收](history-references.md#ref-c7d5393110c1)。
> - 200/300 s 软目标未完全达到；更新 profile 后按收益收敛停止，未进入第三/四轮生产优化。
> - OneRec 32 token 的匹配导出及真实门限继续待实测；P2a 设备/显存及 P4 第二批工程交付已完成，P4 退出待复审，P3 排序保持。

> 上游：全部设计文档 ｜ 实现方：codex（约束见 [12](12-codex-handoff.md)）
> 原则：**先验证最大风险，再冻结契约，再堆能力**；**主线先行，扩展随后**（[D20](DECISIONS.md#d20--分阶段推进主线先行)）。每个阶段有明确的进入条件、退出条件与需求归属。
> 本计划**不锚定外部里程碑日期**（[D21](DECISIONS.md#d21--范围边界暂不做与不做清单) B4）：阶段只按退出条件推进，外部里程碑只作排期参考。

## 0. 阶段总览

```text
主线：推理全链路诊断（性能 + 精度，腾昇 + NVIDIA）
S1 Spike-C ✅ ──► P0 store+schema ──► P1 Python 关联链 ──► P2 设备关联 ──► P4 精度差分
  (风险验证)        (冻结契约)         (op/route/compile)    (kernel/comm)    (tensor/oracle)
                                              └──────── P4 只硬依赖 P1，可与 P2 并行 ────────┘
                                                                                    │
扩展：主线完成后依次推进                                                             ▼
P5 规模化 ──► P6 第二家厂商 + v1 退役 ──► P3 故障+在线+长稳
(campaign/报告/索引)  (平头哥；"厂商无关"的检验)    (health/watch/lifecycle)
```

S1 必须最先做；主线四个阶段完成前不启动扩展阶段；P6 是对"厂商无关"这一设计主张的真正检验。

**阶段编号是标识，不是顺序**：P3 排在最后，但保留原编号，以免破坏决策记录、任务单与 progress 报告中的引用（D20）。各阶段表中的「层级」行标明它属于主线还是扩展。新需求先进 [00 §6 需求池](00-requirements.md#6-需求池参考文档补充)，不直接加入阶段交付。

### 0.1 v1 能力清单（P5 之前的排序依据）

按 [Q12](DECISIONS.md#q12--p5-之前是否按-v1-能力清单排序)（用户 2026-10-29 裁决），P5 之前的工作以下表为清单排序。表由 v1 `inference/support.json` 的自述整理（[10-28 复审页 §5](history-references.md#ref-da9c641ffb09)），每项开工时先对照 v1 代码与产物复核一次。本表也是 [P6](#7-p6--第二家厂商与-v1-退役) 验收 ③ 对照检查表的底稿；D5 的退役条件不变。

“验证面”列按 [D54](DECISIONS.md#d54) 第 4 条（用户 2026-10-31 批准）新增，写成“模型 × 芯片 × 规模”。✅ 只表示该验证面上的登记范围通过。项目验收面是 ≥500 个模型、≥3 家国产芯片效果验证、≥5 家接口覆盖；A100 只作参照，不计入国产芯片家数。

| 序 | v1 能力 | v2 现状（2026-10-31） | 验证面（模型 × 芯片 × 规模） | 归属与依赖 |
|---|---|---|---|---|
| — | Qwen3-Embedding 模型级／块级精度与 A100 参考包 | ✅ 360/360 逐字节复现；三组跨芯片 | Qwen3-Embedding-0.6B × 910C（A100 参照）× 单卡 batch 1 | P4 首批（D51） |
| — | 算子级张量精度（v1 为 not_implemented） | ✅ introduced 与放大；口径由 T11.3 修订 | 同上 | P4 首批 → P4.1 |
| — | 主机侧回退率与算子级性能（v1 为 not_implemented） | ✅ 主机侧 | Qwen3-Embedding-0.6B × 910C × 单卡；A100、多卡、其他模型未验证（[P1 已知限制](history-references.md#ref-567cd9829d3a)） | P1／P1.1 |
| 1 | 导入速度（generate 与设备视图的共同前置） | T12.2 两轮优化完成；T13 OneRec 单卡／TP2 真实来源导入与恢复完成，离线等待继续按实际使用路径优化；历史超时和软目标原判保留 | Qwen3-Embedding 52 万条来源；OneRec-8B 单卡／TP2 来源 × 910C、A100 | P2a：[T12.2 验收](history-references.md#ref-c7d5393110c1) → 设备/显存视图与 OneRec 目标样本 |
| 2 | layer 性能、设备 profiler 热点与设备侧回退率 | 登记范围内工程验收完成：两厂商设备热点、逐层／路由工作量和官方显存视图成立；设备头条按类别／项目归属（`link_status`）拆分并带窗口身份，见 T14.2 重出与窗口身份 | Qwen3-Embedding-0.6B × 910C／A100 × 单卡 batch 1 | [T12.4/T12.5 联合交付](history-references.md#ref-9596bcfb2a96)；[T14.2](history-references.md#ref-1fa80036fb14) |
| 3 | OneRec-8B generate：prefill／decode、逐 token 首差异、32 步性能 | T13 登记范围完成：12 输入、首分歧步包含规则、32 步性能与恢复 | OneRec-8B × 910C（单 die、TP2）／A100 单卡 × batch 1、32 token | [T13 验收](history-references.md#ref-409c11becc2f) |
| 4 | A100 性能基线与跨设备性能比 | T13 匹配单卡基线完成，5 预热／30 测量／3 重复；无同规模 A100 TP2 对照 | OneRec-8B × A100 单卡 vs 910C 单 die | [四条件性能](history-references.md#ref-1e0b6b8ba394) |
| 5 | TP2／HCCL 多卡精度与性能；FlagCX 通信对照 | T13 两 rank 工程验收完成；设备通信明确 ID、设备忙闲与时钟的工程结果见 T14.1；padding、链路量保持不可用 | OneRec-8B × 910C × 2 卡；A100 2 卡 NCCL profile 的采集与导入见 T14.1 | [T13 最终矩阵](history-references.md#ref-829993fd7da4)；[T14.1](history-references.md#ref-7271dc8d7c92)，P2 剩余项见 §4 |
| 6 | preview 策略（FlagGems 分组搜索） | 部分：一次 run 的 `route.coverage` 给出回退清单（D8） | Qwen3-Embedding-0.6B × 910C × 单 run | 隔离验证二次 run，P2b 之后 |
| 7 | DLRM、DeepFM 推荐模型 | ☐ | — | 模型接入，P2b 之后 |
| 8 | FlagTree 编译器对照 | ☐ | — | 组件对照，序 7 之后 |
| 9 | FX／ONNX 图导出 | ☐ | — | 排在最后，开工前核对它在 v1 中的实际使用者 |

第二家国产芯片的契约探针 P6-0（昆仑芯 P800，单卡 Qwen3-Embedding-0.6B）已按用户批准计划完成 12 样本 coverage、块级 accuracy、官方 profile、有限开销和恢复验证，交付 [十二项契约与正式适配缺口](history-references.md#ref-d7f75b6039b0)。P800 接线仅在冻结源码实验副本中；生产默认入口未改变。它不改变 D20 中 P6 的位置，也不构成完整 P6 退出。

P4 第二批 replay、`layers.27` 下钻与跨 run sid 对齐已交付，并经 T13.1 集中复审纳入 P4 登记范围退出。[退出裁决](history-references.md#ref-49da047de8fe)。

## 1. S1 — Spike-C：CANN 算子级关联验证

| 项 | 内容 |
|---|---|
| 目的 | 判定 [04 §3.3](04-correlation.md) 的 H3 在 CANN 上是否可行，确定 [04 §6](04-correlation.md) 的降级级别 |
| 方案 | [11-spike-cann-correlation.md](11-spike-cann-correlation.md) |
| 进入条件 | 具备腾昇环境（镜像 `flagrt/ascend-operator-runtime:0.1.0-cann9.0-py311-torch2.10-arm64` 或其 0.2.0/flagCX 分支） |
| 退出条件 | 交付 `spikes/cann_correlation/REPORT.md`，明确给出 `host_device_link` 的取值与可用规则清单；[03](03-event-schema.md) 与 [04](04-correlation.md) 据此回填，并由设计者裁决后冻结 **L-核心** |
| 需求 | R53（前置风险） |
| 层级 | 主线 |
| 状态 | ✅ 已完成并通过设计者审核（2026-10-10）：单卡与 A100 完成；TP2 HCCL/FlagCX 18 次完成；8 卡暂缓。裁决与口径修正见 [DECISIONS Q7](DECISIONS.md#q7--spike-c-实测回填与冻结边界)，规范变更见 D11–D14。**schema 按 [D11](DECISIONS.md#d11--schema-分三层冻结不是一次性整体冻结) 分层冻结，不整体冻结**。后续 T1–T3（2026-10-11）已执行：T1 的设计门禁**未通过**（证据冲突被报成 linked），契约已按 [D15](DECISIONS.md#d15--t1t3-之后的契约回填r1r8) 回填 |

**S1 不产出主包代码**，只产出 `spikes/` 下的一次性脚本与报告。

## 2. P0 — store 与 schema

| 项 | 内容 |
|---|---|
| 交付 | `store/`（schema 定义、writer、reader、manifest、compact）、`flagprof/context.py`、`attach()` 骨架、registry 框架、CLI 的 `record`/`inspect` 骨架；**基座项（2026-10-14，[T8](history-references.md#ref-24a77204cd82)）**：官方 profiler 采集会话（腾昇 + NVIDIA，[D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合)）、来源账本与三层身份及导入完成标记（[D24](DECISIONS.md#d24--导入契约身份分层来源账本完成标记证据索引)）、原生关系视图（[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)）、store 公共只读接口与批量证据回查 |
| 验收 | ① schema 校验器对全部 13 类事件与两类记录（[03 §2](03-event-schema.md#2-事件信封)）通过；② **kill -9 后 store 可读**，截断行被识别，标 `truncated`；未正常关闭的 writer 丢弃量报下界并标 `tail_unknown`（N2）；③ **写入容量**（[D31](DECISIONS.md#d31--p0-吞吐验收改为三项并报退役单一-accepted-速率)，用户 2026-10-19 采纳），三项并报，均在登记环境测：**③a 热路径成本**：在线 emit 的每事件成本（中位与 p99，µs），含为把事件交给消费端所做的全部工作；按 D31 补充（用户 2026-10-19 采纳）分两部分报告——emit 调用成本（受控协议）与被测线程进度比 R(r)（计入同进程消费端占用 GIL 的时间）；P1 前只报告，不设数值门槛，门槛由 P1 探针开销预算（N1）反推。**③b 并发无损持续速率**：生产者与排空并发、生产者持 GIL 繁忙时，对全部登记在线画像（P-perf 的 G0/G1/G2、P-dispatch-window 的 G1/G2）每个 3 次全部 drop=0 且 flush 完成；另报告节拍扫描得到的无损持续上限 μ_lossless。**③c 突发吸收**：报告队列容量 C 与 C/(λ−μ)；可吸收量按 D31 补充以实测倍数扫描确定（公式只作交叉检查，冲突以实测为准）；登记窗口画像（计数框：`aten::` 37,850 条/窗、与 N 同负载的窗口间隔 1.2496 s、窗间被测线程保持忙、≥10 窗）须不超过可吸收量。（2026-10-19 按 D31 补充改写 ③a/③b/③c 措辞。）原“单 writer ≥ 50 万事件/秒入队”退役为历史基准，T7/T9.3 的 failed 记录保留；为连续性，[T9.5](history-references.md#ref-06e993298ce0) 仍重跑原 100k heartbeat 微基准并报告，不作判定。（**历史（至 2026-10-19）**，原 ③ 文字：“③ 写入吞吐基准：单 writer ≥ 50 万事件/秒入队，**同时**报告持续落盘速率、积压与丢弃 —— 不能只用入队速度代表 store 能力”）；④ 队列溢出产生周期性累计 `drop_report`；双线程受控乱序下，分片内 `seq` 跳变即确定丢弃；⑤ **可解码不变式**：分片中途出现新签名、签名字典超预算、尾部连续丢弃后 kill，存活记录全部可解码；⑥ **writer 身份**：同 rank 的被测进程与 counter 进程并存、两进程并发 attach、同 rank 进程重启，`corr`/`seq`/分片/writer 片段均不冲突；⑦ 在线尾读与离线 reader 得到相同的记录集合，包括封口与压缩期间的尾读；⑧ 真实 CANN / NVIDIA fixture 经薄适配写入 store，单条 `link`、coverage 与原始证据能互相追溯；⑨ **无 torch 环境**中 `store/` 可导入并通过测试（N3）。以下 ⑩–⑬ 为 2026-10-14 增补（[T8](history-references.md#ref-24a77204cd82)）：⑩ **采集会话**（[D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合)）：在腾昇与 A100 上原样使用官方 profiler 采集。要求：单一会话所有者生效；至少 2 个窗口各自导出、校验、封存；无效产物（空、截断、日志报错）判为 invalid；窗口中 SIGKILL 后，已封存窗口完整、未封存窗口标 `lost`；会话包装相对裸官方 profiler 的开销在预登记门槛内（历史：腾昇按 [D26](DECISIONS.md#d26--采集成本的验收口径) 以结构论证豁免，原"不可判定"结论保留。2026-10-18 起：T9.3 已对新 attach 实测 passed（腾昇唯一 16 block 复测），不沿用该豁免，见 D26 的适用范围说明与[退出矩阵](history-references.md#ref-e0872549b0c8)）；输出与无 profiler 基线逐位相同。⑪ **原生关系与项目归属分立**（[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #25）：没有项目 marker 的有效原生链，可查询到原生 owner 与作用域，且不制造项目 corr；已有真实项目关联的样本，`kernel.link` 逐字节不变；[04 §3.7](04-correlation.md#37-原生归属原生调用关系) 的反例行为全部成立。⑫ **来源账本、身份与完成标记**（#26–#28）：每条来源记录恰有一个处置并守恒；单条无法表达的记录不导致整份失败；只读 store 中的登记表即可得到真实硬件、采集器与解析身份；导入完成可判定，中途失败读作 `partial`。⑬ **公共只读接口**：身份、项目绑定、原生关系、账本、两套覆盖率、证据回查都可经接口得到；批量回查与逐条回查逐字节相同，且每份原件只校验一次；接口在无 torch 环境下可导入，不 import `vendors/` 与 `correlate/`；T7 时期的 store 可读（legacy） |
| 需求 | 全部的基础 |
| 层级 | 主线 |
| 依赖 | L-核心按 [D15](DECISIONS.md#d15--t1t3-之后的契约回填r1r8) 回填后冻结（[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结)）即可开工；验收 ⑧ 依赖 [T4](history-references.md#ref-3f0433ab6328)；⑩–⑬ 依赖 [T8](history-references.md#ref-24a77204cd82)，主线设备当前格式的补口依赖 T8.1、T8.2；格式适用范围与静默失败显式化依赖 [T8.3](history-references.md#ref-18cd774a5eb9)（[D28](DECISIONS.md#d28--格式适用范围显式化采集配置入身份导入判定适用性静默失败转显式)；2026-10-16 首轮 Gate 未通过，返工轮按 [D29](DECISIONS.md#d29--判定一致性结构证据优先于名称契约条目须被执行或被测试) 通过，遗留 L1–L6 转 T9 §0）。L-字段在 P0 结束时定为 1.0；T8 的表示形式先在其 Gate 1 定为 `1.0-rc`。验收 ③ 的门槛推导原待 [Q9](DECISIONS.md#q9--p0-吞吐门槛在原生优先路径下如何推导)（历史；Q9 于 2026-10-19 关闭）；2026-10-18 起由 [D31](DECISIONS.md#d31--p0-吞吐验收改为三项并报退役单一-accepted-速率)（设计者提议：③ 改为 ③a 热路径成本 / ③b 并发无损持续速率 / ③c 突发吸收三项并报，退役单一 accepted 速率）承接，**2026-10-19 用户采纳**，③ 的新文字见上，③b/③c 的达标依赖 [T9.5](history-references.md#ref-06e993298ce0)（任务单已写，用户 2026-10-19 确认下发（待用户提交工作树后开工））。历史（至 2026-10-19）：裁决之前现行文字与门槛不变，③ 维持 failed |
| 状态 | **当前（2026-10-27，P0 收尾）**：✅ **P0 已退出，L-字段定为 `1.0`**（[D45](DECISIONS.md#d45)，用户 2026-10-27 批准，即 P0 退出复审结论）。①②④–⑬ 沿用原证据；③ 按 D45 第 4 条判为登记低负载范围达标：③a 已报告，③b W5 十五格全过，③c 名义窗口无损（不承诺 2 倍余量），A5 X86 有效通过、aarch64 以 94.44% 供给下零回退观察接受；W7 有效失败降为跟踪基准。已知限制见 [退出矩阵](history-references.md#ref-5bfcf8206c30)。<br>历史（2026-10-27 收尾前）：T9.8 原始 e385f495 正式校准与独立身份整合后闭环均已完成；T9.7 在线代理／离线共享优化、O2/O3、最终双解释器／134 故障条件、W3–W6/W8、X86 A5 与真实入口已交付。新四对 A/A 合格，正式十对 W7 中位 0.946409 低于原门 0.963973，依用户决定非阻塞且不再修复；aarch64 W9 供给仅 94.44%，低于 98%，原五分钟余量不足重验，停止并保留 blocker。W1 65 分钟、W8 30 分钟已获批准，总八小时／十二小时不变。工程原门、完整测量通过、P0③与 P0 退出均未成立。 详见 [Gate 材料](history-references.md#ref-a1968734253f)。<br>历史（至 2026-10-26）：P0 未退出。T7 的①②④–⑨及T8–T8.3基座继续由回归保护；T9.0一致性、T9.1证据生产者/inspect、T9.2 Context/attach/registry/record已通过独立Gate。T9.3窗口策略及SDK封存/子进程修复已实现，最终真机验收执行完成；Q9实测单writer accepted约20.46–21.28万/s，未达50万/s。完整①–⑬逐项证据及最新阻塞见[退出矩阵](history-references.md#ref-e0872549b0c8)。L-字段保持1.0-rc。<br>**2026-10-18 设计者 Gate 结论**（[设计者审核](history-references.md#ref-79d0492278f2)；上文为 codex 实现方口径，其“独立Gate”为实现方内部复核）：T9.0 通过；T9.1 通过，附测试缺口；T9.2 有条件通过（record 自动注入范围由 [D32](DECISIONS.md#d32--record-自动注入范围只注入-wrapper-直接启动的-python-目标) 追认，待用户确认；corr 唯一性、fork 后附件可见、record 完成判定各层谓词缺独立测试）；T9.3 工程/测量闭环成立，但两项 P0 条件只在验收 harness 路径成立、生产入口未贯通：⑩ 的“日志报错判 invalid”在 record 路径未接线，② 的 truncated/lost 不经 inspect 暴露；另 inspect 对不存在路径退出 0（生产缺陷）、验收工具退出码仍有同族漏洞。P0 未退出，L-字段保持 1.0-rc（与 codex 结论一致）；③ 仍为 failed（旧口径，T7/T9.3 记录保留），D31 已采纳（用户 2026-10-19，③ 改按三项并报判定，新口径的测量由 [T9.5](history-references.md#ref-06e993298ce0) Stage 0 基线承担）。**下一步**：[T9.4](history-references.md#ref-e525188761b6) 生产入口收口（任务单已编写，用户 2026-10-19 确认下发（待用户提交工作树后开工）；不依赖 D31）；[T9.5](history-references.md#ref-06e993298ce0)（任务单已写，用户 2026-10-19 确认下发（待用户提交工作树后开工）；Stage 0 可与 T9.4 并行，Stage 1 起须在 T9.4 合入之后）。<br>历史（至 2026-10-19）：“③ 仍为 failed，D31 待用户裁决……之后 T9.5（待 D31 裁决后编写）。” |

**2026-10-23 环境与性能处置（用户已确认）**：后续存储环境及性能处置按 [D40](DECISIONS.md#d40) 执行。绝对容量数值随环境报告，约定在线画像下的无损、突发吸收与资源要求继续验收；离线耗时增加单独记录。历史结果不追溯改判。

## 3. P1 — Python 侧关联链

| 项 | 内容 |
|---|---|
| 交付 | dispatch（each + summary）、module、route、compile、user 探针；`span` 树重建；`perf.op`、`perf.layer`、`route.coverage`、`compile.health` 分析器；最小 HTML 报告 |
| 验收 | ① Tier 0 跑通一个 HF embedding 模型，产出算子清单 + 回退清单；② `op→route→compile` 的 corr 关联率 100%（纯 Python，无借口）；③ **开销实测**：探针 off/on 的端到端耗时对照写入报告（N1）；④ 自动选层与重复层折叠生效，coverage 如实记录 |
| 需求 | R53（主机侧）、R31（回退率）、R47、R45、R1；需求池 X1（主机侧部分）、X2 |
| 层级 | 主线 |
| 状态 | **当前（2026-10-28）**：✅ 设计者复审判为登记范围内通过并退出（[D49](DECISIONS.md#d49)，用户 2026-10-28 批准）；耗时头条与回退清单由 P1.1 [T10.4](history-references.md#ref-3f47daaaa459) 修补后对外交付。<br>历史（2026-10-27）：▶ 进行中。首批三张任务单 [T10.1](history-references.md#ref-29067dfd8983)、[T10.2](history-references.md#ref-51d0d35bc759)、[T10.3](history-references.md#ref-d60f95337bf0) 已下发，设计决定见 [D46](DECISIONS.md#d46)（预设组成、两级 dispatch、route 判定、选层折叠、主机桥接、分析集、seal 身份传播、N1 只报告）；验收 ①②④ 为门，③ 只报告（D46 第 8、9 条）；终点为设计者 P1 退出复审。<br>历史（2026-10-27 下发前）：☐ 未开始；P0 已于 2026-10-27 退出，可开工。首个任务单按 [D45](DECISIONS.md#d45) 第 7 条：dispatch 默认 summary 聚合、开销按真实模型 C0/C1 口径、主机桥接为本阶段交付、seal 身份传播在首个任务内修复、route/回退清单需 FlagGems 环境 |

> P1 交付通用 `coverage` 扫描入口，并由 T10.3 在登记的 HF embedding 模型与环境上验收“算子清单 + FlagGems 回退清单”。500 模型覆盖是规模化目标，批量接入、支持范围与完成证据由 P5 承接（[D48](DECISIONS.md#d48)）；P1 的清单与导出可先供 FlagGems 开发组使用。

## 4. P2 — 设备侧与通信

| 项 | 内容 |
|---|---|
| 交付 | `vendors/ascend/` 完整适配（trace 导入器的通信等补全、link rules、telemetry；profiler 会话已于 2026-10-14 移至 P0，见 T8）、comm 探针、有来源的主机↔设备时钟换算、`perf.phase`/`perf.gap`/`memory`/`comm` 分析器。`correlate/linker.py` 已在 P0 交付（T4/T7），P2 只在其上补规则与通信关联。counter 探针（独立进程）按 [D54](DECISIONS.md#d54) 第 1 条移至 P3 在线监控（2026-10-31） |
| 验收 | ① kernel/memop/comm 事件由真实 trace 导入并通过 schema 校验，导入器独立产出原始清单（`import_completeness` 可算）；② 级别分布、`linked_count_ratio`/`linked_time_ratio` 有实测数值且与 S1 结论一致，[T4](history-references.md#ref-3f0433ab6328) 的回归套件在 `correlate/linker.py` 上通过；③ 关联规则**无任何时间邻近逻辑**（代码审查 + 测试断言）；④ ~~counter 在 profiler 冲突区间正确标 `blocked`~~ 随 counter 探针移至 P3（D54 第 1 条）；⑤ 跨 rank `(group, group_seq)` 对齐在 2 卡上验证；8 卡移出当前退出，出现 8 卡模型需求时另行登记（D54 第 1 条；原文为“在 2 卡/8 卡上验证”）。**完整 P2 退出**（D54 第 1 条）= 已登记 TP2 + 设备通信明确 ID 连接或显式不可用原因 + 时钟来源判定 + 设备忙闲（`perf.gap`） |
| 需求 | R53、R31、R86/R88（单边）；需求池 X1、X3（设备级） |
| 层级 | 主线 |
| 状态 | **当前（2026-11-02，D55）**：P2 按 D54 登记范围 `passed_registered_small_scope`，通过并退出。[退出矩阵与边界](history-references.md#ref-0bb32f584712)。T12/T13/T14 设备、显存与通信链路及请求时钟补充支撑本次退出；硬件同步误差上界未知，8 卡按需、counter 属 P3。D54 对外交付前的逐类阅读检查另行登记。 |

<details>
<summary>2026-11-02 退出前的 P2 状态（历史原文）</summary>

| 项 | 内容 |
|---|---|
| 状态 | **当前（D54 范围）**：完整 P2 未退出。T12.1–T12.5 已交付；T13 已补 OneRec 匹配来源、单卡基线、TP2 HCCL／FlagCX 与跨 rank 组序号核对。退出范围按 D54 第 1 条收窄（8 卡按需、counter 移至 P3）。T14 增补设备通信明确 ID、时钟来源能力判定与 `perf.gap`；工程状态、同来源区间与未知误差界的范围见 [T14 交付](history-references.md#ref-1dc484c35db2)。[完整 P2 退出证据清单](history-references.md#ref-f9d7201880de)供设计者起草退出提议。工程完成不自动改变本行的 P2 未退出状态。 |

</details>

## 5. P4 — 精度差分

| 项 | 内容 |
|---|---|
| 交付 | tensor 探针、Tier-1 profile（`boundaries`/`mask`/`inputs.recipe`）、seal 模式与 `--inputs-from`、`oracle.*` 全部分析器、`oracle.identity_diff` |
| 验收 | ① 同一模型在 NVIDIA 与腾昇上 seal 对照，产出逐层误差表与首个发散层；② `call_key` 在两边完全对齐，失配项如实列出；③ identity diff 能拦住"环境差异未声明"的对比；④ 敏感算子清单与 `op_perf` 按同一 sig 对齐 |
| 需求 | **R52**、R86/R88、R37；需求池 X4、X5（只报失配） |
| 层级 | 主线（主线的最后一个阶段） |
| 依赖 | P1（span/sig）；P2 可选（用于性能一并对照），因此 P4 可与 P2 并行 |
| 进入时裁决 | 张量采集与数值比对是否复用 msprobe：[Q10](DECISIONS.md#q10--p4-的张量采集与数值比对是否复用-msprobe)。已裁决：不复用为数据来源（[D50](DECISIONS.md#d50) 第 1 条，用户 2026-10-28 批准） |
| 状态 | **当前（2026-10-30）**：T13.1 集中复审通过，P4 `passed_registered_small_scope` 退出，见上方裁决。<br>历史（2026-10-29）：首批复审通过，判为 `passed_registered_small_scope`（[D51](DECISIONS.md#d51)，用户 2026-10-29 批准）：①–③ 与块内 `introduced` 定位在登记范围内成立。P4.1 [T11.3](history-references.md#ref-3c84559c539d) 已完成工程验收，待设计者复审，修订成对哨兵、ULP 统计、头条指标、布局属性与页面。第二批 T11.4/T11.5（replay、`layers.27` 下钻、双侧跨 run sid 对齐）已于 2026-10-30 完成工程验收；P4 退出提交设计者复审。<br>历史（2026-10-28）：首批 [T11.1](history-references.md#ref-6deef0520efd)、[T11.2](history-references.md#ref-97b0743ec8d6) 已下发（D50，用户 2026-10-28 批准）。首批交付 ①–③ 与块内 `introduced` 算子定位，④ 只报告。<br>历史：☐ 未开始 |

## 6. P5 — 规模化

| 项 | 内容 |
|---|---|
| 交付 | `campaign` 命令、矩阵展开、断点续跑、campaign 作业域内的重试与跳过（**依据 run 的退出状态与超时**，不依赖 P3 的规则引擎；基于 Finding 的隔离在 P3 完成后接入 `campaign` Responder）、`_index/`、批量汇总报告、[07 §4](07-analyzers-and-reports.md) 的全部机器可读导出 |
| 验收 | ① 一批（≥ 50 个）模型 × 2 厂商 × flaggems off/on 的 campaign 跑完并出汇总；② 中断后续跑只执行未完成格子；③ 导出清单被 FlagGems/FlagTree/FlagCX 实际消费一次（有反馈记录）；④ `inspect runs --where` 查询可用 |
| 需求 | R89、R92–R97、R31 的反馈闭环、R106–R108 |
| 层级 | 扩展（主线之后的第一个阶段） |
| 进入条件 | 主线（P0、P1、P2、P4）全部退出 |
| 状态 | ☐ 未开始 |

## 7. P6 — 第二家厂商与 v1 退役

| 项 | 内容 |
|---|---|
| 交付 | 第二家国产芯片的 `vendors/<v>/`（候选见 [DECISIONS](DECISIONS.md) 待定项）；v1 能力对照检查表（以 [§0.1](#01-v1-能力清单p5-之前的排序依据) 为底稿，Q12）；v1 标记为 deprecated |
| 验收 | ① 接入该厂商**未修改** `store/`、`correlate/` 框架、`analyzers/` 中的任何一行 —— 这是对 [02](02-architecture.md) P6 原则的检验，若做不到则说明抽象有问题，必须回头改 schema/接口；② 该厂商通过 [05 §4.2](05-probes-and-vendors.md#42-接入一家新厂商的验收清单) 的全部验收项（当前 11 项，含身份、来源守恒、当前采集器贯通与逐跳解释）；③ v1 的全部有效能力在 v2 中有对应项（逐条打勾） |
| 需求 | 5 厂商覆盖、R18/R22 |
| 层级 | 扩展 |
| 进入条件 | P5 退出 |
| 状态 | ☐ 未开始 |

**v1 退役条件**（决策 D5）：P6 的 ③ 全部打勾，且 v2 在至少 2 家厂商上稳定运行一个完整 campaign。在此之前 `inference/` 保持可用，不删除。

> 由于 P3 排在 P6 之后（D20），v1 能力对照中若有依赖 P3 的项（如监控、异常现场），在 P6 时逐项登记为「待 P3」，不为了退役而把 P3 提前拉进 P6。

## 8. P3 — 故障检测与在线通道

| 项 | 内容 |
|---|---|
| 交付 | health 探针、`watch` 命令、规则引擎（[08 §2](08-online-and-fault.md) 的 8 条规则）、`log`/`exporter` Responder、`findings.jsonl` 协议冻结、lifecycle span 与 `stability.drift` 起步 |
| 验收 | ① 故障注入测试：人为制造 hang（持锁）、OOM、kill 某 rank、拔通信，规则在阈值内检出且**证据正确**；② 在线尾读在活跃分片上无误读（不完整行跳过）；③ 长稳：连续运行 ≥ 24 h，store 体积受控，漂移可见 |
| 需求 | R32、R33（起步）、R77–R79；需求池 X12 |
| 层级 | 扩展（最后推进，D20） |
| 进入条件 | P6 退出 |
| 状态 | ☐ 未开始 |

## 9. 状态表（由 codex 回填）

| 阶段 | 状态 | 证据（PR / 报告 / 测试） | 完成日期 |
|---|---|---|---|
| S1 Spike-C | ✅ 已审核；后续 T1 设计门禁未通过，契约已回填 | [实测报告](history-references.md#ref-e1bd695b52c5)、[Q7](DECISIONS.md#q7--spike-c-实测回填与冻结边界)、[D15](DECISIONS.md#d15--t1t3-之后的契约回填r1r8) | 2026-10-11 |
| P0 store+schema | ✅ **2026-10-27：P0 已退出，L-字段 `1.0`**（[D45](DECISIONS.md#d45)）；③ 按登记低负载范围达标，附已知限制。历史（至 2026-10-27 收尾前）：**2026-10-26**：P0 未退出，L-字段 1.0-rc；T9.4 收口通过，T9.6 登记验收通过，T9.5d 后 S9 已收口；③ 按 D31 ③a/③b/③c 待 Stage 1（B2X 仍是 `spikes/` 原型，生产 Writer 未接入）。历史（至 2026-10-26）：T9.0–T9.2 Gate通过；T9.3最终真机验收执行完成。Q9③未达标，P0未退出，L-字段1.0-rc。（上述 Gate 为 codex 内部复核；设计者 Gate 2026-10-18：T9.0、T9.1 通过，T9.2 有条件通过，T9.3 生产入口未贯通，转 T9.4） | [退出矩阵](history-references.md#ref-e0872549b0c8)、[D45](DECISIONS.md#d45)、[退出矩阵](history-references.md#ref-e0872549b0c8)、[T9.3执行记录](history-references.md#ref-61e29505d51c)、[设计者审核（2026-10-18）](history-references.md#ref-79d0492278f2) | 2026-10-27 |
| 　P0 · T9.4 生产入口收口 | **实现、验收与用户授权独立代理 Gate 通过**（范围 M1–M9、A1–A10，不代表 T9.5 容量放行或 P0 退出）。历史（至 2026-10-20）：用户 2026-10-19 确认下发（待用户提交工作树后开工）（不依赖 D31） | [T9.4](history-references.md#ref-e525188761b6)、[最终 Gate](history-references.md#ref-d5d1b746e820)、[裁决依据](history-references.md#ref-5558e1d8a7f3) | 2026-10-20 |
| 　P0 · T9.5 Writer 消费端移出 GIL 与 ③ 三项验收 | **2026-10-26**：Stage 0 完成，经 T9.5b Gate 选 B2X 类、不保留 A（[D39](DECISIONS.md#d39--t95b-gate-裁决选-b2x-类不保留-astage-1-前插-t95creader-流式化t96为-p0-退出前置)）；T9.5c、T9.5d 见下两行。B2X 仍是 `spikes/` 原型，Stage 1（生产接入与 ③a/③b/③c 验收）未开工，将由新的 Writer Stage 1 任务单承接（下发中）。历史（至 2026-10-26）：用户 2026-10-19 确认下发（待用户提交工作树后开工）（Stage 0 可与 T9.4 并行） | [T9.5](history-references.md#ref-06e993298ce0)、[D31](DECISIONS.md#d31--p0-吞吐验收改为三项并报退役单一-accepted-速率)（用户 2026-10-19 采纳） | |
| 　P0 · T9.5c B2X 语义修复与确认测量 | 实现与取证交付，硬条件未通过（V2 I3a、S2、S7 有效覆盖、S5 参照）；失败项由 T9.5d 承接，原证据保留 | [T9.5c](history-references.md#ref-cdbf854eab27)、[Gate 输入](history-references.md#ref-3124ca1c1b92) | 2026-10-23（未通过） |
| 　P0 · T9.5d 生命周期与 S9 收口 | S9 已通过；G1/F11 按用户决定非阻塞，可继续后续生产推进；P0 未退出 | [S9 收口交付](history-references.md#ref-03dff9b5e6ad)、[D42](DECISIONS.md#d42) | 2026-10-25 |
| 　P0 · T9.6 Reader 流式化与有界内存 | 登记验收通过（生产 `flagprof/store`）；2026-10-26 当前 HEAD 源码身份过期已重新绑定（独立旧实现 N0 差分相等） | [T9.6](history-references.md#ref-6d7e83e32eee)、[验收报告](history-references.md#ref-318933687dbe)、[重新绑定](history-references.md#ref-4f341f4eb690) | 2026-10-23 |
| 　P0 · T9.7 Writer Stage 1 生产接入 | 已交付并关闭：进程外代理写者接入生产；W1–W6、W10、X86 A5 通过，W8 完整报告；W7 有效失败（0.946409 < 0.963973）按 D45 降为跟踪基准，W9 aarch64 A5 零回退观察按 D45 接受 | [T9.7](history-references.md#ref-418019448c44)、[报告](history-references.md#ref-0ffd66d1ab4a)、[Gate 输入](history-references.md#ref-a1968734253f) | 2026-10-27 |
| 　P0 · T9.8 真实推理闭环校准（P1 预演） | 已交付并关闭：真实轻量记录约 70 条/s，C1/C0 吞吐比中位 0.995；四段闭环与独立复算通过；时钟、身份传播与主机桥接缺口归入 P1/P2 | [T9.8](history-references.md#ref-104a0f28ea18)、[报告](history-references.md#ref-e93d1c4566d0) | 2026-10-27 |
| P1 Python 关联链 | ✅ 设计者复审：登记范围内通过并退出（[D49](DECISIONS.md#d49)，用户 2026-10-28 批准）；报告口径由 T10.4 修补 | [退出矩阵](history-references.md#ref-567cd9829d3a)、[复盘页](history-references.md#ref-8578565fdbb8) | 2026-10-28 |
| 　P1 · T10.1 探针框架与主机侧关联链 | 实现及A1–A15证据已交付，编译未覆盖能力单列 | [报告](history-references.md#ref-05b7b467c2dd) | 2026-10-27 |
| 　P1 · T10.2 分析器、span树与最小报告 | 实现及B1–B14证据已交付，两档小型离线规模检查通过 | [报告](history-references.md#ref-c0ac3cd26c4e) | 2026-10-27 |
| 　P1 · T10.3 真实模型验收与退出证据 | 小规模C1–C8及N1初步观察已交付；待设计者复审 | [报告](history-references.md#ref-86ccff91a5b8) | 2026-10-27 |
| 　P1 · T10.4 P1.1 报告可信度修补 | 工程执行完成；回退清单 5 键，L1/L0 吞吐比 1.0032 | [报告](history-references.md#ref-c6b5dc8115ca) | 2026-10-28 |
| P2 设备关联 | T12.1–T12.5 与 T13 OneRec／TP2 登记范围工程完成；完整 P2 未退出。按 D54 收窄退出范围后，设备关联、时钟判定与 `perf.gap` 的工程结果见 T14.1/T14.2；P2 退出另行裁决 | [T13 验收](history-references.md#ref-409c11becc2f)、[T14.1](history-references.md#ref-7271dc8d7c92) | 2026-10-31 |
| 　P2a · T12.2 prepare 提速 | 工程完成：映射复用、精确整数索引；M 三对/S/Reader/双解释器 CI 通过；两软目标未完全达到，收益收敛停止 | [报告](history-references.md#ref-c7d5393110c1) | 2026-10-29 |
| 　P2a · T12.3 校验与读取复用 | 已完成；S 两对对照、双厂商与双解释器回归通过 | [任务](history-references.md#ref-f7a5d79a28b5) | 2026-10-29 |
| 　P2a · T12.4/T12.5 设备与显存 | 登记范围工程验收完成；两厂商真实数据、独立复算和双解释器 CI 通过 | [设备](history-references.md#ref-dccf909d6621)、[显存](history-references.md#ref-9187026d3646) | 2026-10-30 |
| 　P4 · T11.4/T11.5 第二批 | 工程验收完成；经 T13.1 集中复审纳入 P4 登记范围退出 | [联合报告](history-references.md#ref-9596bcfb2a96) | 2026-10-30 |
| P4 精度差分 | T13.1 按 D50–D52 登记小范围通过并退出；首批、T11.3 和第二批集中复审成立 | [退出裁决](history-references.md#ref-49da047de8fe) | 2026-10-30 |
| P5 规模化 | ☐ | | |
| P6 第二厂商 | ☐ | | |
| P3 故障+在线 | ☐ | | |

## 10. 不在本轮范围内

范围边界的唯一出处是 [D21](DECISIONS.md#d21--范围边界暂不做与不做清单)，这里只列条目，理由与重新打开的条件见该决策：

- 统一运行时"监控诊断模块"本体，以及运行时各组件的观测契约（B1）
- 精度修正、自动调优、自动修复 —— 本工具只诊断（B2）
- 恢复动作的执行：重试、迁移、隔离交给运行时或调度方（B3；campaign 作业域内的重试与跳过除外）
- 锚定外部里程碑日期（B4）
- TensorFlow / ONNX Runtime 探针（B5，[Q2](DECISIONS.md#q2--tensorflow--onnx-runtime-的范围)）
- 训练、微调、强化学习场景 —— schema 已预留 `tensor.role=grad`，但不实现（B6，[Q4](DECISIONS.md#q4--训练场景是否纳入)）
- vLLM 等推理引擎适配（B7，[Q6](DECISIONS.md#q6--vllm-等引擎的-attach-时机与版本策略)）
- 自研 UI / 时间轴渲染器 —— `perf.timeline` 只产出数据，渲染用现成方案（B8）
- 图模式内部的算子级归因（B9，[04 §5](04-correlation.md)）

参考文档中其余尚未排期的要点在 [00 §6 需求池](00-requirements.md#6-需求池参考文档补充) 中，按层级等待排期，不属于"排除"。


## 前置跨厂商 Spike 的执行证据（2026-10-14）

用户确认的 P800/PPU S0–S3 隔离 Spike 已取得两家真实 M0、完整原生模型矩阵、控制、三路导入评估及接口反例，见 [执行报告](history-references.md#ref-fa068bec36ec) 和 [契约建议](history-references.md#ref-57f7fa1a5488)。原生关系可用但缺合法项目 corr；PPU 注释和 P800 CLI 格式/资源语义存在明确边界。Gems 模型、精度工具等条件能力按证据保留 blocked/unknown/not_tested。

本项完成的是提前取样的受限研究，不代表完整厂商产品化、完整 P6 或 P0 性能通过。生产 store/linker/schema 未修改，既有阶段顺序和退出条件保持原状；公共入口与身份方案只有未应用草案，采纳仍需后续决策。

**2026-10-14 设计者审核**：依据本 Spike、profiler 复用实验，以及 GPT、GLM 的独立审查，新增 [D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合)–[D24](DECISIONS.md#d24--导入契约身份分层来源账本完成标记证据索引)、Q8–Q9、[附录 A](DECISIONS.md#附录-al-核心冻结清单p0-开工即冻结) #25–#28，基座冻结工作下发为 [T8](history-references.md#ref-24a77204cd82)。在 T8 中，P800/PPU 样本只作离线回归样本，生产 adapter 仍属 P6；阶段顺序（D20）不变。

> **2026-10-20 T9.4 收尾**：用户授权的独立代理复核通过，修复、现场生产入口与干净回归已绑定提交；见 [裁决依据](history-references.md#ref-5558e1d8a7f3)。T9.5 分组协议已按 D36 预登记，1,280 窗口 A/A 完整执行后噪声超限；获批的 64 进程 CPU/GC 诊断未重现长尾、结论 inconclusive，尚未形成选型或生产改造验收；P0 ③ 仍未通过，L-字段保持 1.0-rc。

> **2026-10-20 恢复推进**：T9.5 恢复方案已批准并先提交环境预登记；当前 CPU 分配未满足资格，暂不运行四格/R60。空闲 NPU 使用已获授权，但不等同 CPU 隔离。见 [环境资格证据](history-references.md#ref-0f2e9c2e603b)。T9.4 Gate 保持通过，T9.5 Stage 0/P0 均未完成。

> **2026-10-20 CPU 授权后的最新结果**：用户允许使用当前空闲 CPU，已先修订为共享宿主有界诊断并完成全部128窗口；末次CPU负载门失败，最终环境影响未排除。G2消费者参与关联明显，但不是正式5%校准通过；见[结果](history-references.md#ref-8e107007aa49)。累计1,633进程，未选型/未开始Stage1。

> 2026-10-23 [D41](DECISIONS.md#d41)：用户授权先冻结任务单并实现 T9.6、T9.5c。Reader 退出项限于登记画像下消除普通在线事件全量常驻且满足预算；T9.5c 阶段性 CI、计数高水位与参照资格按修订任务单。Stage 1 与 P0 仍需各自 Gate。

> **2026-10-23 T9.6 验收完成**：生产流式读取及 finalize 在登记画像下通过全部内存与响应门，独立旧实现差分通过；[验收包](history-references.md#ref-318933687dbe)。随后在此冻结生产树上完成了 T9.5c 取证；其硬条件未通过，见文首最新回填。

## 历史状态记录


<details>
<summary>本轮交付前状态（原文）</summary>

> **当前状态（2026-10-29，P4 首批复审通过，P4.1 与 P2a 开工）**
> - P1：已退出（[D49](DECISIONS.md#d49)）；P1.1 [T10.4](history-references.md#ref-3f47daaaa459) 工程执行完成。
> - P4：首批在登记范围内通过复审（[D51](DECISIONS.md#d51)，用户 2026-10-29 批准）；P4.1 [T11.3](history-references.md#ref-3c84559c539d) 修订 oracle 误差口径与页面；第二批排在 T12.1 之后。
> - P2a：[T12.1](history-references.md#ref-6ce6c9c166a0) 导入定点分析已完成工程交付；导入门限 N、X 在其交付后由用户给出，目标负载为 v1 OneRec-8B 生成 32 个新 token。
> - P5 之前的工作按 [§0.1 v1 能力清单](#01-v1-能力清单p5-之前的排序依据) 排序（[Q12](DECISIONS.md#q12--p5-之前是否按-v1-能力清单排序)，用户 2026-10-29 裁决）。P3 不提前（[Q11](DECISIONS.md#q11--p3故障检测与热加载是否提前)）。
> - 上一版状态块（2026-10-28）：P1 退出，P4 首批下发，P2 拆为 P2a（导入可用性 + 显存视图）与 P2b（通信、时钟、跨 rank），导入速度是 P2 入口条件（D49 第 5 条）。

</details>

### 2026-10-28 P1 退出复审前的状态块（原文）

> **当前状态（2026-10-27，P1 小规模工程验收交付）**
> - T10.1/T10.2生产实现与T10.3小规模功能验收材料已交付，P1 **待设计者复审**，入口为 [退出矩阵](history-references.md#ref-567cd9829d3a) 与 [正式报告](history-references.md#ref-86ccff91a5b8)。
> - 当前范围按用户最新修订：共用一次最终源码smoke，R1–R3/C0G/C0，N1-a三对与N1-b一对。系统性性能统计与大规模容量验证另行安排；不作性能达标结论。
> - D47的schema1.1/dev/nested与静态route_table、D48的生命周期/身份/有界分析均已实现。编译失败/失效/进程内复用、R3非零corr分支仍未覆盖；时钟unavailable，240万来源记录的历史有效超时和固定镜像归档失败保留。
> - P0既有设计者裁决保持；P2/P4的排期由P1复审决定。此前下发记录与原失败按各自历史材料保留。


### 2026-10-27 P1 首批下发前的状态块（原文）

> **当前状态（2026-10-27）**
> - **P0 已退出，L-字段定为 `1.0`**（[D45](DECISIONS.md#d45)，用户 2026-10-27 批准）。③ 按登记低负载范围达标并附已知限制（[退出矩阵](history-references.md#ref-e0872549b0c8)）；T9.7 Writer Stage 1 与 T9.8 真实闭环校准已交付，W7 有效失败降为跟踪基准，W9 aarch64 A5 以零回退观察接受。
> - 下一阶段：**P1 Python 侧关联链**。首个任务单按 D45 第 7 条编写：dispatch 默认 summary 聚合；开销按真实模型 C0/C1 口径；主机桥接是 P1 的交付内容；seal 身份传播在首个任务内修复；route/回退清单需要装有 FlagGems 的环境。
> - 2026-10-27 之前的状态块原文移至文末[历史状态记录](#历史状态记录)。

### 2026-10-27 P0 收尾前的状态块（原文）

> **当前状态（2026-10-27）**
> - D33 完整 B 链路已整合为正式默认，限定实际 ext4 工作区；双厂商冷导入、默认读取与资源 smoke 通过。D33 9164e171 已提交并交接，T9.7 生产接线已完成；见[整合与交接](history-references.md#ref-a80dbb351dc8)。
> - 已完成：T9.4 生产入口收口（[独立代理最终 Gate](history-references.md#ref-d5d1b746e820)，2026-10-20）；T9.6 Reader 流式化登记验收通过（2026-10-23）；T9.5d ext4 正式验收后 S9 已收口，G1 极差 0.035036 与 F11 由用户按 [D42](DECISIONS.md#d42) 处置为非阻塞。
> - 未完成：T9.5c 硬条件原失败保留；T9.7 W7 有效回归失败、W9 缺有效 aarch64 A5，正式验收未通过；P0 未退出，L-字段保持 1.0-rc（[退出矩阵](history-references.md#ref-e0872549b0c8)）。
> **2026-10-27 D44 本轮交付**：T9.8 原始 e385f495 正式校准与独立身份整合后闭环均已完成；T9.7 在线代理／离线共享优化、O2/O3、最终双解释器／134 故障条件、W3–W6/W8、X86 A5 与真实入口已交付。新四对 A/A 合格，正式十对 W7 中位 0.946409 低于原门 0.963973，依用户决定非阻塞且不再修复；aarch64 W9 供给仅 94.44%，低于 98%，原五分钟余量不足重验，停止并保留 blocker。W1 65 分钟、W8 30 分钟已获批准，总八小时／十二小时不变。工程原门、完整测量通过、P0③与 P0 退出均未成立。 见 [T9.8 报告](history-references.md#ref-e93d1c4566d0)、[T9.7 报告](history-references.md#ref-0ffd66d1ab4a)及 [Gate 材料](history-references.md#ref-a1968734253f)。
> - 2026-10-26 前叠放在文首的带日期状态横幅，原文移至文末[历史状态记录](#历史状态记录)。

### 2026-10-26 状态块合并前叠放的横幅（原文）

以下为 2026-10-26 状态块合并前叠放在文首的带日期状态横幅，按原叠放顺序（新→旧）原文保留，仅作历史语境；当前状态以文首“当前状态”块为准。

> **2026-10-25 最新：S9 已收口；G1 极差与 F11 已由用户接受为非阻塞项，可继续后续生产推进。** a04/a05 补证：store 329、correlate 79、conformance 21 全过，417-case/50,040 次比较与最终字节一致完成。原正式结果保留；本轮未执行生产接入，P0 未退出。以下早期失败/pending 条目保留历史语境，不再以当前 G1/F11 阻碍推进。 [D42](DECISIONS.md#d42)；[收口证据](history-references.md#ref-03dff9b5e6ad)。


> **2026-10-23 实施回填**：T9.6 登记验收通过；T9.5c 实现与取证已交付，候选硬条件未通过。145 个登记 case 全部执行、155 个观测；V2 有 9 次 I3a 失败，S2 仅 8/15；S7 70k 有效覆盖不足，S5 参照不合格。最终 CI 292/79/21 全部通过不覆盖这些反证。Stage 1 与 P0 未放行。 [T9.5c Gate 输入](history-references.md#ref-3124ca1c1b92)。

> 2026-10-20 授权修订：[D34](DECISIONS.md#d34--t94t95-读取批次与测量边界补充)与 T9.4 M8/M9、T9.5 §4.7 补充公共读取校验、首批屏障、质量/身份和测量/关闭边界。P0 未退出；历史验收与旧状态保留。当前任务具体规则以两份修订任务单为准。


> 2026-10-18 执行更新：T9.0–T9.2 已实现并通过独立 Gate；T9.3 生产实现完成，最终真机验收执行完成。Q9③ accepted吞吐未达500,000/s，P0未退出。以[当前退出矩阵](history-references.md#ref-e0872549b0c8)为准；旧阶段记录保留原日期。

<!-- D44 superseded entry, retained history -->
> - **下一步：Writer Stage 1 与真实闭环校准：[T9.7](history-references.md#ref-418019448c44)（Stage 1 生产接入）、[T9.8](history-references.md#ref-104a0f28ea18)（真实闭环校准）已下发（[D43](DECISIONS.md#d43)），待 codex 开工。** 逐项状态见 [§9 状态表](#9-状态表由-codex-回填)。


<details>
<summary>T12.2 交付前入口状态（原文）</summary>

> **当前状态（2026-10-29，T11.3/T12.1 工程交付）**
> - P0 按 D45 在登记低负载范围退出；P1 按 D49 退出，P1.1 T10.4 已交付。
> - P4 首批在登记范围内通过 D51 复审；[T11.3](history-references.md#ref-3c84559c539d) 六报告、字段映射和双解释器 CI 完成，待设计者复审。[验收](history-references.md#ref-4ea9cd3ce263)。完整 P4 退出仍需第二批。
> - P2a 进行中：[T12.1](history-references.md#ref-6ce6c9c166a0) S/M 成本、profile 与单参数对照完成，缓存增大未观察到提速。[交付](history-references.md#ref-a103fc7f312b)。下一步由用户定门限后登记 T12.2。
> - OneRec 目标为 32 个新 token，当前仅有明确假设的敏感性区间；L 代表性条件未满足而不执行。P3 不提前，P5 前按 v1 能力清单排序。决策见 [D51](DECISIONS.md#d51)。

</details>


<details>
<summary>2026-10-31 T13 复盘回填前的 v1 能力表与 P2 状态（原文）</summary>

| 序 | v1 能力 | v2 现状（2026-10-29） | 归属与依赖 |
|---|---|---|---|
| — | Qwen3-Embedding 模型级／块级精度与 A100 参考包 | ✅ 360/360 逐字节复现；三组跨芯片 | P4 首批（D51） |
| — | 算子级张量精度（v1 为 not_implemented） | ✅ introduced 与放大；口径由 T11.3 修订 | P4 首批 → P4.1 |
| — | 主机侧回退率与算子级性能（v1 为 not_implemented） | ✅ 主机侧 | P1／P1.1 |
| 1 | 导入速度（generate 与设备视图的共同前置） | 部分：T12.2 的 52 万条外层 5.4 min，prepare −36.31%；OneRec 门限待实测。历史 52 万条约 6.5 min、240 万条超 15 min 的原记录保留 | P2a：[T12.2 验收](history-references.md#ref-c7d5393110c1) → 设备/显存视图与 OneRec 目标样本 |
| 2 | layer 性能、设备 profiler 热点与设备侧回退率 | 登记范围内工程验收完成：两厂商设备热点、逐层／路由工作量和官方显存视图成立 | [T12.4/T12.5 联合交付](history-references.md#ref-9596bcfb2a96)，P2b 留后续 |
| 3 | OneRec-8B generate：prefill／decode、逐 token 首差异、32 步性能 | ☐ | generate 任务（P2a 之后），依赖序 1、2 与 P4 采集 |
| 4 | A100 性能基线与跨设备性能比 | ☐（P4 只做了 A100 精度） | 随序 2、3 在 NVIDIA 上补采，不单独成阶段 |
| 5 | TP2／HCCL 多卡精度与性能；FlagCX 通信对照 | ☐ | P2b，依赖序 1 |
| 6 | preview 策略（FlagGems 分组搜索） | 部分：一次 run 的 `route.coverage` 给出回退清单（D8） | 隔离验证二次 run，P2b 之后 |
| 7 | DLRM、DeepFM 推荐模型 | ☐ | 模型接入，P2b 之后 |
| 8 | FlagTree 编译器对照 | ☐ | 组件对照，序 7 之后 |
| 9 | FX／ONNX 图导出 | ☐ | 排在最后，开工前核对它在 v1 中的实际使用者 |

P4 第二批（replay 只针对输入已不同的算子、`layers.27` 下钻、跨 run sid 对齐）是 P4 退出条件，不属于 v1 对齐清单，按 D51 第 4 条排在 T12.1 之后，可与序 2 并行。


| 状态 | **当前（2026-10-29）**：▶ P2a 进行中。T12.1 定位完成；[T12.2](history-references.md#ref-61027047fd40) 两轮优化与登记验证完成，M prepare −36.31%、完整导入 −20.63%，软目标未完全达到且按收益收敛收尾。T12.4/T12.5 设备/显存视图登记范围内工程验收完成（2026-10-30）。OneRec 32 token 的匹配真实样本及实际导入门限继续待实测。P2b 未开始。 |

| P2 设备关联 | P2a 设备/显存工程验收完成；T12.1—T12.5 已交付，后续 OneRec 门限和 P2b 单列 | [T12.2 验收](history-references.md#ref-c7d5393110c1) | |

</details>
