# 框架与协作导航

当前生产根目录是 `FlagPerf/inference`，Python 包名保持 `flagprof`。首次使用读 [模块首页](../README.md) 与 [使用说明](usage.md)；接手开发按“数据从哪来、如何落盘、如何消费”的顺序阅读。

## 阅读顺序

1. [02 总体架构](02-architecture.md)：目标进程 → 探针 / 厂商采集 → Store → 分析器 → 报告。
2. [03 事件 schema](03-event-schema.md) 与 [04 关联脊梁](04-correlation.md)：理解事件、身份、时钟、来源、关联和公共读取接口。
3. [05 探针与厂商](05-probes-and-vendors.md)、[06 模型接入](06-model-onboarding.md)、[07 分析器与报告](07-analyzers-and-reports.md)：按负责模块深入。
4. [10 路线图](10-roadmap.md)、[12 交接约束](12-codex-handoff.md)、[DECISIONS](DECISIONS.md)：核对阶段范围和已经决定的接口。

## 框架文档

| 编号 | 文档 | 主要问题 |
|---|---|---|
| 00 | [需求追溯](00-requirements.md) | 项目要解决哪些问题 |
| 01 | [v1 诊断](01-diagnosis.md) | 原有耦合与重构原因 |
| 02 | [总体架构](02-architecture.md) | 分层、数据流和接入模式 |
| 03 | [事件 schema](03-event-schema.md) | 公共数据契约与演进 |
| 04 | [关联脊梁](04-correlation.md) | 算子、主机、设备、layer 的证据关联 |
| 05 | [探针与厂商](05-probes-and-vendors.md) | 采集边界和厂商适配职责 |
| 06 | [模型接入](06-model-onboarding.md) | 模型语义如何进入公共框架 |
| 07 | [分析器与报告](07-analyzers-and-reports.md) | 从事件得到可解释结论 |
| 08 | [在线与故障](08-online-and-fault.md) | 在线通道、恢复与热加载设计 |
| 09 | [CLI 与批量运行](09-cli-and-campaign.md) | 用户入口和批量编排设计 |
| 10 | [路线图](10-roadmap.md) | 阶段、依赖、范围和退出条件 |
| 11 | [CANN 关联 Spike](11-spike-cann-correlation.md) | 架构关键假设及其验证设计 |
| 12 | [交接约束](12-codex-handoff.md) | 实现、验证和文档维护规则 |

[决策记录](DECISIONS.md) 与以上文档一同迁入，因为接口和阶段判断依赖其中的决定。[设计历史入口](design-history.md) 保留原 PLAN 索引和带日期的状态记录。

## 当前状态与文档读法

截至本次 2026-11-02 来源快照，P0/P1/P2/P4 已按各自登记范围退出；P2 的最新裁决见 [D55](DECISIONS.md#d55)。这描述原开发阶段的裁决。迁移本身完成代码发布与入口整合，设备覆盖、性能和长期运行结论仍以对应记录为依据。

00–12 是持续演进的框架材料，其中保留计划、历史状态、曾提议的命令以及开发工作区路径。当前可执行参数以 `python -m flagprof --help` 和 [使用说明](usage.md) 为准；P3 在线恢复 / 热加载、P5 与其他后续内容按路线图推进。正文中的历史实验链接指向 [出处索引](history-references.md)，本次发布保留引用身份，原件继续存于本地。

## 两人如何协作

可以沿数据接口拆分工作：一人负责采集与导入（`probes/`、`vendors/`、`offline/`），另一人负责读取与消费（`store/view.py`、`analyzers/`、`report/`）。`store/schema.py`、身份、时钟、关联规则、`registry.py` 和 CLI 配置是共同接口，相关修改由双方先确认契约，再分别实现。

每人使用自己的功能分支，通过 PR 合入 `dev-1.0`；`main` 保留上游同步职责。PR 说明触发条件、改变后的行为、验证方式和适用范围。相同的公共事实由一个模块提供，消费者通过 StoreView 或注册入口读取。

本次按交付范围保留生产代码与文档，回归套件和实验记录留在本地验证工作区。提交前由验证负责人运行匹配版本的本地套件，记录被测 commit 和结果；设备相关改动再运行对应环境中的小样本完整链路。公开仓库可独立执行 CLI、采集、导入与报告。

功能或契约变化同步更新使用说明、相应编号文档和决策记录。公共目录的当前协作说明以本页为入口；编号文档中的旧 `personal/` 同步路径用于解释原开发流程。
