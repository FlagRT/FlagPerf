# 框架与协作导航

当前生产根目录是 `FlagPerf/inference`，Python 包名保持 `flagprof`。首次使用读 [模块首页](../README.md) 与 [使用说明](usage.md)；接手开发按“数据从哪来、如何落盘、如何消费”的顺序阅读。

P800 候选适配见 [环境、命令与验证边界](../../docs/kunlunxin/inference-v2.md)，不沿用 NVIDIA/Ascend 的硬件资格结论。

## 阅读顺序

1. [02 总体架构](02-architecture.md)：目标进程 → 探针 / 厂商采集 → Store → 分析器 → 报告。
2. [03 事件 schema](03-event-schema.md) 与 [04 关联脊梁](04-correlation.md)：理解事件、身份、时钟、来源、关联和公共读取接口。
3. [05 探针与厂商](05-probes-and-vendors.md)、[06 模型接入](06-model-onboarding.md)、[07 分析器与报告](07-analyzers-and-reports.md)：按负责模块深入。
4. [10 路线图](10-roadmap.md)、[DECISIONS](DECISIONS.md)：核对阶段范围和已经决定的接口。

## 框架文档

| 编号 | 文档 | 主要问题 |
|---|---|---|
| 00 | [需求追溯](00-requirements.md) | 项目要解决哪些问题 |
| 02 | [总体架构](02-architecture.md) | 分层、数据流、接入模式和目录布局 |
| 03 | [事件 schema](03-event-schema.md) | 公共数据契约与演进 |
| 04 | [关联脊梁](04-correlation.md) | 算子、主机、设备、layer 的证据关联 |
| 05 | [探针与厂商](05-probes-and-vendors.md) | 采集边界和厂商适配职责 |
| 06 | [模型接入](06-model-onboarding.md) | 模型语义如何进入公共框架 |
| 07 | [分析器与报告](07-analyzers-and-reports.md) | 从事件得到可解释结论 |
| 08 | [在线与故障](08-online-and-fault.md) | 在线通道、恢复与热加载设计 |
| 09 | [CLI 与批量运行](09-cli-and-campaign.md) | 用户入口和批量编排设计 |
| 10 | [路线图](10-roadmap.md) | 阶段、依赖、范围和退出条件 |
| — | [决策记录](DECISIONS.md) | 接口与阶段判断依据的已登记决定 |

## 当前状态与文档读法

截至 2026-11-02，P0/P1/P2/P4 已按各自登记范围退出；P2 的最新裁决见 [D55](DECISIONS.md#d55)。设备覆盖、性能和长期运行结论以对应记录为依据，v1 中尚未实现的能力见 [版本迁移](migration.md#未实现项)。

框架文档是持续演进的材料，其中保留计划、历史状态和曾提议的命令。当前可执行参数以 `python -m flagprof --help` 和 [使用说明](usage.md) 为准；P3 在线恢复 / 热加载、P5 与其他后续内容按路线图推进。历史资料已从当前目录移除；正文中的历史引用指向删除前的固定 Git 提交，实验出处见 [出处索引](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md)，原件保存在维护者的本地开发仓库。

## 两人如何协作

`FlagPerf/inference` 是公开的生产代码，按本仓库的规范单独进行版本管理。维护者另有私有的开发和测试目录（`FlagPerf_advance/inference_v2`），用于实验、完整回归和验收取证；该目录不公开，也不是本目录的上游分支。两边的生产代码通过 PR 同步，`flagprof/` 的逐字节来源记录在 [版本迁移](migration.md)。

可以沿数据接口拆分工作：一人负责采集与导入（`probes/`、`vendors/`、`offline/`），另一人负责读取与消费（`store/view.py`、`analyzers/`、`report/`）。`store/schema.py`、身份、时钟、关联规则、`registry.py` 和 CLI 配置是共同接口，相关修改由双方先确认契约，再分别实现。相同的公共事实由一个模块提供，消费者通过 StoreView 或注册入口读取。

每人使用自己的功能分支，通过 PR 合入 `dev-1.0`；`main` 保留上游同步职责。每个 PR：

1. 通过 CPU 冒烟测试 `tests/test_smoke.py`（PR 触发 GitHub Actions，也可在 `inference/` 下执行 `python3 -B -S -m unittest discover -s tests -v`）；
2. 涉及采集、厂商适配、导入或报告口径的改动，附上 [设备最小验收](usage.md#设备最小验收) 的命令、被测 commit 与结果；
3. 在说明中写明触发条件、改变后的行为、验证方式和适用范围。

功能或契约变化同步更新使用说明、相应编号文档和决策记录。编号文档中的 `personal/` 路径指维护者本地的同步文档，用于解释原开发流程。
