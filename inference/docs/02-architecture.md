> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书链接转入 [历史引用索引](history-references.md)，其出处保留。

# 02 总体架构

> 上游：[00 需求](00-requirements.md)、[01 诊断](01-diagnosis.md)、[DECISIONS](DECISIONS.md) D1–D5、D22–D24
> 下游：[03 schema](03-event-schema.md)、[04 关联](04-correlation.md)、[05 探针](05-probes-and-vendors.md)、[06 模型接入](06-model-onboarding.md)、[07 分析器](07-analyzers-and-reports.md)、[08 在线/故障](08-online-and-fault.md)、[09 CLI](09-cli-and-campaign.md)

## 1. 设计原则

| # | 原则 | 针对的 v1 问题 / 需求 |
|---|---|---|
| P1 | **契约是唯一不可替换的资产**。契约 = 规范事件 schema + 来源身份 + 版本化派生证据（原生关系、来源账本、证据索引）+ 公共只读接口。探针、分析器、报告都可替换，契约冻结后只做兼容演进（2026-10-15 按 [D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)、[D24](DECISIONS.md#d24--导入契约身份分层来源账本完成标记证据索引) 的落地结果改写，原文为「schema 是唯一不可替换的资产」） | nsys 的经验：采集与分析之间有一份公开、带版本、自描述的契约，两侧各自换代只需遵守它的当前版本（[D1](DECISIONS.md#d1--参照-nsys-的schema-即产品但扩展为三合一)；原写「`.nsys-rep` 的表没变」是事实错误，2026-10-14 补改） |
| P2 | **探针挂进程，不挂模型**。任何探针都不得 import 模型相关代码 | [01 §2.2](01-diagnosis.md)；R92–R97 的 500 模型 |
| P3 | **分析是 store 上的纯函数**。分析器只经 store 的公共只读接口读取（事件 + 版本化只读元数据与派生视图），不直接读运行目录里的文件、不读 `ext`、不解析厂商原件（[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)） | [01 §2.3](01-diagnosis.md) |
| P4 | **语义可选、逐级增强**。没有任何模型配置时也必须产出有用结果（Tier 0） | 500 模型接入成本 |
| P5 | **关联靠 id，不靠时间**。关联不上就如实标注，覆盖率对用户可见 | v1 原则继承；N6 可审计 |
| P6 | **厂商差异只在 `vendors/<v>/`**。schema、关联规则框架、分析器以及 `vendors/` 下的公共模块中，不出现厂商名分支或厂商类型字符串；厂商分类表带版本号，由 `vendors/<v>/` 注入（[D25](DECISIONS.md#d25--冻结语义与输出不冻结厂商分类表)）。[T8.1](history-references.md#ref-5fcd4691137d) 已把这些白名单移入 `vendors/<v>/format.py`，经 `SourceFormat` 注入，CI 以字面量检索断言保证；格式指纹须覆盖全部注入钩子（[D27](DECISIONS.md#d27--格式解释的身份覆盖全部钩子兼容验收到连接键)） | 5 家厂商 |
| P7 | **观察（attach）与实验（seal）并存不合并**。两者共享全部探针与 schema，只在"输入与环境是否被封存校验"上不同 | D2 |
| P8 | **设备采集复用官方 profiler，不自研、不 fork**。工具负责会话（谁启停、采哪个窗口、产物是否有效）与采集之后的读取、关联、记账 | 5 厂商 × 版本矩阵；[D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合) |

## 2. 分层

```text
┌──────────────────────────────────────────────────────────────────────┐
│ L5 报告      report shell + panels（HTML / Markdown / 机器可读导出）      │ 07
├──────────────────────────────────────────────────────────────────────┤
│ L4 分析器    perf / route / layer / oracle / comm / memory / fault ...   │ 07, 08
│              声明 requires，经 store 公共只读接口读取，输出结论 + coverage │
╞════════ 冻结契约：事件 schema v1 + store 公共只读接口（D23） ════════╡ 03
│ L3 关联      corr 分配（在线，TLS）+ linker（离线，项目归属）            │ 04
│              + 原生关系（离线，导入时生成，D23）；span 树 / scope 栈      │ 06
├──────────────────────────────────────────────────────────────────────┤
│ L2 store     分片 JSONL + manifest；可追加、可尾读、崩溃可读             │ 03
│              + 派生视图：来源账本、原生关系、证据索引（D24）              │
├──────────────────────────────────────────────────────────────────────┤
│ L1 采集      官方 profiler 会话（复用，D22）/ vendor-trace 导入器         │ 05
│              探针：module / route / compile / comm / tensor / counter /   │
│              health / dispatch（按需）                                   │
├──────────────────────────────────────────────────────────────────────┤
│ L0 目标接入  attach（进程内 API）/ wrap（注入启动命令）/ launch（内置运行器）│ 02 §3
│              + 可选 seal（封存输入与身份）                               │ 06 §5
└──────────────────────────────────────────────────────────────────────┘
          厂商适配层 vendors/（横切 L0–L2）：官方 profiler 会话、trace 导入与来源账本、项目/原生关联规则、遥测、通信与编译身份   05 §4
```

层间只允许向下依赖。具体约束：

- L4/L5 **不得** import torch 或任何厂商 SDK（N3 离线分析）。
- L1 探针之间不得互相 import；共享状态只通过 L3 的 TLS corr 上下文。
- L0 的三种接入方式最终都调用同一个 `attach()`，区别只在谁负责启动目标进程。
- 导入到 `StoreView` 的路径上，除来源账本、原生关系外，还产出**格式适用性与支持范围评估**（[D28](DECISIONS.md#d28--格式适用范围显式化采集配置入身份导入判定适用性静默失败转显式)、[D29](DECISIONS.md#d29--判定一致性结构证据优先于名称契约条目须被执行或被测试)）。它与外层导入状态、分析能力是三件不同的事：`format_assessment()` 外层 `status` 说明导入事务是否完成，内层 `assessment.status`（契约内 / 契约外 / 无法判定）说明解释前提与支持范围是否成立，分析能力由各分析器的 `requires` 另判（[07 §1](07-analyzers-and-reports.md) 第 10 条）。

### L2 当前读取实现（2026-10-30）

Reader 默认 retain=True 保持快照；retain=False 的 stream 逐条交付普通在线事件，生成器耗尽后更新 report。提前关闭保留进度，同一 Reader 只允许一个活动 stream。T9.6 消除了登记画像下普通在线事件的全量常驻；随后 D33 将完整 B 链路的导入、finalize、默认 inspect 与设备分页/迭代接入有界路径，工作区显式准入实际 ext4。

T9.6 的 M1/M2 三次峰值均低于 128 MiB，见 [验收](history-references.md#ref-318933687dbe)，该数字绑定当时的读取画像。显式列表 API、原生关系整读、签名/控制状态及 compact 仍有随规模增长的部分。T10.2 已按 D48 为分析与报告交付有界读取、磁盘索引与流式导出，见 [P1 交接](history-references.md#ref-f4f11a91c40b)。T12.2 合并重复 prepare 工作，T12.3 合并校验遍历并通过 Store 外的身份绑定凭据复用成功校验；文件或校验器身份变化时重验，Reader 仍校验实际事件分片，见 [读取复用](history-references.md#ref-4921dede4be3)。P0 的退出结论沿用 [D45](DECISIONS.md#d45)。

## 3. 目标接入：attach / wrap / launch

| 方式 | 用法 | 机制 | 典型场景 |
|---|---|---|---|
| attach | `flagprof.attach(run, vendor, ...)` | 进程内创建 Writer、Context 与官方会话；P1 增加自动探针 | 用户推理脚本、Notebook |
| wrap | `flagprof record --store <dir> --vendor <id> -- python serve.py ...` | 默认 final 由目标包装入口调用 attach；显式 windowed 使用 sitecustomize/bootstrap。自动接入限 wrapper 直接启动的 Python 目标（D32 已确认） | 用户推理脚本；多进程接入方案在 P5 |
| runner | `flagprof record ... -- python -m flagprof.runner hf:<dir> --task embedding` | 已实现 HF embedding 运行器，在预热前登记根模型；独立 runner 支持无探针执行及组件开关 | 登记的单模型场景；generate 待接入，批量调度在 P5 |

> **当前实现（2026-10-30）**：默认 final 在关闭时导出，目标无需调用 `step()`；显式 windowed 保留。`record-config.json` 向 final 目标传入生效配置，runner 与普通脚本共用 attachment。module／dispatch／route／compile 以及 P4 tensor 采集已交付；`perf`／`perf-lite` 默认不启用逐调用 dispatch，精确项目调用连接可显式选择 `each-window`。具体默认值与参数见 [P1 使用说明](history-references.md#ref-0a9e456a7414) 和 [设备／显存／replay 使用说明](history-references.md#ref-5ef59ff0c610)。自动接入范围仍遵循 D32 的直接 Python 目标。
>
> 官方显存通过显式 `--profile-memory` 启用；分析侧经公共 StoreView 读取设备与显存证据。replay 通过冻结输入包产生新的标准 Store，再与独立性能 run 按精确签名连接。导入与查询使用 ext4；D52 的 A100 独立 overlay 采集例外须显式 `--capture-only --capture-filesystem overlay`，传输校验后仍在 ext4 消费。工程验收与 P2／P4 阶段状态见 [联合交付](history-references.md#ref-9596bcfb2a96)。

<details>
<summary>2026-10-27 P1 实施前的实现快照（原文）</summary>

> **当前实现（2026-10-27，P1 代码尚未实施）**——以代码与 `python3 -m flagprof record --help` 为准：
>
> - **命令**：`record --store <dir> --vendor <id> [--preset {perf,modules}] [--capture-mode {final,windowed}] -- <python 目标>`。默认 final；windowed 显式保留窗口参数。`perf-lite/coverage` 与探针参数由 T10.1 交付。
> - **接入范围**：D32 已于 D45 获用户确认。final 由 `offline.capture.capture_target` 在执行直接 Python 目标前调用 attach；windowed 的 bootstrap 用完整 proc_key 核对目标父进程。后代 worker 不自动采集，fork 继承的 Writer/Context 不可用；多进程由 P5 设计。
> - **配置传递**：windowed 经 `record-config.json`、`FLAGPROF_RECORD_CONFIG` 与 bootstrap。当前 final 单独传 vendor/preset，并清除该环境变量；T10.1 按 D48 补齐同源配置的显式传递与生效记录。
> - **进程内 API**：`attach(run, vendor, ..., capture_mode='final')` 返回管理 Writer、Context 和官方会话的句柄。final 在关闭后导出，不要求目标调用 `step()`；显式 windowed 仍按 step 推进。自动 module/dispatch/route/compile 探针目前列为 unavailable，待 T10.1 实现。
> - **导入与位置**：目标退出后由独立进程完成导入。完整 B 链路使用实际 ext4 的 capture/Store/work/scratch，默认有界读取；旧窗口路径仍显式可选。

</details>

**P1 接入机制（已实现）**：runner 与普通脚本共用 attachment。runner 在模型准备完成后显式 `register_model`；普通脚本由 module 调用入口惰性发现。根前向计数独立于 user scope，并固定在各线程/Task 的根帧上；组件启停与探针观察分开。具体契约与反例见 [T10.1 §3](history-references.md#ref-ccef0d040b8c)。登记的 HF embedding 已完成 P1 验收，批量模型与服务引擎接入按 P5 排期。

`flagprof` 为暂定包名，见 [DECISIONS](DECISIONS.md) Q1。

## 4. 两种运行模式：observe 与 seal

| | observe（默认） | seal（`--seal`） |
|---|---|---|
| 目的 | 广覆盖、看清楚 | 可复现、可比较 |
| 输入 | 目标自己的输入 | 封存的输入（文件 + 哈希），按 [06 §5](06-model-onboarding.md) 的 input recipe 生成 |
| 环境身份 | 记录但不校验 | 记录并校验；与对照 run 的身份差异必须只落在声明的实验变量上 |
| 适用接入方式 | attach / wrap / launch | launch（必须由工具控制输入）；wrap 仅当目标脚本接受工具提供的输入文件 |
| 可做的结论 | 性能画像、算子覆盖、回退清单、故障 | 以上 + FlagGems off/on 对照、A100 vs 国产芯片差分（R52/R86/R88） |

seal 不是第二套代码路径：它 = observe + input recipe + 身份校验 + 对照实验矩阵。

## 5. 一次 run 的数据流

```text
目标进程                                              分析机（可离线）
┌────────────────────────────────────┐
│ 用户/内置代码                        │
│   └─ aten 调用 ──► dispatch 探针 ──┐ │
│        │ (TLS corr)                │ │
│        └─► FlagGems ──► route 探针 ─┤ │
│              └─► Triton ─► compile ┤ │
│   module hook ───► module 探针 ────┤ │   events/rank-r/*.jsonl ──► compact ──► 分析器 ──► 报告
│   dist.* ─────────► comm 探针 ─────┼─┼─►  manifest.json        ▲
│   官方 profiler 会话 ─► 原始 trace ┼─┼──► vendor-trace 导入器 ─┘(离线：来源账本 + linker + 原生关系)
└────────────────────────────────────┘ │
旁路进程：counter 探针、health 心跳 ────┘
在线：supervisor 尾读 events ──► health 规则 ──► Responder（重试/隔离/告警）   08
```

要点：

1. Python 层探针（dispatch/route/compile/module/comm/tensor）**在线**写事件，并在写入时就带上 corr。
2. 设备侧事件来自官方 profiler 的原始 trace（会话由工具启停，[D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合)），**离线**由导入器转成 kernel/memop 事件。同一次导入同时产出两类关联，两者分立（[D23](DECISIONS.md#d23--原生关系作为一等证据与项目绑定分立)）：
   - linker 按 [04 §3.6](04-correlation.md#36-linker-算法候选并集与判定) 把设备事件归到**项目 corr**（有项目生产者时）；
   - 原生关系按 [04 §3.7](04-correlation.md#37-原生归属原生调用关系) 把设备事件归到**原生调用与作用域**（只要来源提供显式关系）。
   每条来源记录的去向记入来源账本（[D24](DECISIONS.md#d24--导入契约身份分层来源账本完成标记证据索引)）。
   同一次导入还产出格式适用性与支持范围评估，随派生摘要持久化，经 `StoreView.format_assessment()` 读取；它与导入事务状态、分析能力分立，`complete` 与 `within` 都不自动表示分析可用（[05 §4](05-probes-and-vendors.md#4-厂商契约)）。
3. counter 与 health 由独立旁路进程写入，不阻塞目标进程（继承 v1 monitor 的设计）。
4. perf 场景下，主机侧"哪个调用产生了哪个 kernel"优先取自官方 profiler 自带的算子记录（cpu_op）与原生 id；`dispatch(each)` 只在按需模式开启（D22）。没有项目探针时，性能诊断可以到**原生调用**一级（T8 实测：A100 新采集 9,260 条 compute 的原生 owner 为 100%）。**模块作用域**只有在 trace 中已经有作用域注释时才可得：同一份 A100 新采集没有这类注释，作用域覆盖为 0%。层级归属须由 P1 的 module 探针提供，或经评估后取自官方 profiler 自带的模块记录。2026-10-15 更正：原文写「性能诊断仍可到原生调用与模块作用域一级」，高估了无探针时的能力。

**SDK 产物生命周期（当前实现，T9，2026-10-18）**：官方 profiler 的产物经过三个所有权阶段，每一阶段只有一个所有者：

1. **working 目录（SDK 所有）**：官方 profiler 把当前窗口导出到这里。工具不假设它此后不变——SDK 在 `stop` 之后可能改写原件，也可能起分析子进程。
2. **sealed-window 快照（我们所有）**：窗口导出后，工具把 working 目录复制到同级的暂存目录，逐文件计算 SHA 并 fsync；复制前记录文件身份清单，复制后、校验后各核对一次（校验后同时核对 working 目录未被改写）；目录 fsync 后改名为 `sealed-<窗口名>`。窗口记录里 `path` 指向快照，`working_path` 保留 SDK 的原目录。复制失败或期间原件变化时，窗口**保持 `exporting`**，不得声明有效，也不登记快照。
3. **owner 退出后独立导入**：被测进程（SDK 所有者）退出后，由独立进程读取快照导入，不读 working 目录（`record` 流程为 目标退出 → 独立导入 → finalize）。

代价：sealed 副本使逻辑字节 **+100%**，复制/哈希/fsync/校验也计入窗口边界的停顿（构成见 [05 §4](05-probes-and-vendors.md#4-厂商契约) 的"生命周期义务"）。快照语义不依赖具体厂商；"sealed"在此指窗口快照，与 store 分片封口（[03 §6.4](03-event-schema.md#64-封口与尾读交接)）、实验 seal（[§4](#4-两种运行模式observe-与-seal)）是三件不同的事。

## 6. 目录布局

```text
inference_v2/
  docs/PLAN/          本设计文档集
  flagprof/           主包（暂定名）
    store/            schema.py  writer.py  reader.py  manifest.py  compact.py  view.py（公共只读接口，D23）
    correlate/        context.py（TLS corr/scope；当前实现：在 flagprof/context.py）  linker.py（项目归属）  native.py（原生关系，D23）  coverage.py
    probes/           dispatch.py module.py route.py compile.py comm.py tensor.py counter.py health.py
    vendors/          base.py  session.py  import_common.py  ascend/  nvidia/
                      （厂商适配：官方 profiler 会话、trace 导入器与来源账本、项目/原生关联规则、遥测）
    semantics/        profile.py（Tier-1 YAML）  plugin.py（Tier-2 接口）  inputs.py（input recipe）
    runner/           launch.py（内置运行器）  inject/sitecustomize.py（当前实现：flagprof/_bootstrap.py，由 record 的临时 sitecustomize 调用）  seal.py
    analyzers/        perf.py route.py layer.py oracle.py comm.py memory.py fault.py coverage.py
    report/           shell.py  panels/
    campaign/         matrix.py  index.py  scheduler.py
    online/           supervisor.py  rules.py  responder.py
    registry/         probes/ analyzers/ vendors/ components/   （descriptor JSON；设计目标，当前实现：flagprof/registry.py，BUILTINS + --plugins，见 05 §5）
    cli.py
  profiles/           <model>.yaml   Tier-1 模型声明
  plugins/            <model>/       Tier-2 插件
  spikes/             一次性验证代码（不进入主包），首个为 cann_correlation/
  tests/
```

依赖方向：`cli → runner/campaign/online → probes/vendors → correlate → store`；`report → analyzers → store`。`analyzers/`、`report/`、`store/` 三个子包必须在无 torch 环境中可导入。

## 7. 模型接入阶梯（概要）

| Tier | 投入 | 获得 | 详见 |
|---|---|---|---|
| 0 | 0 行 | 算子清单与签名、kernel 耗时、回退率、通信、显存、counter、按 module 树的逐层归因 | [06 §2](06-model-onboarding.md) |
| 1 | 约 20 行 YAML | 声明层边界、对比张量、有效区域掩码、输入 recipe → 逐层精度差分、首个发散层 | [06 §3](06-model-onboarding.md) |
| 2 | Python 插件 | 自定义生成循环、自定义并行、特殊输入编码 | [06 §4](06-model-onboarding.md) |

## 8. 与需求的对应

| 需求 | 架构中的承接点 |
|---|---|
| R53 Profiling 助手 | L1 官方 profiler 会话 + vendor-trace + route/compile（dispatch 按需）+ L3 关联（项目 + 原生）+ L4 perf/layer 分析器 |
| R52 Oracle–Device 差分 | L0 seal + L1 tensor 探针 + L4 oracle 分析器（run ⨝ run） |
| R31 可归因指标与回退率 | L1 route/comm + L3 关联 + L4 route 分析器的导出清单 |
| R32 故障检测与恢复 | L1 health + L2 可尾读 + online/supervisor + Responder |
| R33 热加载与长稳 | lifecycle span + 长稳分析器 + store 轮转 |
| 5 厂商 | 厂商适配层，契约见 [05 §4](05-probes-and-vendors.md) |
| 500 模型 | Tier 0 零配置 + launch 运行器 + campaign |

## 9. 已知的架构风险

| 风险 | 影响 | 应对 |
|---|---|---|
| CANN 上算子级 host→device 关联可能不可行（尤其 FlagGems 的 Triton kernel） | R53 的"关联 Backend"降级为 layer 级 | 已做 [Spike-C](11-spike-cann-correlation.md)，结论决定 [04](04-correlation.md) 与 [03](03-event-schema.md) 的冻结内容。**2026-10-15 更新**：已验证场景下可行——历史 eager 样本与 T8 新采集（T8.2 之后）两种主线设备都能逐调用原生归属。FlagGems Triton kernel 的项目归属仍待 P1/P2 |
| 厂商格式或采集配置变化让解释静默失效（T8 分类、T8.1 默认值、T8.2 主机身份三次出现） | 数据进来了，某个结论静默为 0，或设备分母 R 自己缩小，导入仍是 `complete` | 格式指纹覆盖全部钩子（[D27](DECISIONS.md#d27--格式解释的身份覆盖全部钩子兼容验收到连接键)）；采集配置显式化、格式契约与导入适用性判定、轨道角色审计（[D28](DECISIONS.md#d28--格式适用范围显式化采集配置入身份导入判定适用性静默失败转显式)，T8.3）；与关联可行性分开看待 |
| Python dispatch 探针开销大（生成式模型每 token 数千次 aten 调用） | 性能数字失真 | perf 场景改用官方 profiler 自带的算子记录，`dispatch(each)` 按需开启（[D22](DECISIONS.md#d22--采集层复用官方-profiler不做源码级融合)）；探针分级（`summary`/`each`）；开销实测写入报告，见 [05 §2.1](05-probes-and-vendors.md) |
| 官方 profiler 自身开销不可忽略（2026-10 实验：同步前向 +36% 到 +116%） | 诊断采集的计时不代表无 profiler 时的性能 | 性能基准 run 与诊断采集分开；诊断按窗口采集；开销按采集配置实测登记，不跨 workload 外推（[D14](DECISIONS.md#d14--扰动按指标类分流不是整份-run-的布尔标签)、D22） |
| 图模式执行（CUDA Graph / ACL Graph / torch.compile）绕过 Python dispatch | 图内算子不可见 | 显式标记 `graph_replay` 区间；首版不承诺图内算子级归因，见 [04 §5](04-correlation.md) |
| attach 与 seal 的需求冲突导致代码分叉 | 维护成本翻倍 | P7 原则：seal = observe + 输入封存 + 身份校验，不允许出现 seal 专用探针 |
| 500 × 5 × N 的 run 组合爆炸 | 结果无法管理 | campaign 索引从 P5 开始建，但 run_id 与 manifest 在 P0 就按可索引设计 |
