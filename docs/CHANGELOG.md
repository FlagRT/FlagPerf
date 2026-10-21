# 适配与文档变更记录

## 2026-10-21 — P800 Operation 完整矩阵与文档复核（未发布）

- Docker/local 各完成 728 项规划中的 308 个适用组合尝试；保留数值、编译和运行失败，干扰与独立补测分列，不宣称原 Docker 整轮持续独占。
- 主 README、Operation 入门及厂商导航同步最新覆盖、最小命令和报告位置，区分 Base 发布范围与 Operation 的 `validated:false`。
- 登记公共精确比较统计缺口 `OP-CHECK-001`，保留间歇 trace 限制与六组计数器 not-supported；不改写程序判定或历史证据。
- 说明仓库 Case 扩展与任意外部自定义 kernel 接入的区别；不增加新能力或宣称完成硬件验收。

## 2026-10-20 — P800 Operation 文档分层与责任说明（未发布）

- 厂商 README 保留运行入口和必要边界，机制、覆盖和历史失败迁入独立文档；保留失败与源码身份，不改写 evidence。
- 更正最新 div FP16 与历史 tanh FP16 缺事件记录的批次归属，说明计数器未接入不等于硬件不支持。
- 明确原生/FlagGems 错误的证据与归因边界，local 代表验证为七 case/14 组合；独占全量待资源和框架问题处理后再执行，本轮不安排 Ascend 实机回归。

## 2026-10-20 — Operation 各芯片最小运行说明（未发布）

- 入门指南补齐 Ascend、P800、NVIDIA 及四个旧厂商 local 兼容入口的预览/实际执行命令，明确镜像、解释器、设备预约与支持边界。
- 说明 P800 专用 runtime profile 与 workload/profiling 的区别，以及省略 profile 会选择旧昆仑芯入口。
- 对齐 Ascend/P800 厂商指南的最小原生案例，说明两者特权参数差异；不修改运行行为、已有证据或 validated 状态。

## 2026-10-15 — P800 Operation 公开契约与交付边界（未发布）

- 公开分发测试增加 P800 专用 profile 的 Docker/local 文档示例、全类型 728/308/420 分母和多设备/尺寸规划，纳入 vendor README 的链接与私有路径检查。
- 现有 CI 增加 P800 Operation 标准库门禁与负例测试；不要求设备、Docker 或 torch，不更改 Ascend 命令。
- 明确评估覆盖与全通过的区别；原生路由、共享性能、profiling 和已知算子失败仍按实际范围保留，Operation 保持 `validated:false`。

## 2026-10-14 — P800 Operation Day 08/09 评估与离线交付核对（未发布）

- 完成 52 case、308 个适用组合的 Docker daily 尝试，保留 275 partial、27 failed、6 blocked 及 420 个类型不适用项。
- 完成 7 个 local 代表 case、14 个双路径 smoke；local 结果独立列出，不与 Docker daily 分母合并。
- 按 Day 06/07/08 的三个源码身份分别重建离线报告，核对 728 个唯一组合、源 evidence 哈希、复放命令和确定性；Operation 继续保持 `validated:false`。
- 保留底层 kernel/编译/数值失败及两次 Docker 查询异常；不将评估覆盖写成 runtime 晋级，不重新解释 Base/Toolkit 历史资格。


## 未发布：模型分析与算子诊断更新

- 图导出的输入准备与执行均使用 CPU，宿主容器不再要求设备/驱动挂载；FX 重载核对原生输出一致。
- 配置 YAML 错误、推荐资产缺失及离线命令不适用参数提供明确提示；补齐 A100 虚拟环境调用与固定参考输入使用说明。
- Inference 新增 DeepFM、DLRM、OneRec-8B，采用模型、厂商、组件与注册描述分层；保留合成资产准备工具。
- 接入 A100 精度与 single/TP2 性能参考、schema 4 核心/附件、跨设备比较；NVIDIA 通用入口收敛为参考生产。
- 性能默认开启独立管理面监控；成功无异常的普通任务默认精简原始证据，支持显式完整保留。
- 新增 CPU 离线 analyze、历史参考转换、成本预检、分片明细和受监督 CUDA trace 分析。
- 补充四模型、参考包、监控、离线恢复与版本迁移的公开使用说明。

- 新增单卡/TP 逐层性能采集：独立基准、模块计时、显存、设备事件与路由证据。
- 新增 `report` 性能视图、层/rank/shape 筛选、portable 证据包和有界 profiler 重新分析；兼容旧报告重建入口。
- Preview 默认采用轻量证据、分组搜索和动态预算，续探可复制通过摘要校验的缓存组；执行升级需重新 preview。
- 补齐逐层使用、Preview 和指标来源指南；公开配置继续由使用者提供镜像、模型和编译器资产。

- Inference 提供 Qwen3-Embedding-0.6B 的模型/模块精度、单卡与 TP 整体性能、独立通信采样、三组件比较和图导出。
- Preview 使用 schema 3 策略，支持检查点续探与执行/分析身份区分；环境由用户配置，厂商编译器资产按实际来源封存。
- 整理模型、执行、分析和报告目录；新增面向首次使用的说明，旧 Inference 从工作树移除，提供历史恢复方式。
- Operation 增加独立显存与 profiling、双路径性能比较、公共诊断与显式 replay、进度及报告更新；保留 `--profile` 兼容别名。
- Base 的 ToolBox 路径改为通用示例，使用者通过自己的配置指定本机位置。
- Generate 原有设备监控脚本移入自身 `monitoring/`，消除对旧 Inference 目录的依赖，保持监控行为。

## 2026-10-10 — 昆仑芯 P800 Base 与 Toolkit 集成（待发布）

- 接入 P800 runtime、显式物理选卡、PCI/UUID 绑定、监控和受限容器清理。
- Base 覆盖 10 个可执行 Case 与 5 个提前跳过 Case；支持单卡计算/传输/内存、2–8 rank AllReduce 和双 rank P2P。
- Toolkit 增加 12 个独立 XBLAS/XRE 测试，保留载荷、计时、正确性、监控与诊断合同。
- D2D 总览分为设备内存读写带宽与 runtime memcpy 拷贝吞吐两行，保持原摘要格式。
- 共享配置按 generic、vendor、chip、override 合并；旧 Ascend 入口、租约和历史报告读取继续兼容。
- 补齐中文使用与验证文档，公开模板移除站点路径，声明宿主 PyYAML 依赖；开发记录与原始证据不随源码分发。

锁定 runtime 经维护者确认设为 validated:true / partial，绑定明确验证范围与摘要哈希；
当前版本无需 candidate opt-in，历史失败和单次资格状态保留。历史测量、稳定性和项目例外见
[P800 验证说明](kunlunxin/validation.md)。本次文档整理与离线检查没有重跑硬件。

## 2026-09-12 — Ascend Base 与 Operation 集成（待发布）

- 同步 Base 短时默认配置、容量测试结束等待配置及 `npu-smi` 进程占用检查；保留设备 lease 和独立状态。
- 同步单机双 rank FlagCX P2P 资格矩阵、校准、故障检查及失败会话恢复，正式协议命名为 `p2p-single-node-v1`。
- 将校准依赖和资格记录整理为可分发摘要；正式计划不依赖私有结果目录，绑定配置、范围与镜像身份。
- 接入 Operation 通用 CLI、52 个 Case、CPU 参考、路由取证、独立计时、失败诊断及多批次报告。
- 同步 add/sub/rsub 和 triu 工作量线性计数修正；保留原 Case 算子和其他公式，反向三倍仍为估算。
- 旧 SSH Operation 入口迁入 `operation/legacy/`；旧 schema 结果保留读取兼容，不提供重复的单算子执行入口。
- 补充公开文档导航、Ascend 环境与支持范围、P2P 协议、验证方法和公开覆盖摘要；清理本机路径与内部文档链接。
- 新增无需设备和私有镜像的公开分发检查，完整回归仍在锁定运行镜像执行。

此次更新没有重跑 NPU。历史实机记录与本次离线验证分别说明；镜像分发、完整硬件复验和远端 CI 状态
不由代码同步自动完成。详见 [验证与维护](ascend/validation.md)。

## 2026-09-02 — Ascend Base 首次集成

加入 CANN 9 / Torch-FL Base 适配、统一 Benchmark/Toolkit 入口、运行身份锁、设备选择与租约、
独立报告及测量窗监控。当前支持范围以本次更新后的使用指南和 runtime 记录为准。
