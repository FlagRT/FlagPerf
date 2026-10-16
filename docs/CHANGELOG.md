# 适配与文档变更记录


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

## 2026-10-15 — 平头哥 PPU Base 集成（待发布）

- 接入平头哥 PPU（ZW810E）Base runtime：非特权有界容器、显式物理选卡、UUID 绑定、逐卡监控与清理；统一 `run.py` 入口与 Ascend/昆仑芯共享资格审计。
- Base 覆盖 21 个 case×变体条目：18 项可执行（6 精度计算、H2D/D2H×锁页/非锁页×基线/跨 NUMA、带宽/容量、2–16 rank AllReduce 与恰好 2 rank P2P），3 项启动前跳过（FP8 算子路由失败、两个跨机 Case）。
- INT8 经 `torch._int_mm` K-major 前置进入调优内核；行主序回退证据保留在受控结果根并建议上游跟进，不作为对外数字。AllReduce 单迭代用时对 rank 数非单调，高 rank 资格须在各自窗口内重校 ITERS。
- 锁定镜像保持 `release_stage:candidate / validated:false`，每轮 run 显式 `--allow-candidate-runtime`；16 rank 资格与 validated 复评列为维护者后续动作，不自动晋级。
- 补齐中文使用与验证文档及全部 15 个 case 目录的芯片 README；站点信息不入公开文档，原始测量保留在受控结果根，不随源码分发。

本次文档整理与离线检查没有重跑硬件；历史测量、稳定性与项目例外见
[PPU 验证说明](thead/validation.md)。

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
