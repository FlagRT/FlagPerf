# 适配与文档变更记录


## 2026-10-10 — 昆仑芯 P800 Base 与 Toolkit 集成（待发布）

- 接入 P800 candidate runtime、显式物理选卡、PCI/UUID 绑定、监控和受限容器清理。
- Base 覆盖 10 个可执行 Case 与 5 个提前跳过 Case；支持单卡计算/传输/内存、2–8 rank AllReduce 和双 rank P2P。
- Toolkit 增加 12 个独立 XBLAS/XRE 测试，保留载荷、计时、正确性、监控与诊断合同。
- D2D 总览分为设备内存读写带宽与 runtime memcpy 拷贝吞吐两行，保持原摘要格式。
- 共享配置按 generic、vendor、chip、override 合并；旧 Ascend 入口、租约和历史报告读取继续兼容。
- 补齐中文使用与验证文档，公开模板移除站点路径，声明宿主 PyYAML 依赖；开发记录与原始证据不随源码分发。

runtime 保持 candidate / validated:false；历史测量、稳定性和项目例外见
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
