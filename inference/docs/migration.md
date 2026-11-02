# 从 v1 迁入 flagprof v2

本次用 `FlagPerf_advance/inference_v2/flagprof` 的生产实现替换 `FlagPerf/inference` 的旧模型评测框架。框架文档 00–12 迁入本目录，同时保留其依赖的 `DECISIONS.md` 与设计历史索引。

## 使用方式变化

| 原入口 / 资产 | 当前方式 |
|---|---|
| `run.py list/preview/accuracy/performance/export`、`run_inference.py` | `run.py` 或 `python -m flagprof` 的 `record/launch/import/inspect/page/report/replay` |
| `config/*.yaml`、四模型 registry、旧 `ModelAdapter` | 被测 Python 程序或 `flagprof.runner`；采集配置由 CLI / attach 显式传入 |
| 宿主入口自动启动模型容器 | 在准备好的设备运行环境中执行采集；离线消费使用标准库环境 |
| 旧模型/参考包与结果协议 | 使用对应历史版本读取；新版本围绕 capture、Store、身份和关联证据组织 |
| 旧监控和热加载说明 | 当前已实现能力见 [模块首页](../README.md)，后续在线能力见 [08](08-online-and-fault.md) |

旧命令与旧结果没有自动转换。旧目录中的用户资产继续由其所有者管理，历史路径不批量改写。新运行使用新输出路径，封存证据保留原身份。

## 来源和迁移边界

来源是 2026-11-02 冻结的本地工作区：`FlagPerf_advance` 基础 commit 为 `df452e18ac60cba8182126a776aa615723cdae1a`，同时纳入冻结时尚未提交的生产更新。生产模块、厂商 JSON 契约与登记资源按快照复制；公开目录增加了统一脚本入口、安装元数据和协作说明。

本次没有合并 `FlagPerf_advance` 的 Git 历史。它继续作为本地开发与验证仓库；正式协作代码在 `FlagPerf`。测试、验收工具、实验原件、模型权重、镜像和运行产物留在各自原位置。来源清单和逐文件摘要在本地迁移记录中保存。

00–12 正文保留设计与历史语境，仅补充迁移阅读说明并调整链接。指向本次已迁入文件的链接在公开目录内解析；其余历史引用进入 [出处索引](history-references.md)，索引记录原路径和引用页。未来协作使用 [文档导航](README.md) 中的目录和分支约定。

## 恢复旧版

迁移前正式仓库提交为 `f1786cab7562b0be4eb2850d7affcbbec23ba316`。在包含该历史的仓库根目录，可建立独立的旧版工作区：

```bash
git worktree add --detach ../FlagPerf-inference-v1 f1786cab7562b0be4eb2850d7affcbbec23ba316
```

旧版模型评测、配置、预览和结果说明均可在那里读取。若为浅克隆，先获取该提交。运行仍需对应版本的模型、输入及设备环境。

更早的 SSH / 编译引擎框架保留于 `3e7c558b6f56e6ea5f9c8b318852d97d1517a14c`，可按相同方式建立独立历史工作区。
