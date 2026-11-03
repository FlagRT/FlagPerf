# 从 v1 迁入 flagprof v2

2026-11-02 起，`FlagPerf/inference` 由 flagprof v2 的生产实现替换旧模型评测框架。框架文档、其依赖的 `DECISIONS.md` 与设计历史一同迁入本目录。

## 使用方式变化

| 原入口 / 资产 | 当前方式 |
|---|---|
| `run.py list/preview/accuracy/performance/export`、`run_inference.py` | `run.py` 或 `python -m flagprof` 的 `record/launch/import/inspect/page/report/replay` |
| `config/*.yaml`、四模型 registry、旧 `ModelAdapter` | 被测 Python 程序或 `flagprof.runner`；采集配置由 CLI / attach 显式传入 |
| 宿主入口自动启动模型容器 | 在准备好的设备运行环境中执行采集；离线消费使用标准库环境 |
| 旧模型/参考包与结果协议 | 使用对应历史版本读取；新版本围绕 capture、Store、身份和关联证据组织 |
| 旧监控和热加载说明 | 当前已实现能力见 [模块首页](../README.md)，后续在线能力见 [08](08-online-and-fault.md) |

旧命令与旧结果没有自动转换。旧目录中的用户资产继续由其所有者管理，历史路径不批量改写。新运行使用新输出路径，封存证据保留原身份。

## 未实现项

v2 按 [10 路线图 §0.1](10-roadmap.md#01-v1-能力清单p5-之前的排序依据) 的 v1 能力清单推进，D5 的退役条件不变。截至 2026-11-02，以下 v1 能力在本目录中尚未提供或只部分提供；需要时从 [旧版](#恢复旧版) 运行：

| v1 能力 | 本目录现状 | 归属 |
|---|---|---|
| DLRM、DeepFM 推荐模型（v1 用合成权重与输入） | 无内置 runner 与登记验证 | 路线图 §0.1 序 7，P2b 之后 |
| FlagTree 组件开关与编译器对照 | 无；`--flaggems` 为唯一组件开关，FlagCX 作为 `launch --comm-backend` 的通信后端提供 | 序 8 |
| preview 分组搜索与联合策略校验 | 部分：一次 run 的 `route.coverage` 给出回退清单，无二次搜索 | 序 6 |
| `export` FX / ONNX 图导出 | 无 | 序 9 |
| A100 TP2 性能基线 | 只有匹配的单卡基线；无同规模 A100 TP2 对照 | 序 4–5 |
| TP 通信 padding 与链路量 | 不可用，报告保留原因 | 序 5 |
| 独立设备监控（v1 默认开启的 dcmi 遥测与 worker CPU/RSS） | 无；counter / health 旁路归入 P3 | [08 在线与故障](08-online-and-fault.md) |
| v1 参考包读写与 `repack-reference` | 不读取 v1 参考包；数值参照改为 accuracy run 与 `report oracle` | 不迁移 |

## 来源和迁移边界

`flagprof/` 与私有开发仓库 `FlagPerf_advance` 提交 `ac23227c14a76a578449c216118a360496ad513f` 中的 `inference_v2/flagprof/` 逐字节一致（含厂商 JSON 契约与登记资源）。公开目录另外增加统一脚本入口、安装元数据、CPU 冒烟测试和协作说明。

本目录是公开的生产代码，按本仓库规范单独进行版本管理；`FlagPerf_advance/inference_v2` 是维护者私有的开发与测试目录，保留完整回归套件、验收工具和实验原件，不合并其 Git 历史。两边的生产代码通过 PR 同步，同步后在本节更新来源提交。模型权重、镜像和运行产物留在各自原位置。

框架文档保留设计与历史语境。2026-11-03 移除 `docs/history/` 目录；正文中的历史引用改为删除前固定 Git 提交中的链接，[出处索引](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md)保留原路径和引用页。协作方式见 [文档导航](README.md#两人如何协作)。

## 恢复旧版

迁移前正式仓库提交为 `f1786cab7562b0be4eb2850d7affcbbec23ba316`。在包含该历史的仓库根目录，可建立独立的旧版工作区：

```bash
git worktree add --detach ../FlagPerf-inference-v1 f1786cab7562b0be4eb2850d7affcbbec23ba316
```

旧版模型评测、配置、预览和结果说明均可在那里读取。若为浅克隆，先获取该提交。运行仍需对应版本的模型、输入及设备环境。

更早的 SSH / 编译引擎框架保留于 `3e7c558b6f56e6ea5f9c8b318852d97d1517a14c`，可按相同方式建立独立历史工作区。
