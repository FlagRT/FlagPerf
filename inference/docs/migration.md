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

## v1 能力对照

v2 按 [10 路线图 §0.1](10-roadmap.md#01-v1-能力清单p5-之前的排序依据) 的 v1 能力清单推进，D5 的退役条件不变。下表更新至本次成熟功能增量；仍未迁入的能力可从 [旧版](#恢复旧版) 运行：

| v1 能力 | 本目录现状 | 归属 |
|---|---|---|
| DLRM、DeepFM 推荐模型（v1 用合成权重与输入） | 无内置 runner 与登记验证 | 路线图 §0.1 序 7，P2b 之后 |
| FlagTree 组件开关与编译器对照 | 2026-11-10 已同步登记范围的组件选择、精度与性能链路；见 [组件对比](components.md) | 序 8 |
| preview 分组搜索与联合策略校验 | 部分：一次 run 的 `route.coverage` 给出回退清单，无二次搜索 | 序 6 |
| `export` FX / ONNX 图导出 | 无 | 序 9 |
| A100 TP2 性能基线 | 只有匹配的单卡基线；无同规模 A100 TP2 对照 | 序 4–5 |
| TP 通信 padding 与链路量 | 不可用，报告保留原因 | 序 5 |
| 独立设备监控（v1 默认开启的 dcmi 遥测与 worker CPU/RSS） | 无；counter / health 旁路归入 P3 | [08 在线与故障](08-online-and-fault.md) |
| v1 参考包读写与 `repack-reference` | 不读取 v1 参考包；数值参照改为 accuracy run 与 `report oracle` | 不迁移 |

## 来源和迁移边界

首次迁入时，`flagprof/` 与私有开发仓库 `FlagPerf_advance` 提交 `ac23227c14a76a578449c216118a360496ad513f` 中的 `inference_v2/flagprof/` 逐字节一致。此后公开目录独立增加 P800、平头哥适配，以及输入身份、窗口时钟、OneRec TP、进程生命周期等修复；当前版本由这些公开改动和下述成熟功能增量共同组成。

## 2026-11-10 成熟功能增量

以公开提交 `e47be5ff56b7f1f4d72da585b46562225e0e26ed` 为基础，按功能迁入开发提交 `748fb9a405ce6dc5f947f18feb2d284806c6cfb2`，并单独补齐后续的 `launch --preset coverage`。保留公开仓库的厂商适配和修复，开发仓库历史继续独立管理。

- 精度：launch 的 model/layer/JSON 边界与算子下钻、Qwen embedding TP2 runner、按同 rank 配对的组 Oracle，以及字段级身份差异声明。
- 组件：record/launch/runner 的 FlagTree 选择、launch 的 FlagGems 参数与 Ascend FlagCX 快捷入口、实际编译器/JIT/通信实现身份。
- 正确性：collective 未完成边界不采快照并失效旧来源、replay 编译器及 rank 检查、有效配置重复解析，以及显式组件失败时停止目标。
- 性能：按实际根前向（含预热）统计 Qwen 通信，组内设备通信比较排除成员编号，保留负载与窗口身份检查。

Store schema 保持 1.2。monitor/watch、telemetry、资源与联合时间线分析及其 schema 1.3 扩展不在本次公开同步中。历史精度的来源 partial、性能格式 outside/partial 继续保留；本次迁移不增加设备覆盖或加速比结论。

验证采用公共 CPU 回归、固定镜像 CPU torch/Gloo 测试和已归档 Store 报告复验；具体结果见 [验证记录](validation-components-20261110.md)。登记设备数据为 Qwen 六组精度对比、四条件性能链路及 OneRec 32-token 回归。

本目录是公开的生产代码，按本仓库规范单独进行版本管理；`FlagPerf_advance/inference_v2` 是维护者私有的开发与测试目录，保留完整回归套件、验收工具和实验原件，不合并其 Git 历史。两边的生产代码通过 PR 同步，同步后在本节更新来源提交。模型权重、镜像和运行产物留在各自原位置。

框架文档保留设计与历史语境。2026-11-03 移除 `docs/history/` 目录；正文中的历史引用改为删除前固定 Git 提交中的链接，[出处索引](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md)保留原路径和引用页。协作方式见 [文档导航](README.md#两人如何协作)。

## 恢复旧版

迁移前正式仓库提交为 `f1786cab7562b0be4eb2850d7affcbbec23ba316`。在包含该历史的仓库根目录，可建立独立的旧版工作区：

```bash
git worktree add --detach ../FlagPerf-inference-v1 f1786cab7562b0be4eb2850d7affcbbec23ba316
```

旧版模型评测、配置、预览和结果说明均可在那里读取。若为浅克隆，先获取该提交。运行仍需对应版本的模型、输入及设备环境。

更早的 SSH / 编译引擎框架保留于 `3e7c558b6f56e6ea5f9c8b318852d97d1517a14c`，可按相同方式建立独立历史工作区。
