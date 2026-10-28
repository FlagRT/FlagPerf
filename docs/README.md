# FlagPerf 文档导航

| 使用目的 | 入口 |
|---|---|
| Ascend 环境、支持范围与公开验证记录 | [Ascend 适配指南](ascend/README.md) |
| 昆仑芯 P800 环境、Base/Toolkit/Operation 范围与验证 | [P800 适配指南](kunlunxin/README.md) |
| P800 算子最小运行、完整矩阵与已知问题 | [运行入口](../operation/vendors/kunlunxin/README.md)、[验证与限制](kunlunxin/operation-validation.md) |
| 平头哥 PPU 环境、支持范围与验证记录 | [PPU 适配指南](thead/README.md) |
| PPU 算子最小运行、运行栈准备与验收边界 | [运行入口](../operation/vendors/thead/README.md)、[机制与验收](thead/operation.md)、[失败与归因边界](thead/validation.md#九个失败的对照实验与归因边界) |
| 基础算力、内存、传输与厂商诊断 | [Base 使用指南](../base/README.md) |
| 算子正确性、路由、性能与失败诊断 | [Operation 使用指南](../operation/README.md) |
| 模型输出差异、A100 参考、性能与监控 | [Inference 使用指南](../inference/README.md) |
| Ascend 单机双 rank 通信资格验证 | [P2P 协议](ascend/p2p.md) |
| 适配维护、离线回归与发布检查 | [Ascend 验证](ascend/validation.md)、[P800 验证](kunlunxin/validation.md)、[PPU 验证](thead/validation.md) |
| 正式功能与文档更新记录 | [变更记录](CHANGELOG.md) |

适配和扩展资料按领域维护：[Base](base/base-case-doc.md)、
[Operation](operations/operations-case-doc.md)、[Training](training/specifications/standard-case-spec.md)、
[Inference](inference/inference-case-doc.md)、[Generate](generate/generate-case-doc.md)。
旧集群说明仅适用于保留该入口的领域；旧 Inference 通过 [历史版本](inference/migration.md) 恢复。当前单机入口不读取 `host.yaml` 或建立 SSH 连接。

使用命令集中在各领域指南；环境和支持声明集中在各厂商指南；协议阈值以代码及配置为准。
变更行为、默认参数或验证范围时，同时更新对应指南和变更记录，避免复制多套相互冲突的命令表。
