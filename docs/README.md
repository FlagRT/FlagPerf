# FlagPerf 文档导航

| 使用目的 | 入口 |
|---|---|
| Ascend 环境、支持范围与公开验证记录 | [Ascend 适配指南](ascend/README.md) |
| 基础算力、内存、传输与厂商诊断 | [Base 使用指南](../base/README.md) |
| 算子正确性、路由、性能与失败诊断 | [Operation 使用指南](../operation/README.md) |
| 单机双 rank 通信资格验证 | [P2P 协议](ascend/p2p.md) |
| 适配维护、离线回归与发布检查 | [验证与维护](ascend/validation.md) |
| 正式功能与文档更新记录 | [变更记录](CHANGELOG.md) |

原有框架资料继续按领域维护：[Base](base/base-case-doc.md)、
[Operation](operations/operations-case-doc.md)、[Training](training/specifications/standard-case-spec.md)、
[Inference](inference/inference-case-doc.md)、[Generate](generate/generate-case-doc.md)。
旧集群说明仅适用于明确选择 SSH 入口的使用者。当前单机入口不读取 `host.yaml` 或建立 SSH 连接。

使用命令集中在各领域指南；环境和支持声明集中在 Ascend 指南；协议阈值以代码及配置为准。
变更行为、默认参数或验证范围时，同时更新对应指南和变更记录，避免复制多套相互冲突的命令表。

P800 single-card identity, telemetry and cleanup: [bounded preflight](../base/docs/p800-preflight.md).
