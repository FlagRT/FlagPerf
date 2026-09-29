# Ascend 适配指南

Base / Operation 当前适配采用 CANN 9.0.0、Python 3.11.15、PyTorch 2.10、Torch-FL、triton_ascend 和 FlagGems，
Toolkit 使用 MindCluster ToolBox 26.1.0。目标硬件验证记录来自 Ascend 910C / A3。
Torch-FL 在此组合中拥有 `PrivateUse1`，设备名为 `flagos:<index>`；不能向这套锁定镜像混装 `torch_npu`。

Inference 使用独立的 torch_npu 运行栈，配置方式见 [Inference 环境准备](../inference/environment.md)。
不要将两个模块的镜像要求混用。

## 入口和支持范围

所有示例均从 FlagPerf 仓库根目录执行。

| 能力 | 推荐入口 | 当前边界 |
|---|---|---|
| Base Benchmark | `python3 base/run.py benchmark run ...` | 保留 Case、配置合并、rank、计时和结果公式；一次调用选择一个 Case |
| Ascend Toolkit | `python3 base/run.py toolkit run ...` | 单机厂商测量与诊断；独立权限、测量、监控和诊断状态 |
| Operation | `python3 operation/run.py run --vendor ascend ...` | 52 个 Case 已接入；每个 dtype/路径的通过状态单独报告 |
| Inference | `python3 inference/run.py accuracy / performance ...` | Qwen3-Embedding-0.6B；同设备配置的模型/模块差异及整体性能；环境单独准备 |
| P2P 资格验证 | `python3 base/vendors/ascend/torch_fl_2.10_flagcx/run_p2p_qualification.py` | 单机双 rank、指定拓扑/方向/大小；默认只输出计划 |

各模块的支持范围独立；不由 Base/Operation 记录推定 Training、Generate、Inference 或其他厂商的运行结果。
Operation 是单算子输入上的 CPU 参考检查及路由取证，没有实现 NVIDIA—Ascend 模型逐层差分、
自动重试、节点隔离或模型热更新。Operation 的独立 profiling 仅报告可关联的目标窗口，不代表全部硬件 kernel 覆盖。

## 环境准备

| 用途 | 镜像 tag | 本次核对的本地镜像 ID |
|---|---|---|
| Base 常规 Case、Toolkit、Operation | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64` | `sha256:d948410966b0dfdfaf4f9c95b9b14bdb7c4279cb85ccd7f9157f0fb5d1c6d397` |
| Base P2P 通信 | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | `sha256:3b9e08f231d0e80d37341e375e6bb13066e03a67794b78a3d2874c1770c6b6bc` |

两者是已经在维护环境中构建并核对的镜像身份。它们当前没有可核对的 `RepoDigests`，
本项目没有据此承诺匿名 `docker pull` 可用。公开代码与可公开获取的运行镜像是两项独立交付。
运行器会比对实际镜像 ID，不能仅把新镜像标记为同一个 tag 就沿用旧验证结论。

源码、依赖 pin、Dockerfile 和检查脚本分别位于
[operator runtime](../../base/vendors/ascend/torch_fl_2.10/) 与
[communication runtime](../../base/vendors/ascend/torch_fl_2.10_flagcx/)。
构建 operator 镜像需要锁文件指定的上游基础镜像访问权限，当前底座来自需授权的 registry；
还需获取锁定的 Torch-FL、FlagGems 源码和依赖制品。不要把代码开放误读成底座制品已获分发许可。

构建脚本接受 `TORCH_FL_SOURCE_REPO`、`FLAG_GEMS_SOURCE_REPO`，通信镜像接受
`FLAGCX_SOURCE_REPO`、`FLAGCX_JSON_SOURCE_REPO`，均指向操作者准备的本地源码目录。
它们仍核对固定 commit、patch 和父镜像身份，不会自动更新到其他版本。
自行重建得到不同镜像 ID 时，应作为新 runtime 验证并记录，不能直接复制旧 `validated=true`。

宿主需有 Docker、可用驱动、`npu-smi` 和 `fuser`。Base 配置是 JSON 格式的 YAML 子集：
复制 [本机配置](../../base/configs/ascend910_cann9_local.yaml) 为自行管理的配置文件，按实际宿主填写
`expected_device_ids`、挂载和 `toolbox_host_path`，通过 `--config` 使用。ToolBox 默认路径为
`/usr/local/Ascend/toolbox`；自定义配置应放在版本库外。示例设备编号必须替换成已预约、空闲的设备，
物理 NPU 与逻辑 Device 映射以实时 `npu-smi info -m` 为准。

## 最小静态检查

```bash
python3 base/run.py benchmark run --case computation-FP16 --npu-ids 7 --dry-run
python3 base/run.py toolkit run --case computation-FP16 --npu-ids 7 --dry-run
python3 operation/run.py list
python3 operation/run.py run --vendor ascend --device-ids 14 --case abs --dry-run
```

这些命令不查询设备、不启动 Docker。真正运行的权限、资源治理及参数见
[Base 指南](../../base/README.md) 和 [Operation 指南](../../operation/README.md)。
短时配置尽量降低日常占用；两分钟是每个组合的目标，不是全套测试耗时保证。
容量/OOM Case 仍需要显式高风险授权；P2P 资格矩阵另有约 30 分钟纯测量预算。

## 如何阅读验证记录

P2P 的资格摘要来自既有单机实验，正式名为 `p2p-single-node-v1`，其工作量、方向、阈值及
镜像身份没有因命名整理而变化。原始 P2P 日志当前不在分发树中，随代码提供的是有来源哈希的
维护者记录，不能把它称为本次重新实测或第三方独立复现。详细口径见 [P2P 协议](p2p.md)。

Operation 的[公开覆盖摘要](operation-coverage.json)记录 2026-09-09 的历史结果：
308 个适用组合中 277 passed、24 blocked、5 failed、2 partial，另有 420 个不适用项。
两个来源运行及其封存文件已在整理时核对 SHA-256；原始日志、输入张量及机器身份未随代码分发。
这是跨批次汇总，不能声称集成后源码重新跑过全部组合。失败解释见
[Ascend Operation 支持说明](../../operation/vendors/ascend/README.md)。

Base 短时配置、Toolkit 监控及其他 dtype 的验证范围不得从一次 FP16 或一次 P2P 成功外推。
`passed`、`partial`、`blocked`、`failed` 与 `not-run` 表示不同事实，正式命名不改变状态门禁。
本次离线集成验证及复验方法见 [验证与维护](validation.md)。
