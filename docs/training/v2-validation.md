# Training v2 验证与已知限制

日期：2026-11-11。起点为 `zhiyu/model-adaptation@3417e54d`，开发分支 `zhiyu/perf-train`。验证目录包含多轮独立运行，不覆盖失败、不合并计数冒充一次验收。

## 实测状态

| 项目 | 结果 | 解释 |
|---|---|---|
| CPU 标准训练、累积、AdamW 与 Dropout 恢复 | 通过 | 与无观察标准 PyTorch 更新对照；连续/恢复的参数与优化器指纹一致 |
| CPU 两进程 Gloo DDP | 通过 | 两 rank 完成、最终模型一致；不作为硬件通信性能 |
| PPU 单卡五类 preset | 通过 | `perf-lite/coverage/accuracy/modules/perf` 均执行三 optimizer steps，每步累积两 microbatches；与无采集 loss、梯度、参数和优化器状态逐位一致 |
| PPU 官方内存与报告 | 通过 | perf + profile-memory 的 memory 面板为 ok；五种 preset 的训练阶段面板为 ok；inspect --verify 与无 torch/SDK 报告通过 |
| PPU 显式 BF16 / FP16 AMP | 通过 | 分类功能训练完成，FP32 参数、标准 autocast/GradScaler；不是自动降精度补救，不据此声明所有模型支持 |
| PPU 单卡检查点恢复 | 通过 | 中断在 optimizer 边界，恢复后最终参数/优化器与连续运行逐位一致 |
| PPU 两卡 NCCL 兼容 DDP | 通过 | 无采集、perf-lite、perf 分别完成；各次两 rank 参数相同，逐 rank 与无采集基线的训练状态逐位一致 |
| PPU Qwen3-0.6B | 通过功能验证 | 原本地权重、FP32/eager、固定 token 输入、SGD lr=1e-4、两步标准 causal LM；无采集/官方采集 loss、最终模型和优化器指纹一致，HTML 报告成功 |
| P800 klx2 物理卡 0、klx1 物理卡 7 | 未通过健康前置 | 固定 FlagRT v2 镜像内，在 torch import 后、设备模型搬运前停滞，硬上限后 rc=137；没有进入本训练框架。不能声明 P800 训练适配完成 |

所有结果保持 `not_qualified`。这是功能与观察透明性验证，不是完整模型训练收敛认证或独占性能资格。

## 环境与证据索引

| 平台 | 身份和环境 | 证据目录内容 |
|---|---|---|
| PPU | PPU-ZW810E；物理 0/1 → 容器逻辑 0/1；驱动 1.5.1-1d747a；FlagRT 镜像 ID `1ba7ead15378`；torch 2.10 / SDK 2.1.0 / Python 3.12 | `health-ppu1.*`、`suite-v3/`、`suite-v5/`、`final-suite/`、`ppu-ddp/`、CPU/Gloo 与回归输出 |
| P800 | 同一已实测镜像 ID `cac5ddae5f62`，torch 2.9.0+cu129；klx1 物理 7 为 minor 6，klx2 物理 0 为 minor 0 | `health.stdout/stderr/returncode` 或 `health-xpu0.*`、pre/postflight、独立小脚本 |

证据归档根使用 `perf-train-evidence-20261111` 名称。它是本次实验资产，不是 Git 仓库内自动存在的路径；机器地址、账户、凭据不写入公开文档。公开源码提供最小复现命令；需要复核原始大体积 Store/模型快照时从项目维护者取得对应归档并核对 SHA-256。不能把没有入库的目录引用当作公开下载承诺。

## 回归

| 回归 | 结果 | 说明 |
|---|---|---|
| 既有 Inference 标准库 smoke（python -S） | 129 tests，40 skipped，OK | 在无 torch/SDK 导入条件运行；包含新增离线训练阶段负例，跳过项不计通过 |
| 训练语义 + 训练分析 + 原精度/replay 定向回归 | 27 tests，OK（2026-11-11 最终定向回归） | 不与上行或模型设备实测重复相加 |
| 原 TP2 embedding CPU 用例 | 环境失败保留 | PPU 镜像无设备映射时，Transformers TP 加载尝试 CUDA 初始化并报 driver shutting down；在完全未修改的 `3417e54d` 归档上同样复现。不是本次引入，亦不改测试掩盖 |

## 保留的问题与范围

| 编号 | 事实 | 下一步 / 判定边界 |
|---|---|---|
| TRAIN-P800-001 | 两台 P800 机器的最小标准 PyTorch 健康探针未完成 | 先恢复可用设备窗口并复测同镜像；不重置共享驱动、不清理他人任务；该平台仍待验证 |
| TRAIN-ENV-001 | 首轮官方采集因 scratch 位于容器 overlay 被 ext4 门禁拒绝 | 失败保留，指定 ext4 scratch 后通过；使用手册已明确 |
| TRAIN-ENV-002 | 上述既有 TP2 embedding 无设备 CPU 测试在 PPU 镜像失败 | 精确起点基线已复现；需要独立处理环境选择，不能称全部旧回归全绿 |
| 观察范围 | forward module 树不能直接当作 backward 层归属；Python collective 不包含全部 DDP C++ reducer 调用 | 训练显式范围内逐调用覆盖已补，跨线程不自动传递；设备关联缺失照常 partial/unavailable |
| 模型范围 | 内置分类功能示例和 HF causal LM；不含正式数据集收敛、CTR 完整训练辅助目标、FSDP/ZeRO/TP/PP | 新模型需明确输入、标签、loss、优化器和训练协议再单独验收 |
| 性能范围 | accuracy 快照开销、首次编译/初始化、共享宿主 | 不自动给独占资格，不把诊断时长当成优化后性能，不合计 rank-local 吞吐 |

新增代码版权署名为 **Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)**；已有上游文件保留原版权声明。
