# Training v2 验证与已知限制

日期：2026-11-11（2026-11-12 增补 CTR 训练、embedding/reranker 训练任务与 P800 探针收官）。起点为 `zhiyu/model-adaptation@3417e54d`，开发分支 `zhiyu/perf-train`。验证目录包含多轮独立运行，不覆盖失败、不合并计数冒充一次验收。

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
| PPU 单卡 CTR 训练（14 架构） | 通过 | 密封 `ctr-package-v1` 语料 + 派生标签；全部 14 模板架构 `--task ctr` 端到端训练完成（返回码全零）；逐架构 loss 见 [v2.md](v2.md) |
| PPU 单卡 CTR 训练五 preset + AMP | 通过 | dlrm 基线 3 步累积 2；`perf-lite/coverage/modules/perf/accuracy` 观测后与无采集基线零容差逐位一致（573 行对比 pass）；显式 BF16 AMP 独立运行完成，不与 FP32 基线对比 |
| CTR 训练与推理权重互挂 | 相等 | 训练协议 `ctr_weight_sha256` = 推理 `oracle_identity.weight_sha256` = `4b20358e…a90`（同种子同构建器）；两台 PPU 均复验 |
| PPU 双机 CTR 训练 A/A | 通过 | thead1 与 thead2 独立部署同一提交分别全量套件；跨主机零容差对比 pass（573 行），`final_model_sha256` 双机同为 `63b7af75…61f9` |
| PPU embedding 微调（Qwen3-Embedding-0.6B） | 通过 | 确定性合成三元组组语料（32 组×3 负例，seed 531）上 `--task embedding`，last/cls 两种 pooling 各 3 步 AdamW（累积 2）端到端完成（返回码全零）；InfoNCE loss 区 FP32、温度 0.05 进协议 |
| PPU reranker 微调（Qwen3-Reranker-0.6B / bge-reranker-base） | 通过 | 复用推理侧密封 rerank 包，相关性标签由 sample id 派生；`causal_yesno`（AutoModelForCausalLM，末有效位置 [no,yes] 两列 CE）与 `sequence_logit`（AutoModelForSequenceClassification，BCEWithLogits）两种语义各 2 步完成；协议记录包 oracle/inputs/contract/template 与 `rerank_weight_sha256` |
| PPU embedding 官方采集观测与对比 | 通过 | perf-lite 采集 embedding 训练后，与无采集基线零容差对比 pass（4654 行） |
| PPU 逐算子微基准（诊断） | 完成 | `op_micro.py` 14 算子 @ B8 S512 H1024 vocab151669 bf16，中位数 50 iters；产出 `op-micro-ppu.json`；诊断脚本，不入采集协议 |
| P800 两台机器空闲卡健康探针 | 未通过健康前置 | 固定 FlagRT v2 镜像内最小 PyTorch 探针在 torch import 后、首个设备内核前停滞，硬上限后 rc=137；四轮独立实验覆盖两台机器共 6/6 长空闲物理卡同型挂死（楔死卡可被其他租户唤醒跑满，非永久损坏），证据链闭环，定性为厂商级阻塞。没有进入本训练框架，不能声明 P800 训练适配完成 |

所有结果保持 `not_qualified`。这是功能与观察透明性验证，不是完整模型训练收敛认证或独占性能资格。

## 环境与证据索引

| 平台 | 身份和环境 | 证据目录内容 |
|---|---|---|
| PPU | PPU-ZW810E；物理 0/1 → 容器逻辑 0/1；驱动 1.5.1-1d747a；FlagRT 镜像 ID `1ba7ead15378`；torch 2.10 / SDK 2.1.0 / Python 3.12 | `health-ppu1.*`、`suite-v3/`、`suite-v5/`、`final-suite/`、`ppu-ddp/`、CPU/Gloo 与回归输出 |
| PPU（CTR，双机） | thead1 物理 0 与 thead2 物理 2，同一镜像与身份文件；证据根 `perf-train-evidence-20261112` | `ctr-suite-v1/`（两台同构：baseline、14 架构、五 preset record/compare/inspect/report、AMP、推理 oracle 权重互挂、跨主机 comparison、`ctr-regression-20261112/` 回归输出） |
| PPU（embedding/rerank） | thead1 物理 0（thead2 复验见下行），同一镜像；证据根同上 | `model-tasks-v1/`（语料生成、last/cls embedding、两种 rerank 语义、perf-lite record/compare、`op-micro-ppu.json`、`regression-20261112/` 权威回归输出） |
| P800 | 同一已实测镜像 ID `cac5ddae5f62`，torch 2.9.0+cu129；klx1 minor 6/0/3、klx2 minor 0/2/1/3/6 四轮探针 | 各轮 `probe*` stdout/stderr/returncode、pre/postflight、独立小脚本与容器生命周期记录 |

证据归档根使用 `perf-train-evidence-20261111` 与 `perf-train-evidence-20261112` 名称。它是本次实验资产，不是 Git 仓库内自动存在的路径；机器地址、账户、凭据不写入公开文档。公开源码提供最小复现命令；需要复核原始大体积 Store/模型快照时从项目维护者取得对应归档并核对 SHA-256。不能把没有入库的目录引用当作公开下载承诺。

## 回归

| 回归 | 结果 | 说明 |
|---|---|---|
| 既有 Inference 标准库 smoke（python -S） | 160 tests，65 skipped，OK（2026-11-12，PPU 容器内 Linux） | 在无 torch/SDK 导入条件运行；新增 16 项 embedding/rerank 训练任务测试中 12 项依赖 torch 在此条件下跳过、4 项纯契约测试通过，跳过项不计通过 |
| 训练语义 + 训练分析 + CTR/embedding/rerank 任务 + 原精度/replay 定向回归 | 53 tests，OK（2026-11-12，含 10 项 CTR、16 项 embedding/rerank 任务测试） | 命令为 `python3 -B -m unittest tests.test_training tests.test_training_analysis tests.test_training_ctr tests.test_training_embedding tests.test_training_rerank tests.probes.test_accuracy tests.probes.test_replay -v`（仓库 `inference/` 目录）；不与上行或模型设备实测重复相加 |
| 原 TP2 embedding CPU 用例 | 环境失败保留 | PPU 镜像无设备映射时，Transformers TP 加载尝试 CUDA 初始化并报 driver shutting down；在完全未修改的 `3417e54d` 归档上同样复现。不是本次引入，亦不改测试掩盖 |

## 保留的问题与范围

| 编号 | 事实 | 下一步 / 判定边界 |
|---|---|---|
| TRAIN-P800-001 | 两台 P800 机器 6/6 长空闲物理卡在固定镜像内首内核前同型挂死（四轮探针证据链闭环）；挂死卡可被其他租户唤醒跑满，排除永久损坏 | 厂商级阻塞，交厂商定位；恢复可用设备窗口后按同镜像复测；不重置共享驱动、不清理他人任务；该平台仍待验证 |
| TRAIN-ENV-001 | 首轮官方采集因 scratch 位于容器 overlay 被 ext4 门禁拒绝 | 失败保留，指定 ext4 scratch 后通过；使用手册已明确 |
| TRAIN-ENV-002 | 上述既有 TP2 embedding 无设备 CPU 测试在 PPU 镜像失败 | 精确起点基线已复现；需要独立处理环境选择，不能称全部旧回归全绿 |
| 观察范围 | forward module 树不能直接当作 backward 层归属；Python collective 不包含全部 DDP C++ reducer 调用 | 训练显式范围内逐调用覆盖已补，跨线程不自动传递；设备关联缺失照常 partial/unavailable |
| 模型范围 | 内置分类功能示例、HF causal LM、密封 CTR 语料模板训练（14 架构，派生标签 + 逐架构 loss）、embedding 微调（InfoNCE，三种 pooling）与 reranker 微调（两种打分语义）；不含正式数据集收敛、FSDP/ZeRO/TP/PP | CTR/embedding 语料为确定性合成（不是真实点击或检索数据），rerank 标签由 sample id 派生；验证微调通路与观测闭环，不声明任务精度；新模型需明确输入、标签、loss、优化器和训练协议再单独验收 |
| TRAIN-MODEL-001 | Qwen3-Reranker-8B 全参数微调超出单卡显存：FP32 参数 + AdamW 一二阶矩 + 梯度约 128 GB > 96 GB（PPU-ZW810E）；DDP 每 rank 复制全量不解决分片 | 暂不可在本框架训练，如实记录；FSDP/ZeRO 与 bf16 参数协议属范围外/协议变更，需用户明确授权后再评；同构通路已由 Qwen3-Reranker-0.6B（causal_yesno）实测覆盖 |
| TRAIN-MODEL-002 | OneRec-1.7B 训练适配暂缓：模型权重未在 PPU 双机供给（thead1 全盘检索无 OneRec 目录），推理侧 runner 标注 inference-only；生成式推荐端到端训练目标（多任务/强化学习）超出当前标准训练任务定义 | 待权重供给与官方训练协议明确后再评；不把推理模板裸加 `.train()` 冒充训练适配 |
| TRAIN-MODEL-003 | 搜广推 15 模型清单中两个多模态模型（openai/clip-vit-large-patch14、Qwen3-VL-Embedding-8B）按本轮范围约定排除 | 不在本轮训练适配范围；其余 13 个目标去向：9 个 CTR 架构（DIN/DeepFM/DCN-Mix/ESMM/DLRM/DSSM/LightGCN/SASRec/PLE）由 `--task ctr` 覆盖，Qwen3-Embedding-0.6B 与 bge-m3 由 `--task embedding` 覆盖（bge-m3 见下行实测状态），Qwen3-Reranker-8B 见 TRAIN-MODEL-001（0.6B 同构通路已实测），OneRec-1.7B 见 TRAIN-MODEL-002 |
| 性能范围 | accuracy 快照开销、首次编译/初始化、共享宿主 | 不自动给独占资格，不把诊断时长当成优化后性能，不合计 rank-local 吞吐 |

新增代码版权署名为 **Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)**；已有上游文件保留原版权声明。
