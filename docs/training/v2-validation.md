# Training v2 验证与已知限制

日期：2026-11-11（2026-11-12 增补 CTR 训练、embedding/reranker 训练任务与 P800 探针收官；同日晚间增补 FSDP 分片训练、注册表 --task 缺省解析与 causal-lm 通路；2026-11-19 增补三平台全量矩阵：P800 训练与观察态全量、Ascend 910C 全量、通信画像报表实测、FlagScale 真实集成与算子级画像，证据随 2026-11-17/18 三平台交付材料归档；2026-11-23 增补合并 PR #15 后的节点监控与 rank 级结果报告实车）。起点为 `zhiyu/model-adaptation@3417e54d`，开发分支 `zhiyu/perf-train`。验证目录包含多轮独立运行，不覆盖失败、不合并计数冒充一次验收。

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
| PPU embedding 微调（Qwen3-Embedding-0.6B、bge-m3） | 通过 | 确定性合成三元组组语料（32 组×3 负例，seed 531）上 `--task embedding`，last/cls 两种 pooling 各 3 步 AdamW（累积 2）端到端完成（返回码全零）；Qwen3 走 AutoModel 回退 CausalLM 主干，bge-m3（XLMRoberta）走标准 `AutoModel`（权重经 sha256 对账的 modelscope 镜像补齐）；InfoNCE loss 区 FP32、温度 0.05 进协议 |
| PPU reranker 微调（Qwen3-Reranker-0.6B / bge-reranker-base） | 通过 | 复用推理侧密封 rerank 包，相关性标签由 sample id 派生；`causal_yesno`（AutoModelForCausalLM，末有效位置 [no,yes] 两列 CE）与 `sequence_logit`（AutoModelForSequenceClassification，BCEWithLogits）两种语义各 2 步完成；协议记录包 oracle/inputs/contract/template 与 `rerank_weight_sha256` |
| PPU embedding 官方采集观测与对比 | 通过 | perf-lite 采集 embedding 训练后，与无采集基线零容差对比 pass（4654 行） |
| PPU 双机 embedding/rerank A/A | 通过 | thead2 独立复跑同款套件（8 项返回码全零，模型与密封包 sha256 中转核对）；跨主机 JSON 级逐字段对照一致：`final_model_sha256` 双机同 `e2dbfd7b…cd4b`、`final_optimizer_sha256` 同 `149b5d11…7e39`、两 rerank `rerank_weight_sha256` 同（`73991b03…`/`b2b6114f…`）、逐步 loss 逐位相同；0.6B 模型逐步快照为 GB 级，不做跨机全量搬移，对照粒度如实记录为结果 JSON |
| PPU 两卡 FSDP 分片（classification） | 通过 | thead1/thead2 各自独立双跑逐位可复现；final optimizer 指纹四跑双机逐位同（`47393cfa…`）；final model 按 rank 分片留证、各 rank 双机逐位同（脚本形态 rank-0 `0643d70f…`/rank-1 `33093750…`，CLI 形态 rank-0 `bbe255a5…`/rank-1 `f3b2667d…`）；summary 按 `sharding`/`check_mode` 识别分片并输出 complete |
| PPU 两卡 FSDP vs DDP（0.6B rerank） | 数值逐位一致 | 复用密封 rerank 包，FSDP 与 DDP 对照的逐步 loss 轨迹逐位同（rank-0 `[0.0005846787244081497, 49.59575653076172]`）；最终模型指纹差异为 rank 局部 shard 与全量副本的语义差，非数值分歧；双机结果一致 |
| PPU FSDP 扩展路径（thead2 fsdp-ext） | 通过 | performance 模式与 accuracy 数值逐位同且按预期不含 step 快照；FSDP+累积 2 与 DDP+累积 2 逐步 loss 逐位一致；4 卡 FSDP classification summary complete（无重复块 root-only 包装属预期）；compare 对协议不匹配与形状不匹配均诚实拒绝，`--declare` 不可绕过 |
| PPU 两卡 FSDP 分片（Qwen3-Reranker-8B 全参，performance 模式） | 通过（rc=0，2 步完成） | FP32 训练态约 128 GB 分片进双 96 GB 卡（每 rank 约 64 GB），torchrun 2 rank；protocol `sharding=fsdp`、`fsdp_wrapping=per-block+root`；五分片权重全部通过 modelscope API sha256 关卡（含 shard4 双写事故修复后复验）；rank-0 逐步 loss [0.0124926, 0] |
| PPU 单卡 causal-lm（OneRec-1.7B，注册表 --task 缺省 + 缺省语料自动生成） | 通过（双机双跑 rc=0） | thead1/thead2 各 explicit 与 registry 两形态共四跑：final_model_sha256 四方一致 467145ae260d9598…、逐步 loss 四方逐位 [13.533725, 21.052334]；protocol task_source explicit/registry 对比落证；权重 4263793656 字节 sha256 cabc217fad6129ab… 与 modelscope API 权威值逐字同，索引文件 sha 双机同 |
| PPU 逐算子微基准（诊断） | 完成 | `op_micro.py` 14 算子 @ B8 S512 H1024 vocab151669 bf16，中位数 50 iters；产出 `op-micro-ppu.json`；诊断脚本，不入采集协议 |
| P800 训练全量（klx1） | 通过（套件 28/28） | CTR 14 架构全绿、DDP 2/4/8 卡（含 100 步长跑，稳态 1816 samples/s）、embedding（last/cls、bge-m3）、rerank 两语义、FSDP 0.6B r2–r4、8B r2–r4（16 卡 47 s）、checkpoint/resume（单卡+4 卡）、AMP bf16、租户共存稳定性（rendezvous 换端口） |
| P800 观察态全链（klx1） | 通过 | 零配置观察态（`FLAGPROF_TRAIN_OBSERVE`）全链打通，与无采集基线零漂移对照 MATCH；store 114 MB、trace 192k 事件、comm 计数 171/87 与 store 交叉一致；设备内核事件缺失为厂商 kineto 限制（见下表） |
| Ascend 910C 训练全量（ascend1） | 通过 | CTR 13/14（dien：`DynamicGRUV2` 仅 FP16，FP32 不支持，厂商算子规格）；DDP 8 卡与 16 卡（各 20 步）；FSDP 0.6B 2 卡；8B FSDP 4 卡；embedding（last/cls、bge-m3）、rerank 两语义；checkpoint/resume（补 npu RNG 分支后）；稳定性 10/10 轮（5×DDP16 100 步、3×DDP8 150 步、2×8B FSDP r4）；AMP 未单测（套件外） |
| Ascend 910C 观察态全链 | 通过 | vendor 按设备自动识别（npu → 内置 ascend session，torch_npu.profiler）；store 88 MB、manifest complete |
| 通信画像报表（模型层 §5.5 四条，PPU rerank06b 双卡） | 完成 | 通信与计算分列：每卡 172 comm（allgather 114 次/4.14 GB + reducescatter 58 次/4.77 GB）双 rank 对称；按 rank 归集：逐 rank 分键；等待与带宽：wait 中位 5.1–523.7 ns（到达偏斜），带宽按 payload/传输段派生、厂商分解栈逐行标注不作线上值；同并行度：data 相 2.363–2.379 s 占 step 墙钟 38.5–39.8%（world_size 入档）。P800 因无设备内核事件 wait/带宽不可得、910C 按行为级论证，均如实标注 |
| FlagScale 真实集成（train_retrieval.py，PPU 双卡） | 通过 | bge-m3、DDP 2 卡：train_loss 0.116918 / validation 0.000107；store 1.6 GB manifest complete；comm=`{model/allreduce: 2}`（评测归约；DDP 梯度 Reducer 按设计不可见）、phases data 28 / forward 27 / backward 27 / optimizer 27 / step 27；开/关采集两跑 `model.safetensors` sha256 逐字节相同（`4b346bb7…`）。工件见 `inference/tools/flagscale_integration/`，指南见 [FlagScale使用FlagPerf最佳实践](FlagScale使用FlagPerf最佳实践.md) |
| 8B FSDP2 外部循环演示（适配器） | 通过 | 手写 FSDP2 循环接 `ObservationSession`：双 rank 每步 all-gather/reduce-scatter 逐调用入账（合计 allgather 730 / reducescatter 370）；开/关两跑 399/399 张量值一致（DTensor 文件字节不同为分片元数据差异，按值对比） |
| 算子级画像（封存 trace 聚合，`kernel_profile.py`） | 完成 | FlagScale 运行：183,647 内核 / 104 种 / 通信占比 52.0%（pccl AllReduce RING 243 次、twoShot 351 次、elementwise add 21,120 次等）；8B FSDP2：87,478 内核 / 69 种 / 45.5%（AllGather 730 / ReduceScatter 370，与通信事件计数吻合） |
| 训练观察态节点监控与 rank 级结果报告（PPU DDP2 FSDP，2026-11-23） | 通过（双车 rc=0） | 带 `FLAGPROF_TRAIN_MONITOR=on`：store 根单监控会话（rank0 属主，`monitor-ref.json` + `monitor/` + `targets/` 逐 rank 注册），每 rank run 目录内 relocate 出旁路引用，跑完自动渲染 `<run>-report/`（report.html/report.md/result.json）；训练面板 status ok（2 步、11 区间全 complete、data/zero_grad/forward/backward/optimizer 计数齐全），perf.device/perf.layer/perf.timeline/resource/memory/communication 等推理级面板同套可用，报告标题按训练负载呈现；不带监控的对照车 store 无任何 monitor 引用、报告照常生成；拓扑/诊断状态与设备身份降级（thead `physical_identity` 缺失记 unknown）逐字落 attachment state |
| 训练观察态节点监控与在线诊断（Ascend 910C DDP2 FSDP rerank-0.6B，2026-11-23） | 通过（双车 rc=0） | vendor=ascend 走 **agent 拓扑**：rank0 自起独立 `flagprof watch` 监督进程（会话 `monitor-session/1`，vendor=ascend，结束 status=stopped——自起进程回收验证），双 rank 注册为 target（history 2 条），DCMI 遥测真实采样（aggregates 含 `host.memory.*` 与 `chip:N` 逐分钟聚合），诊断引擎产出 2 条 confirmed finding（`monitor.gap` 覆盖缺口，schema `monitor-finding/1`）；rank store 内 DiagnosticProbe 发出 788 条 `health/heartbeat` 事件 + 258 条 comm + 124 条训练 span；attachment state 记 `monitor_topology=agent`、`monitor_diagnose=on`、设备身份 `chip:0 phyid 0 available`（ascend vendor 有 `physical_identity`）；报告 training ok（3 步）、`perf.timeline ok`、面板全套；无监控对照车无会话痕迹、报告照常 |
| 官方流水并行引擎（torch.distributed.pipelining） | 未验证（厂商栈挂死） | 官方 PP 引擎在 PPU 带/不带观察两配方均第一步挂死（NCCL RAS 转储留档），P800 同型；不绕过、不改写，如实记录为引擎级 PP 未验证；DDP/FSDP2 通路不受影响 |
| P800 两台机器空闲卡健康探针 | 未通过健康前置 | 固定 FlagRT v2 镜像内最小 PyTorch 探针在 torch import 后、首个设备内核前停滞，硬上限后 rc=137；四轮独立实验覆盖两台机器共 6/6 长空闲物理卡同型挂死（楔死卡可被其他租户唤醒跑满，非永久损坏），证据链闭环，定性为厂商级阻塞。没有进入本训练框架，不能声明 P800 训练适配完成（2026-11-17 更新：klx1 正常卡窗口内 P800 训练与观察态全量已通过，见新增行；空闲卡同型挂死仍交厂商，本行保留为当时记录） |

所有结果保持 `not_qualified`。这是功能与观察透明性验证，不是完整模型训练收敛认证或独占性能资格。

## 环境与证据索引

| 平台 | 身份和环境 | 证据目录内容 |
|---|---|---|
| PPU | PPU-ZW810E；物理 0/1 → 容器逻辑 0/1；驱动 1.5.1-1d747a；FlagRT 镜像 ID `1ba7ead15378`；torch 2.10 / SDK 2.1.0 / Python 3.12 | `health-ppu1.*`、`suite-v3/`、`suite-v5/`、`final-suite/`、`ppu-ddp/`、CPU/Gloo 与回归输出 |
| PPU（CTR，双机） | thead1 物理 0 与 thead2 物理 2，同一镜像与身份文件；证据根 `perf-train-evidence-20261112` | `ctr-suite-v1/`（两台同构：baseline、14 架构、五 preset record/compare/inspect/report、AMP、推理 oracle 权重互挂、跨主机 comparison、`ctr-regression-20261112/` 回归输出） |
| PPU（FSDP 分片，双机） | thead1 物理 0/1 与 thead2 物理 2/3，同一镜像与身份文件；证据根同上 | thead1：`fsdp-ppu/`、`regression-fsdp/`（权威回归）、`fsdp-8b/`、`fsdp-onerec/`；thead2：`fsdp-ppu/`、`fsdp-ppu-b/`（跨主机 A/A 对照）、`fsdp-ext/`（performance 模式、FSDP+累积、4 卡 FSDP、compare 诚实性）；同一改动含 summary.py 分片感知修复与 `task_source` 协议字段 |
| PPU（embedding/rerank，双机） | thead1 物理 0 与 thead2 物理 2，同一镜像与身份文件；证据根同上 | `model-tasks-v1/`（两台同构：语料生成、last/cls embedding、两种 rerank 语义、perf-lite record/compare、`op-micro-ppu.json`）；thead1 另含 `regression-20261112/` 权威回归输出与 `bgem3-v1/`（bge-m3 cls/last 补测） |
| P800 | 同一已实测镜像 ID `cac5ddae5f62`，torch 2.9.0+cu129；klx1 minor 6/0/3、klx2 minor 0/2/1/3/6 四轮探针 | 各轮 `probe*` stdout/stderr/returncode、pre/postflight、独立小脚本与容器生命周期记录 |
| P800（训练与观察态全量，2026-11-17） | klx1，同一镜像 `cac5ddae5f62`；优先空闲卡 | 训练套件逐项输出、观察态 store/trace、稳定性长跑与共存记录（随三平台交付材料归档） |
| Ascend 910C | Atlas 800T A3（8 卡 × 每卡 2 chip）；ascend1 / ascend2；CANN 9.x（宿主版本挂 `/usr/local/Ascend/cann-9.0.0` 别名）；flagrt `ascend-operator-runtime` torch2.10 arm64 镜像 ID `62bcaae3f839` | 回归与训练全量、观察态 store、稳定性 10 轮、`op-micro-ascend910c.json`（随三平台交付材料归档） |

证据归档根使用 `perf-train-evidence-20261111` 与 `perf-train-evidence-20261112` 名称。它是本次实验资产，不是 Git 仓库内自动存在的路径；机器地址、账户、凭据不写入公开文档。公开源码提供最小复现命令；需要复核原始大体积 Store/模型快照时从项目维护者取得对应归档并核对 SHA-256。不能把没有入库的目录引用当作公开下载承诺。

## 回归

| 回归 | 结果 | 说明 |
|---|---|---|
| 既有 Inference 标准库 smoke（python -S） | 197 tests，89 skipped，OK（2026-11-13 合并 inference 侧评审提交后的权威回归 regression-fsdp5，PPU 容器内 python -S；torch 守卫用例按既有模式环境性跳过） | 在无 torch/SDK 导入条件运行；含 CTR/embedding/rerank/模型注册表新测试中不依赖 torch 的项，其余在此条件下跳过，跳过项不计通过 |
| 训练语义 + 训练分析 + CTR/embedding/rerank/注册表/分片任务 + 原精度/replay 定向回归 | 79 tests，OK（2026-11-13 最终权威回归 regression-fsdp5，逐模块实测：training 17、analysis 5、ctr 10、embedding 8、rerank 8、registry 13、sharding 9、accuracy 8、replay 1；含 summary 分片识别/步骤覆盖校验、Roberta 式包装发现、task_source CLI 缺省与缺省语料目录拼接组合断言） | 命令为 `python3 -B -m unittest tests.test_training tests.test_training_analysis tests.test_training_ctr tests.test_training_embedding tests.test_training_rerank tests.test_training_registry tests.probes.test_accuracy tests.probes.test_replay -v`（仓库 `inference/` 目录）；不与上行或模型设备实测重复相加 |
| 原 TP2 embedding CPU 用例 | 环境失败保留 | PPU 镜像无设备映射时，Transformers TP 加载尝试 CUDA 初始化并报 driver shutting down；在完全未修改的 `3417e54d` 归档上同样复现。不是本次引入，亦不改测试掩盖 |
| 全量回归（三平台实机，2026-11-17） | 230 passed / 0 failed × 4（thead1、thead2、klx1、ascend1；33–54 s/机） | 同一提交在 PPU/P800/910C 三平台一致通过；910C 于 flagrt 容器内运行 |
| P800 夜间全量口径（更宽，236 项） | 232 passed / 3 failed / 1 skipped | 3 项失败经 thead 同树对照定性为既有全量口径测试债（非 P800 平台差异、非本分支引入），如实保留未修；与上一行 230 项子集口径分别记录，不合并计数 |
| 合并 dev-1.0（PR #15）后全量（PPU 容器，2026-11-23 @ `9b635461`） | 377 passed / 12 failed / 1 skipped | 12 项失败为 PR #15 自带 monitor/final-lifecycle/components 测试，在**未修改的 dev-1.0 `cc85edc8`** 同容器逐名复现同一批失败（上游环境性，非合并/训练侧引入）；训练观察监控/报告新测试（tests/test_training_observe.py 4 项）全绿 |
| 合并后全量（Ascend 910C 容器，2026-11-23 @ `cddb2302`） | 376 passed / 11 failed / 3 skipped | 失败集为同一 PR #15 继承家族的 monitor/final-lifecycle 子集（PPU 12 项中的 11 项；TP2 components 项在本平台计入 skip），训练线与新增观察监控测试全绿；运行于租户共存的 flagrt 镜像（--privileged 重跑） |
| 合并后全量（P800 新机 klx1 容器，2026-11-24 @ `8a9a9565` 树） | 377 passed / 15 failed / 3 skipped | 15 = 同一 PR #15 继承家族（monitor 5 + final_lifecycle 10 含 5 个 subtest），训练线与新增观察监控测试全绿；三平台失败集同族、均非训练侧引入 |
| 训练观察态节点监控与 rank 级结果报告（P800 新机 DDP2 FSDP rerank-0.6B，2026-11-24） | 通过（三车：监控车/CTR 冒烟/守卫车） | P800 vendor=thead → local 拓扑：单监控会话 + 双 rank target 注册 + 每 rank `<run>-report/`（训练面板 ok、3 步全 ok、`perf.timeline ok`、11 面板同套）；`monitor_topology=local`/`monitor_diagnose=off` 如实落 state，`register_device` 记 unknown（thead vendor 无 `physical_identity`，与 thead 平台一致）；对照车零监控痕迹、报告照常；CTR 密封包单卡车 3/3 complete，`ctr_config/inputs/weight_sha256` 与 `oracle_id` 逐字记录——**同时验证了旧机 528 G 迁移数据的字节完整性**。DDP bootstrap 超时改为 `FLAGPROF_TRAIN_DIST_TIMEOUT` 可调（默认 120 不变） |
| 新机空闲卡首内核挂死与多设备枚举不稳（klx1，2026-11-24） | 厂商级阻塞（与 TRAIN-P800-001 同族，新机复现） | 8 节点单节点隔离探针：**5/8 健康（xpu1/4/5/6/7，单卡 1.6 s 通过）、3/8 挂死（xpu0/2/3，首内核挂死且 SIGTERM 不免疫需 SIGKILL，rc=137）**；多设备容器枚举在卡空闲期不稳定（`invalid device ordinal`，唤醒后可复现通过）；单卡训练实测 `wait for noc idle timeout`（status 299）。操作口径：先探针唤醒再发车，选已验证健康节点。证据：klx1 `/data1/dzy-align/node-probe.log`、`align-e2e/` 各车 stderr 原文 |
| 训练逐层精度差分（oracle）A/A 与非扰动（三平台，2026-11-24 @ `8a9a9565` 树） | 通过 | `FLAGPROF_TRAIN_ORACLE_AT=auto` 抓 forward 边界输出（Qwen3-0.6B FSDP 双卡 = embed_tokens + 28 层，29 边界 × 3 步 = 87 事件）；**A/A 自检**：PPU 87/87 exact、P800 87/87 exact（blob 逐元素）、**Ascend 29 exact + 58 differs（step1 起 max_abs 1.9e-09→6.4e-02 随深度/步数累积——TRAIN-ASCEND-002 平台非确定性在逐层粒度的直接呈现，如实记录）**；**非扰动**：开/关 oracle 两跑 `training-result.json` 逐位一致（PPU cls+rr、P800 rr 双 IDENTICAL；Ascend 平台自身非确定，以采集纯读 + PPU/P800 逐位证据论证）；单测 3/3（PPU/Ascend 容器双确认）。命令与口径见 v2.md「训练逐层精度差分」 |
| 训练 oracle 跨厂商 join（rerank-0.6B FSDP 双卡，store 跨机汇总到 thead1，2026-11-24） | 完成 | **PPU↔Ascend**：87 行全 blob-elementwise（逐元素 256k-4096 样本张量级），exact 1（embed_tokens step0 逐位同）/differs 86，**首个发散 = step 0 `model.layers.0` 输出（max_abs 1.14e-05）**，逐步 max-of-max_abs 0.37→0.80→62.8；**PPU↔P800**：87 行全 blob-elementwise，exact 1/differs 86，首个发散同点（1.14e-05 同源量级），0.36→0.72→56.1——两厂商形态一致、量级可读，发散随深度与步数放大即为厂商核差异的直接量化；**P800↔Ascend**：87 行 exact 1/differs 86（第三对补齐） |
| 训练 oracle 梯度通道与报告面板（PPU，2026-11-24） | 通过 | join `--snapshots-left/right` 指向 accuracy 快照目录：**310 参数 × 3 步 = 930 梯度行全部逐位 exact**（A/A）；跨厂商按行 verdict 标注。oracle.json 置于 run 目录后 `flagprof report` 重新渲染即出「训练逐层精度差分（join）」与「首个发散」面板（87 行，report.html/report.md 同套） |
| 训练 oracle 算子钻取与重放（PPU 容器，2026-11-24） | 通过（A/A） | `FLAGPROF_TRAIN_ORACLE_OPS=<module>`：TorchDispatchMode 钻取该模块 forward 的逐算子输入/输出（busy 守卫隔离采集自身 CPU 快照操作，污染 0）；`oracle replay` 从输入 blob 重执行并与捕获输出 digest 对比——同机重放 match=True；跨厂商重放 = 在对方栈对同一 store 副本执行同一 CLI（输入即捕获值），输出 digest 对比即敏感算子判定；身份字段记录捕获/重放两侧 torch/cuda 版本 |

## 保留的问题与范围

| 编号 | 事实 | 下一步 / 判定边界 |
|---|---|---|
| TRAIN-P800-001 | 两台 P800 机器 6/6 长空闲物理卡在固定镜像内首内核前同型挂死（四轮探针证据链闭环）；挂死卡可被其他租户唤醒跑满，排除永久损坏 | 厂商级阻塞，交厂商定位；恢复可用设备窗口后按同镜像复测；不重置共享驱动、不清理他人任务；该平台仍待验证 |
| TRAIN-ENV-001 | 首轮官方采集因 scratch 位于容器 overlay 被 ext4 门禁拒绝 | 失败保留，指定 ext4 scratch 后通过；使用手册已明确 |
| TRAIN-ENV-002 | 上述既有 TP2 embedding 无设备 CPU 测试在 PPU 镜像失败 | 精确起点基线已复现；需要独立处理环境选择，不能称全部旧回归全绿 |
| 观察范围 | forward module 树不能直接当作 backward 层归属；Python collective 不包含全部 DDP C++ reducer 调用 | 训练显式范围内逐调用覆盖已补，跨线程不自动传递；设备关联缺失照常 partial/unavailable |
| 模型范围 | 内置分类功能示例、HF causal LM、密封 CTR 语料模板训练（14 架构，派生标签 + 逐架构 loss）、embedding 微调（InfoNCE，三种 pooling）、reranker 微调（两种打分语义）与两卡 FSDP 参数分片；不含正式数据集收敛、ZeRO/TP/PP | CTR/embedding 语料为确定性合成（不是真实点击或检索数据），rerank 标签由 sample id 派生；验证微调通路与观测闭环，不声明任务精度；新模型需明确输入、标签、loss、优化器和训练协议再单独验收 |
| TRAIN-MODEL-001 | Qwen3-Reranker-8B 全参 FP32 训练态约 128 GB > 单卡 96 GB，DDP 每 rank 复制不可行；两卡 FSDP 分片（每 rank 约 64 GB）已实测通过（performance 模式，2 步，rc=0，权重过 sha 关卡） | 正式数据集收敛、多机扩展与 bf16 参数协议属范围外/协议变更，需用户明确授权后再评；0.6B 同构通路已逐步逐位验证 |
| TRAIN-MODEL-002 | OneRec-1.7B 权重已供给（4263793656 字节，sha256 与 modelscope API 权威值逐字同）；标准 causal-lm 微调通路单卡实测通过：显式 --task 与注册表缺省两形态结果逐位一致，缺省语料自动生成 | 生成式推荐完整训练目标（多任务/强化学习）仍属范围外，按注册表 note 声明；不把推理模板裸加 .train() 冒充训练适配 |
| TRAIN-MODEL-003 | 搜广推 15 模型清单中两个多模态模型（openai/clip-vit-large-patch14、Qwen3-VL-Embedding-8B）按本轮范围约定排除 | 不在本轮训练适配范围；其余 13 个目标去向：9 个 CTR 架构（DIN/DeepFM/DCN-Mix/ESMM/DLRM/DSSM/LightGCN/SASRec/PLE）由 `--task ctr` 覆盖，Qwen3-Embedding-0.6B 与 bge-m3 由 `--task embedding` 覆盖（均已实测），Qwen3-Reranker-8B 见 TRAIN-MODEL-001（0.6B 同构通路已实测），OneRec-1.7B 见 TRAIN-MODEL-002——至此 13 个非多模态目标全部实测覆盖 |
| TRAIN-P800-002 | P800 厂商 kineto 不产设备内核事件：`XPU_CUPTI_ENABLE_DEVICE=0` 配方实测报 `etiEventSamplingSetMode error 17`，默认配方 attach 成功但 trace 无 kernel 事件（驱动 5.0.21.47 两配方一致） | 交厂商联合升级；通信计数与字节账完整可用，wait/带宽与内核画像在该平台不可得，如实标注不冒充 |
| TRAIN-ASCEND-001 | 910C CTR `dien` 的 GRU 映射融合算子 `DynamicGRUV2` 仅支持 FP16（CANN 9.0/9.1/9.2 一致），FP32 训练不支持 | 厂商算子规格缺口交算子团队；框架不静默换 dtype；其余 13 架构正常（PPU/P800 同架构正常，属平台差异） |
| TRAIN-ASCEND-002 | 910C 同一命令逐跑 final sha 非确定（控制实验：两次无采集 base 运行 sha 均不同） | sha 级零漂移判据在 Ascend 不适用；观察不扰协议以行为级证据（loss/梯度轨迹）论证 |
| TRAIN-ASCEND-003 | ascend2 租户容器上机后 HCCL NPU 网卡绑定失败（`EI0020`，环境级） | 该机 FSDP/观察态冒烟待租户退场复测；能力已由 ascend1 覆盖 |
| TRAIN-PP-001 | 官方流水引擎（torch.distributed.pipelining）在 PPU/P800 厂商栈第一步挂死 | 引擎级 PP 未验证，如实留档（PPU 两配方日志）；诊断性 p2p/集合通信 demo 通过，但不据此声明 PP 支持 |
| TRAIN-MONITOR-001 | 训练观察态在线诊断（agent 拓扑）要求厂商设备遥测（ascend/nvidia）；thead 平台与推理当日能力一致保持 off，本地边车会话可用。agent 拓扑已在 Ascend 910C 真实训练实车验证（2026-11-23，见上表）；nvidia 平台训练侧未实车（无 nvidia 训练机）；P800 复测时附带验证 | thead 平台不虚报诊断能力；`register_device` 在 thead 记 unknown（vendor 缺 `physical_identity`）、ascend 记 `chip:N available` |
| 性能范围 | accuracy 快照开销、首次编译/初始化、共享宿主 | 不自动给独占资格，不把诊断时长当成优化后性能，不合计 rank-local 吞吐 |

新增代码版权署名为 **Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)**；已有上游文件保留原版权声明。
