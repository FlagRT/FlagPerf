# thead (PPU) 模型适配：包、目标与 runner 任务

适用范围：FlagPerf/inference 分支 `zhiyu/model-adaptation`，PPU-ZW810E（thead1/thead2）。
配套方案：`平头哥-Inference-v2-15模型适配实施方案-20261109.md`；阶段证据见各阶段报告与执行流水。

## 1. 三种执行面，一套 canonical 实现

| 面 | 入口 | 产物 | 用途 |
|---|---|---|---|
| attach 目标 | `targets/target_reranker.py` / `targets/template_ctr.py` / `targets/target_embedder.py`（薄壳） | `target-result.json` | `flagprof record` 挂探针采集 |
| runner 任务 | `flagprof.runner --task rerank / ctr / embedding` | `runner.json` | 无 probe 基线、report oracle 配对 |
| 批量验收 | `tools/batch_driver.py --config tools/models.yaml` | `battery-summary.{md,json}`（文件名沿用历史前缀） | 15→500 模型扩量（方案 §8.2） |
| 包构建器 | `targets/make_rerank_package.py` / `make_ctr_package.py` | 封签包 | 输入固化 + oracle_id |

canonical 实现在 `flagprof/runner/{rerank,ctr,rerank_common}.py`；attach 薄壳与 runner 任务共用同一执行体（`run_ctr`/`run_rerank`），锚点等价由结构保证：同包同种子下 attach 与 runner 的 `score_batches_sha256` 必须逐位相同。批量验收对每行跑两遍（A/A），分值摘要兼容顶层字段（ctr/rerank/attach）与 run 层 outputs（embedding）。

## 2. 包契约 v1

- **rerank-package-v1**：`prepared/inputs.pt`（sample_id + 变长 input_ids）+ manifest（files sha256 / oracle_id / score_semantics / template_sha / contract.batch_size）。语义两种：`sequence_logit`（Bert 类重排）、`causal_yesno`（生成类重排，tokenizer 需含字面 yes/no）。
- **ctr-package-v1**：Criteo 规范形四字段（dense f32[13]、sparse i64[26]∈[1,999]、hist_ids i64[20]（0=pad）、hist_mask bool[20]），seed 可复现，manifest 含 schema_sha + `contract.archs`。历史包 `ctr-v1`（10 架构）保留；`ctr-v1b` 扩至 14 架构（inputs.pt 与 ctr-v1 逐字节同、oracle_id 相同——contract 不入 oracle 身份，扩合同零漂移）。
- **ctr 模型权重**：随机初始化（`torch.manual_seed(--seed)`，默认 531）——CTR 模板无上游预训练权重，identity 以 `weight_sha256`（state_dict 排序名+字节，bf16 先 `view(int16)`）钉真实权重（lightgcn 的包派生图 buffer `norm_adj`/`id_to_node` 一并入钉）。

## 3. 模型/架构清单

**检索/重排（Batch A，4）**：bge-reranker-large（sequence_logit）、Qwen3-Reranker-0.6B（causal_yesno，三段模板）、bge-large-zh-v1.5（runner `--pooling cls`）、jina-embeddings-v3（runner `--trust-remote-code` 受控厂商代码加载 + `use_flash_attn=False` 原生注意力 + mean 池化）。

**第 15 名目**：Qwen3-Embedding-0.6B（runner embedding `--pooling last`，对应官方 last-token 池化语义）。

**HuggingFace Tier-0 探针（标准架构直接接入验证）**：bge-small-zh-v1.5（cls）、m3e-base（mean）、bge-reranker-base（sequence_logit，与 -large 同属 XLM-R 词表族，复用同一输入包）。

**CTR（Batch B/C，10）**：dlrm（Meta 点积交互）、din（ActivationUnit+dice）、widedeep、deepfm、dcn、esmm（[cvr,ctr,ctcvr] 三列）、mmoe、ple（两级 CGC）、bst（TransformerEncoder）、dien（GRU packed + AUGRU；aux loss 只喂训练，推理路径省略）。全部严格镜像参考实现（torch-rechub 1.0.0 / Meta dlrm），输出已 sigmoid 的架构（esmm/mmoe/ple/bst/dien）transform 不二次 sigmoid。

**CTR 扩展（2026 年度计划，+4，包 `ctr-v1b`）**：dcnmix（torch-rechub DCNv2 parallel + CrossNetMix 低秩 MOE：tanh 低秩空间、softmax 专家门控、残差；forward 内 sigmoid→PRE_SIGMOID）、dssm（Huang et al. 2013 双塔：用户塔=dense+hist 掩码均值池化、物品塔=sparse 后 13 列均值，tanh MLP、L2 归一 cosine∈[-1,1] 即终分，rechub 无 dssm 属论文镜像、字段映射偏差记录 docstring）、sasrec（pmixer/SASRec.pytorch post-LN：item/pos padding_idx=0 的 pad 位零值排除、PointWise Conv1d×2、score=<末位隐向量,target 嵌入> raw logit）、lightgcn（kuandeng/LightGCN 官方：纯传播 D^-1/2·A·D^-1/2 无自环无激活、层均值组合、random_normal(0.01) 初始化；图由包行确定性派生，构建需包行、每 batch 注入 user_idx=样本索引）。

## 4. 精度证据协议（attach 侧）

1. **A/A 双跑**：同包同种子同命令连跑两次，`score_batches_sha256` 与 `oracle_identity` 必须逐位一致。
2. **跨机复验**：双机各 A/A 后 `score_batches_sha256` 跨机比对（bf16、跨驱动逐位一致为达标线）。
3. `flagprof report oracle` 精度面板只认 runner 任务契约；attach 证据 = A/A 协议（runner promotion 后 runner 级可用 oracle 配对）。
4. `record status=complete` 不代表 target 成功——必须核 `exit_code` 与产物文件。

## 5. 常用命令（PPU 单卡容器）

```bash
# attach（accuracy preset，挂采集）
flagprof record --vendor thead --preset accuracy --store S --work W --scratch X -- \
  python3 -B targets/template_ctr.py --package packages/ctr-v1 --arch dlrm \
  --seed 531 --device cuda:0 --dtype bfloat16 --outputs OUT

# runner 任务（可脱离 record 单跑）
python3 -B -m flagprof.runner --task ctr --package packages/ctr-v1 --arch dlrm \
  --seed 531 --device cuda:0 --dtype bfloat16 --outputs OUT
python3 -B -m flagprof.runner --task rerank --package packages/bge-reranker-seq \
  --model hf:models/bge-reranker-large --score sequence_logit \
  --device cuda:0 --dtype bfloat16 --outputs OUT

# embed（runner 内建，cls 池化）
python3 -B -m flagprof.runner --task embedding --pooling cls hf:models/bge-large-zh-v1.5 \
  --inputs texts.jsonl --batch-size 4 --dtype bfloat16 --device cuda:0 --outputs OUT

# embed + 受控厂商代码加载（jina 类；详见 flagprof/runner/vendor.py docstring）
python3 -B -m flagprof.runner --task embedding --pooling mean --trust-remote-code \
  hf:models/jina-embeddings-v3 --inputs texts.jsonl --batch-size 4 \
  --dtype bfloat16 --device cuda:0 --outputs OUT
```

CPU 套件：`docker run -w /w/FlagPerf/inference -e PYTHONPATH=/w/FlagPerf/inference IMG python3 -B -m pytest tests -q`。

## 6. 已知边界与决策记录

- **厂商代码（trust_remote_code）受控加载**（2026-11-10 用户授权后取代"仅限 attach 目标"旧决定）：runner `--task embedding --trust-remote-code` 显式开关；加载只认模型目录内已在磁盘的代码（本地包绕行，不走 transformers 动态模块缓存——4.57 只拷一级相对导入）；auto_map 必须是本地条目（拒绝 `repo--class` 远程前缀）；每个被消费的 `*.py` sha256 钉入 oracle identity（`code_files`）；flash-attention 强制关闭。标准模型带此 flag 是无害 no-op；非 embedding 任务传此 flag 直接 argparse 报错。canonical 实现唯一：`flagprof/runner/vendor.py`（attach 目标同源引用）。
- jina-v3 下载侧 provenance：`localization-provenance.json`（原始 download-provenance 不改）。
- DIEN：`pack_padded_sequence(enforce_sorted=False)` + bf16 GRU 在 PPU 实测可用（ppu1 微基准探针）；pad 位 attention 精确为 0 → 隐状态不受 pad 影响。
- deepfm 随机初始化下 FM 项量级可饱和 sigmoid 至 0/1 端点——闭区间断言，非缺陷。
- fixed.load 保持文本包专用；rerank/ctr 包由 runner canonical loader 负责（identity 契约在 oracle_id/inputs_sha256 层统一）。
