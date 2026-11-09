# thead (PPU) 模型适配：包、目标与 runner 任务

适用范围：FlagPerf/inference 分支 `zhiyu/model-adaptation`，PPU-ZW810E（thead1/thead2）。
配套方案：`平头哥-Inference-v2-15模型适配实施方案-20261109.md`；阶段证据见各阶段报告与执行流水。

## 1. 三种执行面，一套 canonical 实现

| 面 | 入口 | 产物 | 用途 |
|---|---|---|---|
| attach 目标 | `targets/target_reranker.py` / `targets/template_ctr.py` / `targets/target_embedder.py`（薄壳） | `target-result.json` | `flagprof record` 挂探针采集 |
| runner 任务 | `flagprof.runner --task rerank / ctr / embedding` | `runner.json` | 无 probe 基线、report oracle 配对 |
| 批量电池 | `tools/batch_driver.py --config tools/models.yaml` | `battery-summary.{md,json}` | 15→500 模型扩量（方案 §8.2） |
| 包构建器 | `targets/make_rerank_package.py` / `make_ctr_package.py` | 封签包 | 输入固化 + oracle_id |

canonical 实现在 `flagprof/runner/{rerank,ctr,rerank_common}.py`；attach 薄壳与 runner 任务共用同一执行体（`run_ctr`/`run_rerank`），锚点等价由结构保证：同包同种子下 attach 与 runner 的 `score_batches_sha256` 必须逐位相同。批量电池对每行跑两遍（A/A），分值摘要兼容顶层字段（ctr/rerank/attach）与 run 层 outputs（embedding）。

## 2. 包契约 v1

- **rerank-package-v1**：`prepared/inputs.pt`（sample_id + 变长 input_ids）+ manifest（files sha256 / oracle_id / score_semantics / template_sha / contract.batch_size）。语义两种：`sequence_logit`（Bert 类重排）、`causal_yesno`（生成类重排，tokenizer 需含字面 yes/no）。
- **ctr-package-v1**：Criteo 规范形四字段（dense f32[13]、sparse i64[26]∈[1,999]、hist_ids i64[20]（0=pad）、hist_mask bool[20]），seed 可复现，manifest 含 schema_sha + `contract.archs`（10 架构）。
- **ctr 模型权重**：随机初始化（`torch.manual_seed(--seed)`，默认 531）——CTR 模板无上游预训练权重，identity 以 `weight_sha256`（state_dict 排序名+字节，bf16 先 `view(int16)`）钉真实权重。

## 3. 模型/架构清单

**检索/重排（Batch A，4）**：bge-reranker-large（sequence_logit）、Qwen3-Reranker-0.6B（causal_yesno，三段模板）、bge-large-zh-v1.5（runner `--pooling cls`）、jina-embeddings-v3（厂商代码包加载 + `use_flash_attn=False` 原生注意力 + mean 池化）。

**CTR（Batch B/C，10）**：dlrm（Meta 点积交互）、din（ActivationUnit+dice）、widedeep、deepfm、dcn、esmm（[cvr,ctr,ctcvr] 三列）、mmoe、ple（两级 CGC）、bst（TransformerEncoder）、dien（GRU packed + AUGRU；aux loss 只喂训练，推理路径省略）。全部严格镜像参考实现（torch-rechub 1.0.0 / Meta dlrm），输出已 sigmoid 的架构（esmm/mmoe/ple/bst/dien）transform 不二次 sigmoid。

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
```

CPU 套件：`docker run -w /w/FlagPerf/inference -e PYTHONPATH=/w/FlagPerf/inference IMG python3 -B -m pytest tests -q`。

## 6. 已知边界与决策记录

- jina-v3：transformers 4.57 动态模块缓存只拷一级相对导入 → 厂商代码以本地包绕行加载（不走动态缓存）；auto_map 去远程前缀；provenance 记录在 `localization-provenance.json`（原始 download-provenance 不改）。
- DIEN：`pack_padded_sequence(enforce_sorted=False)` + bf16 GRU 在 PPU 实测可用（ppu1 微基准探针）；pad 位 attention 精确为 0 → 隐状态不受 pad 影响。
- deepfm 随机初始化下 FM 项量级可饱和 sigmoid 至 0/1 端点——闭区间断言，非缺陷。
- fixed.load 保持文本包专用；rerank/ctr 包由 runner canonical loader 负责（identity 契约在 oracle_id/inputs_sha256 层统一）。
