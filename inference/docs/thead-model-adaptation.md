# PPU（thead）模型适配：模型清单、验证结果与使用方法

本文回答三个问题：**这套适配覆盖哪些模型、精度怎么验证的、怎么在你自己的机器上复跑**。代码在分支 `zhiyu/model-adaptation`，目标硬件为同型号 PPU-ZW810E 服务器（下文称"双机"）。适配过程的工作日志在团队内部仓库，本文只保留可复现的内容。

## 1. 四种入口，一套实现

| 你想做什么 | 用什么 | 产物 |
|---|---|---|
| 挂探针采集单模型精度 | attach 目标（`targets/` 下薄壳脚本） | `target-result.json` |
| 不挂探针跑基线 | `python3 -B -m flagprof.runner --task ctr/rerank/embedding …` | `runner.json` |
| 批量验证一整份模型名册 | `tools/batch_driver.py --config tools/models.yaml` | `battery-summary.md/.json` |
| 把测试输入固化成密封包 | `targets/make_rerank_package.py`、`make_ctr_package.py` | 包目录 + `manifest.json` |

设计要点：attach 与 runner 共用同一个执行体（`flagprof/runner/{rerank,ctr}.py`），减少了两套执行实现产生语义漂移的机会；同包同种子、相同运行配置下的逐位一致性由 A/A 实测验证。批量验证的每一行都连跑两遍（A/A），两遍不一致即判失败。

名册文件有两份：`tools/models.yaml`（15 模型 + 10 CTR 架构）与 `tools/models-hf2.yaml`（2026-11-11 从 HuggingFace 扩选的 6 个模型）。

## 2. 测试输入为什么用"密封包"

模型评测最怕输入不一致：改一个 token、换一种截断，结果就不可比。所以每个任务的输入连同指纹一起封进包：

| 包 | 内容 | 适用任务 |
|---|---|---|
| rerank-package-v1 | 逐对 query-doc 的 token ids + manifest（文件 sha256、oracle_id、打分语义） | 重排 |
| ctr-package-v1 / ctr-v1b | Criteo 规范形四字段（dense/sparse/hist_ids/hist_mask），固定种子可复现 | CTR 架构 |

两点说明：

- CTR 模板没有上游预训练权重，用固定种子（默认 531）随机初始化；真实权重以 `weight_sha256` 钉进身份记录。
- `ctr-v1b` 把支持的架构从 10 个扩到 14 个，但输入文件与 `ctr-v1` 逐字节相同、oracle_id 不变——扩名单对历史结果零影响。

## 3. 模型清单（28 项）

### 检索 / 重排（5 个）

| 模型 | 任务 | 加载方式 |
|---|---|---|
| bge-reranker-large | 重排 sequence_logit | 标准 |
| Qwen3-Reranker-0.6B | 重排 causal_yesno（三段模板） | 标准 |
| bge-large-zh-v1.5 | 句向量，cls 池化 | 标准 |
| jina-embeddings-v3 | 句向量，mean 池化 | 厂商代码受控加载 |
| Qwen3-Embedding-0.6B | 句向量，last 池化（对应官方语义） | 标准 |

### HF 标准架构探针（3 个）

bge-small-zh-v1.5（cls）、m3e-base（mean）、bge-reranker-base（sequence_logit，与 -large 同词表族，复用同一输入包）。用途：验证"标准架构不经适配直接接入"的路径。

### HF 扩面（6 个，2026-11-11）

bge-base-zh-v1.5（cls）、all-MiniLM-L6-v2（mean）、e5-small-v2（mean）、chinese-roberta-wwm-ext（mean）、ms-marco-MiniLM-L-6-v2（重排）、jina-reranker-v2-base-multilingual（重排 + 厂商代码）。用途：扩大模型覆盖面，其中 jina-reranker-v2 是第一个需要厂商代码的重排模型。

### CTR（14 种架构）

| 架构 | 参考实现 | 要点 |
|---|---|---|
| dlrm | Meta | 点积交互 |
| din | torch-rechub | ActivationUnit + dice |
| widedeep / deepfm / dcn | torch-rechub | 宽窄路径 / FM 项 / 交叉网络 |
| esmm | torch-rechub | 三列输出 [cvr, ctr, ctcvr] |
| mmoe / ple | torch-rechub | 多专家门控 / 两级 CGC |
| bst / dien | torch-rechub | TransformerEncoder / GRU+AUGRU |
| dcnmix | torch-rechub DCNv2 | CrossNetMix 低秩 MOE |
| dssm | Huang et al. 2013 论文 | 双塔 cosine ∈ [-1,1]（rechub 无此模型，按论文镜像，字段映射偏差在 docstring） |
| sasrec | pmixer/SASRec.pytorch | pad 位零值排除，输出 raw logit |
| lightgcn | kuandeng/LightGCN | 纯传播无激活，图由包行确定性派生 |

所有架构严格镜像参考实现；自带 sigmoid 的架构（esmm/mmoe/ple/bst/dien）不做二次 sigmoid。

## 4. 精度怎么验

三级协议，逐级加严：

1. **A/A**：同一命令连跑两遍，输出摘要与身份记录必须逐位一致。
2. **跨机**：两台机器各跑 A/A 后，分数指纹跨机比对——bf16 下逐位一致是达标线，不是"近似就算"。
3. **跨厂商**（CTR 扩展架构）：在昆仑芯 P800 上复跑对照。

注意：`record status=complete` 只说明采集流程走完，不代表模型跑成功——必须看 `exit_code` 和产物文件本身。

### 当前验证结果

**HF 扩面 6 模型**（2026-11-11，双机各 A/A ×2，分数指纹跨机逐位一致，全过）：

| 模型 | 任务 | 分数指纹 |
|---|---|---|
| bge-base-zh-v1.5 | 句向量 | `14174c43e547d506` |
| all-MiniLM-L6-v2 | 句向量 | `05aa167fdc93f6db` |
| e5-small-v2 | 句向量 | `4286b9513ec0c501` |
| chinese-roberta-wwm-ext | 句向量 | `3b41fd05cff4e47d` |
| ms-marco-MiniLM-L-6-v2 | 重排 | `d075ef7665c0d624` |
| jina-reranker-v2-base-multilingual | 重排（厂商代码） | `b760f9c50163cf14` |

**主名册 22 行**（15 模型名目 + 3 个 HF 探针 + 4 个 CTR 扩展架构）双机全过，老架构与历史基准值全对（零扰动回归）。跨厂商矩阵：lightgcn 三方逐位一致；dcnmix/dssm/dlrm 为 PPU = P800-CPU ≠ P800-XPU；sasrec 为 PPU = P800-XPU ≠ P800-CPU。

## 5. 常用命令（单卡容器）

```bash
# runner 任务：CTR（可脱离 record 单跑）
python3 -B -m flagprof.runner --task ctr --package packages/ctr-v1 --arch dlrm \
  --seed 531 --device cuda:0 --dtype bfloat16 --outputs OUT

# runner 任务：重排（标准模型）
python3 -B -m flagprof.runner --task rerank --package packages/bge-reranker-seq \
  hf:models/bge-reranker-large --score sequence_logit \
  --device cuda:0 --dtype bfloat16 --outputs OUT

# 句向量（cls 池化）
python3 -B -m flagprof.runner --task embedding --pooling cls hf:models/bge-large-zh-v1.5 \
  --inputs texts.jsonl --batch-size 4 --dtype bfloat16 --device cuda:0 --outputs OUT

# 厂商代码受控加载（jina 类，embedding / rerank 两个任务支持）
python3 -B -m flagprof.runner --task rerank --score sequence_logit --trust-remote-code \
  --package packages/jina-reranker-v2-seq hf:models/jina-reranker-v2-base-multilingual \
  --batch-size 4 --dtype bfloat16 --device cuda:0 --outputs OUT

# attach（需要挂探针采集时）
flagprof record --vendor thead --preset accuracy --store S --work W --scratch X -- \
  python3 -B targets/template_ctr.py --package packages/ctr-v1 --arch dlrm \
  --seed 531 --device cuda:0 --dtype bfloat16 --outputs OUT
```

包任务现在执行 `--flaggems` / `--flaggems-include`、`--flagtree` / `--vendor-compiler`、`--numerics`，并记录实际配置。FlagTree 在导入 torch 之前选择；FlagGems 的 `on` 仍要求非空 include 清单。挂采集器时，显式组件参数必须与采集器配置相同。`--repeats` 会实际执行多轮；包任务的 token/padding 已由输入包固定，`--inputs`、`--pooling` 等不适用选项会报错。

CLS pooling 缺省使用右填充；显式左填充时按 attention mask 选择首个有效 token。`last` / `mean` 的既有默认填充保持不变。模板要求首个有效 token 就是模型的 CLS token。

**Oracle 身份与报告。** CTR 和 rerank 都记录批次形状、样本顺序、probe 算法、软件、数值策略和组件来源。CTR 必须匹配架构配置、初始权重、种子和打分语义；rerank 必须匹配权重、打分语义、模板、yes/no token ID、attention 和厂商代码。两者均以 `scores` 注册输出边界。权重或打分语义变化会阻断精度归因；设备与软件等差异沿用既有显式声明规则。旧版本缺少这些身份字段的 Store 需要重新采集，不补写未知字段。

`tests/test_review_regressions.py` 用真实 CPU 采集、Store 和报告验证 CTR 及两种 rerank 打分路径，并覆盖身份冲突、配置生效与清理、厂商代码授权、子目录指纹、CLS 默认值和 BF16 输入下的 FP32 loss。GitHub `Inference CPU smoke` 增加独立 `model-contracts` 作业，运行这些测试和训练语义回归，不下载模型或使用设备。

CPU 测试套件：`python3 -B -m pytest tests -q`（在 `inference/` 下，配好 `PYTHONPATH`）。

## 6. 决策与边界

**厂商代码（trust_remote_code）受控加载。** 有些模型自带 Python 代码（`config.json` 的 `auto_map` 指向自定义建模文件）。默认一律不执行模型自带的代码；确要执行时用 `--trust-remote-code` 显式打开（runner 的 embedding、rerank，以及 `target_embedder.py` / `target_reranker.py` 入口一致；ctr/generate 传了直接报错）。打开后也不是放任执行：

- 只加载模型目录里**已经在磁盘上**的代码（绕开 transformers 动态模块缓存——它在 4.57 只拷一级相对导入，多级导入必坏）；
- auto_map 必须是本地条目，拒绝指向远端的 `repo--class` 写法；
- 递归记录模型目录内 `.py` 文件的 SHA-256，以相对路径为 `code_files` 的键；模块命名空间包含目录与代码指纹，避免代码更新后复用旧子模块；
- 厂商的 flash-attention 路径强制关闭，走原生注意力；
- rerank 按打分语义选类：sequence_logit → `AutoModelForSequenceClassification`，causal_yesno → `AutoModelForCausalLM`。

标准模型带这个 flag 等于没带。实现只有一处：`flagprof/runner/vendor.py`。首个实测模型 jina-reranker-v2-base-multilingual：双机各 A/A ×2、跨机逐位一致。

其余边界：

- **下载 provenance**：每个模型记录下载渠道、文件清单与 sha256（`download-provenance.json`；jina-v3 的在 `localization-provenance.json`）。
- **DIEN**：`pack_padded_sequence(enforce_sorted=False)` + bf16 GRU 在 PPU 实测可用；pad 位 attention 精确为 0，隐状态不受 pad 影响。
- **deepfm**：随机初始化下 FM 项可能把 sigmoid 推到 0/1 端点，断言用闭区间——这是预期行为，不是缺陷。
- **fixed.load** 保持文本包专用；rerank/ctr 包由 runner 的 canonical loader 负责（共用输入身份字段，并分别校验任务专属的权重与打分契约）。
