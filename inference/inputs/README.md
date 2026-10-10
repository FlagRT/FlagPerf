# 固定最小输入集

本目录提供新编写的固定中英文、长短句、技术及日常文本，供 Qwen3-Embedding 与 OneRec 使用。

额外用例：空文本、重复样本、混合语言、标点换行、超过 256 tokens 的截断输入。默认 12 条，batch size 4，max_length 256，左 padding。每行 JSON 包含唯一字符串 `id` 和字符串 `text`；这是数值验证输入，不是任务评分数据集。

tokenization 只发生在 prepare。原始 token 数、截断标志和文本保存在 samples.json；序列没有任何 token 时明确补 EOS，避免将全 padding 向量当作有效样本。off/on 加载同一 inputs.pt，样本 ID 和 attention mask 是配对依据。Layer 指标排除 padding，完整捕获产物仍保留 padding 位置。

OneRec 使用原始文本，不套 chat template。DeepFM/DLRM 的输入由各自资产准备工具生成，格式与模型边界见[模型指南](../../docs/inference/models.md)。
