# 文本输入格式

每行一个 JSON 对象，包含唯一字符串 `id` 和字符串 `text`：

```json
{"id": "example-1", "text": "如何评估模型推理性能？"}
{"id": "example-2", "text": "Measure the same workload on both paths."}
```

用 `--input-path /path/to/texts.jsonl` 或配置 `inputs.path` 选择自己的输入。
[minimal.jsonl](minimal.jsonl) 提供 12 条原创示例文本，含中英文、长短句、空文本、重复文本和截断情况；
它用于数值和执行比较，不是任务评分数据集。
默认每批 4 条、最长 256 token、左 padding。

tokenization 只在 prepare 执行；两侧读取相同的 tokenized 输入，样本 ID 和 attention mask 用于配对。
样本信息保存原 token 数和截断情况；无 token 输入按模型 EOS 处理，模块级差异排除 padding。
完整捕获张量仍保留 padding 位置。输入内容会进入自己的运行目录。
