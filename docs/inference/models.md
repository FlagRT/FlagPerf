# 模型准备与比较边界

模型决定输入、数值边界、吞吐单位和 TP 策略。四模型共享入口，但不同模型的指标不能直接混用。
本页的镜像和模型路径为占位值，执行前必须替换。命令从 `inference/` 目录运行。

| 模型 / 配置 | 输入与模型级精度边界 | 并行与图导出 |
|---|---|---|
| Qwen3-Embedding-0.6B / `config/default.yaml` | 文本；最后有效 token 的 pooled 与 FP32 L2 归一化 embedding | single；TP2/4/8 参数；single FX/ONNX |
| DeepFM / `config/deepfm.yaml` | 固定离散特征；输出头前 hidden，概率可选 | single、TP2；single FX |
| DLRM / `config/dlrm.yaml` | dense 加变长稀疏 ID 袋；top MLP 输出头前 hidden，概率可选 | single、TP2；single FX |
| OneRec-8B / `config/onerec.yaml` | 原始文本；最终 RMSNorm 后、LM head 前 hidden | single、TP2；无图导出 |

参数支持不代表每种设备规模、组件组合都已完成硬件验证。DeepFM/DLRM 的合成权重和输入用于分析工具运行，不提供 CTR/AUC 或训练后业务质量。

## Qwen3-Embedding

本地提供模型目录，在 `config/local.yaml` 中填写路径。默认 BF16、eager、batch 4、最大长度 256、左 padding；KV cache 不适用。
完整 forward、pooling 和归一化属于总体测量；输入驻留设备后开始计时。层选择可用 `layers.0` 或 `layers.0.self_attn.q_proj`。

## DeepFM 与 DLRM

在具有 CPU PyTorch、NumPy、safetensors 的环境中执行一次准备；launcher 虚拟环境不包含这些依赖，复用镜像的完整命令见[环境指南](environment.md)：

```bash
python3 -m benchmarks.deepfm.prepare_assets --output runtime_assets/deepfm-v1
python3 -m benchmarks.dlrm.prepare_assets --output runtime_assets/dlrm-v1
python3 run.py accuracy --config config/deepfm.yaml --image your-compatible-image:tag \
  --levels model layer --output result/deepfm-accuracy
python3 run.py performance --config config/dlrm.yaml --image your-compatible-image:tag \
  --parallelism tp --devices 0 1 --level layer --output result/dlrm-tp-layer
```

已存在的资产目录会拒绝覆盖。宿主只有 launcher 依赖时，可在兼容镜像中准备并将输出目录持久挂载。
默认各 128 条合成样本，FP32、batch 32。DeepFM 输入 `[B,5]`，默认 hidden `[B,400]`；DLRM 默认 dense `[B,13]`、26 个稀疏字段、hidden `[B,32]`。
DLRM 每字段使用 `values[N]` 和 `offsets[B+1]`，空袋为零，重复 ID 逐次求和；interaction 支持 dot/cat。

完整前向仍计算概率并检查有限性。`model.options.compare_probability: true` 才追加概率比较；正标量 cosine 通常为 1，应结合幅度误差。
吞吐为 samples/s，token 和 KV cache 不适用。TP2 按 embedding 维度分片后 AllGather，MLP 按输入列分片后 AllReduce；DLRM 允许合法的不等长输入列分片。
TP4/8、非 FP32、TP export、ONNX 和任务评分会提前拒绝。

## OneRec

```bash
python3 run.py accuracy --config config/onerec.yaml --model-path /absolute/path/to/OneRec-8B \
  --image your-compatible-image:tag --levels model layer --output result/onerec-accuracy
python3 run.py performance --config config/onerec.yaml --model-path /absolute/path/to/OneRec-8B \
  --image your-compatible-image:tag --parallelism tp --devices 0 1 --level layer --output result/onerec-tp-layer
```

原始文本不套 chat template。默认 BF16、eager、greedy、batch 1、左 padding、max length 256。
精度遵守 EOS，比较前缀截至首次选 token 分歧（包含分歧步）；之后继续生成与有限值检查，完整轨迹保留。
性能每请求从空 KV cache 开始，固定生成 32 个新 token，EOS 不提前中止；可修改 `model.options.max_new_tokens`，修改后需新 preview。
报告分开列出完整生成、prefill/首 token、decode、生成 token/s 和 KV/allocator；首 token 不含加载、分词、传输或服务排队，KV 字节不与 allocator 峰值相加。
跨设备生成可能提前分歧，数值可比范围以该对结果的前缀为准，不代表推荐业务质量。

四模型均需自己的 preview。不同模型、输入、TP 或组件组合不能互用策略。FlagTree/FlagCX 的联合策略见[进阶使用](advanced.md)。

跨设备两端使用同一份封存推荐模型资产；仅使用相同 seed 重新生成，不保证资产 manifest 身份相同。工具或来源元数据变化后，应重新生产匹配的参考包。
