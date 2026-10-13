# A100 参考包与跨设备比较

A100 是参考采集端。`oracle` 提供模型规定 dtype 的精度参考，`performance-baseline` 提供独立性能基线；两者分开生产，消费参数互斥。
Qwen3/OneRec 参考为 BF16，DeepFM/DLRM 为 FP32。参考权重与固定输入必须与被测任务一致。

## 1. 在已有 CUDA 环境中生产

`run.py` 启动 Ascend 推理容器；A100 使用已准备环境内的 `run_inference.py`。
先按[环境指南](environment.md)准备 CUDA 虚拟环境。从 `inference/` 执行，模型路径按实际替换；若环境位于其他目录，修改两个 Python 路径：

```bash
ORACLE_PYTHON="$PWD/runtime_assets/a100-venv/bin/python"
BASELINE_PYTHON="$PWD/runtime_assets/a100-performance-venv/bin/python"
"$ORACLE_PYTHON" run_inference.py oracle --config config/a100.yaml \
  --model-path /absolute/path/to/Qwen3-Embedding-0.6B --output result/a100-oracle
"$BASELINE_PYTHON" run_inference.py performance-baseline --config config/a100-performance.yaml \
  --model-path /absolute/path/to/Qwen3-Embedding-0.6B --output result/a100-single
"$BASELINE_PYTHON" run_inference.py performance-baseline --config config/a100-performance.yaml \
  --model-path /absolute/path/to/Qwen3-Embedding-0.6B --parallelism tp --devices 0 1 \
  --output result/a100-tp2
```

使用其他模型时换成 `config/deepfm.yaml`、`config/dlrm.yaml` 或 `config/onerec.yaml`，另传 `--vendor nvidia`。
精度 oracle 只在单卡生产，三个组件均 off，独立重复一致、输出有限及包验证完成后才能发布。
性能支持 single/TP2，原生组件，独立总体、层时延、显存与 profiler；默认 5/30/3 采样，不与精度 oracle 混为同一数值设置。

## 2. 搬移核心包

新参考使用 schema 4：生产目录的 `package/` 是可独立消费的核心包，`package-evidence/` 是摘要绑定的诊断附件。
核心包含固定输入、合同、必要输出或性能观测及逐文件摘要，不包含模型权重和原始 trace。日常比较只需完整核心目录；完整诊断需同时保留附件。
历史 schema 1～3 继续只读消费，可用[离线转换](offline.md)生成新核心。

## 3. 在 Ascend 消费

```bash
python3 run.py accuracy --config config/local.yaml --oracle /absolute/path/to/oracle/package \
  --input-path /absolute/path/to/oracle/package/inputs.jsonl \
  --levels model layer --output result/a100-accuracy
python3 run.py performance --config config/local.yaml --baseline /absolute/path/to/baseline/package \
  --input-path /absolute/path/to/baseline/package/inputs.jsonl \
  --level layer --output result/a100-performance
```

原生消费不需要组件策略。组件 both 比较时，先使用同一参考参数、设备和组件组合执行 preview，再传 verified 策略。
固定输入由包提供，不重新分词。复制完整默认配置后，`inputs.path` 是显式输入，程序会检查它与参考包的摘要一致；上面的 `--input-path` 指向包内输入，可避免本机默认文本与参考输入不同。
preview 也使用相同的 `--input-path` 和参考参数。模型、dtype、batch_size、max_length、padding_side、模型 options、并行度与计时合同必须匹配；这些配置不会由包自动覆盖，失配时停止。
single 基线用于 single，TP2 基线用于 TP2；两端使用同一份封存权重及资产目录，不仅是同一模型名称或 seed。
精度 TP 只比较模型支持的完整边界，各 rank 分别对参考，不拼接任意分片。

## 4. 阅读比较

`comparison.json` 保留同设备 off→on；`oracle-comparison.json` 列出 A100→被测路径。
`performance-comparison.json` 的倍率方向为 **被测/A100**，同时展示总体、shape、rank/layer、独立重复波动和可比显存。
两厂商通信 trace 的 NCCL kernel 与 CANN collective 时间不直接计算倍率。OneRec 按首次 token 分歧限制可比前缀，完整生成轨迹仍保存。
比较正确与业务精度达标、稳定性能收益是不同结论；业务阈值由用户定义。参考生产保留完整现场，普通消费任务沿用证据精简规则。
