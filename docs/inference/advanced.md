# TP 与组件进阶使用

命令从 `inference/` 运行。先用原生路径确认模型和环境，再为当前设备、模型、输入与组件组合生成策略。

## 张量并行

```bash
python3 run.py accuracy --config config/local.yaml --parallelism tp --devices 0 1 \
  --levels model layer --output result/tp-native
python3 run.py performance --config config/local.yaml --parallelism tp --devices 0 1 \
  --level layer --output result/tp-layer
```

`--devices` 顺序确定 rank 顺序；不与 `--device` 混用。`config/tp.yaml` 提供 TP 超时、共享内存和 HCCL 环境覆盖，真实模型和镜像仍需用户配置。
Qwen3 提供 TP2/4/8 参数，其他三模型只提供 TP2。TP 参数支持不代表全部组合的硬件验证；模型分片语义见[模型指南](models.md)。

## FlagTree 与 FlagCX

FlagTree 比较需固定 FlagGems on；FlagCX 在 TP 中使用，固定其他组件；每组都重新生成联合策略。例如：

```bash
python3 run.py preview --config config/local.yaml --flaggems on --flagtree both --output result/tree-preview
python3 run.py performance --config config/local.yaml --flaggems on --flagtree both \
  --policy result/tree-preview/preview/policy.yaml --level layer --output result/tree-layer
python3 run.py preview --config config/local.yaml --parallelism tp --devices 0 1 \
  --flagcx both --output result/cx-preview
python3 run.py performance --config config/local.yaml --parallelism tp --devices 0 1 --flagcx both \
  --policy result/cx-preview/preview/policy.yaml --level layer --output result/cx-layer
```

策略只能用于相同固定组件组合。共同入选集合按两侧共同验证，不能拼接两份独立策略。
actual JIT/collective 调用是组件参与证据；eager 路径可能不调用 Triton 编译器。unknown 保留预算、资源或证据原因。

## 图导出与已准备环境

```bash
python3 run.py export --config config/local.yaml --formats fx onnx --output result/export
```

导出为首个固定 batch 的 CPU 原生图，输入准备和导出均不初始化加速设备。`run.py export` 使用用户镜像但不挂载设备或驱动；已有 CPU 环境可直接使用 `run_inference.py export`。
FX 会重载并检查原生输出一致；ONNX 需安装 onnx、onnxscript，只检查结构，不验证 ONNX Runtime 数值。支持范围由模型决定，TP export 不支持。
已处于准备好的容器时使用 `run_inference.py`；在该环境重新 preview，不能把其他镜像策略视为自动兼容。
A100 仅使用 oracle/performance-baseline，见[参考指南](references.md)。
