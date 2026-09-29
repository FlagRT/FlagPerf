# Inference 配置参考

[返回入门指南](../../inference/README.md)

配置读取顺序为：**默认配置 → `--config` 指定的覆盖文件 → 显式 CLI 参数**。
默认配置是 `inference/config/default.yaml`，accuracy 和 performance 共用模型、输入和设备设置，分别保存自身选项。
`config/tp.yaml` 是 TP 场景的轻量覆盖文件；不会为 TP 复制一份独立的精度/性能参数。

配置中的相对路径按**定义该值的配置文件**所在目录解析；继承值保留其来源目录。
CLI 相对路径按当前工作目录解析；输出中的 `effective.yaml` 保存最终绝对路径。
未知字段、错误类型和不合法组合会报错，不会静默采用其他模式。

| 配置区块 | 主要字段 | 说明 |
|---|---|---|
| `model` | `path`、`dtype`、`attention` | 用户模型目录；BF16/FP16/FP32；当前 attention 为 eager |
| `inputs` | `path`、`batch_size`、`max_length`、`padding_side` | JSONL 文件；默认 4 条/批、最长 256 token、左 padding |
| `accuracy` | 三组件开关、`levels`、`layers`、`worst_samples` | 默认仅记录原生模型输出；可比较模型与指定模块 |
| `performance` | 三组件开关、`level`、`warmup_rounds`、`measure_rounds`、`repeats` | 只支持 total；默认预热 5 轮、测量 30 轮、重复 3 次 |
| `runtime` | `vendor`、`device`、`parallelism`、`devices`、`timeout_seconds` | 单卡用 device，TP 用 devices；超时作用于 worker 阶段 |
| `container` | `image`、`shm_size` | 用户兼容镜像；默认共享内存 4g，TP 示例为 8g |
| `vendors.ascend` | `vendor_compiler`、`env` | 只读编译器资产及统一环境变量 |
| `policy` | `path` | 正式 FlagGems on/both 使用的 verified 策略 |
| `preview` | 三组件开关、`budget_seconds`、`resume_from` | 默认探测预算 3600 秒；可选只读续探来源 |
| `export` | `formats` | fx、onnx；单卡接口、CPU 原生图 |

YAML 中的 `off`、`on`、`both` 请加引号，例如 `flaggems: 'off'`，避免被 YAML 当作布尔值。
三组件开关在 accuracy、performance、preview 中分别配置；preview 会主动探测 FlagGems，开关默认 off 不表示跳过探测。
一次最多一个组件为 `both`。

## 常用参数

```bash
# 以下命令在 inference 目录执行，基于已准备的 config/local.yaml。
# 单路径记录：不产生双路径差异。
python3 run.py accuracy --config config/local.yaml --device 0 --flaggems off

# 只比较指定模块，策略来自相同环境、模型和输入下的 preview。
python3 run.py accuracy --config config/local.yaml --device 0 \
  --flaggems both --policy result/preview/preview/policy.yaml \
  --levels layer --layers layers.0 layers.13 layers.27

# 缩短一次性能运行；短轮次用来确认运行链路。
python3 run.py performance --config config/local.yaml --device 0 \
  --flaggems off --warmup-rounds 1 --measure-rounds 3 --repeats 2
```

`--image` 覆盖 `container.image`；`--model-path` 和 `--input-path` 覆盖对应路径。
`--device` 与 `--devices` 不混用。性能用 `--level total`，精度用 `--levels model layer`。
模型级输出包括 pooled vector 和最终 embedding；`--layers all` 表示所有 Transformer blocks。
模块名字来自当前模型，不是通用硬件算子名称。

`--task-score` / `--data-path` 是保留接口，任务评分尚未实现，指定后明确报错。
不同命令只验证自身的专属选项，不会用另一模式的采样参数替代当前配置。
