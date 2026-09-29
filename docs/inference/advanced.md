# Inference 进阶使用

[返回入门指南](../../inference/README.md)

以下命令在 `inference` 目录执行，默认已准备好 `config/local.yaml`、模型、镜像和编译器资产。
物理设备编号仅为示例，每次运行都应选当前空闲设备，输出目录使用新名称。

## 单机 TP

TP 将同一个模型分到多张卡，共同处理同一批输入。使用本机配置时，可直接覆盖并行方式和设备集合：

```bash
python3 run.py preview --config config/local.yaml --parallelism tp --devices 0 1 \
  --output result/tp-preview
python3 run.py accuracy --config config/local.yaml --parallelism tp --devices 0 1 \
  --flaggems both --levels model layer --policy result/tp-preview/preview/policy.yaml \
  --output result/tp-accuracy
python3 run.py performance --config config/local.yaml --parallelism tp --devices 0 1 \
  --flaggems both --policy result/tp-preview/preview/policy.yaml --output result/tp-performance
```

也可用 `--config config/tp.yaml --model-path /absolute/model --image your-image:tag --devices 0 1`，
获得 TP 示例中的共享内存、超时和 HCCL 设置。`--config` 每次读取一个覆盖文件，不会隐式合并 local.yaml 与 tp.yaml。
单卡策略不能直接用于 TP；TP 策略必须覆盖全部 rank。
`--communication-profile-rounds` 调整每个 repeat 的独立通信采样轮数，默认 1。

## 比较 FlagTree 或 FlagCX

最多一个组件为 `both`。联合 preview 在各环境探测后取共同函数集合，再在每个环境、每个 rank 完整复验。
不能把两份函数集合不同的独立策略拼成编译器或通信对照。

```bash
# 固定通信后端，比较编译器。
python3 run.py preview --config config/local.yaml --parallelism tp --devices 0 1 \
  --flagtree both --flagcx on --output result/tree-preview
python3 run.py performance --config config/local.yaml --parallelism tp --devices 0 1 \
  --flaggems on --flagtree both --flagcx on \
  --policy result/tree-preview/preview/policy.yaml --output result/tree-performance

# 固定编译器，比较通信后端；使用另一份匹配策略。
python3 run.py preview --config config/local.yaml --parallelism tp --devices 0 1 \
  --flagtree on --flagcx both --output result/cx-preview
python3 run.py accuracy --config config/local.yaml --parallelism tp --devices 0 1 \
  --flaggems on --flagtree on --flagcx both --levels model \
  --policy result/cx-preview/preview/policy.yaml --output result/cx-accuracy
```

FlagTree on 选择编译器，不额外启用 `torch.compile`。FlagCX on 选择模型通信组后端；Gloo 协调组与模型通信组分开。
是否实际参与以报告中的取证为准。

## Preview 续探

```bash
python3 run.py preview --config config/local.yaml --device 0 \
  --budget-seconds 300 --output result/preview-seed
python3 run.py preview --config config/local.yaml --device 0 \
  --resume-from result/preview-seed --budget-seconds 300 --output result/preview-next
```

来源只读，新目录不能与来源相互包含。保留原模型、输入、设备顺序及组件组合；TP 或联合比较两次需使用相同参数。
续探先校验来源摘要和执行身份，重新执行原生基线并复验旧接受集合，通过后才继续候选探测。
旧集合失效会停止，不会悄悄减少函数集合。中断项、未探测项、历史未知项按保存的顺序处理；明确排除项不重试。

预算从 prepare 完成后起算，覆盖所有环境的基线、恢复复验、候选决策和最终复验；准备与清理另列。
预算不足会保存可恢复进度，必要复验未完成时不会发布 verified 策略。
新目录复制必要的轻量证据与 tokenized 输入，大张量/trace 可能仍引用来源，因此不能把续探目录当作独立完整备份。
旧 schema 1/2 仅用于历史报告读取，不能直接执行或续探。

## 图导出

```bash
python3 run.py export --config config/local.yaml --device 0 \
  --formats fx onnx --output result/export
```

图表示首个固定 batch、当前 dtype 的 CPU 原生 forward、pooling 和归一化。
FX 保存后重新加载并核对输出；ONNX 做结构检查，不提供 ONNX Runtime 或厂商编译引擎数值验证。
导出保持单卡入口；准备阶段仍需要所选运行环境，不能把该命令当作完全无设备的离线模型转换工具。

## 离线重建报告

```bash
python3 tools/replay_reports.py --source result/performance --output result/reports-rebuilt
```

不执行模型、不改变历史判定和策略；在新目录生成报告，并校验原记录摘要未变。
可通过 `--source` 后的多个目录重建多份报告，各来源目录名必须不同。
输出通过符号链接读取原始数据，依赖原目录继续存在。报告中记录原结果与当前生成器身份。
