# Preview：获得策略、理解进度和继续探测

[返回入门指南](../../inference/README.md)

Preview 寻找当前模型、输入、设备与运行栈下能一起执行的 FlagGems 函数集合。默认原生基线记录候选，随后分组探测，最终对接受集合独立复验。得到 `verified` 策略后，accuracy 和 performance 可在相同执行身份下使用它。

## 开始与续探

以下命令在 `inference` 目录执行，环境已经按入门指南配置：

```bash
python3 run.py preview --config config/local.yaml --device 0 \
  --budget-seconds 600 --output result/preview-seed

python3 run.py preview --config config/local.yaml --device 0 \
  --resume-from result/preview-seed --budget-seconds 900 \
  --output result/preview-next
```

这些预算用于分段投入时间，不保证一次就能生成策略。默认预算为 3600 秒，单 worker 默认上限 900 秒。先看终端 `Results:` 对应的 `report.md`；正式运行只使用其中明确标记 `verified` 的 `preview/policy.yaml`。

预算从输入准备和可选缓存导入结束后起算，覆盖基线、恢复复验、候选搜索和必要的最终复验。准备、复制和超时清理另列，因此总墙钟时间可能更长。必要复验未完成时保存进度，但不发布 verified 策略。

续探来源只读，新目录不能与来源相互包含。保持模型、输入、设备顺序、背景组件、搜索策略和预算模式一致。程序校验来源摘要，重新运行原生基线并复验旧接受集合，通过后继续保存的搜索队列。旧集合失效会停止，不自动缩小集合。

## 默认选项各自改变什么

| CLI 参数 | 默认值 | 使用含义 |
|---|---|---|
| `--preview-evidence` | `lightweight` | 全量输入、全部 block、pooling 和 embedding 仍检查有限值；成功试验不保存完整输出张量 |
| `--resume-cache` | `auto` | 从指定续探来源复制身份与摘要匹配的 Triton 缓存组，改写副本路径 |
| `--preview-search` | `grouped` | 成功组整体接受，失败组继续拆分；最终集合独立复验 |
| `--preview-budget` | `adaptive` | 分开估计冷候选与暖复验成本，并为实际待复验集合预留时间 |

`forward-checks.json` 保存样本、边界、shape、dtype、检查元素数与非有限值数量，协调进程逐 rank 验证完整性。`full` 额外保留完整输出张量，两种模式的检查门禁一致。

缓存只复制到本次新目录，不共享可写缓存。损坏或缺失的缓存组回到冷编译；必需进度证据损坏仍会拒绝续探。`--resume-cache off` 可用于排查缓存影响。

需要同版本逐项调度作对照时，另开目录：

```bash
python3 run.py preview --config config/local.yaml --device 0 \
  --preview-search sequential --preview-budget fixed \
  --preview-evidence full --output result/preview-sequential
```

分组成员共享组合执行证据；接受集合可能与逐项顺序不同，不能据此认定每个成员在任意组合下独立可用，也不保证最大覆盖。资源故障、缺失命中或证据不完整不会冒充明确不兼容。

## 遇到 unknown 时先看原因

| 原因 | 下一步 |
|---|---|
| `not_scheduled` / `estimate_does_not_fit` | 尚未执行或估计成本放不进剩余额度；查看预算记录，再增加续探预算 |
| `worker_timeout` | 单次试验超时；检查最后阶段及 worker 日志 |
| `resource_failure` | 检查设备占用、OOM 或通信资源 |
| `evidence_incomplete` | 检查批次、边界、rank 和摘要是否完整 |
| `not_observed` | 没观测到函数命中；当前输入不足以确认该函数可用 |

详细进度在 `preview/checkpoint.json`，阶段成本和搜索记录在 `preview/budget.json`，缓存导入情况在 `preview/cache-import.json`，worker 阶段记录在各试验的 `stages.json`。阶段墙时包含初始化与编译等工作，不能当成独立 kernel 时间。

## 联合比较与版本变化

FlagTree/FlagCX 比较按[进阶指南](advanced.md)准备联合 preview：各环境探测后取共同集合，再在每个环境、每个 rank 完整复验。两个独立策略不能拼成受控对照。

策略格式为 schema 3，当前检查点 envelope 为 schema 2。格式号相同也不意味着执行身份相同：升级执行源码后必须重新 preview，旧策略和检查点不能修改摘要后继续使用；旧结果仍可阅读。纯分析和报告源码单独记录分析身份。

续探目录保留部分历史大张量/trace 的来源引用，不等于完整独立备份。`verified` 只确认当前输入下执行、函数命中和有限值检查，应用精度应另跑 accuracy 并结合业务阈值判断。
