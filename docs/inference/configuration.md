# 配置与命令参考

配置优先级为 **config/default.yaml → --config 指定文件 → CLI**。场景文件仅覆盖差异；相对路径按所在 YAML 目录解析，CLI 路径按当前工作目录解析。
复制为 `config/local*.yaml` 保存本机设置，文件已被 Git 忽略。模型和镜像默认为空，真实运行由用户填写。
`--config` 选择一个场景文件，不提供多文件叠加；重复传入时只读取最后一个。使用模型或 TP 场景文件时，另通过 CLI 填写镜像/模型，或复制该场景为本机配置；其他本机配置不会自动叠加。

| 区块 | 主要设置 |
|---|---|
| model | name、path、dtype、attention、模型 options |
| inputs | path、batch_size、max_length、padding_side、模型 options |
| accuracy | 组件开关、levels、layers、worst_samples、keep_tensors |
| performance | 组件开关、level、layers、采样、monitor、keep_evidence、variability |
| runtime | vendor、device 或 devices、parallelism、seed、worker 和分析超时、output_root |
| container | image、shm_size |
| vendors | 厂商 env、profile、编译器资产等适配配置 |
| reference | path（精度）或 performance_path（性能），互斥 |
| preview | budget_seconds、resume_from、evidence_mode、resume_cache、search_strategy、budget_mode |
| policy / export | 策略路径 / 模型支持的导出格式 |

## 常用 CLI

| 参数 | 使用规则 |
|---|---|
| `--image` / `--model-path` / `--input-path` | 覆盖本机资源；image 用于 Ascend 宿主 Docker 入口 |
| `--device N` | single 的物理设备号 |
| `--parallelism tp --devices N M` | 物理设备及 rank 顺序；不与 device 混用 |
| `--levels model layer` | accuracy 的边界层级 |
| `--level total\|layer --layers ...` | performance 的层级与模块选择 |
| `--flaggems/--flagtree/--flagcx off\|on\|both` | 默认 off，一次最多一个 both |
| `--oracle` / `--baseline` | 导入精度 / 性能参考，二者互斥 |
| `--keep-tensors` / `--keep-evidence` | accuracy / performance 保留原始现场 |
| `--monitor on\|off --monitor-interval-ms N` | performance 的独立监控，默认 on / 1000 ms |
| `--warmup-rounds --measure-rounds --repeats` | 默认 5 / 30 / 3；修改需结合统计稳定性解读 |
| `--budget-seconds --resume-from` | preview 的预算与只读续探来源 |
| `--source --output` | report/analyze/repack-reference 的来源与新目录 |
| `--dry-run` | 静态成本估计，不建立 verified 执行身份 |

`runtime.timeout_seconds` 限制单个 worker；`runtime.analysis_timeout_seconds` 默认 900，限制独立 CUDA 后处理。
`--analysis-timeout-seconds` 用于性能、report 重解析或 analyze；须为正整数。YAML 中组件和 monitor 的 `'on'` / `'off'` 要加引号，`keep_tensors` / `keep_evidence` 等布尔值使用不加引号的 `true` / `false`。
DeepFM/DLRM 可选 `model.options.compare_probability`；OneRec 可选 `model.options.max_new_tokens`。其他未知配置会拒绝。
模型 TP、dtype 和 export 支持由注册适配器校验；详情见[模型指南](models.md)。
