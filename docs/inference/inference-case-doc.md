# Inference 目录与扩展边界

用户从 [入门指南](../../inference/README.md) 开始；本文说明代码职责及修改边界。
当前提供 Qwen3-Embedding-0.6B 的 PyTorch/Transformers eager 适配，能力清单由
[support.json](../../inference/support.json) 和 `run.py list` 提供。

| 位置 | 职责 |
|---|---|
| `run.py` / `run_inference.py` | 宿主 Docker 入口 / 已准备环境内入口 |
| `config/` / `inputs/` | 默认与场景配置 / 小型 JSONL 输入示例 |
| `models/qwen3_embedding/` | 模型加载、tokenization、pooling 与模块选择 |
| `engines/` | forward、输出捕获及候选调用清单 |
| `runtime/` | 配置、任务目录、进程生命周期、preview、TP、性能、分组 preview、缓存导入及层/通信采集 |
| `vendors/` | 设备 API、编译器选择及通信后端选择 |
| `analysis/` | 数值差异、比较证据判断、离线层统计与设备/通信解析 |
| `reporting/` | 单卡/TP/组件/层级报告、筛选视图及只读重建 |
| `tools/` | 资产准备与离线报告重建 |

一次运行的顺序是：解析配置 → 宿主解析镜像 ID 并只读挂载输入 → prepare 封存输入与身份 →
独立 worker 执行各路径 → 分析结果 → 写入报告。正式性能与逐调用取证、通信 profiler 在不同执行阶段完成。
每次运行保留独立目录，避免后续分析覆盖执行证据。

执行源码扫描覆盖根入口、runtime、vendors、engines、models、analysis、reporting。
`runtime/common.py` 的显式分析清单将纯分析/报告文件与执行身份分开；新文件默认归执行身份。
新增源码根目录时必须扩展扫描边界，不能通过移出扫描目录规避策略失配。

新增模型应先明确输入、输出、padding、模块边界和 TP 分片语义，再添加适配并更新配置校验与能力清单。
新增厂商应在 vendors 实现设备和依赖边界，不将厂商特有调用散布到通用分析或报告中。
目录整理不改变误差公式、总级计时窗口、工作量计数和策略复验规则。

旧模型案例、任务评分器、编译引擎及 SSH 入口已移出当前工作树，恢复方式见 [迁移说明](migration.md)。
