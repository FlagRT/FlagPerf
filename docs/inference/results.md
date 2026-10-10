# 如何阅读 Inference 结果

先核对模型、输入、组件、设备/rank、dtype、并行方式和比较对象，再看数值。终端的 `Results:` 指向独立结果目录。

## 状态与比较对象

先检查 `result.json` 的执行和分析状态，失败或 partial 时阅读原因和日志；质量状态 needs_review/invalid_evidence 也需单独处理。
completed 说明规定流程结束，不代表业务精度达标或性能稳定。监控状态和逐指标有效点数另见监控报告。

同设备 `comparison.json` 比较固定条件下的 off→on。导入参考后，`oracle-comparison.json` 比较 A100→被测路径，保留同设备差分；
`performance-comparison.json` 统一使用被测/A100 倍率。两种参考不能同时传入，参考 dtype 与数值设置由模型定义。

精度展示 MSE、MAE、最大绝对误差、cosine、相对/归一化误差、分位数及局部异常。零范数情况下某些指标未定义，不能当作零误差或通过。
概率标量的 cosine 可能缺乏区分度。OneRec 只比较首次 token 分歧之前及分歧那一步，之后的完整轨迹用于另外检查。

## 性能口径

静态模型总体窗口从输入驻留设备开始，覆盖模型规定的完整前向和同步；加载、分词、搬运及独立取证分开。
Qwen3 覆盖 forward、pooling 和归一化。DeepFM/DLRM 吞吐为 samples/s；Qwen3 另列固定文本工作量对应吞吐。
OneRec 默认固定生成 32 个新 token，分别列出完整生成、prefill/首 token、decode 和新生成 token/s；实际步数以配置为准，首 token 不含服务排队。

层级时延为包含子模块的窗口，父子耗时不能相加。框架 allocator 的窗口峰值不是模块独占分配，也不是整张设备占用。
KV 字节、allocator 峰值、设备 HBM 遥测来源不同，不能直接相加。主计时、层计时、显存、路由和 profiler 为独立采样。

独立重复波动与调用尖峰分别展示；少于三次独立重复不形成 repeat-CV 判断。查看比值时同时检查样本、计时、规模及缺失项。
NCCL kernel 和 CANN collective 时间不直接计算跨厂商倍率。Python 路由命中不等于硬件 kernel 覆盖率。

## 保存与离线查看

成功且分析完整、无数值异常或需复查测量的普通任务默认精简输出 tensor、原始 trace 与缓存。
固定输入、数值、日志、配置、源码、派生归因与异常信息保留。keep 选项、失败、partial、异常和参考生产保留完整现场。
精简后可以重绘和 portable 搬移；缺少原始 tensor/trace 时，不能重新计算相应指标或重解析。

| 文件或入口 | 用途 |
|---|---|
| result.json / report.md | 状态、条件、结论和明细导航 |
| comparison.json | 同设备组件差分 |
| oracle-comparison.json / performance-comparison.json | A100 精度 / 性能比较 |
| layer-index.md | 逐层、shape、rank 及热点导航 |
| monitor.html / monitor/ | 曲线、指标状态和监控原始记录 |
| prepared/ | 固定输入及模型 workload |
| artifacts/ | 分片明细，与引用 JSON 一起保存 |
| package/ / package-evidence/ | 参考核心 / 诊断附件 |

策略绑定执行身份。此次目录与执行代码更新后，历史策略需要重新 preview；历史报告和参考包按读协议继续使用。
操作见[逐层指南](layer.md)、[A100 参考](references.md)和[离线恢复](offline.md)。
