# 指标来源矩阵：level × 指标 × 来源

本页覆盖四模型、总体/逐层性能、参考比较和独立监控的指标来源/API。具体执行版本以该次封存的 source-snapshot 为准；代码接入不等同于某次运行已成功采集。

**阅读结论：total主性能来自同步主机计时和框架分配器；layer分别采集模块主机/Event窗口、独立显存和硬件trace；TP通信来自独立profiler重放；路由与组件参与来自Python调用取证；精度来自封存tensor的CPU FP64比较。** `npu-smi info`另存运行前后环境快照，未用于生成主时延或框架峰值显存。

OneRec 自回归工作负载的口径见 §14：总吞吐的 token 分子是新生成 token，另外记录提示输入 token；精度按样本比较到首次 token 分歧，包含分歧步。前文中的单步 forward、pooling 与有效输入 token 吞吐适用于静态模型。

## 1. level、状态与证据路径

| 模式 / level | 当前范围 | 阅读入口 |
|---|---|---|
| performance / total，单卡 | 固定输入串行批次的时延、吞吐、阶段、显存、传输和独立调用取证 | §2–4、§6、§14 |
| performance / total，单机 TP | 一个逻辑批次的全局时间包络；逐 rank 资源；独立通信采样及模块归因 | §2–6、§14 |
| performance / layer，单卡与单机 TP | 完整模型内的模块窗口、独立显存/trace/路由观测；实测状态见具体实现记录 | §11、§14 |
| performance-baseline / performance --baseline | A100 单卡或 TP=2 完整封包与同规模离线总量/逐层比较 | §12、[A100 参考指南](references.md) |
| performance / request、operator | CLI 尚不接受这些 level；没有对应正式性能模式 | §9 |
| accuracy / model | 适配器声明的模型边界；Qwen3 pooled/embedding、DeepFM/DLRM head 前 hidden（probability 显式开启）、OneRec RMSNorm hidden | §7、§14、§15 |
| accuracy / layer | 选定模块边界输出；TP 在相同 rank 内配对 | §7、§14 |
| accuracy / A100 oracle，model + layer | A100 固定参考与被测各 side/rank 完整边界；CPU FP64，公式沿用 §7 | [A100 参考指南](references.md) |
| preview / 组件审计 / 运行辅助 | 候选与策略状态、实际调用、预算、环境快照；不是新增性能 level | §6、§8 |
| export / CPU 原生复验 | FX 重载数值复验；ONNX 结构检查 | §8 |

本页的 `total` 表示模型整体性能，配置见[配置参考](configuration.md)。性能的 `--level total` 与精度的 `--levels model layer` 是两个接口，见[参数定义][配置]；TP 通信分类展示集合调用归属；模块耗时以独立 layer 采集为准，不能从通信分类推导逐算子时延。适用设备范围以 [support.json](../../inference/support.json) 和具体运行身份为准；NVIDIA/A100 专用于参考生产，实际比较需满足模型和参考合同。


本页的“已接入”表示代码存在实际采集/分析路径；每次运行仍需检查 `status`、原始产物与覆盖。**未采集、未独立采集、不适用、未实现、未定义和实测零值分别保留**。派生值必须有原始输入；估算不标成实测。配置值、身份摘要、错误文本和 shape/dtype 作为解释指标所需的上下文，不全部另立为数值指标。

下表中的路径均相对一次运行的输出根目录；`<side>` 是该次被比较组件的 off/on，而非恒指 FlagGems，具体组合在 `comparison-context.json`。

| 简写 | 原始 / 汇总产物 | 字段定位规则 |
|---|---|---|
| R | `result.json` | 最终汇总；`report.md` 是其与相邻证据的展示投影 |
| W | 单卡：`<side>/repeat-XX/result.json`；TP：`<side>/repeat-XX/rank-N/result.json` | 一次计时 worker；相邻 `batches.jsonl` 保存逐批数据 |
| G | TP：`<side>/repeat-XX/result.json`、`batches.jsonl` | 按 rank 合并的全局批次，W 仍保留 rank 本地数据 |
| C | `profiles/<side>/repeat-XX/communication.json` | 独立通信分析；逐 rank 字段在 `ranks.<N>`；R 的 `profiles` 也含该结果 |
| A | `audit/<side>/`，TP 再下钻 `rank-N/` | `inventory.json`、`route.json`、`gems-calls.jsonl`、`components.json`；辅助原生取证在该侧 `native/` |
| D | `comparison.json` | 精度；TP 下钻 `ranks.<N>`；原始 tensor 为 `<side>/[rank-N/]batch-XXXX.pt` |
| O | `oracle-comparison.json` | A100 差分；下钻 `candidates.<side>.ranks.<N或single>`；参考为外部包的 `outputs/*.safetensors`。成功后被测 tensor 默认清理，删除摘要见 `tensor-retention.json` |

矩阵中的 `*` 表示动态键、`[]` 表示列表元素，二者是文档定位记法，不是新增 JSON 字段。来源链接指向实现文件，表内给出 API 或函数名以便定位。完整采样规则以该次封存的 `source-code/`、`source-snapshot.json` 和 `effective.yaml` 为准。

### 模型和设备 SDK 口径

Qwen3 与 DeepFM 共用上述公式和采样默认值。DeepFM 默认模型边界为输出头前 `hidden[B,400]`，精度 layer 默认为 embedding 与三个 MLP 隐藏块末端；性能 layer 仍覆盖 linear/embedding/fm/mlp。显式 `model.options.compare_probability: true` 时追加 `probability[B]`，每样本提取为标量 `[]`。正概率标量 cosine 通常为 1，不提供有效的幅度误差判断；原公式保持不变。TP=2 的完整输出在两 rank 复制，按 rank 比对但不重复累计全局样本吞吐；分支与隐藏层依赖由 `workload.json` 保存。样本/s 的分子为实际样本处理次数；token、token/s 为 null，KV cache 不适用。

公共流程读取 `LoadedModel.total_seconds`，CPU 加载/设备迁移仅在适配器独立采集时填写；直接设备加载可以有 null 子阶段及 `unmeasured_reason`。设备指标经 `DeviceSession` 标准接口采集，实际能力和来源在 `memory.measurement`；未提供的内存/峰值为 null，不能解读为零。要求 layer 事件、流或峰值测量却无相应能力时，拒绝该请求。新参考使用 schema 4 核心/附件，protocol_schema_version 保留模型合同；边界样本轴、有效区域、布局、shape 和计时语义均封存，历史包只读兼容。详见[模型指南](models.md)。

## 2. 总 level：时延、吞吐、阶段与完成情况

一次性能执行：准备固定 tokenized inputs → 加载模型并迁入输入 → 不计时有限性检查与首批输出拷贝 → 预热 → 正式批次计时 → 汇总。路由审计使用独立 worker，TP 通信使用另一次 profiler worker；这些时间不加入主计时。

| level / 场景 | 指标与单位 | 来源 / API | 计算与采集边界 | 字段 / 原始证据 |
|---|---|---|---|---|
| total / 单卡 | 批次时延，ns；均值，ms | `time.perf_counter_ns()`、`backend.synchronize()`；[计时采集] `run()` | 前同步在起点之前；计时包含 model forward、pooling、FP32 normalization、末尾同步；不含加载、tokenization、拷贝、预热、落盘与取证 | W `batches[].latency_ns`、`summary.latency_mean_ms`；`batches.jsonl` |
| total / TP | rank 批次起止与时延，ns | 同一宿主机 `perf_counter_ns()`、设备同步；[TP采集] `run()` | 每 rank 分别记录；Gloo 控制 barrier 位于计时窗外；模型 collective 在 forward 内 | W `batches[].start_ns/end_ns/latency_ns` |
| total / TP | 全局批次时延、rank 进入偏斜，ns | [TP采集] `global_batches()`，派生 | 时延 = `max(end_ns) - min(start_ns)`；偏斜 = `max(start_ns) - min(start_ns)`；先验证各 rank 批次顺序、工作量与输入元数据一致，不是 rank 时长之和 | G `batches[].latency_ns/rank_entry_skew_ns` |
| total / 两种场景 | 测量批次、样本处理次数、有效输入 token 数 | `len(rows)`、batch 样本 ID 数、`attention_mask.sum()`；[计时采集]、[TP采集] | 每轮遍历全部输入；重复处理同一文本重复计入工作量；TP 全局分子只计一次，不能乘 rank 数；token 不是生成 token | W/G `batches[].samples/tokens`；`summary.batches/samples/tokens` |
| total / 两种场景 | 累计主测量时间，s；平均批次时延，ms | [计时采集] `summarize()`，派生 | `T = sum(latency_ns)/1e9`；均值 = `1000*T/批次数`；T 不含批次间落盘、barrier 等间隙，不是整个命令墙钟时间 | W/G/R `summary` 中 `measured_seconds`、`latency_mean_ms`（R 下先取 side） |
| total / 两种场景 | p50 / p90 / p99，ms | [计时采集] `percentile()`，派生 | 排序后位置 `(n-1)*p/100`，相邻点线性插值；R 合并全部 repeat 的原始批次后重算，不平均各组分位数；这些是批次分位数 | `summary.latency_p50_ms/latency_p90_ms/latency_p99_ms` |
| total / 两种场景 | 样本/s、有效输入 token/s | [计时采集] `summarize()`，派生 | 样本处理次数/T、有效输入 token 数/T；包含实际尾批大小 | `summary.samples_per_second/tokens_per_second` |
| total / 对照 | 主耗时比、吞吐比及百分比变化 | [协调汇总] `performance()`；[组件报告] `overview()` | R 保存 `T_off/T_on` 与 `throughput_on/throughput_off`；报告“on 用时为 off 的几倍”取前者倒数，变化为 `(比值-1)*100%`；一次只比较一个组件 | R `comparison.off_over_on_measured_time/on_over_off_samples_per_second`；报告派生展示 |
| total / 分组 | shape / repeat 各组性能及 on/off 比 | [评估分析] `grouped_performance()`，派生 | 按 repeat/cycle/batch 配对，并核对样本数、token、shape、sample IDs；再调用 `summarize()`；缺形状不猜测，配对失败不生成分组比 | R `assessment.performance_groups.shape[]/repeat[]`：`paths`、`on_over_off_latency/on_over_off_throughput` |
| total / 跨重复组 | 均值 ± 样本标准差、最小–最大范围 | `statistics.mean/stdev`；[单卡报告]、[TP报告] `stat()/span()` | 从各完整组对应指标计算；SD 分母为 `n-1`，n=1 不适用；不是置信区间，也不是合并批次标准差；阶段、显存、传输和通信展示复用此规则 | 报告展示；输入来自 W/G/C 各组，不新增底层测量 |
| total / 单卡 | 模型 CPU 准备、模型总加载，s | `perf_counter_ns()`、`AutoModel.from_pretrained(...).eval()`；[计时采集]、[模型适配] | CPU 准备窗还包含 CPU 驻留检查与 storage 统计；总加载到 `.to(device)` 后同步结束，含分配与迁移；独立迁移见 §4 | W `model_cpu_load_seconds/model_load_seconds` |
| total / TP | 模型总加载，s | `perf_counter_ns()` 包围 `load_model()` 与末尾同步；[TP采集] | Transformers `tp_plan='auto'` 加载包含读取、分片与放置；每 rank 分列，无独立 CPU 加载或权重 H2D 时间 | W `model_load_seconds` |
| total / 两种场景 | 预热时间，s | `perf_counter_ns()`；[计时采集]、[TP采集] | 所有预热轮的 forward、同步及循环开销；单卡窗口还包含写 warmup 阶段记录；不属于正式时延 | W `warmup_seconds`；轮数来自 `effective.yaml` |
| total / 两种场景 | 完成/预期批次、失败数量、退出与超时 | 采集循环、`subprocess` 状态及 [TP采集] `aggregate()`；[协调汇总] | 预期 = 输入批次数 × 测量轮数。成功 worker 写 `failure_count=0`；单卡异常未统一输出失败批次数。TP 异常计数是 failed/missing rank 结果数，launcher 异常可能使运行失败而该数为 0；不能据此推出无失败 | W/G `completed_batches/expected_batches/failure_count`；失败时的 `failure_count_scope`、`status/exit_code/timed_out`；`stage.json`、`exit.json`、日志 |
| total / 运行审计 | 执行、分析、主测量与通信状态 | [协调汇总] `run()`；[评估分析] `assess()` | 执行与分析状态分别保存；`measurement_status` 仅按 summary 是否存在判断，不能替代整体成功判定；TP 通信需每侧每组 profile 完整，否则 incomplete，单卡为 not_applicable | R `execution_status/analysis_status/status`、`assessment.measurement_status/communication_status` |

以上采集均已接入；失败字段缺失时保留缺失，部分写出的 JSONL 仅作故障证据，不拼成“完整成功”的性能摘要。

## 3. 总 level：显存来源与阶段

`backend` 在 Ascend 为 `torch.npu`、NVIDIA 分支为 `torch.cuda`，由[设备接入]显式选择。单卡使用逻辑设备 0，TP 使用本地 rank。以下分配器 API 测的是当前进程的框架分配器；不是整张物理卡所有进程及运行库的总占用。

| level / 场景 | 指标与单位 | 来源 / API | 计算与边界 | 字段 / 状态 |
|---|---|---|---|---|
| total / 单卡、TP 每 rank | 权重及 buffer storage，B | `model.parameters()/buffers()`、`Tensor.untyped_storage()` 的 `data_ptr()/nbytes()`；[计时采集] `tensor_storage_bytes()` | 按设备类型、设备索引、storage 地址去重；统计实际模型张量 storage，TP 为本 rank；包含 buffer，不能简单称为参数文件大小 | W `memory.weight_storage_bytes`；已接入 |
| total / 单卡、TP 每 rank | 加载前/后 allocated、reserved，B | `backend.memory_allocated(device)`、`memory_reserved(device)`；[计时采集]、[TP采集] | 加载前与加载同步完成后各一次快照 | W `memory.before_model_load/after_model_load.{allocated_bytes,reserved_bytes}`；已接入 |
| total / 单卡、TP 每 rank | 加载后 allocated 增量，B | 上述两次快照相减；[单卡报告]、[TP报告] | `after_model_load.allocated_bytes - before_model_load.allocated_bytes`；可能含其它初始化分配，不是权重 storage 的另一份可相加内存 | 报告派生展示；已接入 |
| total / 单卡、TP 每 rank | 正式窗口基线、峰值、结束值，B | `memory_allocated/reserved`、`reset_peak_memory_stats`、`max_memory_allocated/reserved`；[计时采集]、[TP采集] | 预热后取基线并重置峰值；测量完成读取峰值与结束快照；峰值不覆盖加载、预热阶段 | W `memory.measurement_baseline/measurement_peak/measurement_end.{allocated_bytes,reserved_bytes}`；已接入 |
| total / 单卡、TP 每 rank | 峰值减基线，B | 分配器快照相减 | allocated/reserved 各自 `peak - baseline`；包含临时张量、工作区等，未独立归因纯 activation | W `memory.peak_increment_bytes.{allocated_bytes,reserved_bytes}`；已接入 |
| total / 静态模型 | KV cache 显存 | Qwen3 `use_cache=False`；DeepFM/DLRM 无自回归 cache | 不用 0 字节代替“不适用”；OneRec 的实际 KV 观测另见 §14 | 静态模型不适用 |
| total / 设备物理内存 | 分配器外内存、独立 activation/workspace 峰值 | 尚无对应独立采集 | `npu-smi` 快照见 §8，不能替代正式窗口内的进程归因与峰值 | 未采集 |

报告 B→KiB 除以 `1024`，B→MiB 除以 `1024²`。权重 storage 已占用 allocated 的一部分，不能与 allocated 相加；也不能跨 rank 累加峰值后称为某个时刻的全局峰值。

## 4. 总 level：数据传输

所有时间都是**主机发起复制至设备同步结束的窗口**，包含 API、分配及同步开销。逻辑字节说明复制对象规模；当前没有物理总线流量、独立 DMA kernel 时间或线路带宽计数。

| level / 场景 | 对象、方向 | 来源 / API | 范围、时间与字节口径 | 字段 / 状态 |
|---|---|---|---|---|
| total / 单卡 | 模型参数和 buffer，CPU→所选设备 | `model.to(device)`、`backend.synchronize()`、`perf_counter_ns()`；[计时采集] | CPU 建模后整模型迁移一次；逻辑字节来自迁移前去重 CPU storage，不含文件读取量或总线协议开销 | W `transfer.model_weights.seconds/logical_bytes`；已接入 |
| total / TP | 权重读取、分片及放置 | Transformers TP loader；[TP采集]、[模型适配] | 只记录总加载时间和加载后本 rank storage；不能用后者补写迁移字节，也不能用总时间补写 H2D 时间 | W `transfer.model_weights.status=not_separately_measured`；未独立采集 |
| total / 单卡、TP 每 rank | `input_ids`、`attention_mask`，CPU→所选设备 | 每个 tensor `.to(device)`，前后同步，`perf_counter_ns()`；[计时采集]、[TP采集] | 预热前每个准备好的 batch 拷贝一次，逐批窗口求和；逻辑 B = 所有输入 tensor 的 `numel()*element_size()` 之和；不是每个测量轮的传输总量 | W `transfer.input.seconds/logical_bytes`；已接入 |
| total / 单卡、TP 每 rank | 首个 batch 的 FP32 embedding，所选设备→CPU | `embedding.cpu()`、同步、`perf_counter_ns()` | 预热前额外跑一次首批 forward 并先同步，之后才开始复制计时；额外 forward 不计入复制时间；逻辑 B = embedding `numel()*element_size()` | W `transfer.output_embedding.seconds/logical_bytes`；已接入 |

单卡还保存 `objects/logical_link/physical_link/scope`；`physical_link=None` 表示未知。TP 的输入/输出记录只有 `seconds/logical_bytes/scope`，表中的对象和方向由当前采集代码确认，不能声称历史 JSON 存在相同元数据字段。TP 各 rank 的输入副本分别记录，不能把它们重复算入模型吞吐。

## 5. 总 level / TP：独立通信采样与模块归因

采集入口为 [TP采集] `run(profile=True)`：`torch_npu.profiler.profile()` 使用 CPU/NPU activities、`record_shapes=True`、`_ExperimentalConfig(profiler_level=ProfilerLevel.Level1)`，经 `tensorboard_trace_handler()` 输出文件。`schedule(wait=0, warmup=len(data), active=len(data)*communication_profile_rounds, repeat=1)`；预热步不计入 active collective 捕获。

以 Qwen3 为例，归因链为：Attention `self_attn.o_proj` / MLP `mlp.down_proj` 模块 hook → 被包装的 `torch.distributed.all_reduce` → `record_function` CPU marker → enqueue/dequeue correlation 或经核验的同线程直接 launch → CANN connection ID → HCCL 设备事件。其他模型按适配器声明的模块和 collective 类型关联，OneRec 与 DLRM 另见 §14–15。缺失、多义关联不按事件顺序猜配，分析输出 `partial`。即使模型组是 FlagCX，设备分析仍依赖实际出现的 CANN/HCCL 证据，不推定能解析任意通信后端。

| level / 场景 | 指标与单位 | 来源 / API | 计算与边界 | 字段 / 原始证据 |
|---|---|---|---|---|
| total / TP 采样、每 rank | 模型 collective 次数、类别、逻辑载荷 B | 模块 `register_forward_pre_hook/register_forward_hook`、包装 `dist.all_reduce`、`dist.get_backend()`、`tensor.numel()*element_size()`；[通信捕获] | 只记录 active 采样步；类别为 attention_output / mlp_output / unattributed；逻辑量不是 AllReduce 算法实际线路流量 | profile rank 目录 `collectives.json` 的 `category/module/collective/backend/elements/logical_bytes`；C `ranks.*.events[]` |
| total / TP 采样、每 rank | 预期/观测/已归因 collective 数 | [TP采集]、[通信分析] `analyze_rank()` | 当前模型预期 = active 批数 × 2 × block 数；与 wrapper、marker、HCCL、communication 条目及关联结果核对；前提是当前 Qwen3 TP 结构 | profile rank `result.json` 的 `profile_batches/expected_model_collectives`；C `ranks.*.expected_collectives/observed_collectives/attributed_collectives` |
| total / TP 采样、每事件 | 主机 API scope 与起点，ns | `trace_view.json` 中 collective marker 的 `dur/ts`；[通信分析] | trace 的 μs 通过 `Decimal(str(value))*1000` 转 ns；主机 API scope 不是设备执行时长 | C `ranks.*.events[].host_api_scope_ns/host_api_start_ns` |
| total / TP 采样、每事件 | 设备事件起点和时长，ns | `trace_view.json` 的 `hcom_*` 事件 `ts/dur` | 同样 μs→ns；保留 `correlation_id/connection_id/attribution_method` 以复核映射 | C `ranks.*.events[].device_start_ns/device_elapsed_ns` |
| total / TP 采样、每事件 | Elapse、Transit、Wait、Synchronization、Idle，ms | profiler 原始 `communication.json` → `Communication Time Info`；[通信分析] | 原样读取五个 `* Time(ms)` 字段；缺字段标不完整。Wait 与 Synchronization 可能重叠，不能直接相加；该 Transit 也不能与链路矩阵 Transit 混为一项 | C `ranks.*.events[].time` |
| total / TP 采样、每事件 | 起始时间 μs、Wait / Synchronization 比例等附带字段 | 同一 `Communication Time Info` 字典透传 | 当前示例含 `Start Timestamp(us)`、`Wait Time Ratio`、`Synchronization Time Ratio`；本工具未重算其分母，不把它们称为正式主时延占比 | C `ranks.*.events[].time`；已保留，非独立新增测量 |
| total / TP 采样、每类别每 rank | collective 次数、逻辑量、五类时间之和 | [通信分析] `categories`，派生 | 在同一 rank/category 内累加事件；任一必要值缺失，相应总值保留未知；事件时长之和不等于去重墙钟时间 | C `ranks.*.categories.*`：`collectives/logical_payload_bytes/device_elapsed_ms_sum/transit_time_info_ms_sum/wait_ms_sum/synchronization_ms_sum/idle_ms_sum` |
| total / TP 采样、传输类型 | 传输量 MB、时间 ms、带宽 GB/s、包比例和大小分布 | 原始 `communication.json` → `Communication Bandwidth Info`，字典透传 | 当前示例含 `Transit Size(MB)/Transit Time(ms)/Bandwidth(GB/s)/Large Packet Ratio/Size Distribution`，按 RDMA/HCCS/PCIE/SDMA/SIO 等原始键保存；不是硬件线路计数，本工具未重新定义包比例或分布 | C `ranks.*.events[].bandwidth_by_transport`；字段/单位沿原始文件 |
| total / TP 采样、有向链路 | 链路任务量 MB、时间 ms、transport | profiler `communication_matrix.json`；[通信分析] `matrix_totals()` | 仅取 `*-total@group`、`src_rank==当前rank` 的有向边，LOCAL 单列；不叠加 top/middle/bottom 与 total，也不把重复的 HCCS/SDMA 视图相加 | C `ranks.*.links[].src_rank/dst_rank/transport/transit_size_mb/transit_time_ms` |
| total / TP 采样、每 rank | 通信区间并集、计算区间并集、交集，ms | `trace_view.json`；[通信分析] `union_ns()/overlap_ns()` | 通信用已读取的 HCCL 事件；计算只筛 `Task Type` 为 AI_CORE、MIX_AIC、MIX_AIV、AI_VECTOR_CORE；区间去重；无计算事件时计算与交集为 None | C `ranks.*.intervals.communication_union_ms/compute_union_ms/communication_compute_overlap_ms` |
| total / TP 采样、跨 rank | 对应 collective 主机进入偏斜，ns | CPU marker 起点；[通信分析] `analyze()` | 各 rank 完整后按 sequence/cycle/batch/module/payload 配对，计算起点 max-min；同机诊断线索，不证明慢 rank 的根因 | C `rank_entry_skew[].host_entry_skew_ns` |
| total / TP 采样、报告汇总 | 区间范围、最大主机进入偏斜，ms | [TP报告] `communication()` 的 `span()/max()`，派生 | 区间范围为各完整 rank/采样组的最小–最大值；偏斜取完整进程组采样中所有已配对 collective 偏斜的最大值并 ns→ms；不是跨 rank 合并设备时钟 | 报告“同一采样窗口内的区间与到达差异”；原值在 C `ranks.*.intervals` 与 `rank_entry_skew[]` |
| total / TP 采样 | 归因覆盖、完整性及报告重复组统计 | [通信分析] 与 [TP报告] `coverage()/stat()` | 报告“已归因/预期”来自计数；多组资源/类别/链路按完整组展示均值、样本 SD、范围；部分采样保留 issues 与缺失状态 | C 的 `status/issues`、逐 rank 状态；报告派生展示 |

所有通信行均为已接入、依赖该次采样证据。原始文件在 `profiles/<side>/repeat-XX/rank-N/profiler/` 下，确切相对路径记录于 C `ranks.*.sources`。CANN 的 MB/GB 标记按原始文件保留，不能当作 KiB/MiB 擅自换算。透传字典可能随运行时版本变化，表中的已见字段不是新建固定 schema。

**不能跨窗口相除：** 同一次正式 batch 的时延与独立 profiler batch 的通信时间不是同一观测，不能相除作为通信占比；区间交集也不是关键路径节省时间。各 rank 耗时相加同样不能解释为模型墙钟耗时。

## 6. 调用级回退与组件参与审计

性能模式的这些取证在 A 中完成，不进入正式计时；accuracy/preview 的 forward worker 也保存对应证据。TP 需要保留每 rank 结果；汇总 Python 调用数可以跨 rank 累加，但不是模型样本数或设备 kernel 数。

| level / 场景 | 指标与单位 | 来源 / API | 计算与边界 | 字段 / 原始证据 |
|---|---|---|---|---|
| total 辅助 / preview | 设备顶层 ATen 种类、调用数 | `torch.utils._python_dispatch.TorchDispatchMode.__torch_dispatch__`；[执行取证] `Inventory` | 只统计涉及非 CPU tensor 的成功返回调用；跳过 FlagGems 包装函数内部嵌套调用；每种最多保留 32 个不同签名，签名数量不是全量分布 | `inventory.json`：`<aten_key>.calls/signatures`；种类数为键数 |
| 各模式 / FlagGems | 允许、注册、实际命中函数及调用次数 | `flag_gems.only_enable`、`GeneralOpRegistrar.register_impl` 包装；[执行取证] `GemsRoute` | 允许来自策略；包装在函数进入时累计，可包含随后失败的调用；“注册成功”与“被调用”分别记录。`gems-calls.jsonl` 每函数只写首次签名，不能用文件行数当总调用数 | `route.json` / worker `route.allowed/registered/actual_function_calls/observed/last_error`；`gems-calls.jsonl` |
| total 辅助 / FlagGems 开启侧 | 策略调用分类计数及占比 | [协调汇总] `route_summary()`：原生候选映射 + 本侧 inventory + policy + 函数命中 | 类别为 excluded、unverified、uncovered、allowed_function_observed、ambiguous；每类按 ATen calls 加权，占比分母是该侧全部顶层设备 ATen 调用 | R `route_summaries.<side>.counts/observed_top_level_aten_calls/details`；旧单轴结果可能为 `route_summary` |
| total 辅助 / FlagGems 开启侧 | 策略明确原生调用比例 | 同上，派生 | `(excluded + unverified + uncovered) / observed_top_level_aten_calls`；分母为 0 则 None；某函数被命中过不证明映射到它的每次 ATen 调用都走了 FlagGems | `route_summaries.<side>.policy_native_call_ratio` |
| total 辅助 | 硬件 kernel fallback 数量、比例、耗时占比 | 无对应逐 kernel 路由归因 | TP 存在通信/计算 trace 也没有建立 FlagGems fallback 的 kernel 映射；不能从 Python 调用分类推算 | `hardware_kernel_fallback_ratio=None`、`hardware_kernel_fallback_status=not_collected`；未采集 |
| 各模式 / 编译器 | JIT run、warmup、可选 launch hook 次数 | 包装 `triton.runtime.jit.JITFunction.run`；可选 `triton.knobs.runtime.launch_enter_hook`；[组件取证] | JIT 计数在原调用成功返回后增加；launch hook 仅在 API 可用时记录，否则 None；不是编译次数、编译耗时或硬件 kernel 覆盖率 | `components.json` 的 `compiler.jit_run_calls/jit_warmup_calls/launch_hook_calls/observation` |
| 各模式 / 模型通信组 | collective 次数、后端与逻辑 tensor 大小 | 包装 `dist.all_reduce/all_gather/all_gather_into_tensor/reduce_scatter_tensor`、`dist.get_backend()`；[组件取证] | 记录成功返回的 API 调用，排除控制组 barrier；大小由被记录 tensor 元素数×元素字节数得到，未推导算法网络流量；单卡通信参与为不适用 | `components.json` 的 `communication.model_collective_calls/collectives[].logical_bytes/observed_backends/status` |
| 对照 / 逐侧逐 rank | 组件是否实际参与 | [评估分析] `participation()` | 核对执行完成、无数值异常及对应函数/JIT/通信后端证据；得到 observed、not_observed、incomplete 等状态；preview 不评价组件收益，单卡 FlagCX 为不适用 | R `assessment.comparison.status/reason` |

编译器 provider、包版本、代码路径和通信组身份是上述计数的上下文，来自[组件适配]与 `components.json`，不能只靠开关值判断实际执行。当前未独立采集编译耗时、autotune 时间或缓存命中率。

## 7. 精度：model / layer

来源链：固定 `inputs.pt` → forward → 模型 pooled/embedding 或 `register_forward_hook` 捕获模块输出 → `.detach().cpu()` 与 `torch.save()` 封存 → [协调汇总] `records()` 排除层输出 padding 并转 FP32 NumPy → [数值指标] `metrics()` 转 CPU FP64 计算。模型输出不按 token mask 裁剪；TP 只在同 rank、同样本、同边界做 off/on 配对，不拼接分片或混合 rank。

下表适用于 model 的 pooled/embedding 及每个选定 layer 边界。除异常计数外，指标均无量纲；`a=off`、`b=on`。off 是所比较组件关闭时的同设备参照，未建立绝对真值或精度通过阈值。

| level | 指标 | 来源 / API | 公式 / 聚合范围 | D 字段 |
|---|---|---|---|---|
| model、layer | MSE、MAE、最大绝对误差 | `np.asarray(dtype=np.float64)`、`np.isfinite/abs/mean/max`；[数值指标] | 仅两侧都有限的元素对：`mean((b-a)^2)`、`mean(abs(b-a))`、`max(abs(b-a))`；无有限对时 None | `samples[]` / `aggregate.<boundary>` 的 `mse/mae/max_abs` |
| model、layer | 平均/最大相对误差 | `np.maximum`、`np.mean/max` | `abs(b-a)/max(abs(a),1e-12)` 后求均值/最大值；近零参照可放大相对误差 | `relative_mean/relative_max`；顶层 `relative_denominator_floor` |
| model、layer | 余弦相似度 | `np.linalg.norm`、`np.dot` | 完整向量归一化后点积；非有限元素、空 tensor、零范数或范数乘积溢出均返回 None 并记录原因；不以有限子集余弦代替完整向量 | `cosine_similarity/cosine_undefined_reason` |
| model、layer | 全局误差与逐样本余弦均值/最低值 | [数值指标] `compare_records()`、`np.concatenate/mean` | 全局先拼接同边界各样本有效位置的数组再重算指标，按元素而非样本等权；全局 cosine 是拼接向量 cosine；另对已定义的样本 cosine 求均值/min | `aggregate.<boundary>`；`sample_cosine_mean/sample_cosine_min/cosine_scope` |
| model、layer | 有效/总元素、参照零元素，个 | `np.isfinite`、布尔数组求和 | 有效数是双方有限的元素对数；总数是该次比较的全部元素；相对分母下限不改变参照零元素计数 | `valid_elements/total_elements/reference_zero_elements/statistics_scope` |
| model、layer | NaN、正/负 Inf 数；零范数和计算溢出 | `np.isnan/isposinf/isneginf`、范数及有限性检查 | 两侧分别计数；误差统计计算溢出的项改为 None；不等同于应用精度不达标 | `special.off/on.{nan,positive_inf,negative_inf}`、`zero_norm`、`overflow_metrics` |
| model、layer | 形状配对状态、最差样本与异常记录数 | shape 比较、指标排序；[数值指标]、[协调汇总] | shape 不同返回 `shape_mismatch`；各误差降序、cosine 升序取 `worst_count`；异常记录以样本×边界为单位，含余弦未定义与统计溢出，不是异常元素个数 | `worst.<boundary>.<metric>[]`、`anomalies[]`；R `pairing_status/numerical_anomalies` |

所有上述指标已接入。forward 的 `torch.isfinite()` 还产生 worker `numerical_anomalies[]`（样本/边界的非有限输出记录）；它与 D 的详细 NaN/Inf 元素计数和异常记录集合不是同一个统计量。逐层输出差异已可查看，但当前未实现自动“首个显著发散层”判定或误差因果归因。

## 8. Preview、环境与导出辅助来源

这些记录用于说明探测成本、证据完整性和运行条件，不能替代模型性能或精度指标。

| 场景 | 指标 / 状态 | 来源 / API | 计算与边界 | 产物字段 |
|---|---|---|---|---|
| Preview | 候选、接受、排除、未知、共同集合数量 | [执行取证] `candidates()` 以 inventory 与 `flag_gems._FULL_CONFIG` 相交；[续探] `finish()`；[评估分析] `coverage()` | 每环境分别保存；共同集合取接受集合交集。R `accepted` 为共同集合大小，`excluded/unknown` 为各环境列表长度之和，未跨环境去重；unknown 原始列表可含“单侧接受但未进入共同集合”，assessment 会单列此类 | `preview/policy.yaml` 的 `include/profiles.*.{candidate_count,accepted_include,excluded,unknown}`；R `accepted/excluded/unknown`、`assessment.policy` |
| Preview 续探 | 继承接受、该次新增接受、续探代数 | [续探] 检查点状态及集合差 | `new_include` 为当前接受中不在继承集合的项；继承覆盖不是该次新增探测收益；代数是恢复谱系，不是总试验次数 | `policy_profiles.*.inherited_include/new_include`、R `resume.generation` |
| Preview 轻量证据 | 每样本/边界检查规模与非有限值数量 | `torch.isfinite()`、CPU tensor 的 `numel/shape/dtype`；[preview_evidence](../../inference/runtime/preview_evidence.py) 校验完整性 | 完整输入与全部层仍执行；检查元素数为有效 token×hidden size（pooled/embedding 为 hidden size）。最多保存 16 个异常坐标，数量不截断；默认不保存完整输出 tensor，非精度阈值验收 | `forward-checks.json` 的 `batches[].checks[]`；worker `forward_evidence.sha256`；准备清单 `forward-inputs.json` |
| Preview unknown | 是否已启动、额度来源、最后阶段、细分原因 | [续探] 调度/失败状态；协调器与逐 rank 结果 | 未启动、估计成本放不进额度、总/搜索额度、worker timeout、资源故障、确认未完和证据不完整分别记录，不能混为算子失败；历史粗粒度原因不反推 | policy/checkpoint `unknown[].reason/attempted/limit_sources/last_stages`；trial `limits/limit_sources` |
| Preview worker 阶段 | 初始化、身份、加载、前向采集、检查、写证据与 TP 清理秒数 | `time.monotonic()`；[Stages](../../inference/runtime/preview_evidence.py) | 开始即原子落盘，结束记录 duration/status；异常终止缺失的时长显示未采集，另列缺失记录数，不补成 0。worker 阶段包含在协调器 process_seconds 中，不能重复相加；前向含编译和 CPU 采集，finite_checks 含末端输出 CPU 复制；TP rank 可能重叠，不加总为整体时长 | 每 worker `stages.json`；`budget.trials[].diagnostics.*.stages/last_stage` |
| Preview 协调与存储 | setup/process/postprocess 秒、输出 tensor/check 字节 | 协调器 monotonic 窗口、文件 `stat().st_size` | 环境资产检查是 setup 子集；进程时间含启动/退出。字节是逻辑文件大小，不是物理 I/O 流量；不含输入、日志和编译缓存 | trial `orchestration_timing`；`diagnostics.*.tensor_bytes/check_bytes` |
| Preview 续探缓存 | 导入完整组数、复制字节、导入/封存秒与跳过原因 | [CacheLedger](../../inference/runtime/preview_cache.py)；文件 SHA256、monotonic | 只从显式同身份来源复制完整组并改写路径；损坏组冷编译。导入单列准备成本、在探测预算之前；封存在探测预算内但不含于 trial invoke 秒数 | `preview/cache-import.json`；`budget.cache_import`、trial `cache_seal_seconds`；R `preparation_seconds.cache_import` |
| Preview 实际缓存访问 | 查找、完整读取、写组次数 | `FileCacheManager.get_group/put_group` 的 Python 钩子；[StackAudit](../../inference/runtime/stack_audit.py) | 完整读取要求索引中全部成员存在；不等于硬件 kernel 命中率，写组不等于精确编译次数；进程异常未生成 components 时缺失，不补 0 | worker `components.json.cache_groups`；trial `diagnostics.*.cache_groups` |
| Preview | 该次预算、已计费秒数、各阶段 trial 时间/次数/超时次数 | `time.monotonic()`；[续探] `Budget/trial/finish`；[单卡报告] `add_resume_evidence()` | 预算在 prepare 后开始；`charged_seconds=当前时刻-预算起点-cleanup`；trial 秒数为 invoke 整段耗时减超时清理，包括 worker 周转。报告按 phase 汇总该次 trials，不是 forward 净时间 | `preview/budget.json` 的 `seconds/charged_seconds/trials[].seconds/allowance/completed/timed_out` |
| Preview | 剩余/候选额度、预留与建议下次预算，s | [续探] `Budget.remaining/reserve/estimate/room`，计算/估算 | 当前 fixed 预留 `min(预算*25%,2*环境数*worker超时)`；成本取同阶段最近最多八次成功观测最大值×1.5，无同阶段时退到其它成功观测；超时 allowance 提供下限，估计最终受 worker timeout 限制；建议值是各环境必要阶段估计之和向上取整，不保证完成 | `preview/budget.json` 的 `mode/skipped[].estimated_seconds/estimated_minimum_next_budget_seconds`；剩余额度主要为运行内计算，未统一输出独立字段 |
| 各模式 / 准备 | 恢复校验/复制、prepare 时间，s | `time.monotonic()`；[组件编排] `run_stack()` | 包围恢复检查与轻量复制、环境输入准备和比较上下文生成；前一窗口在无恢复时也有很小的框架开销；不加入模型主时延 | R `preparation_seconds.resume_validation_and_copy/prepare` |
| worker / launcher | 进程墙钟、超时清理时间，s；退出码、超时标志 | `time.monotonic()`、`subprocess.Popen/wait`、超时进程组清理；[公共工具] `execute()` | 墙钟含进程启动、执行、等待及已发生的超时清理；cleanup 字段专指超时清理分支，不是正常进程组释放总耗时 | 对应目录 `exit.json` 的 `wall_seconds_diagnostic_only/cleanup_seconds/exit_code/timed_out`；协调器另投影 `worker_wall_seconds/cleanup_seconds`；预算累计清理在 `budget.json` |
| 环境辅助 / Ascend | 运行前后设备与进程快照 | 宿主 `subprocess.run(['npu-smi','info'])`；[宿主入口] | 保存原始 stdout+stderr，不做连续采样。TP 报告 `snapshot_processes()` 从可解析快照映射进程所在物理设备；未生成利用率/温度/功耗/显存时间序列或平均值 | `npu-before.txt/npu-after.txt`；[TP报告] 条件说明；无统一监控数值字段 |
| 输入辅助 | 输入条数、原始 token 长度范围、截断数、batch shape | `AutoTokenizer`、`tokenize()`；[模型适配]；报告计算 | 未截断 token IDs 决定原始长度，`len(ids)>max_length` 标记截断；实际批次 shape 取 tensor；不能把原始长度当正式 token 吞吐分子 | `prepared/samples.json` 的 `original_tokens/truncated`；`inputs.pt`、性能 `batches[].input_shape/sample_ids`；报告派生范围/计数 |
| export / FX | 重载误差、完全相等状态 | `torch.export.export/save/load`、`torch.equal`、复用 `metrics()`；[工作进程] `export_graphs()` | CPU 原生、首个准备 batch、当前 weight dtype；比较 pooled/embedding；误差公式同 §7，另要求 round-trip exact equality；不是设备性能或 FlagGems 路由证据 | `export/result.json` 的 `formats.fx.metrics/roundtrip`；`fx-nodes.json` 是图结构清单 |
| export / ONNX | 结构校验状态 | `torch.onnx.export`、`onnx.checker.check_model`、`onnx.load`；[工作进程] | 当前只检查结构，未运行 ONNX runtime 数值验证；节点清单不是实际设备 kernel 清单 | `formats.onnx.structural_check/numerical_runtime_validation`、`onnx-nodes.json` |

预算耗尽且必要复验未完成时不能发布 verified 策略；verified 的含义是当前输入下执行完成、所选函数逐 rank 命中且被检查边界有限，不代表精度阈值达标。schema 3 恢复证据包括历史继承项，阅读时必须区分该次 trials 与累计 checkpoint。

预算在 prepare 与可选缓存导入之后开始，导入计入准备成本。阶段与字节字段、模式开关及使用边界见[Preview 指南](preview.md)。历史运行没有这些字段时保持未采集，不能据旧 trial 墙时补推各阶段时间。

## 9. 尚未提供的指标与范围

| 请求范围 | 当前状态 | 已有数据不能替代它的原因 |
|---|---|---|
| 在线 request 调度、排队、服务 TTFT / TPOT | 未实现 | OneRec 已记录离线 prefill/首 token、decode 与完整生成时延（§14）；这些不含服务排队或调度，不能当作在线服务指标 |
| 逐算子时延与FX/ONNX图节点对应 | 未实现正式operator level | 新增layer模块窗口及kernel关联不等于已有算子图节点映射 |
| 纯 kernel 总时延、硬件 kernel fallback 比例及原因 | 未采集相应完整归因 | 主时延含主机及同步开销；Python dispatch 数不等于 kernel 数 |
| 总线物理字节、物理链路带宽、纯传输 kernel 耗时 | 未采集 | tensor 逻辑量、同步拷贝窗和 CANN 链路任务量的计量对象不同 |
| 硬件连续峰值、短 batch 独占温度/功耗/利用率 | 未提供连续或独占归因 | 已接入独立轮询遥测（§16）；采样极值不等于硬件连续峰值，查询区间不能分配为短 batch 独占消耗 |
| 单卡模型集合通信、静态模型 KV cache | 不适用 | 单卡执行无模型 collective；Qwen3/DeepFM/DLRM 不使用自回归 cache，OneRec 的 KV 观测见 §14 |

## 10. 核对该次结果

从 result.json / report.md 进入该次原始记录与明细，核对采集代码、字段、公式、单位及缺失语义。历史来源只读；报告重绘不会补造缺少的观测。

## 11. Layer 性能：独立采样、关联证据与额外指标

本节对应[逐层使用](layer.md)。`performance --level layer` 保存无插桩模型基准及以下独立采集阶段；基准仍使用 §2–6 的原口径。`total` 不自动增加层级采集。每次运行按具体配置、来源和覆盖状态解读。

简写：L 为 `layer/<phase>/<side>/repeat-XX/`，TP 再加 `rank-N/`；层原始记录在 `L/layers.jsonl`，批次记录在 `L/batches.jsonl`，worker 汇总在 `L/result.json`。全量索引为 `result.json.layer` / `layer/summary.json`；trace关联为 `L/attribution.json`。

| level / 指标 | 来源 / API | 采集边界、公式及字段 |
|---|---|---|
| layer：实际调用与完成状态 | module pre/post hook；[层级采集] `Scopes` | 每次调用记录module、call_index、rank、repeat、cycle、batch、sample IDs和输入shape；失败worker保留日志/阶段，不能从最后模块猜根因 |
| layer：主机调用窗口 ns | `time.perf_counter_ns()` | `host_end_ns-host_start_ns=host_ns`；模块调用窗口，不含批末补充同步 |
| layer：设备流窗口 ns | `torch.npu.Event` / `torch.cuda.Event` 的 `record/elapsed_time` | `device_ns=elapsed_time_ms*1e6`；批末统一同步后读取。起止流不同标为stream_mismatch，不静默降级为主机计时 |
| layer：均值、p50/p90/p99 ms | [层级统计] `statistics()`，复用线性插值percentile | 按module/rank/shape分组全部原始有效调用；`timing.groups[].paths.<side>.device/host`，不平均repeat分位数 |
| layer：调用/s、等效样本/s、有效token/s | 次数或对应工作量除以层窗口总秒数 | `calls_per_second/samples_per_second/tokens_per_second`；不是端到端吞吐，TP不累加各rank分子 |
| layer：本地权重storage bytes | parameters/buffers → `untyped_storage().nbytes()`，按设备/地址去重 | `modules.json[].weight_storage_bytes`；包含子模块，父子/共享权重之间不再相加 |
| layer：进出allocator、峰值、峰值增量、净变化 bytes | `memory_allocated/reserved`、`reset_peak_memory_stats`、`max_memory_allocated/reserved` | layer_memory阶段，非嵌套模块分组完整重放，边界同步；`before/after/peak/peak_increment/net_change`；进程窗口指标，不是层独占量 |
| layer：kernel/通信/拷贝事件数及时间 ns | `torch_npu.profiler` CPU/NPU活动、Level1、`trace_view.json`；[层级trace] | CPU模块record_function→flow端点/唯一connection_id→设备任务；`events[].event_index/owners/start_ns/duration_ns`；按调用保存时长和与区间并集 |
| layer：通信对象、量、时间与链路 | §5原始collective、communication.json、communication_matrix.json | `attribution.json.communication`；逻辑payload与CANN链路任务量不同；未能归属到层的链路字段保持rank级 |
| layer：普通拷贝方向、字节 | 设备trace明确提供的copy kind和size/bytes字段 | `events[].direction/bytes`；未知保留null/unknown；通信内部Memcpy不重复累计；模块输出大小不作为搬运量 |
| layer：FlagGems调用及策略原生比例 | 独立审计的模块作用域＋真实注册函数包装＋顶层TorchDispatch计数 | `layer.routes.<side>.<rank>.<module>`；包含子模块；classification分母为顶层设备ATen数。硬件kernel fallback未采集 |
| layer：off/on设备时间比 | [层级统计] `summarize_layer()` | 对齐repeat/cycle/batch/rank/module/call_index，并核对sample IDs/shape/工作量；`on_over_off_device_time=sum(on)/sum(off)`，无完整配对不生成比值 |
| layer：**额外**覆盖率 | 去重设备事件和唯一模块marker | `profiles.<side>.<repeat>.<rank>.coverage`的分母是计算kernel、collective整体窗口和可识别普通Memcpy；底层队列/SDMA/通信实现任务保存在auxiliary_events并另报auxiliary_coverage；profiler控制任务另计排除数，未知任务不丢弃 |
| layer：**额外**采样扰动 | 插桩批次与原基准相同逻辑输入配对 | `layer.overhead[].instrumented_over_baseline`；两次独立运行的窗口总时间比，包含环境波动，不自动扣减层时间 |
| layer：**额外**TP rank差异 | 同一逻辑层调用的各rank设备窗口 | `timing.rank_imbalance[]`保留max/min/spread；各设备时钟不拼接，最大局部窗口不是全局层时延 |
| layer：**额外**解析恢复状态与耗时 | [层级trace] `recover_trace()`；独立进程 `torch_npu.profiler.profiler.analyse`；进程墙钟 | 仅对唯一原始Ascend采集目录的副本导出一次，300秒超时；`profiler-recovery/recovery.json`保存原始文件集合/hash、进程状态及source_unchanged。没有重跑模型，解析耗时不进入性能指标；CANN导出需要驱动管理接口，普通report视图不触发导出 |

profile_memory不作为该次窗口峰值来源；显存来自独立allocator采样。Profiler时长与Event窗口不混为同一指标，且不从二者的差值推导“框架开销”。层选择有父子嵌套时，事件可包含在多条层记录中，但覆盖率分母按设备事件去重。跨rank、跨重放或跨不同来源的时间不直接相加。

[层级采集]: ../../inference/runtime/layer_capture.py
[层级统计]: ../../inference/runtime/layer_analysis.py
[层级trace]: ../../inference/runtime/layer_trace.py
[配置]: ../../inference/runtime/config.py
[计时采集]: ../../inference/runtime/performance.py
[TP采集]: ../../inference/runtime/tp.py
[模型适配]: ../../inference/benchmarks/qwen3_embedding/model.py
[设备接入]: ../../inference/vendors/device.py
[协调汇总]: ../../inference/runtime/coordinator.py
[评估分析]: ../../inference/runtime/assessment.py
[单卡报告]: ../../inference/runtime/reporting.py
[TP报告]: ../../inference/runtime/tp_reporting.py
[组件报告]: ../../inference/runtime/stack_reporting.py
[通信捕获]: ../../inference/runtime/communication.py
[通信分析]: ../../inference/runtime/communication_analysis.py
[执行取证]: ../../inference/inference_engine/pytorch.py
[组件取证]: ../../inference/runtime/stack_audit.py
[组件适配]: ../../inference/vendors/stack.py
[数值指标]: ../../inference/runtime/metrics.py
[续探]: ../../inference/runtime/preview.py
[组件编排]: ../../inference/runtime/stack.py
[公共工具]: ../../inference/runtime/common.py
[宿主入口]: ../../inference/runtime/host.py
[工作进程]: ../../inference/runtime/worker.py


### Layer使用体验派生视图

| 输出 | 来源与公式 | 范围 |
|---|---|---|
| 整模型on/off变化 | 完整配对批次的on时长和/off时长和 | 封存运行完整工作量，按输入shape另列 |
| 层观察范围 | 有效Event原始调用min/max | module/rank/shape/side |
| 独立重复波动 | 每个worker重复的层均值，再计算样本标准差 | 至少两次完整重复时展示标准差 |
| 按shape扰动 | 同repeat/cycle/batch/sample配对后，插桩模型时长和/无插桩时长和 | side/repeat/rank/shape；包含环境波动 |
| 关联设备事件均值 | 独立profiler各调用的已关联duration_sum取均值 | 与层计时、显存分别采集；计数和覆盖状态随详情展示 |
| 展示筛选 | 已有模块、逻辑rank、输入shape精确匹配 | 原始记录与整模型工作量保留 |
| 重新分析状态 | 原始采集副本的trace唯一性及模块/通信关联检查 | 新分析状态与来源执行状态分别登记 |

生成器为`runtime/layer_experience.py`，详细使用见[逐层指南](layer.md)。

## 12. A100 性能基线与跨厂商同规模比较

| 指标或证据 | 来源 / 原始字段 | 公式与边界 |
|---|---|---|
| A100 正式总体时延 | `runtime/performance.py` 的同步 `perf_counter_ns`；TP 为 `runtime/tp.py` 的逐 rank start/end | TP 取 `max(end)-min(start)`，逻辑样本只计一次；无逐层插桩和 Profiler |
| 均值与 p50/p90/p99 | `metrics.json.total.<side>[]` / 两端原始数值摘录的 `latency_ns` | 全部完整批次；分位数线性插值，不删除离群值，不是请求级服务尾延迟 |
| 样本 / 有效 token 吞吐 | `samples`、`tokens` 与同一批次的 `latency_ns` | 分子求和 / 时延求和；TP 不乘 rank 数，tokens 不含 padding |
| 独立重复波动 | 按 `repeat` 分组的均值 | 不同服务器不配对；重复均值的样本标准差，n=1 时未定义 |
| 层时延比较 | `metrics.json.layer.<side>[]` 的 `device_ns`；`runtime/layer_capture.py` 当前流 Event | 同 rank/module/shape 的包含式流窗口；不可跨层/跨 rank 相加为整体时延 |
| 框架显存比较 | `memory.<side>[].measurement_peak`；层 `peak_increment` | 同 rank、同 allocator 字段比较观察最大值；不同采样量影响观察峰值机会；不是整卡所有内存 |
| CUDA compute/copy/NCCL kernel | `*.pt.trace.json` 的 category、correlation、External id、ts/dur | `runtime/cuda_trace.py` 显式唯一关联到主机 launch 与模块 marker；不按时间邻近猜测 |
| CUDA 层事件覆盖 | `attribution.json.coverage` | 主机 user_annotation 与 GPU 镜像标记分开；多义关联保持 unknown/partial |
| NCCL 模型通信逻辑 payload | `collectives.json` 的元素数×element_size；CUPTI 参数作核验 | 每 rank 实际 all_reduce 输入；每采样逻辑批次归一化；不是物理链路流量 |
| NCCL 设备时间 | `nccl_kernel_duration_ns` / `nccl_kernel_ms_sum` | kernel 时间与 CANN collective elapsed 不同源，不给跨厂商时间倍率；wait/transit 保持不适用 |
| 被测 / A100 倍率 | `performance-comparison.json.paths` | 时延、吞吐、同来源内存各自计算；零分母/缺失为 null，不给通过阈值 |
| 固定输入及可比合同 | `manifest.json.contract`、`prepared/inputs.pt`、`forward-inputs.json` | 模型/输入内容、批次、dtype、语义、种子、TP/设备数、预热和计时协议严格一致 |
| 设备、线程和数值 profile | `provenance/identity.json`、`performance-environment.json` | 驱动、拓扑、CPU/affinity、依赖及实际 TF32/归约等值；不是额外性能指标 |
| 保存策略 | `retention.json` | 普通成功无异常任务按规则精简，参考生产保留现场；派生数值可读取，缺失 tensor/trace 时不能重新计算或解析 |

参考生产、消费和适用边界见 [A100 参考指南](references.md)。原同设备 off/on 指标与 A100 比较分别保存，不相互覆盖。

## 13. 诊断解释层（保留既有指标）

| 指标或证据 | 来源 | 口径与边界 |
|---|---|---|
| Kernel 热点 | 已保存 attribution.json 的计算事件 | 每来源事件 ID 去重，同路径/rank/筛选分母；模块上下文包含式，不相加为主计时 |
| 模块路由上下文 | layer.routes 的独立覆盖审计 | 全输入函数调用数；不声称与某个 kernel 一一对应 |
| 三方精度 | oracle-comparison.json 与独立 comparison.json | A100→off、A100→on、同设备 off→on；第三者不能由前两者相减 |
| 样本改善/退化 | 同 rank、同边界、同样本的指标 | 直接比较数值方向，不设通过阈值；缺失保持未定义 |
| 参考 RMS / 归一化 RMSE | CPU FP64 有限配对元素 | sqrt(mean(reference²))；误差 RMS / 参考 RMS；零参考未定义 |
| 绝对误差 p50/p90/p99 | 同一有限配对元素 | 线性插值精确分位数 |
| 近零参考 | abs(reference)≤1e-12 | 数量、占比及该处最大绝对误差；阈值沿用旧分母下限，不是容差 |
| 样本等权 / 元素权重 | 逐样本差分记录 | 有定义样本指标平均；有效元素数/全体有效元素数，保留两种口径 |
| 物理板卡/芯片映射 | 该次 npu-smi 前后快照 | 宿主运行时编号与板卡/芯片分别展示；缺失/冲突未知，不按编号猜测 |

旧差分缺少解释字段时展示“未记录”；有 tensor 和参考包才能重新计算。详细定义与使用见 [结果指南](results.md)。

## 14. OneRec 生成执行

本节是自回归工作负载的新增口径；静态 Qwen3 embedding 与 DeepFM 继续沿用其封存元数据。OneRec 默认固定 32 个新 token 的性能负载由 1 次 prefill 和 31 次 decode 构成，精度模式单独遵守 EOS。

| 指标/证据 | 来源与字段 | 范围 |
|---|---|---|
| 生成模式、phase/step、active、token、停止原因 | `execution.json.requests[].steps/samples`；模型 ExecutionSession | 每请求/每样本；prefill step 0，decode step 1..N-1 |
| 完整有限值覆盖 | `forward-checks.json` schema 2，`steps[].checks[]` | 逐有效样本/逐选定边界/逐步；包含分歧后尚未停止的步，轻量模式同样执行 |
| RMSNorm hidden/各层误差 | `comparison.json` / `oracle-comparison.json`；generation_analysis→metrics CPU FP64 | prefill 有效提示位置、decode 当前有效位置；按每对比较首次 token 分歧截断并包含分歧步 |
| prefill/decode 差分及逐步误差 | `generation.phase_comparisons`、`step_metrics`、`trajectories` | 保留各样本配对步数、两侧生成长度/停止原因；不同配对范围不计算改善/退化 |
| 完整生成时延 | W/G `batches[].latency_ns`，同步 host envelope | 驻留输入、空 KV、固定生成；包含 greedy 与控制回传；不含服务准备、独立检查和 profiler |
| prefill/首 token | `summary.*.generation.prefill_mean_ms/first_token_mean_ms` | 第一个生成 token 的同步模型窗口；不是服务 TTFT |
| decode 均值/分位数/速率 | `generation.decode_mean_ms/decode_p50_ms/decode_p90_ms/decode_p99_ms/decode_tokens_per_second` | 全部实测 decode 步窗口；可按 phase/step/KV 长度细分；短采样分位数仅描述 |
| 输入 token/s | `generation.input_tokens_per_prefill_second` | mask 有效输入 token / prefill 秒数；TP 逻辑输入计一次 |
| 生成 token/s | `generation.generated_tokens_per_second` 与总 `tokens_per_second` | 固定新 token 数 / 完整生成秒数；TP 不乘 rank 数 |
| KV 本地 storage | W `generation_cache.steps[].cache` | 首批输入的额外非计时重放，实际 keys/values storage 去重；含 padding，逐 rank 独立 |
| 阶段 allocator 峰值与增量 | W `generation_cache.steps[].allocator_peak/allocator_peak_increment` | 与 KV 同一次独立重放；含权重/临时量/工作区；KV 已包含，不能相加 |
| 阶段层窗口 | `layer.timing.groups[]` 及原始 rows 中 phase/generation_step/kv_length_before | 每模块/每 rank/每 shape/每阶段/每步/KV 长度；decode 同 shape 不混合不同位置 |
| LM head AllGather | `collectives.json` collective/input_logical_bytes/output_logical_bytes；C `events` | 完整词表 logits 聚合；与 attention/MLP AllReduce 分型关联，逻辑字节不是物理流量 |
| A100 生成精度/性能参考 | schema 4 核心/附件；protocol_schema_version=3 的 workload、step index、CPU tensors 和固定性能步数 | 包摘要与模型合同分别校验；旧 schema 1～3 按读协议兼容，具体设备和质量状态以参考包为准 |

生成总时延、阶段时延、层窗口、KV storage 与 allocator 互有包含关系，不能相加为新指标。使用与边界见[模型指南](models.md)。

OneRec TP 的 HCCL TensorList AllGather 另外记录 `backend_elements`、`backend_input_buffer_bytes`、`backend_input_padding_bytes` 与 `payload_relation`。它们由封存版本/设备、输入输出 tensor 描述（包括输出 shape 必须相同）和 CANN `count/data_type` 严格配对得到。
分析器只对核验条件匹配的 Torch-NPU 构建按 512 字节对齐解释本地 buffer；构建、设备或 tensor 形状不匹配时保留未知，见[通信分析器](../../inference/vendors/ascend/communication.py)。这是本地 runtime buffer 大小，不能直接作为 wire traffic；`logical_bytes` 仍是模型实际输入字节数。

## 15. DLRM 与变长推荐输入

DLRM 沿用已有数值公式、同步主机计时、独立层事件/内存/profile 与保存协议。与 DeepFM 共用推荐适配层，但输入增加每字段变长 ID 袋；CPU 离线参考根据封存布局校验，不加载 DLRM 代码。实际验证范围见[模型指南](models.md)与 support.json。

| 指标或契约 | API/计算与原始字段 | 解读边界 |
|---|---|---|
| 固定 ragged 输入 | `pack_samples` 用 torch.tensor 构造 dense/values/offsets；workload.input_layout、forward-inputs.input_tensors | dense 样本轴 0；values/offsets 样本轴 null；offsets[B+1] 包含末尾 values 长度，不是 padding |
| 输入完整性 | `runtime/input_layout.validate_tensors`；dtype/shape、isfinite、ID 范围、offsets 单调性与终值；模型 validate_batches 核对封存样本 | 离线 schema 2 消费使用同一布局检查；摘要完整不替代来源可信性 |
| 每批稀疏规模 | DLRMAdapter.batch_metadata；`Tensor.numel`、offsets 相邻差、min/max/count(0)；batches[].sparse_fields | indices 包含重复 ID；bag_lengths 按样本顺序；相同 dense shape 不表示相同 ID 工作量 |
| 空袋与重复 | sum EmbeddingBag；整字段空 values 返回零、torch.where 屏蔽空袋；重复 ID 分别求和 | 无补齐或哨兵 ID；模型完整前向窗口计入这些处理 |
| 模型精度边界 | hidden[B,last_top_width]；可选 probability[B]；原 CPU FP64 数值公式 | 默认 hidden 不覆盖输出头；概率 scalar cosine 结合幅度误差读取 |
| 分支模块边界 | bottom_mlp/embeddings/interaction/top 隐藏激活；workload.boundary_relations | bottom 与 embeddings 汇合后才进入 top，不据模块顺序推导逐层误差归因 |
| 总/层吞吐 | 样本处理次数除以对应同步前向/层设备窗口；samples_per_second | token/token/s 为 null；空袋样本照常计数；TP 只计一次全局样本 |
| TP 模型集合调用 | `recommendation.parallel`、通信 Capture/StackAudit；默认每 rank 每批 26 AllGather+5 AllReduce | embedding 按维度、MLP 按输入列真实分片；逻辑输出在各 rank 完整复制；调用数不等于设备通信时间 |
| 同步与归因 | 沿用 vendors.ascend.trace/communication 和独立 layer profile；实际 schema/版本严格匹配 | 未关联或不支持的 payload 合同保持 partial，不以代码支持代替设备证据 |

总级和逐层报告只汇总保存的 sparse_fields；完整逐字段、逐样本长度仍在 forward-inputs.json。samples/s 衡量整个固定推荐工作负载，不能外推到在线 CTR 服务、不同袋长度分布或训练质量。

## 16. 连续监控旁路

本节记录当前实现，按诊断深度分为通用与进阶指标；“通用”表示常用的监控需求，具体可用性仍由厂商 provider 决定。性能命令默认 `--monitor on`、`--monitor-interval-ms 1000`，最小允许间隔为 100 ms。轮询频率不代表传感器刷新频率，查询耗时较长时实际间隔会超过请求值。

采集指标定义见 [Ascend provider](../../inference/vendors/ascend/telemetry.py) 与 [Linux collector](../../inference/runtime/monitor.py)，派生统计见[离线分析](../../inference/runtime/monitor_analysis.py)，batch 关联展示见[时间轴生成器](../../inference/runtime/monitor_report.py)。接入、生命周期与扩展约定见[监控指南](monitor.md)。

### 16.1 通用监控指标

下表以当前 `ascend.dcmi.a3` 后端为准。设备指标属于选定芯片或其共享板卡，进程指标按 worker 记录，宿主指标覆盖整个宿主。DCMI 原始数值随标准单位值一起保存。

| 指标 / 字段 | 来源/API 与单位、换算 | 作用域、用途与解读边界 |
|---|---|---|
| 整体 NPU 活跃度：`ascend.npu.activity` | `dcmi_get_device_utilization_rate`，type 13；% | chip；观察原生 NPU activity 趋势，不表示峰值 FLOPS 达成率 |
| HBM 已用容量：`memory.used` | `dcmi_get_device_hbm_info.memory_usage × 2**20`；B | chip；设备级占用，包含其他进程，与框架 allocated/reserved 分开解释 |
| HBM 总容量：`memory.total` | `dcmi_get_device_hbm_info.memory_size × 2**20`；B | chip；该次 A3 ABI 原值标为 MB，按每单位 `2**20` B 换算 |
| HBM 可用容量：`memory.free` | 同一 HBM info 的 `(memory_size − memory_usage) × 2**20`；B | chip；由总量与已用量相减，不是独立传感器读数，也不保证某次分配一定成功 |
| 板卡功率：`power.draw` | `dcmi_get_device_power_info(card, 0, …)`，原值 × 0.1；W | card；A3 共享管理板卡域，每轮同域只采一次，可能包含未选中的兄弟芯片，不能按 rank 求和或解释为进程功率 |
| 计算频率：`clock.compute` | `dcmi_get_device_frequency`，type 7；MHz | chip；当前 AI Core 频率；频率变化本身不说明节流原因 |
| HBM 频率：`clock.memory` | `dcmi_get_device_frequency`，type 6；MHz | chip；当前 HBM 频率 |
| 芯片温度：`temperature.chip` | `dcmi_get_device_temperature`；degC | chip；芯片传感器温度，用于观察持续负载趋势 |
| HBM 温度：`temperature.memory` | `dcmi_get_device_hbm_info.temp`；degC | chip；HBM 传感器温度 |
| 设备健康：`ascend.health` | `dcmi_get_device_health`；原生 code | chip；0 表示健康，保留非零原生码，不自动推导故障根因 |
| worker CPU：`process.cpu` | `/proc/PID/stat` 的 user+system ticks；`Δticks / SC_CLK_TCK / Δ墙钟秒`；cores | process；1 表示一个核持续忙，多线程可大于 1；首次没有差分基准，值为 null |
| worker RSS：`process.rss` | `/proc/PID/stat` 的 resident pages × `SC_PAGE_SIZE`；B | process；包含共享页，不对多个 worker 的 RSS 去重求和 |
| 宿主可用内存：`host.memory.available` | `/proc/meminfo` 的 `MemAvailable × 1024`；B | host；整个宿主的可用内存，不是容器内存限额或 worker 配额 |
| CLI 降级容量占用率：`memory.capacity_used` | `npu-smi info -t usages` 的 `HBM Usage Rate(%)`；% | chip；仅 CLI 后端提供，保留圆整百分比，不反推 `memory.used` 或 `memory.free` |

物理设备通过 `dcmi_get_card_id_device_id_from_phyid` 映射到 card/chip，再关联 rank 与 power domain；不使用设备号除二等算术猜测。当前后端验证 A3 双芯片管理卡布局，换硬件/ABI 后需重新核对结构和单位。HBM 查询返回错误或 `memory_usage > memory_size` 时，该次 HBM 容量与温度记录均不可用。worker PID 通过 PID namespace、namespace PID 和进程 start ticks 对应宿主 `/proc`，避免容器编号差异或 PID 复用导致错误归属。

### 16.2 进阶监控指标

#### 硬件细分指标

| 指标 / 字段 | 来源/API 与单位 | 作用域、用途与解读边界 |
|---|---|---|
| AI Core 活跃度：`ascend.aicore.activity` | `dcmi_get_device_utilization_rate`，type 2；% | chip；保留厂商 AI Core 原生口径 |
| Cube 活跃度：`ascend.cube.activity` | 同一 API，type 14；% | chip；观察 Cube 活动趋势 |
| Vector 活跃度：`ascend.vector.activity` | 同一 API，type 12；% | chip；观察 Vector 活动趋势 |
| HBM 带宽活跃度：`ascend.hbm_bandwidth.activity` | 同一 API，type 10；% | chip；观察带宽活动趋势，与容量占用率不同，不据此反推读写 GB/s |

这些原生百分比保留 `ascend.*` 命名，不能直接相加，也不构成跨厂商统一的算力利用率。当前所有五种 activity（含通用指标中的 NPU activity）均配置 profiler 冲突屏蔽，规则见 §16.4。

#### 派生统计与已有计时的关联

| 指标 / 字段 | 来源与计算口径 | 解读边界 |
|---|---|---|
| 阶段关联：`phases`、`phase_statistics` | 公共 `worker_start/stage/worker_end` 事件；保存 worker、rank、action、stage、components、repeat、side，按查询区间与阶段重叠关联 | 按实体/rank 匹配；进程还需匹配同一 worker，host 可关联各 rank。跨阶段查询可能关联多个阶段，不可将阶段统计直接相加 |
| 样本数：`samples`、`valid_samples` | `series` 按 entity+metric 分组；`phase_statistics` 再按阶段分组。`samples` 计该组全部样本，`valid_samples` 仅计 `status=ok` 且 value 非 null 的样本 | 顶层 `samples` 为全部遥测记录条数，不是轮询轮数；轮数见 inventory 的 `iterations` |
| 采样均值与极值：`sample_mean`、`minimum`、`maximum` | 对该组有效值计算算术均值/min/max；无有效值时为 null | 采样点统计，不是时间加权均值或硬件连续峰值；健康码等枚举应按原生码读取，不能据均值推断健康程度 |
| 轮询间隔与空洞：`median_poll_interval_ms`、`max_poll_gap_ms` | 同组所有样本的 `read_start_ns` 排序，相邻差换算为 ms 后取 median/max；不足两点为 null | 包含值不可用的采样记录；表示查询间隔，不代表源读数已刷新或有效值覆盖间隔 |
| 不可用与冲突点数：`unavailable_samples`、`profiler_conflict_samples` | 顶层计数；前者统计派生视图中 status 非 ok 的记录，后者统计被 profiler 规则屏蔽的记录 | 冲突点包含在不可用点中，不能相加；原始文件保留原读数与错误原因 |
| 时钟一致性：`clock_aligned`、`max_clock_offset_difference_ns` | 比较事件与 collector 的时钟实现，以及各自 UTC−单调时间锚点的偏移差 | 偏移差门槛 50 ms；这是保守一致性检查，不是精度保证 |
| 正式阶段采样不足：`insufficient_resolution_phases` | 检查 action 为 `performance` 且 stage 为 `measurement` 的阶段；须存在至少一个 chip 级指标有 ≥2 个有效点 | 不足阶段列入数组并令监控 partial；不代表每项指标都有两个点，也不自动延长负载；当前此门槛不覆盖 `layer_timing/layer_memory` action |
| batch 时延：页面 `batch.latency` | 复用各 performance worker 的 `batches.jsonl` 中 `start_ns/end_ns/latency_ns`；时延为 `latency_ns / 1e6` ms | 原有同步主机计时，未由 monitor 重新测量；profiler batch 不纳入此面板，时钟不一致时不叠加阶段/batch |

### 16.3 后端选择与 CLI 降级能力

优先使用 DCMI；初始化或发现阶段遇到已处理的库加载、接口缺失或运行错误时尝试 `ascend.npu_smi.a3`，记录 `fallback_reason`。CLI 使用 `npu-smi info -m` 的标签映射物理设备，单条查询超时 3 秒；采集开始后不静默更换来源。两条路径均不可用时仍可保存主机采样，监控状态按证据标为 partial/failed。

| CLI 查询与原始标签 | 输出字段 / 换算 |
|---|---|
| `info -t usages -i <card>`：`NPU Utilization(%)` | `ascend.npu.activity`；% |
| 同上：`Aicore Usage Rate(%)`、`Aicube Usage Rate(%)`、`Aivector Usage Rate(%)` | 分别为 `ascend.aicore.activity`、`ascend.cube.activity`、`ascend.vector.activity`；% |
| 同上：`HBM Bandwidth Usage Rate(%)` | `ascend.hbm_bandwidth.activity`；% |
| 同上：`HBM Capacity(MB)`、`HBM Usage Rate(%)` | 分别为 `memory.total`（× `2**20` B）与 `memory.capacity_used`（%） |
| `info -t common -i <card>`：`Aicore curFreq(MHZ)`、`Temperature(C)` | 分别为 `clock.compute`（MHz）与 `temperature.chip`（degC） |
| 同上，chip 0 的 `NPU Real-time Power(W)` | `power.draw`；W，按 card 只记录一次 |

CLI 按标签解析并保存原始查询输出；字段缺失或无法解析时记录 null 与原因。当前 CLI 不提供 `memory.used`、`memory.free`、`clock.memory`、`temperature.memory`、`ascend.health`，不能视为与 DCMI 等量替换。Linux `/proc` 采集不依赖这两种设备后端。读取结果时同时检查 inventory 的 `metrics`、`unavailable`、首次探测及逐样本状态，不能把能力缺项当作零值。

当前 provider 未暴露 `power.limit`、`clock.throttle_reason`、`memory.read_bytes_per_second`、`memory.write_bytes_per_second`；这不表示硬件必然不支持。能量积分、通信链路吞吐、跨厂商统一 FLOPS 利用率及自动根因判断也不在本版输出内。

### 16.4 时间窗口、冲突与状态解释

- **查询窗口**：每点保存 `read_start_ns/read_end_ns`、原值、来源及状态；`source_timestamp_ns`、来源平均窗口和刷新周期未知时为 null。查询区间重叠只用于阶段相关性，不能将阶段均值分配给短 batch。阶段墙钟包含批次间隙，与累计 batch latency 分开解释。
- **时钟检查**：事件和 collector 使用 `perf_counter_ns`；时钟实现不一致、没有事件或 UTC 锚点偏移差超过 50 ms 时，监控为 partial，停止阶段关联及 batch 叠加。保留独立遥测曲线。
- **profiler 冲突**：当前按同 rank 的整个 `profile/layer_profile` worker 生命周期保守屏蔽全部 activity 字段，而非只屏蔽某个 kernel 窗口；原始返回值留在 samples，派生值为 null、曲线断线。若时钟无法对齐且存在 profiler 事件，则整段 activity 保守屏蔽。只识别该次事件，不检测其他任务的 profiler。
- **缺失与零值**：合法 0 保留；API 返回错误、非法百分比、非有限值或字段缺失记录为不可用。截断记录、collector 错误和阶段未闭环保留原因，缺失不填零。
- **完成状态**：无遥测样本为 `failed`；存在采集/时钟/阶段错误、没有有效设备点或存在非预期不可用值时为 `partial`；其余为 `completed`。CPU 首次差分、进程退出/不可访问和正常 profiler 冲突是允许的缺失原因，单独出现不会令监控 partial。能力缺项与这些预期缺失可与 completed 共存，仍须逐指标检查有效点数。
- **状态隔离**：关闭监控为 `not_requested`，不创建 collector/事件目录；结束处理自身异常可返回 `failed`。监控状态写入 `result.json.monitoring`，不改写性能执行/分析结论和退出码；页面生成状态单独写入 `monitor-report-status.json`。collector 心跳超时由 watchdog 终止，生命周期与状态见[监控指南](monitor.md)。

### 16.5 该次监控证据位置

| 文件 | 核验用途 |
|---|---|
| `monitor/inventory.json` | provider、驱动、设备/rank/功率域映射、指标定义、单位、能力缺项和首次探测 |
| `monitor/samples.jsonl`、`monitor/events/*.jsonl` | 完整原始遥测、查询窗口及 worker 阶段事件 |
| `monitor/summary.json` | 监控状态、全程/分阶段统计、时钟与冲突/缺失信息 |
| `monitor/manifest.json` | 监控文件 SHA-256 核验清单 |
| `monitor.html` | 共享时间轴上的实体、阶段、时间范围筛选与原始 batch 时延展示 |

普通 compact 保留监控轻量证据；离线重绘只消费已保存内容，链接式重绘不向来源 monitor 目录写文件，portable 模式物化证据。历史结果未采集的遥测不会被补造。
