# 指标来源矩阵：level × 指标 × 来源

[返回入门指南](../../inference/README.md) · [逐层使用](layer.md)

本文供需要核对计算口径的使用者查阅。先从结果报告确定指标，再按下表找到采集 API、公式和原始字段。实际采集是否完整以该次运行状态为准。

**阅读结论：total主性能来自同步主机计时和框架分配器；layer分别采集模块主机/Event窗口、独立显存和硬件trace；TP通信来自独立profiler重放；路由与组件参与来自Python调用取证；精度来自封存tensor的CPU FP64比较。** `npu-smi info`另存运行前后环境快照，未用于生成主时延或框架峰值显存。

## 1. level、状态与证据路径

| 模式 / level | 当前范围 | 阅读入口 |
|---|---|---|
| performance / total，单卡 | 固定输入串行批次的时延、吞吐、阶段、显存、传输和独立调用取证 | §2–4、§6 |
| performance / total，单机 TP | 一个逻辑批次的全局时间包络；逐 rank 资源；独立通信采样及模块归因 | §2–6 |
| performance / layer，单卡与单机 TP | 完整模型内的模块窗口、独立显存/trace/路由观测 | §10 |
| performance / request、operator | CLI 尚不接受这些 level；没有对应正式性能模式 | §9 |
| accuracy / model | 归一化前 pooled vector 和最终 embedding 的差分 | §7 |
| accuracy / layer | 选定模块边界输出；TP 在相同 rank 内配对 | §7 |
| preview / 组件审计 / 运行辅助 | 候选与策略状态、实际调用、预算、环境快照；不是新增性能 level | §6、§8 |
| export / CPU 原生复验 | FX 重载数值复验；ONNX 结构检查 | §8 |

性能的 `--level total|layer` 与精度的 `--levels model layer` 是两个接口，见[参数定义][配置]。`total` 测量整模型；`layer` 在完整模型中独立采集模块窗口。TP 通信归因不等于逐算子图节点映射。适用范围以 [support.json](../../inference/support.json) 和实际运行身份为准；NVIDIA 接口尚未完成实机验证。

本页的“已接入”表示代码存在实际采集/分析路径；每次运行仍需检查 `status`、原始产物与覆盖。**未采集、未独立采集、不适用、未实现、未定义和实测零值分别保留**。派生值必须有原始输入；估算不标成实测。配置值、身份摘要、错误文本和 shape/dtype 作为解释指标所需的上下文，不全部另立为数值指标。

下表中的路径均相对一次运行的输出根目录；`<side>` 是本次被比较组件的 off/on，而非恒指 FlagGems，具体组合在 `comparison-context.json`。

| 简写 | 原始 / 汇总产物 | 字段定位规则 |
|---|---|---|
| R | `result.json` | 最终汇总；`report.md` 是其与相邻证据的展示投影 |
| W | 单卡：`<side>/repeat-XX/result.json`；TP：`<side>/repeat-XX/rank-N/result.json` | 一次计时 worker；相邻 `batches.jsonl` 保存逐批数据 |
| G | TP：`<side>/repeat-XX/result.json`、`batches.jsonl` | 按 rank 合并的全局批次，W 仍保留 rank 本地数据 |
| C | `profiles/<side>/repeat-XX/communication.json` | 独立通信分析；逐 rank 字段在 `ranks.<N>`；R 的 `profiles` 也含该结果 |
| A | `audit/<side>/`，TP 再下钻 `rank-N/` | `inventory.json`、`route.json`、`gems-calls.jsonl`、`components.json`；辅助原生取证在该侧 `native/` |
| D | `comparison.json` | 精度；TP 下钻 `ranks.<N>`；原始 tensor 为 `<side>/[rank-N/]batch-XXXX.pt` |

矩阵中的 `*` 表示动态键、`[]` 表示列表元素，二者是文档定位记法，不是新增 JSON 字段。来源链接指向实现文件，表内给出 API 或函数名以便定位。完整采样规则以该次封存的 `source-code/`、`source-snapshot.json` 和 `effective.yaml` 为准。

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
| total / 当前模型 | KV cache 显存 | `model(..., use_cache=False)`；[计时采集]、[TP采集] | 当前执行未启用 KV cache，不用 0 字节代替“不适用” | 不适用；无 KV 显存测量字段 |
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

归因链为：Attention `self_attn.o_proj` / MLP `mlp.down_proj` 模块 hook → 被包装的 `torch.distributed.all_reduce` → `record_function` CPU marker → enqueue/dequeue correlation 或经核验的同线程直接 launch → CANN connection ID → HCCL 设备事件。缺失、多义关联不按事件顺序猜配，分析输出 `partial`。即使模型组是 FlagCX，设备分析仍依赖实际出现的 CANN/HCCL 证据，不推定能解析任意通信后端。

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
| Preview 续探 | 继承接受、本次新增接受、续探代数 | [续探] 检查点状态及集合差 | `new_include` 为当前接受中不在继承集合的项；继承覆盖不是本次新增探测收益；代数是恢复谱系，不是总试验次数 | `policy_profiles.*.inherited_include/new_include`、R `resume.generation` |
| Preview 轻量证据 | 每样本/边界检查规模与非有限值数量 | `torch.isfinite()`、CPU tensor 的 `numel/shape/dtype`；[preview_evidence](../../inference/runtime/preview_evidence.py) 校验完整性 | 完整输入与全部层仍执行；检查元素数为有效 token×hidden size（pooled/embedding 为 hidden size）。最多保存 16 个异常坐标，数量不截断；默认不保存完整输出 tensor，非精度阈值验收 | `forward-checks.json` 的 `batches[].checks[]`；worker `forward_evidence.sha256`；准备清单 `forward-inputs.json` |
| Preview unknown | 是否已启动、额度来源、最后阶段、细分原因 | [续探] 调度/失败状态；协调器与逐 rank 结果 | 未启动、估计成本放不进额度、总/搜索额度、worker timeout、资源故障、确认未完和证据不完整分别记录，不能混为算子失败；历史粗粒度原因不反推 | policy/checkpoint `unknown[].reason/attempted/limit_sources/last_stages`；trial `limits/limit_sources` |
| Preview worker 阶段 | 初始化、身份、加载、前向采集、检查、写证据与 TP 清理秒数 | `time.monotonic()`；[Stages](../../inference/runtime/preview_evidence.py) | 开始即原子落盘，结束记录 duration/status；异常终止缺失的时长显示未采集，另列缺失记录数，不补成 0。worker 阶段包含在协调器 process_seconds 中，不能重复相加；前向含编译和 CPU 采集，finite_checks 含末端输出 CPU 复制；TP rank 可能重叠，不加总为整体时长 | 每 worker `stages.json`；`budget.trials[].diagnostics.*.stages/last_stage` |
| Preview 协调与存储 | setup/process/postprocess 秒、输出 tensor/check 字节 | 协调器 monotonic 窗口、文件 `stat().st_size` | 环境资产检查是 setup 子集；进程时间含启动/退出。字节是逻辑文件大小，不是物理 I/O 流量；不含输入、日志和编译缓存 | trial `orchestration_timing`；`diagnostics.*.tensor_bytes/check_bytes` |
| Preview 续探缓存 | 导入完整组数、复制字节、导入/封存秒与跳过原因 | [CacheLedger](../../inference/runtime/preview_cache.py)；文件 SHA256、monotonic | 只从显式同身份来源复制完整组并改写路径；损坏组冷编译。导入单列准备成本、在探测预算之前；封存在探测预算内但不含于 trial invoke 秒数 | `preview/cache-import.json`；`budget.cache_import`、trial `cache_seal_seconds`；R `preparation_seconds.cache_import` |
| Preview 实际缓存访问 | 查找、完整读取、写组次数 | `FileCacheManager.get_group/put_group` 的 Python 钩子；[StackAudit](../../inference/runtime/stack_audit.py) | 完整读取要求索引中全部成员存在；不等于硬件 kernel 命中率，写组不等于精确编译次数；进程异常未生成 components 时缺失，不补 0 | worker `components.json.cache_groups`；trial `diagnostics.*.cache_groups` |
| Preview | 本次预算、已计费秒数、各阶段 trial 时间/次数/超时次数 | `time.monotonic()`；[预算调度] `Budget` 与[续探]的 trial/finish；[单卡报告] `add_resume_evidence()` | 预算在 prepare 与缓存导入后开始；`charged_seconds=当前时刻-预算起点-cleanup`；trial 秒数为 invoke 整段耗时减超时清理，包括 worker 周转。报告按 phase 汇总本次 trials，不是 forward 净时间 | `preview/budget.json` 的 `seconds/charged_seconds/trials[].seconds/allowance/completed/timed_out` |
| Preview | 剩余/候选额度、预留与建议下次预算，s | [续探] `Budget.remaining/reserve/estimate/room`，计算/估算 | 当前 fixed 预留 `min(预算*25%,2*环境数*worker超时)`；成本取同阶段最近最多八次成功观测最大值×1.5，无同阶段时退到其它成功观测；超时 allowance 提供下限，估计最终受 worker timeout 限制；建议值是各环境必要阶段估计之和向上取整，不保证完成 | `preview/budget.json` 的 `mode/skipped[].estimated_seconds/estimated_minimum_next_budget_seconds`；剩余额度主要为运行内计算，未统一输出独立字段 |
| 各模式 / 准备 | 恢复校验/复制、prepare 时间，s | `time.monotonic()`；[组件编排] `run_stack()` | 包围恢复检查与轻量复制、环境输入准备和比较上下文生成；前一窗口在无恢复时也有很小的框架开销；不加入模型主时延 | R `preparation_seconds.resume_validation_and_copy/prepare` |
| worker / launcher | 进程墙钟、超时清理时间，s；退出码、超时标志 | `time.monotonic()`、`subprocess.Popen/wait`、超时进程组清理；[公共工具] `execute()` | 墙钟含进程启动、执行、等待及已发生的超时清理；cleanup 字段专指超时清理分支，不是正常进程组释放总耗时 | 对应目录 `exit.json` 的 `wall_seconds_diagnostic_only/cleanup_seconds/exit_code/timed_out`；协调器另投影 `worker_wall_seconds/cleanup_seconds`；预算累计清理在 `budget.json` |
| 环境辅助 / Ascend | 运行前后设备与进程快照 | 宿主 `subprocess.run(['npu-smi','info'])`；[宿主入口] | 保存原始 stdout+stderr，不做连续采样。TP 报告 `snapshot_processes()` 从可解析快照映射进程所在物理设备；未生成利用率/温度/功耗/显存时间序列或平均值 | `npu-before.txt/npu-after.txt`；[TP报告] 条件说明；无统一监控数值字段 |
| 输入辅助 | 输入条数、原始 token 长度范围、截断数、batch shape | `AutoTokenizer`、`tokenize()`；[模型适配]；报告计算 | 未截断 token IDs 决定原始长度，`len(ids)>max_length` 标记截断；实际批次 shape 取 tensor；不能把原始长度当正式 token 吞吐分子 | `prepared/samples.json` 的 `original_tokens/truncated`；`inputs.pt`、性能 `batches[].input_shape/sample_ids`；报告派生范围/计数 |
| export / FX | 重载误差、完全相等状态 | `torch.export.export/save/load`、`torch.equal`、复用 `metrics()`；[工作进程] `export_graphs()` | CPU 原生、首个准备 batch、当前 weight dtype；比较 pooled/embedding；误差公式同 §7，另要求 round-trip exact equality；不是设备性能或 FlagGems 路由证据 | `export/result.json` 的 `formats.fx.metrics/roundtrip`；`fx-nodes.json` 是图结构清单 |
| export / ONNX | 结构校验状态 | `torch.onnx.export`、`onnx.checker.check_model`、`onnx.load`；[工作进程] | 当前只检查结构，未运行 ONNX runtime 数值验证；节点清单不是实际设备 kernel 清单 | `formats.onnx.structural_check/numerical_runtime_validation`、`onnx-nodes.json` |

预算耗尽且必要复验未完成时不能发布 verified 策略；verified 的含义是当前输入下执行完成、所选函数逐 rank 命中且被检查边界有限，不代表精度阈值达标。schema 3 恢复证据包括历史继承项，阅读时必须区分本次 trials 与累计 checkpoint。

预算在 prepare 与可选缓存导入之后开始，导入计入准备成本。选项及续探规则见[Preview 指南](preview.md)。历史记录没有的阶段字段保持未采集，不能从旧 trial 墙时推算。

## 9. 尚未提供的指标与范围

| 请求范围 | 当前状态 | 已有数据不能替代它的原因 |
|---|---|---|
| request 调度、排队、请求延迟、TTFT / TPOT | 未实现 | 当前是固定文本 embedding 批处理，没有在线请求或自回归生成计时 |
| 逐算子时延与FX/ONNX图节点对应 | 未实现正式operator level | layer模块窗口及kernel关联不等于已有算子图节点映射 |
| 纯 kernel 总时延、硬件 kernel fallback 比例及原因 | 未采集相应完整归因 | 主时延含主机及同步开销；Python dispatch 数不等于 kernel 数 |
| 总线物理字节、物理链路带宽、纯传输 kernel 耗时 | 未采集 | tensor 逻辑量、同步拷贝窗和 CANN 链路任务量的计量对象不同 |
| 连续温度/功耗/利用率、板卡全局显存峰值 | 未接入连续指标采集 | `npu-smi info` 原始前后快照不能恢复运行中的时间序列 |
| 任务评分、A100 Oracle 对比、算子/张量级精度诊断 | 未实现 | 当前为同设备组件 off/on 描述性向量差分；导出结构和 FX 重载不构成跨设备 Oracle |
| 单卡模型集合通信、当前模型 KV cache | 不适用 | 当前单卡执行无模型 collective，forward 显式 `use_cache=False` |

## 10. Layer 性能：独立采样、关联证据与额外指标

`performance --level layer` 保存无插桩整模型基准及以下独立采集阶段，基准沿用 §2–6 的口径。`total` 不自动增加层采集。操作步骤及报告阅读顺序见[逐层指南](layer.md)。

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
| layer：覆盖率（辅助） | 去重设备事件和唯一模块marker | `profiles.<side>.<repeat>.<rank>.coverage`的分母是计算kernel、collective整体窗口和可识别普通Memcpy；底层队列/SDMA/通信实现任务保存在auxiliary_events并另报auxiliary_coverage；profiler控制任务另计排除数，未知任务不丢弃 |
| layer：采样扰动（辅助） | 插桩批次与原基准相同逻辑输入配对 | `layer.overhead[].instrumented_over_baseline`；两次独立运行的窗口总时间比，包含环境波动，不自动扣减层时间 |
| layer：TP rank差异（辅助） | 同一逻辑层调用的各rank设备窗口 | `timing.rank_imbalance[]`保留max/min/spread；各设备时钟不拼接，最大局部窗口不是全局层时延 |
| layer：解析恢复状态与耗时（辅助） | [层级trace] `recover_trace()`；独立进程 `torch_npu.profiler.profiler.analyse`；进程墙钟 | 仅对唯一原始Ascend采集目录的副本导出一次，300秒超时；`profiler-recovery/recovery.json`保存原始文件集合/hash、进程状态及source_unchanged。没有重跑模型，解析耗时不进入性能指标；CANN导出需要驱动管理接口，普通report视图不触发导出 |

层窗口峰值不使用profiler的profile_memory；显存来自独立allocator采样。Profiler时长与Event窗口不混为同一指标，且不从二者的差值推导“框架开销”。层选择有父子嵌套时，事件可包含在多条层记录中，但覆盖率分母按设备事件去重。跨rank、跨重放或跨不同来源的时间不直接相加。

[层级采集]: ../../inference/runtime/layer_capture.py
[层级统计]: ../../inference/analysis/layer.py
[层级trace]: ../../inference/analysis/layer_trace.py
[配置]: ../../inference/runtime/config.py
[计时采集]: ../../inference/runtime/performance.py
[TP采集]: ../../inference/runtime/tp.py
[模型适配]: ../../inference/models/qwen3_embedding/model.py
[设备接入]: ../../inference/vendors/device.py
[协调汇总]: ../../inference/runtime/coordinator.py
[评估分析]: ../../inference/analysis/assessment.py
[单卡报告]: ../../inference/reporting/model.py
[TP报告]: ../../inference/reporting/parallel.py
[组件报告]: ../../inference/reporting/components.py
[通信捕获]: ../../inference/runtime/communication.py
[通信分析]: ../../inference/analysis/communication.py
[执行取证]: ../../inference/engines/pytorch.py
[组件取证]: ../../inference/runtime/stack_audit.py
[组件适配]: ../../inference/vendors/stack.py
[数值指标]: ../../inference/analysis/metrics.py
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

生成器见[层级视图](../../inference/reporting/layer_view.py)，使用与筛选说明见[逐层指南](layer.md)。

[预算调度]: ../../inference/runtime/preview_budget.py
