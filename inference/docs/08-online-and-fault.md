> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书链接转入 [历史引用索引](history-references.md)，其出处保留。

# 08 在线通道、故障诊断与长稳

> 上游：[03 schema](03-event-schema.md) §4.11、[07 分析器](07-analyzers-and-reports.md) ｜ 需求：R32、R33、R77–R79
> 层级：**扩展**，在主线与 P5、P6 之后推进（[D20](DECISIONS.md#d20--分阶段推进主线先行)）。本文是设计储备：主线阶段只需不与之冲突，不提前实现。

## 1. 边界：工具做什么，不做什么

| 做 | 不做 |
|---|---|
| 采集故障信号（health 事件） | 决定集群调度策略 |
| 在规则引擎中**判定**卡死/OOM/通信中断/设备异常 | 直接杀进程、重启节点、摘除节点 |
| 通过 Responder 接口**通知**调度方，并给出证据与建议动作 | 替代 K8s/调度器/MindCluster 的恢复机制 |
| 在自己的 campaign 内部执行重试与跳过 | — |
| 消费已有事件做判定 | 承担统一运行时"监控诊断模块"本体，或定义运行时各组件的观测契约（[D21](DECISIONS.md#d21--范围边界暂不做与不做清单) B1） |

R77–R79 明确恢复动作由调度层负责。v2 的价值在于**可归因的判定**：不是"进程没响应了"，而是"rank 3 在 allreduce(group=G, seq=1721) 上等待 180 s，其余 7 个 rank 已到达，设备利用率为 0"。

## 2. 检测规则

规则读事件流（在线尾读或离线），输出 `health` 事件与 `Finding`。每条规则声明依赖的事件类型与参数。

| 规则 | 依据 | 判定 | 证据 |
|---|---|---|---|
| `hang.heartbeat` | `health(kind=heartbeat)` 周期缺失 > `k × period`，其中 `period` **取自 `meta.run_start.heartbeat_period_ns`**（被测方实际使用的值），不取规则配置的默认值 | 进程卡死或已死 | 最后心跳 ts、最后一个 op 的 corr 与 sid |
| `hang.no_progress` | 有心跳但 `op`/`opsum` 计数无增长，持续 > 阈值；设备侧事件在线可用时（见下），`kernel` 也无增长作为佐证 | 执行停滞（死锁、等锁、等设备） | 停滞起点、停滞时的栈顶 scope |
| `hang.device_idle` | `counter(util)` 持续为 0，但进程存活；设备侧事件在线可用时，无 kernel 作为佐证 | 设备空转 | counter 区间 |
| `comm.timeout` | 某 rank 在同一通信域内的 `group_seq` 落后于其他 rank，且领先的 rank 已在该序号上停留超过阈值（停留时长各 rank 用自己的时钟计，不做跨节点时间比较，[04 §4](04-correlation.md#4-时钟)）；`comm(phase=device)` 在线可用时，再加「enqueue 之后长时间无 device 事件」 | 通信中断/慢节点 | 落后的 rank 列表、group_seq |
| `oom.device` | `health(kind=oom)`，或 `counter(mem_used)` 触顶后出现 `error` | 显存不足 | 峰值前的 memop 归因（哪一层分配的） |
| `device.error` | 厂商遥测的 ECC/链路/温度告警，或驱动返回错误码 | 设备异常 | 原始错误码与 counter 轨 |
| `proc.exit` | `health(kind=process_exit)` 非正常退出码，或 writer 心跳中断而其片段 `state` 仍为 `active` | 进程异常退出 | 退出码、最后事件 |
| `perf.regression` | 长稳窗口内的 p50 耗时相对基线漂移超阈值 | 性能劣化 | 漂移窗口与对比基线 |

规则参数（周期、阈值）在配置中可调，默认值随厂商能力矩阵给出。**每条规则必须给出证据引用**，否则不允许产生 Finding（N6）。

**数据条件：事件类型存在 ≠ 在线可用。** CANN 的 `kernel`/`memop`/`comm(phase=device)` 由离线导入器在 profiler 结束后产出（[05 §4](05-probes-and-vendors.md)），在线尾读期间根本不存在。因此每条规则须声明：

- **在线依赖**：在线判定实际用到的事件类型。依赖设备侧事件的判定在 CANN 上**只能离线做**（事后复盘）；在线判定只用主机侧事件、`counter` 与 `health`。厂商若提供在线设备事件流，由能力矩阵声明，规则再启用对应的佐证条件。
- **水位**：规则引擎对每个来源已读到哪里（各 writer 的已读 `seq`、导入器已导入的范围）。规则只在水位以内判断「缺失」。
- **来源未启用或尚未导入，不等于设备没有完成。** 没有 kernel 事件，可能只是 vendor trace 没开、或导入器还没跑；规则不得据此推出「设备停滞」或「通信未完成」。

### 2.1 为什么判定不放在探针里

探针在目标进程内，进程卡死时探针也卡死 —— 由它判定卡死是矛盾的。因此探针只负责**发心跳与记录异常**，判定一律在进程外（supervisor 或离线分析器）。这是 R32 "识别卡死"能够成立的前提。

## 3. 在线尾读协议

store 的活跃分片是明文 JSONL 且逐行 flush（[03 §6](03-event-schema.md#6-store-布局manifest-与-seal-身份)），这使得在线消费不需要额外的 IPC 通道：

```text
supervisor（独立进程）
  1. 轮询 <run>/writers/ 下的 writer 片段，获知各 writer 的活跃分片
     （片段由该 writer 原子写入，读到的永远是完整版本，03 §6.2）
  2. 对每个活跃分片保持「已读记录数」，增量读新行
  3. 最后一行不完整则本轮跳过，下轮重读（不做部分解析）
  4. 分片内按 seq 检测跳变 → 确定的丢弃
     （取号与入队在同一临界区，单个 writer 的文件顺序就是 seq 顺序，03 §2）
  5. 事件喂给规则引擎 → Finding → Responder
```

**与封口的竞态**（[03 §6.4](03-event-schema.md) 的封口协议）：分片被封口后仍以**原文件名的明文**存在，直到后台压缩完成才改名删除。尾读方据此：

```text
  - 只有分片 sealed=true 且自己的已读记录数等于片段中的 records，才释放该分片
    （按记录数，不按时间戳：ts 可能重复，也可能为 null）
  - 读取中文件消失（已压缩改名）→ 按片段中的新 path 重新打开，跳过已读的前 N 条记录后继续；
    明文的字节偏移不能用于压缩后的文件；解码方式以片段中的 codec 为准
  - 重复读到的记录按身份去重：产生侧事件以 (writer_id, seq)，控制记录按内容键（03 §6.5）
  - 引用 sid 的事件之前，同一分片中一定已有对应的 sig（03 §4.2 的可解码不变式），
    故尾读方在任意分片处接入都能解码
```

writer 未正常关闭（片段 `state` 不是 `closed`）时，尾读方报告的丢弃量是下界，并标 `tail_unknown`（[03 §6.3](03-event-schema.md)）。

对外（R77–R79 的统一观测平台）提供两种出口，协议在 P3 冻结：

- **文件**：`<run>/findings.jsonl` 持续追加，格式稳定；
- **回调**：Responder 插件接口。

## 4. Responder 接口

```python
class Responder(Protocol):
    def on_finding(self, finding: Finding) -> None: ...
```

内置实现：

| Responder | 行为 | 使用场景 |
|---|---|---|
| `log` | 写 findings.jsonl | 默认，总是启用 |
| `exporter` | 推送到外部监控（Prometheus/webhook） | 对接 R77–79 |
| `campaign` | 在批量运行内部执行：标记该 run 失败、保留现场、重试、把该设备/节点加入黑名单跳过 | 500 模型批量扫描（R32 的"重试与隔离"落在工具自己的作业域内）。P5 先于 P3 交付（D20），P3 之前 campaign 只依据 run 的退出状态与超时重试/跳过；本 Responder 的基于 Finding 的隔离在 P3 接入 |
| `abort` | 终止本 run（仅当用户显式开启） | 交互式调试 |

`Finding` 结构：`{rule, sev, ts, scope, evidence[], suggested_action, run_id, rank}`。`suggested_action` 是**建议**（如 `retry` / `isolate_device` / `reduce_batch`），由调用方决定是否采纳。

**故障现场保留**：任何 `sev >= error` 的 Finding 触发保留策略切换（[03 §7](03-event-schema.md)）—— 全量事件与 `vendor_raw` 不被精简。

## 5. 热加载与版本切换（R33）

模型热更新的观测通过 `span(cat=lifecycle)` 表达：

| span name | attrs | 用途 |
|---|---|---|
| `model.load` | `{model_id, version, gen}` | 加载耗时、资源占用增量 |
| `model.swap` | `{from_gen, to_gen, strategy: blue_green/canary}` | 切换耗时、切换期间的请求影响 |
| `model.unload` | `{gen}` | 卸载后的资源回收是否完成 |
| `idle.release` | `{freed_bytes}` | 空闲资源释放 |

配套分析器 `lifecycle`：

- 每次切换的耗时与期间的吞吐跌落；
- **资源回收验证**：对比 `model.unload` 前后的 `counter(mem_used)`，若未回落到基线则标记疑似泄漏（这是热加载场景最常见的问题）；
- 灰度期间新旧代次的性能对比（按 `gen` 分组的 opsum/kernel 聚合）。

v2 只提供**观测与判定**，热加载的执行机制属于服务框架，不在本工具范围内。

## 6. 长稳运行

长期运行对 store 提出额外要求：

| 问题 | 处理 |
|---|---|
| 事件量无上限 | 分片轮转 + 按时间窗精简（[03 §7](03-event-schema.md#7-保留与压缩策略)）：老窗口只留 opsum/counter 聚合与全部 health。精简是显式步骤，窗口记入 `manifest.retention`，依赖逐事件的分析器在被精简的窗口上降级 |
| 探针开销累积 | 长稳场景默认用 `perf-lite` 预设（dispatch=summary） |
| 需要识别缓慢漂移 | `stability.drift` 分析器：按固定窗口计算 p50/p99 耗时、显存、通信时间的趋势，与首个稳定窗口比较 |
| 内存泄漏 | counter 的 mem_used 趋势 + memop 的净分配量趋势 |
| 碎片化 | 若厂商遥测提供碎片指标则纳入 counter，否则记为能力缺失 |

长稳的完整能力（自动基线、趋势告警）在 v2 第一轮之后，P3 只交付**可长期运行不爆盘 + 漂移可见**（[10](10-roadmap.md)）。
