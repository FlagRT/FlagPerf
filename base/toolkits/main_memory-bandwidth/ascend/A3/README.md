# Ascend A3 D2D 主存带宽实验

本 Case 使用 DMI `d2d` 测量 Device 内部数据搬运带宽。权威结果是 manifest 中的原始逐设备 `GB/s` 序列，不再固定读取第 30 行。

## 测量与输出

默认先执行：

```bash
ascend-dmi --bw -t d2d -d <device> -q --fmt json
```

DMI 26.1 在 A3 D2D 模式下固定 size 和 execute-times，显式传入这两个参数会被拒绝。runner 因此先保存默认 Device 0 的完整序列，再对其余已发现逻辑设备逐一追加 `-d <id>`；任一设备不支持或缺失即记为 `partial`，不会把 Device 0 写成全机平均。

| 字段 | 含义 |
|---|---|
| `metrics[]` | DMI 原始带宽、逻辑 Device、`scope=logical-device`、单位和来源格式 |
| `bandwidth_scope` | `metric_scope=logical-device`、`aggregation=none`、所选 Device 集合和 DMI 来源 |
| `[FlagPerf Result]` | 逐逻辑 Device 原样输出 DMI GB/s，不四舍五入、不乘二 |
| `diagnosis` | DMI `bandwidth` 厂商诊断引用 |

后续核验确认：DMI 原始逐 Device 值才是权威测量；旧 `×2` 不能解释为一次单 Device 运行得到的
双 chip/card 实测带宽。当前实现已移除该倍率和 legacy 单标量，标准输出逐逻辑 Device 保留 DMI
原值。旧 README 的“8 卡平均”没有代码证据，已取消。完整分析见
`personal/Ascend-Base-Toolkit-D2D-and-capacity-x2-analysis.md`。

推荐运行：

```bash
python3 base/run.py toolkit run --case main_memory-bandwidth --npu-ids 1 \\
  --allow-privileged-root --allow-disruptive-dmi
```

结果同时保存版本兼容、pre/post health、拓扑、stdout、stderr、退出码和厂商 bandwidth 诊断。本轮仅限单机。

默认还在每条 D2D workload 窗口并行采集 `npu-smi info -t usages`，以厂商字段
`HBM Bandwidth Usage Rate(%)` 作为主要作证数据，`NPU Utilization(%)` 仅作同期上下文。
每个目标至少要求 10 个有效样本且主 DMI 窗口至少重叠 1 个样本；样本不足可原命令追加一次，
但追加 DMI 原值不并入主结果。不计算平均值、兑现率或理论峰值比例。可用
`--data-movement-monitor off` 关闭，此时监控状态为 `not-run`。
