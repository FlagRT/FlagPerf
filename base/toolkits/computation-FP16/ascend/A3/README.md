# Ascend A3 FP16 算力实验

本 Case 使用 MindCluster DMI 26.1.0 测量单机全设备的厂商 FP16 矩阵乘能力。它和 Base Torch FP16 的 shape、软件路径与计时边界不同，不能仅因单位相同就直接相除。

## 测量协议

```bash
ascend-dmi -f -t fp16 --all --et 80 -q --fmt json
```

DMI 输出单位为 `TFLOPS@FP16`。解析器按 JSON 字段或带单位的表头定位指标，允许前置告警和空行，不再固定读取第 4 行第 4 列。保存的 26.1.0 单设备样例 `0/1 ... 752.465 TFLOPS` 只用于 parser 回归，不作为整机标定值。
JSON 数值的厂商原始字面量保存为 `value_raw`，报告原样显示；兼容数值字段不用于舍入或替换该原文。

同一 DMI 进程窗口内按物理 NPU 并行执行 `npu-smi info -t usages`，原样保存逐 Chip
`Aicore Usage Rate(%)`、`Aivector Usage Rate(%)`、`HBM Bandwidth Usage Rate(%)` 和
`NPU Utilization(%)`。监控完整只表示每个目标 Chip 至少有 10 个完整样本且主 DMI 窗口有样本，
不表示达到理论峰值；AIVector 较低也不自动构成失败。

| 证据 | 用途 |
|---|---|
| DMI version/compatibility | 证明工具与软件栈身份 |
| pre/post health | 排除明显设备异常和测试后状态变化 |
| microbenchmark stdout/stderr/rc | 保留原始连续数值与失败信息 |
| aiflops diagnosis | 用厂商阈值判断算力健康 |
| npu-smi topology | 绑定设备与本机拓扑 |
| npu-smi usages JSONL | 同窗 AIC/AIV/HBM/NPU 原始时间序列 |

推荐运行：

```bash
python3 base/run.py toolkit run --case computation-FP16 --npu-ids 1 \\
  --allow-privileged-root --allow-disruptive-dmi
```

完整证据位于结果目录的 `toolkit-evidence/manifest.json`。本 Case 仅限单机；不运行压力测试、复位或跨机通信。若单次样本不足，runner 最多追加一次相同 `--et 80` 负载；追加 TFLOPS 不替换主结果。`report_monitor.md` 提供直接显示原值的逐 Chip 静态 timeline 和 `Sxx` 逐样本表；`--compute-monitor off` 仅用于开关 A/B。
