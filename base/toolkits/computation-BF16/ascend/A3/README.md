# Ascend A3 BF16 算力实验

本 Case 是单机全设备的厂商 microbenchmark，不是 PyTorch workload。支持基线为 CANN 9.0.0、MindCluster ToolBox/DMI 26.1.0 和 Atlas 800T A3（本项目目标服务器为 Ascend 910C）。

## 测量协议

```bash
ascend-dmi -f -t bf16 --all --et 80 -q --fmt json
```

DMI 在 AI Core 上执行厂商定义的 BF16 矩阵乘并按设备执行时间计算 `TFLOPS`。结果解析优先读取 JSON 中与 `BF16/TFLOPS` 对应的字段；文本回退必须先识别 `TFLOPS@BF16` 表头，不再读取固定第 4 行第 4 列。
JSON 数值的厂商原始字面量保存为 `value_raw`，报告原样显示；兼容数值字段不用于舍入或替换该原文。

计算命令运行时，runner 按本轮 `npu-smi map` 对目标物理 NPU 并行轮询
`npu-smi info -t usages`，逐 Chip 原样保存 AICore、AIVector、HBM Bandwidth 和 NPU
Utilization。每个 Chip 要求至少 10 个完整 workload 内样本；不足时最多追加一次相同 `--et 80`
负载。追加 DMI 值单独归档，不进入主 TFLOPS，也不做平均或峰值换算。

```mermaid
flowchart LR
  A[版本/兼容性] --> B[测试前健康]
  B --> C[BF16 --all + npu-smi 同窗采样]
  C --> D[原始 TFLOPS + monitor JSONL]
  D --> G[aiflops 诊断]
  G --> E[测试后健康]
  E --> F[manifest + 原始输出]
```

厂商诊断为 `ascend-dmi --dg --items aiflops`。完整命令、stdout、stderr、退出码、拓扑、逐设备指标和诊断引用保存在 `toolkit-evidence/manifest.json`；`[FlagPerf Result]` 仅用于兼容原聚合器。面向读者的主报告为 `report.md`；`report_monitor.md` 用静态 timeline 色块直接显示四字段原值，并以 `Sxx` 关联逐样本表。

## 运行

推荐统一运行入口：

```bash
python3 base/run.py toolkit run --case computation-BF16 --npu-ids 1 \\
  --allow-privileged-root --allow-disruptive-dmi
```

直接执行 `main.sh` 时须先设置 `FLAGPERF_ALLOW_DISRUPTIVE_DMI=1`。DMI 会占用设备；设备非空闲时 runner 拒绝启动。本轮只验证单机，不覆盖 HCCL 或跨机实验。
