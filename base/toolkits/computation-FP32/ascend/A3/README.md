# Ascend A3 FP32 算力实验

本 Case 在当前 CANN 9.0.0 + ToolBox 26.1.0 基线上调用 DMI 的厂商 FP32 microbenchmark，目标范围是单机全部可见 Ascend 设备。

## 协议与判定

```bash
ascend-dmi -f -t fp32 --all --et 80 -q --fmt json
```

- 原始指标：DMI 报告的 `TFLOPS@FP32`，保留所有设备或 DMI 明确给出的 `all` 聚合值。
- 原值保真：JSON 数值字面量另存为 `value_raw` 并在报告原样显示，不做舍入或替换。
- 解析：JSON 优先；文本只按 `TFLOPS@FP32` 表头和数据行解析，不依赖行号。
- 诊断：运行 `aiflops` 厂商诊断并在 manifest 中建立引用。
- 监控：同窗按物理 NPU 并行轮询 `npu-smi usages`，逐 Chip 保存 AICore、AIVector、HBM
  Bandwidth、NPU Utilization 四个原始百分比。
- 成功：命令退出码为 0、指标为有限正数、设备覆盖明确且诊断通过。
- 不完整：设备覆盖无法证明或任一目标 Chip 少于 10 个完整监控样本时返回 `partial`/退出码 2，
  但有效 DMI 原值的 `measurement_status` 仍保持 `passed`。

```bash
python3 base/run.py toolkit run --case computation-FP32 --npu-ids 1 \\
  --allow-privileged-root --allow-disruptive-dmi
```

每轮同时保存兼容性、pre/post health、拓扑、完整 stdout/stderr、退出码和 SHA-256。本轮不处理 HCCL 和跨机实验。样本不足时最多追加一次相同 `--et 80` 负载，追加 TFLOPS 只进入监控证据；`report_monitor.md` 用静态 timeline 和逐样本表直接展示原值，不计算利用率均值、阈值或峰值兑现率。
