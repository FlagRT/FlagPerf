# Ascend A3 HBM 容量属性实验

本 Case 读取设备管理接口报告的 HBM 容量属性，不通过逐步分配内存寻找 OOM 边界，因此它不是“当前可分配显存”测试。

## 测量协议

runner 根据本机 `/dev/davinciN` 发现结果枚举 A3 card/chip 候选，并保存每次命令：

```bash
npu-smi info -t memory -i <card> -c <chip>
```

解析只匹配带名称的 `HBM Capacity(MB)` 字段。每个成功读取的 card/chip 都以 `scope=chip`、
逻辑 Device、原样 `MB` 数值、完整 stdout/stderr 和退出码进入 manifest；Case 级
`capacity_scope` 明确记录 `metric_scope=chip`、`aggregation=none` 和全部目标。同时运行 DMI
`hbm` 健康诊断，避免把容量属性等同于 HBM 健康。

当前实现已移除旧脚本的 `card0/chip0 ×2` 兼容标量和错误的 MiB 标签，逐 chip
`HBM Capacity(MB)` 是唯一权威容量结果。它不隐式报告 card 或整机容量；若后续需要更大 scope，
必须基于实时 map 枚举结果另行显式求和。完整分析见
`personal/Ascend-Base-Toolkit-D2D-and-capacity-x2-analysis.md`。

```bash
python3 base/run.py toolkit run --case main_memory-capacity --npu-ids 1 \\
  --allow-privileged-root --allow-disruptive-dmi
```

本 Case 仅限单机，不报告 allocator 碎片、业务可用容量或跨机容量。

容量是静态属性，不生成伪造的利用率 timeline。默认作证区保存每个目标的
`HBM Capacity(MB)`、时钟、温度等 `memory` 原字段，并补采 `npu-smi info -t ecc`
中的 ECC/隔离页原字段，关联本轮 pre/post HBM health。任何不支持项均保留原始命令、
退出码和输出并使监控为 `partial`，但不会修改已经成功取得的容量原值。
