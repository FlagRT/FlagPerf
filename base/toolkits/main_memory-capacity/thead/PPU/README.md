# 平头哥 PPU 显存容量实验

本 Case 在容器内运行 `ppu-smi -q` 读取每张选中卡的容量与状态，是静态查询：
不分配显存、不做 OOM 压力、不与 Benchmark 的分配/OOM Case 互替。

## 测量协议

- 逐卡解析 `ppu-smi -q` 输出，记录 Total/Used/Free（MiB）与 UUID。
- UUID 与宿主选择核对，不一致按失败点保留；Used=0 合法（空闲卡的正常状态）。
- 每 repeat 一次完整查询并逐卡归档原始 stdout。
- 指标行带 `capacity-query` 范围标注，监控层按静态查询语义检查
  （自身原始查询即观测，无并发负载要求），不计 10 样本窗。

## 运行与判定

先按 [Toolkit 总说明](../../../_common/thead/PPU/README.md#从宿主运行) 准备镜像、宿主权限与结果目录写权限。

```bash
bash base/toolkits/main_memory-capacity/thead/PPU/main.sh \
  --physical-device-ids 14 \
  --allow-candidate-runtime --repeat 5 \
  --result-root base/result/thead-toolkit
```

`main.sh` 将参数原样转发给宿主 facade `base/run.py toolkit run`。容量不受 samples、
warmup 等计时语义控制；`--smoke` 也不改变容量行为。测量层要求查询成功、UUID 一致、
数值为非负有限数；诊断层为 ppu-smi 健康观测。三层全部通过 Case 才为 passed。

完整契约、工具来源、证据目录与权限路径见 [Toolkit 总说明](../../../_common/thead/PPU/README.md)。
