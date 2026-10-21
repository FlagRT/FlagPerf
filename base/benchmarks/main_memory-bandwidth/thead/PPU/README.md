# 平头哥 PPU Base — main_memory-bandwidth

## 实现和范围

预分配 source/destination 的 Tensor.copy_；每次一读一写，字节量为 2×payload。分配与正确性检查在计时外；不是循环 clone 分配，不声称物理显存峰值。

## 配置

Melements 1024（4 GiB float32 payload），warmup 2，ITERS 4400，MODE qualification。

配置合并顺序为 generic < vendor < PPU < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `main_memory-bandwidth:PPU` 与 PPU host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

按实际字节量和完整同步墙钟计时输出 GB/s 与 GiB/s；单位分别除以 1e9 和 2³⁰。

历史实测与当前限制见 [公开验证说明](../../../../../docs/thead/validation.md)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。单次结果不改变 runtime 声明或扩大验证范围。


## 静态计划

从仓库根目录执行；卡号仅为示例。锁定 runtime 处于 candidate 状态，dry-run 与真实执行均显式 `--allow-candidate-runtime`。

```bash
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case main_memory-bandwidth:PPU --physical-device-ids 7 \
  --nproc-per-node 1 --timeout 300 --allow-candidate-runtime --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
