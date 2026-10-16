# 平头哥 PPU Base — interconnect-P2P_intraserver

## 实现和范围

厂商 NCCL send/recv，恰好两个 rank，方向 rank 0→rank 1。本机型实测没有卡间直连 P2P 路径，NCCL 经主机 staged 转发：对外数字描述 staged 路径，运行证据保留 `P2P=disabled` 标注，不作为直连带宽呈现。rank 不等于物理卡号。正确性检查 payload/序列，外层超时和清理必须完整。

## 配置

消息 256 MiB（Melements 64 float32），warmup 10，ITERS 340，恰好 2 rank。需 internal bridge。

配置合并顺序为 generic < vendor < PPU < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-P2P_intraserver:PPU` 与 PPU host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

单向带宽 = payload_bytes×ITERS/elapsed；不乘 2。每 run 取两个有效 rank 的较小带宽，再对五次取中位数。

历史实测与当前限制见 [公开验证说明](../../../../../docs/thead/validation.md)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。单次结果不改变 runtime 声明或扩大验证范围。


## 静态计划

从仓库根目录执行；卡号仅为示例。锁定 runtime 处于 candidate 状态，dry-run 与真实执行均显式 `--allow-candidate-runtime`。

```bash
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case interconnect-P2P_intraserver:PPU --physical-device-ids 7,8 \
  --nproc-per-node 2 --timeout 300 --allow-candidate-runtime --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
