# 昆仑芯 P800 Base — interconnect-P2P_intraserver

## 实现和范围

FlagCX/BKCL send/recv，恰好两个 rank，方向 rank 0→rank 1。rank 不等于物理卡号。正确性检查 payload/序列，外层超时和清理必须完整。

## 配置

默认配置是 smoke。历史资格 payload 256 MiB、3500 iterations，需独立 override 和 internal bridge。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-P2P_intraserver:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

单向带宽 = payload_bytes×ITERS/elapsed；不乘 2。每 run 取两个有效 rank 的较小带宽，再对五次取中位数。

历史实测与当前限制见 [公开验证说明](../../../../../docs/kunlunxin/validation.md#历史实测范围)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。单次结果不改变 runtime 声明或扩大验证范围。


## 静态计划

从仓库根目录执行；卡号仅为示例。

```bash
python3 base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case interconnect-P2P_intraserver:P800 --physical-device-ids 4,7 \
  --nproc-per-node 2 --timeout 300 --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
