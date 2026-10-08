# Kunlunxin P800 Base — interconnect-P2P_intraserver

## 实现和范围

FlagCX/BKCL send/recv，恰好两个 rank，方向 rank 0→rank 1。rank 不等于物理卡号。正确性检查 payload/序列，外层超时和清理必须完整。

## 配置

默认配置是 smoke。历史资格 payload 256 MiB、3500 iterations，需独立 override 和 internal bridge。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-P2P_intraserver:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../docs/p800-base-handoff.md)。

## 指标和验收

单向带宽 = payload_bytes×ITERS/elapsed；不乘 2。每 run 取两个有效 rank 的较小带宽，再对五次取中位数。

2026-10-08 结果：两卡最慢 rank 的五次中位数 31.1537 GB/s，CV 0.049%，严格通过。 这些数字仅属于固定镜像、设备集合与配置，不能当作所有设备/尺寸的保证。详情和逐组状态见 [验收复核](../../../../docs/p800-base-acceptance-20261008.md)。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。候选 manifest 不因本 case 通过自动晋级。
