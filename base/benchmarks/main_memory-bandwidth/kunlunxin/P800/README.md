# Kunlunxin P800 Base — main_memory-bandwidth

## 实现和范围

预分配 source/destination 的 Tensor.copy_；每次一读一写，字节量为 2×payload。分配与正确性检查在计时外；不是循环 clone 分配，不声称物理 HBM 峰值。

## 配置

Melements 1024（4 GiB float32 payload），warmup 2，ITERS 4200，MODE qualification。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `main_memory-bandwidth:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../docs/p800-base-handoff.md)。

## 指标和验收

按实际字节量和完整同步墙钟计时输出 GB/s 与 GiB/s；单位分别除以 1e9 和 2³⁰。

2026-10-08 结果：2235.9549 GB/s，CV 0.0040%，五次严格通过。 这些数字仅属于固定镜像、设备集合与配置，不能当作所有设备/尺寸的保证。详情和逐组状态见 [验收复核](../../../../docs/p800-base-acceptance-20261008.md)。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。候选 manifest 不因本 case 通过自动晋级。
