# Kunlunxin P800 Base — interconnect-MPI_intraserver

## 实现和范围

FlagCX/BKCL all_reduce SUM，后端 cpu:gloo,cuda:flagcx；provider 接受 2–8 rank，实际设备由 UUID join 绑定。非零输入用于计时外正确性，计时循环用零和系数避免多 rank 溢出。

## 配置

默认 YAML 为 4 MiB、2 iterations 的 smoke。历史五次资格为 256 MiB（Melements 64），两/六卡 3500 次，八卡 4200 次；用独立 override。需 internal bridge。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-MPI_intraserver:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../docs/p800-base-handoff.md)。

## 指标和验收

algbw = message_bytes×ITERS/elapsed；busbw = algbw×2×(world_size−1)/world_size。逐 run 取最慢 rank，不求和，不加额外 NVIDIA 展示系数。

2026-10-08 结果：2/6 卡严格通过，7 卡三次且含 ECC 卡 2；8 卡五次稳定、64.0284 GB/s，但健康 degraded，不是健康资格通过。 这些数字仅属于固定镜像、设备集合与配置，不能当作所有设备/尺寸的保证。详情和逐组状态见 [验收复核](../../../../docs/p800-base-acceptance-20261008.md)。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。候选 manifest 不因本 case 通过自动晋级。
