# Kunlunxin P800 Base — interconnect-d2h

## 实现和范围

预分配设备源和主机目标，循环 Tensor.copy_；完整同步、计时外内容检查。Base 没有 Toolkit 的 1 MiB XRE 分块实现。

## 配置

512 MiB；四份 case_config.measured-*.yaml。pageable/nonblocking 为 ITERS 500、MODE measured，实际有短于 15 秒样本。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-d2h:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../docs/p800-base-handoff.md)。

## 指标和验收

按实际字节量和完整同步墙钟计时输出 GB/s 与 GiB/s；单位分别除以 1e9 和 2³⁰。

2026-10-08 结果：四变体功能测量通过。pageable/nonblocking 的三组短窗口由用户接受；idle/r2 稳定、初组仍 unstable。pinned-blocking 的 idle 通过，final 两组 unstable 保留。 这些数字仅属于固定镜像、设备集合与配置，不能当作所有设备/尺寸的保证。详情和逐组状态见 [验收复核](../../../../docs/p800-base-acceptance-20261008.md)。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。候选 manifest 不因本 case 通过自动晋级。
